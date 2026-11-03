"""Bounded central-directory preflight for pinned Torch file-object ZIP output.

Not a general ZIP reader: one disk, no prefix/comment/extensible ZIP64 sector,
ASCII archive/ names, stored records, and central extras limited to ZIP64.
Every read is fixed-size or checked against an independent metadata allowance.
"""

import struct

FIXED_NAMES = frozenset(
    (
        "data.pkl",
        ".format_version",
        ".storage_alignment",
        "byteorder",
        "version",
        ".data/serialization_id",
    )
)
END_BYTES = 98  # EOCD22 + ZIP64 locator20 + fixed ZIP64 EOCD56
MAX_NAME_BYTES = 128
MAX_EXTRA_BYTES = 28  # header4 + uncompressed/compressed/local-offset uint64


def record_limit(metadata_cap):
    # Independent object allowance, also enforced before save constructs the ZIP.
    return min(100_000, metadata_cap // 512)


def check_save_records(storage_count, metadata_cap):
    count = storage_count + len(FIXED_NAMES)
    if count > record_limit(metadata_cap):
        raise ValueError("ZIP directory record allowance exceeded before serialization")
    if count * (46 + MAX_NAME_BYTES + MAX_EXTRA_BYTES) > metadata_cap:
        raise ValueError("ZIP directory byte allowance exceeded before serialization")


def preflight(read_at, size, metadata_cap, storage_cap):
    """read_at(offset, count) must return exactly count bytes; no eager ZIP parser.

    Returns small directory facts only. Directory/name/extra sets die before the
    unmodified Torch restricted loader runs. metadata_cap independently limits
    data.pkl and total central-directory bytes; the latter includes all headers,
    names, extras and comments (comments are unsupported).
    """

    def read(offset, count):
        if count > max(MAX_NAME_BYTES, MAX_EXTRA_BYTES, 56) or offset < 0 or offset + count > size:
            raise ValueError("ZIP bounded read extent invalid")
        value = read_at(offset, count)
        if len(value) != count:
            raise ValueError("ZIP truncated bounded read")
        return value

    if size < 22 or metadata_cap <= 0:
        raise ValueError("ZIP end record missing")
    end_at = size - 22
    end = struct.unpack("<4s4H2IH", read(end_at, 22))
    signature, disk, directory_disk, disk_count, count, length, offset, comment = end
    if signature != b"PK\x05\x06" or disk or directory_disk or comment:
        raise ValueError("ZIP requires uncommented single-disk end record")
    directory_end = end_at
    sentinel = count == 65535 or disk_count == 65535 or length == 0xFFFFFFFF or offset == 0xFFFFFFFF
    if not sentinel and length > metadata_cap:
        raise ValueError("ZIP directory byte allowance exceeded before read")
    locator_at = end_at - 20
    locator = read(locator_at, 20) if locator_at >= 0 else b""
    has_zip64 = locator[:4] == b"PK\x06\x07"
    if has_zip64:
        _, disk64, end64_at, disks = struct.unpack("<4sIQI", locator)
        if disk64 or disks != 1 or end64_at + 56 != locator_at:
            raise ValueError("ZIP64 locator extent/disk mismatch")
        values = struct.unpack("<4sQ2H2I4Q", read(end64_at, 56))
        sig64, record_size, _, needed, disk64, directory_disk64, disk_n64, n64, size64, off64 = (
            values
        )
        if sig64 != b"PK\x06\x06" or record_size != 44 or needed > 45 or disk64 or directory_disk64:
            raise ValueError("ZIP64 fixed end record unsupported")
        for small, large, marker in (
            (disk_count, disk_n64, 65535),
            (count, n64, 65535),
            (length, size64, 0xFFFFFFFF),
            (offset, off64, 0xFFFFFFFF),
        ):
            if small != marker and small != large:
                raise ValueError("ZIP64 end declarations disagree")
        disk_count, count, length, offset = disk_n64, n64, size64, off64
        directory_end = end64_at
    elif sentinel:
        raise ValueError("ZIP64 sentinel requires locator")
    # Critically these tests precede any central-directory record/body read.
    if length > metadata_cap:
        raise ValueError("ZIP directory byte allowance exceeded before read")
    if not 0 < count <= record_limit(metadata_cap):
        raise ValueError("ZIP directory record allowance exceeded before read")
    if disk_count != count or length < count * 46 or offset + length != directory_end:
        raise ValueError("ZIP directory count/extent mismatch")
    position, names, fixed, first_local = offset, set(), set(), offset
    for _ in range(count):
        if position + 46 > directory_end:
            raise ValueError("ZIP directory record extent mismatch")
        fields = struct.unpack("<4s6H3I5H2I", read(position, 46))
        (
            sig,
            _,
            needed,
            flags,
            method,
            _,
            _,
            _,
            compressed,
            unpacked,
            name_n,
            extra_n,
            comment_n,
            volume,
            _,
            _,
            local,
        ) = fields
        if sig != b"PK\x01\x02" or needed > 45 or flags & ~0x808 or method or volume:
            raise ValueError("ZIP directory record format unsupported")
        if not 0 < name_n <= MAX_NAME_BYTES or extra_n > MAX_EXTRA_BYTES or comment_n:
            raise ValueError("ZIP name/extra/comment allowance exceeded before read")
        record_end = position + 46 + name_n + extra_n
        if record_end > directory_end:
            raise ValueError("ZIP directory variable extent mismatch")
        name = read(position + 46, name_n).decode("ascii")
        extra = read(position + 46 + name_n, extra_n)
        position = record_end
        if not name.startswith("archive/") or name in names:
            raise ValueError("ZIP name/prefix/duplicate unsupported")
        names.add(name)
        key = name[len("archive/") :]
        if key in FIXED_NAMES:
            fixed.add(key)
            limit = metadata_cap if key == "data.pkl" else 128
        elif key.startswith("data/") and key[5:].isascii() and key[5:].isdecimal():
            limit = storage_cap
        else:
            raise ValueError("ZIP record name unsupported")
        needs = [unpacked == 0xFFFFFFFF, compressed == 0xFFFFFFFF, local == 0xFFFFFFFF]
        if extra:
            if len(extra) != 4 + 8 * sum(needs):
                raise ValueError("ZIP64 central extra extent mismatch")
            tag, extra_size = struct.unpack("<HH", extra[:4])
            if tag != 1 or extra_size != len(extra) - 4:
                raise ValueError("ZIP central extra unsupported")
            cursor, resolved = 4, []
            for value, needed64 in zip((unpacked, compressed, local), needs, strict=True):
                if needed64:
                    value = struct.unpack_from("<Q", extra, cursor)[0]
                    cursor += 8
                resolved.append(value)
            unpacked, compressed, local = resolved
        elif any(needs):
            raise ValueError("ZIP64 central sentinel missing extra")
        if compressed != unpacked or unpacked > limit:
            raise ValueError("ZIP pickle/storage/record size allowance exceeded")
        if local + 30 + name_n + compressed > offset:
            raise ValueError("ZIP local record extent exceeds payload area")
        first_local = min(first_local, local)
    if position != directory_end or fixed != FIXED_NAMES or first_local != 0:
        raise ValueError("ZIP directory count/required records mismatch")
    return {"bytes": length, "records": count, "zip64": has_zip64}


class DirectoryTail:
    """Bounded counting-pass tail, never a whole-archive buffer."""

    def __init__(self, metadata_cap):
        self.limit = metadata_cap + END_BYTES
        self.data = bytearray()
        self.size = 0

    def write(self, data):
        # Slice a view, not data, so a large storage callback is never copied.
        tail = memoryview(data)[-self.limit :]
        excess = max(0, len(self.data) + len(tail) - self.limit)
        if excess:
            del self.data[:excess]
        self.data.extend(tail)
        self.size += len(data)

    def read(self, offset, count):
        start = offset - (self.size - len(self.data))
        if start < 0 or start + count > len(self.data):
            raise ValueError("ZIP directory outside bounded counting tail")
        return bytes(memoryview(self.data)[start : start + count])
