"""Stdlib reader for flat and one-level-list parquet columns (strings, int32/int64).

Enough for the pinned Hugging Face GSM8K and MMLU-Pro parquet files (Arrow
writer, Snappy, PLAIN or dictionary encoding) without installing pyarrow. Anything else
(nested schemas, other codecs or encodings) raises instead of guessing.
"""

import struct
import zlib

UNCOMPRESSED, SNAPPY, GZIP = 0, 1, 2
PLAIN, PLAIN_DICTIONARY, RLE_DICTIONARY = 0, 2, 8
DATA_PAGE, DICTIONARY_PAGE, DATA_PAGE_V2 = 0, 2, 3
INT32, INT64, BYTE_ARRAY = 1, 2, 6
REQUIRED, OPTIONAL, REPEATED = 0, 1, 2


def _varint(data, pos):
    shift = result = 0
    while True:
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def snappy_decompress(data):
    size, pos = _varint(data, 0)
    out = bytearray()
    while pos < len(data):
        tag = data[pos]
        pos += 1
        kind = tag & 3
        if kind == 0:  # literal
            length = tag >> 2
            if length >= 60:
                extra = length - 59
                length = int.from_bytes(data[pos : pos + extra], "little")
                pos += extra
            length += 1
            out += data[pos : pos + length]
            pos += length
            continue
        if kind == 1:
            length, offset = ((tag >> 2) & 7) + 4, ((tag >> 5) << 8) | data[pos]
            pos += 1
        elif kind == 2:
            length, offset = (tag >> 2) + 1, int.from_bytes(data[pos : pos + 2], "little")
            pos += 2
        else:
            length, offset = (tag >> 2) + 1, int.from_bytes(data[pos : pos + 4], "little")
            pos += 4
        if offset == 0 or offset > len(out):
            raise ValueError("invalid snappy copy offset")
        for _ in range(length):  # overlapping copies are byte-by-byte by definition
            out.append(out[-offset])
    if len(out) != size:
        raise ValueError("snappy length mismatch")
    return bytes(out)


def _zigzag(n):
    return (n >> 1) ^ -(n & 1)


def _read_value(data, pos, kind):
    if kind in (1, 2):
        return kind == 1, pos
    if kind == 3:
        return struct.unpack_from("<b", data, pos)[0], pos + 1
    if kind in (4, 5, 6):
        n, pos = _varint(data, pos)
        return _zigzag(n), pos
    if kind == 7:
        return struct.unpack_from("<d", data, pos)[0], pos + 8
    if kind == 8:
        n, pos = _varint(data, pos)
        return bytes(data[pos : pos + n]), pos + n
    if kind in (9, 10):
        header = data[pos]
        pos += 1
        size, elem = header >> 4, header & 0x0F
        if size == 15:
            size, pos = _varint(data, pos)
        items = []
        for _ in range(size):
            if elem in (1, 2):  # bools inside lists are one byte each
                items.append(data[pos] == 1)
                pos += 1
            else:
                item, pos = _read_value(data, pos, elem)
                items.append(item)
        return items, pos
    if kind == 12:
        return read_struct(data, pos)
    raise ValueError(f"unsupported thrift type {kind}")


def read_struct(data, pos):
    """Thrift compact struct -> {field_id: value}, and the end offset."""
    fields, last = {}, 0
    while True:
        header = data[pos]
        pos += 1
        if header == 0:
            return fields, pos
        delta, kind = header >> 4, header & 0x0F
        if delta:
            field = last + delta
        else:
            n, pos = _varint(data, pos)
            field = _zigzag(n)
        fields[field], pos = _read_value(data, pos, kind)
        last = field


def rle_hybrid(data, bit_width, count):
    values, pos, width = [], 0, (bit_width + 7) // 8
    while len(values) < count:
        header, pos = _varint(data, pos)
        if header & 1:
            groups = header >> 1
            chunk = int.from_bytes(data[pos : pos + groups * bit_width], "little")
            pos += groups * bit_width
            mask = (1 << bit_width) - 1
            values.extend((chunk >> (i * bit_width)) & mask for i in range(groups * 8))
        else:
            value = int.from_bytes(data[pos : pos + width], "little")
            pos += width
            values.extend([value] * (header >> 1))
    return values[:count]


def plain_byte_arrays(data, count):
    out, pos = [], 0
    for _ in range(count):
        (n,) = struct.unpack_from("<I", data, pos)
        out.append(bytes(data[pos + 4 : pos + 4 + n]))
        pos += 4 + n
    return out


def plain_values(data, count, physical):
    if physical == BYTE_ARRAY:
        return plain_byte_arrays(data, count)
    if physical == INT64:
        return list(struct.unpack_from(f"<{count}q", data, 0))
    if physical == INT32:
        return list(struct.unpack_from(f"<{count}i", data, 0))
    raise ValueError(f"unsupported physical type {physical}")


def leaf_levels(schema):
    """Map dotted leaf path -> levels. Supports flat columns and one-level lists."""
    leaves, pos = {}, 1

    def walk(path, max_def, max_rep, list_def):
        nonlocal pos
        element = schema[pos]
        pos += 1
        repetition = element.get(3, 0)
        max_def += repetition in (OPTIONAL, REPEATED)
        if repetition == REPEATED:
            max_rep += 1
            list_def = max_def
        path = [*path, element[4].decode()]
        children = element.get(5, 0)
        if children:
            for _ in range(children):
                walk(path, max_def, max_rep, list_def)
        else:
            if max_rep > 1:
                raise ValueError("nested lists are unsupported")
            leaves[".".join(path)] = {"top": path[0], "type": element[1], "max_def": max_def, "max_rep": max_rep, "list_def": list_def}

    for _ in range(schema[0].get(5, 0)):
        walk([], 0, 0, None)
    return leaves


def assemble(reps, defs, values, max_def, max_rep, list_def):
    """Rebuild rows from levels: flat values, or one list per row (None = null list)."""
    if max_rep == 0:
        return [next(values) if d == max_def else None for d in defs]
    rows = []
    for r, d in zip(reps, defs, strict=True):
        if r == 0:
            rows.append(None if d < list_def - 1 else [])
        if d >= list_def:
            rows[-1].append(next(values) if d == max_def else None)
    return rows


def _decompress(codec, raw, size):
    if codec == UNCOMPRESSED:
        out = raw
    elif codec == SNAPPY:
        out = snappy_decompress(raw)
    elif codec == GZIP:
        out = zlib.decompress(raw, 16 + zlib.MAX_WBITS)
    else:
        raise ValueError(f"unsupported parquet codec {codec}")
    if len(out) != size:
        raise ValueError("page size mismatch")
    return out


def _levels(buffer, max_level, count, length_prefixed):
    """Decode one RLE level stream; returns (levels, remaining bytes)."""
    if max_level == 0:
        return [0] * count, buffer
    width = max_level.bit_length()
    if length_prefixed:
        (n,) = struct.unpack_from("<I", buffer, 0)
        return rle_hybrid(buffer[4 : 4 + n], width, count), buffer[4 + n :]
    return rle_hybrid(buffer, width, count), b""


def _column(data, meta, leaf):
    codec, total, physical = meta[4], meta[5], meta[1]
    pos = meta.get(11) or meta[9]  # dictionary page first when present
    dictionary, reps, defs, values = None, [], [], []
    while len(defs) < total:
        header, pos = read_struct(data, pos)
        kind, size, compressed = header[1], header[2], header[3]
        raw = data[pos : pos + compressed]
        pos += compressed
        if kind == DICTIONARY_PAGE:
            dictionary = plain_values(_decompress(codec, raw, size), header[7][1], physical)
            continue
        if kind == DATA_PAGE:
            body = _decompress(codec, raw, size)
            count, encoding = header[5][1], header[5][2]
            r, body = _levels(body, leaf["max_rep"], count, True)
            d, body = _levels(body, leaf["max_def"], count, True)
        elif kind == DATA_PAGE_V2:
            v2 = header[8]
            count, encoding, def_len, rep_len = v2[1], v2[4], v2[5], v2[6]
            r, _ = _levels(raw[:rep_len], leaf["max_rep"], count, False)
            d, _ = _levels(raw[rep_len : rep_len + def_len], leaf["max_def"], count, False)
            body = raw[def_len + rep_len :]
            if v2.get(7, True):
                body = _decompress(codec, body, size - def_len - rep_len)
        else:
            raise ValueError(f"unsupported page type {kind}")
        present = sum(level == leaf["max_def"] for level in d)
        if encoding == PLAIN:
            decoded = plain_values(body, present, physical)
        elif encoding in (PLAIN_DICTIONARY, RLE_DICTIONARY):
            if dictionary is None:
                raise ValueError("dictionary page missing")
            decoded = [dictionary[i] for i in rle_hybrid(body[1:], body[0], present)]
        else:
            raise ValueError(f"unsupported encoding {encoding}")
        reps.extend(r)
        defs.extend(d)
        values.extend(decoded)
    return assemble(reps, defs, iter(values), leaf["max_def"], leaf["max_rep"], leaf["list_def"])


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def read_columns(path):
    """Return {top-level column: [value, ...]}: str/int, or list for list columns."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"PAR1" or data[-4:] != b"PAR1":
        raise ValueError("not a parquet file")
    (n,) = struct.unpack("<I", data[-8:-4])
    footer, _ = read_struct(data[-8 - n : -8], 0)
    leaves = leaf_levels(footer[2])
    columns = {leaf["top"]: [] for leaf in leaves.values()}
    for group in footer[4]:
        for chunk in group[1]:
            meta = chunk[3]
            leaf = leaves[b".".join(meta[3]).decode()]
            columns[leaf["top"]].extend(_column(data, meta, leaf))
    for name, values in columns.items():
        if len(values) != footer[3]:
            raise ValueError(f"column {name} has {len(values)} rows, expected {footer[3]}")
        columns[name] = [_decode(v) for v in values]
    return columns


def read_string_columns(path):
    """Flat BYTE_ARRAY-only files (GSM8K); rejects anything else."""
    columns = read_columns(path)
    if any(not all(v is None or isinstance(v, str) for v in vs) for vs in columns.values()):
        raise ValueError("expected string columns only")
    return columns
