import struct

import pytest

from rl.eval import parquet_lite as pl


def test_snappy_literal_and_copy():
    # length 9 = varint 0x09; literal "abc" (tag (3-1)<<2); copy-1 len 6 offset 3:
    # tag = ((6-4)<<2) | 0b01 | ((3>>8)<<5), then offset low byte 3.
    data = bytes([0x09, (3 - 1) << 2]) + b"abc" + bytes([((6 - 4) << 2) | 1, 3])
    assert pl.snappy_decompress(data) == b"abcabcabc"
    # copy-2: tag ((len-1)<<2)|2 then 2-byte little-endian offset
    data = bytes([0x06, (2 - 1) << 2]) + b"xy" + bytes([((4 - 1) << 2) | 2, 2, 0])
    assert pl.snappy_decompress(data) == b"xyxyxy"
    with pytest.raises(ValueError):
        pl.snappy_decompress(bytes([0x05, 0]) + b"a")  # declared 5, produced 1


def test_thrift_compact_struct_with_list_and_binary():
    # struct { 1: i32 = 3 (zigzag 6); 2: binary "hi"; 3: list<i32> [1, 2]; stop }
    data = bytes([0x15, 6, 0x18, 2]) + b"hi" + bytes([0x19, 0x25, 2, 4, 0x00])
    value, end = pl.read_struct(data, 0)
    assert value == {1: 3, 2: b"hi", 3: [1, 2]} and end == len(data)


def test_rle_bitpacked_hybrid_decodes_both_run_kinds():
    # bit width 2: RLE run of 3 x value 1 (header 3<<1), then bit-packed group of 8
    # values 0..3,0..3 (header (1<<1)|1) packed little-endian 2 bits each.
    packed = bytes([0b11100100, 0b11100100])
    data = bytes([3 << 1, 1, (1 << 1) | 1]) + packed
    assert pl.rle_hybrid(data, 2, 11) == [1, 1, 1, 0, 1, 2, 3, 0, 1, 2, 3]


def test_schema_levels_for_flat_and_list_columns():
    # message { optional binary q; optional group options (LIST) { repeated group list { optional binary element } }; required int64 id }
    schema = [
        {4: b"schema", 5: 3},
        {1: 6, 3: 1, 4: b"q"},
        {3: 1, 4: b"options", 5: 1},
        {3: 2, 4: b"list", 5: 1},
        {1: 6, 3: 1, 4: b"element"},
        {1: 2, 3: 0, 4: b"id"},
    ]
    leaves = pl.leaf_levels(schema)
    assert leaves["q"] == {"top": "q", "type": 6, "max_def": 1, "max_rep": 0, "list_def": None}
    assert leaves["options.list.element"] == {"top": "options", "type": 6, "max_def": 3, "max_rep": 1, "list_def": 2}
    assert leaves["id"]["max_def"] == 0


def test_assemble_list_rows_from_levels():
    # rows: ["a", "b"], [], None, ["c"]
    reps = [0, 1, 0, 0, 0]
    defs = [3, 3, 1, 0, 3]
    rows = pl.assemble(reps, defs, iter([b"a", b"b", b"c"]), max_def=3, max_rep=1, list_def=2)
    assert rows == [[b"a", b"b"], [], None, [b"c"]]
    flat = pl.assemble(None, [1, 0, 1], iter([b"x", b"y"]), max_def=1, max_rep=0, list_def=None)
    assert flat == [b"x", None, b"y"]


def test_plain_int64():
    assert pl.plain_values(struct.pack("<qq", 7, -2), 2, 2) == [7, -2]


def test_plain_byte_arrays():
    data = struct.pack("<I", 2) + b"ab" + struct.pack("<I", 0)
    assert pl.plain_byte_arrays(data, 2) == [b"ab", b""]
