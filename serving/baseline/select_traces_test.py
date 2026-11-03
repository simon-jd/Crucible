import hashlib
import os
import time

import pytest

from serving.baseline.select_traces import select_windows


def test_azure_selection_uses_utc_under_host_timezone(monkeypatch):
    raw = (
        b"TIMESTAMP,ContextTokens,GeneratedTokens\r\n"
        b"2023-11-16 18:15:46.6805900,0,0\r\n"
        b"2023-11-16 18:16:16.6805900,4107,594\r\n"
    )
    previous = os.environ.get("TZ")
    try:
        monkeypatch.setenv("TZ", "America/Los_Angeles")
        time.tzset()
        windows = select_windows(raw, "azure", 30)
        assert windows[0]["start"] == "1700158546.6805900"
        assert windows[0]["zero_input"] == windows[0]["zero_output"] == 1
        assert windows[1]["over_context_4096"] == 1
        assert windows[1]["source_rows"] == [2]
        assert windows[0]["csv_bytes"].endswith(b",0,0\r\n")
    finally:
        if previous is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", previous)
        time.tzset()


def test_burst_arrival_only_dense_disjoint_windows_keep_boundary_and_duplicates():
    raw = (
        b"Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type\n"
        b"0,X,1,2,3,Conversation log\n"
        b"100,X,0,0,0,Failed\n"
        b"100,X,2,3,5,Conversation log\n"
        b"110,X,4,5,9,Conversation log\n"
        b"110,X,4,5,9,Conversation log\n"
        b"119,X,6,7,13,Conversation log\npartial"
    )
    a, b = select_windows(raw, "burstgpt", 10, min_rows=2)
    assert a["source_rows"] == [2, 3]
    assert b["source_rows"] == [4, 5, 6]
    assert a["zero_output"] == 1
    assert a["selection_sha256"] == hashlib.sha256(a["csv_bytes"]).hexdigest()
    assert b["first_source_timestamp"] == "110"
    assert b["trailing_idle_seconds"] == "1"


def test_long_windows_give_rounds_large_enough_for_the_one_percent_rule():
    # 100+ requests per round makes "more than 1% late" mean more than one request.
    header = b"Timestamp,Request tokens,Response tokens\n"
    rows = b"".join(b"%d,8,4\n" % (t * 3) for t in range(400))
    a, b = select_windows(header + rows, "burstgpt", 600, min_rows=100)
    assert a["rows"] == b["rows"] == 200
    with pytest.raises(ValueError, match="configuration"):
        select_windows(header + rows, "burstgpt", 601)


def test_selector_rejects_unsorted_input_and_insufficient_disjoint_windows():
    header = b"Timestamp,Request tokens,Response tokens\n"
    with pytest.raises(ValueError, match="sorted"):
        select_windows(header + b"2,1,1\n1,1,1\n", "burstgpt", 10)
    with pytest.raises(ValueError, match="two disjoint"):
        select_windows(header + b"2,1,1\n", "burstgpt", 10)
