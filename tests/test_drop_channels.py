"""Drop channels are an ALLOWLIST, not a side effect of membership: the
message.channels subscription delivers every channel scribe is in, and only the
configured ones may process unmentioned traffic."""

from __future__ import annotations

from scribe.config import Settings


def test_drop_channel_ids_parse_and_trim():
    s = Settings(drop_channels="C0AAA, C0BBB ,,")
    assert s.drop_channel_ids == {"C0AAA", "C0BBB"}


def test_default_is_no_drop_channels():
    assert Settings().drop_channel_ids == set()
