"""m4b packaging. WAV math is exact (stdlib wave); the ffmpeg encode itself only runs
where ffmpeg exists — CI has it via the Dockerfile, laptops via homebrew."""

from __future__ import annotations

import shutil
import wave
from pathlib import Path

import pytest

from scribe.package import ChapterAudio, PackageError, _ffmetadata, build_m4b, wav_seconds


def _write_wav(path: Path, seconds: float, rate: int = 24000) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return path


class TestWavSeconds:
    def test_duration_is_exact_from_the_header(self, tmp_path):
        p = _write_wav(tmp_path / "a.wav", 2.5)
        assert wav_seconds(p) == pytest.approx(2.5)


class TestFfmetadata:
    def test_chapter_offsets_accumulate_in_ms(self, tmp_path):
        a = _write_wav(tmp_path / "a.wav", 1.0)
        b = _write_wav(tmp_path / "b.wav", 2.0)
        c = _write_wav(tmp_path / "c.wav", 0.5)
        meta = _ffmetadata(
            [ChapterAudio("Summary", [a]), ChapterAudio("Full article", [b, c])]
        )
        assert ";FFMETADATA1" in meta
        assert "START=0\nEND=1000\ntitle=Summary" in meta
        # Second chapter starts where the first ended and spans BOTH its files.
        assert "START=1000\nEND=3500\ntitle=Full article" in meta

    def test_metadata_breaking_characters_are_neutralized(self, tmp_path):
        a = _write_wav(tmp_path / "a.wav", 1.0)
        meta = _ffmetadata([ChapterAudio("a=b\nc", [a])])
        assert "title=a-b c" in meta


class TestBuildM4b:
    def test_refuses_empty_input(self, tmp_path):
        with pytest.raises(PackageError, match="no audio segments"):
            build_m4b([], tmp_path / "out.m4b", title="t", author="a", workdir=tmp_path)

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
    def test_encodes_a_playable_m4b_with_both_chapters(self, tmp_path):
        a = _write_wav(tmp_path / "a.wav", 1.0)
        b = _write_wav(tmp_path / "b.wav", 1.0)
        out = tmp_path / "out.m4b"
        total = build_m4b(
            [ChapterAudio("Summary", [a]), ChapterAudio("Full article", [b])],
            out, title="Test Title", author="scribe", workdir=tmp_path,
        )
        assert total == pytest.approx(2.0)
        # MP4 container magic: 'ftyp' at byte 4.
        assert out.read_bytes()[4:8] == b"ftyp"
        assert out.stat().st_size > 1000

    @pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffmpeg not installed")
    def test_cover_is_embedded_as_an_attached_picture(self, tmp_path):
        """scribe#14: Audiobookshelf only extracts a cover that is an attached_pic stream,
        and the chapters must survive the extra input shifting nothing."""
        import json
        import subprocess

        from PIL import Image

        a = _write_wav(tmp_path / "a.wav", 1.0)
        b = _write_wav(tmp_path / "b.wav", 1.0)
        cover = tmp_path / "cover.jpg"
        Image.new("RGB", (64, 64), (200, 30, 30)).save(cover, "JPEG")
        out = tmp_path / "out.m4b"
        build_m4b([ChapterAudio("Summary", [a]), ChapterAudio("Full article", [b])],
                  out, title="T", author="scribe", workdir=tmp_path, cover=cover)

        probe = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_chapters", "-of", "json",
             str(out)], check=True, capture_output=True, text=True).stdout)
        kinds = [s["codec_type"] for s in probe["streams"]]
        assert kinds.count("audio") == 1
        video = [s for s in probe["streams"] if s["codec_type"] == "video"]
        assert len(video) == 1 and video[0]["disposition"]["attached_pic"] == 1
        assert [c["tags"]["title"] for c in probe["chapters"]] == ["Summary", "Full article"]
