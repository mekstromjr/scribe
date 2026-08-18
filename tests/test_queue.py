"""The durable job spool. Filesystem only — no network."""

from __future__ import annotations

import json

import pytest

from scribe.config import Settings
from scribe.queue import Job, complete, enqueue, restore, spool


@pytest.fixture
def settings(tmp_path):
    return Settings(spool_dir=str(tmp_path / "queue"))


def make(settings, label="https://a.example", attachment=None):
    job = Job.new("C123", "1700000000.1", label, label)
    job.attachment = attachment
    enqueue(settings, job)
    return job


class TestRoundTrip:
    def test_enqueued_job_survives_and_restores(self, settings):
        job = make(settings)
        [restored] = restore(settings)
        assert restored == job

    def test_complete_removes_the_record(self, settings):
        job = make(settings)
        complete(settings, job)
        assert restore(settings) == []

    def test_complete_is_idempotent(self, settings):
        job = make(settings)
        complete(settings, job)
        complete(settings, job)  # must not raise

    def test_complete_also_removes_the_attachment(self, settings, tmp_path):
        att = tmp_path / "deck.pdf"
        att.write_bytes(b"data")
        job = make(settings, attachment=str(att))
        complete(settings, job)
        assert not att.exists()


class TestOrdering:
    def test_restores_in_arrival_order(self, settings):
        """Ids are epoch-ms prefixed so lexicographic sort == chronological. A queue that
        replayed out of order would summarize the fifth link before the first."""
        jobs = [make(settings, f"https://{i}.example") for i in range(5)]
        assert [j.id for j in restore(settings)] == [j.id for j in jobs]

    def test_ids_are_unique_within_a_millisecond(self, settings):
        ids = {Job.new("C", "1", "t", "t").id for _ in range(200)}
        assert len(ids) == 200


class TestResilience:
    def test_corrupt_record_is_dropped_not_fatal(self, settings):
        """One unparseable file must not stop every other queued item from resuming."""
        good = make(settings)
        bad = spool(settings) / "0000000000000-deadbeef.json"
        bad.write_text("{not json")
        assert [j.id for j in restore(settings)] == [good.id]
        assert not bad.exists()

    def test_record_with_unknown_fields_is_dropped(self, settings):
        bad = spool(settings) / "0000000000001-deadbeef.json"
        bad.write_text(json.dumps({"nope": 1}))
        assert restore(settings) == []

    def test_empty_spool_restores_nothing(self, settings):
        assert restore(settings) == []

    def test_no_partial_records_are_left_behind(self, settings):
        """enqueue writes to a temp file then renames, so a crash mid-write cannot leave
        a half-written record that fails to parse on restore."""
        make(settings)
        assert list(spool(settings).glob("*.tmp")) == []
