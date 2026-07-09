"""Tests for the job data-integrity certifier (backend/certify.py)."""
import json

import pytest

from backend import certify
from backend.workers import analytics as A
from test_analytics import _synthetic_dashboard


def _make_enriched_job(tmp_path, monkeypatch, no_motionbert):
    """Produce a real enriched_data.json by running the analytics pipeline."""
    monkeypatch.chdir(tmp_path)
    job_id = "certify01"
    out_dir = tmp_path / "outputs" / job_id
    out_dir.mkdir(parents=True)
    with open(out_dir / "dashboard_data.json", "w") as f:
        json.dump(_synthetic_dashboard(n_frames=5, sep=1.0), f)
    A._compute_spatial_analytics_sync(job_id, {})
    return str(out_dir)


def test_certify_passes_on_valid_job(tmp_path, monkeypatch, no_motionbert):
    job_dir = _make_enriched_job(tmp_path, monkeypatch, no_motionbert)
    checks = certify.certify_job(job_dir)
    failed = [c for c in checks if not c.ok]
    assert not failed, f"unexpected failures: {[(c.name, c.detail) for c in failed]}"
    # sanity: it actually ran a meaningful number of checks
    assert len(checks) >= 8


def test_certify_missing_file():
    checks = certify.certify_job("/nonexistent/job/dir")
    assert len(checks) == 1
    assert checks[0].ok is False
    assert "exists" in checks[0].name


def _load(job_dir):
    with open(job_dir + "/enriched_data.json") as f:
        return json.load(f)


def _save(job_dir, data):
    with open(job_dir + "/enriched_data.json", "w") as f:
        json.dump(data, f)


def _check(checks, name_fragment):
    return next(c for c in checks if name_fragment in c.name)


def test_certify_catches_nan_position(tmp_path, monkeypatch, no_motionbert):
    job_dir = _make_enriched_job(tmp_path, monkeypatch, no_motionbert)
    data = _load(job_dir)
    data["frames"][0]["humans"][0]["world_pos"] = [float("nan"), 0.0, 0.0]
    _save(job_dir, data)
    checks = certify.certify_job(job_dir)
    assert _check(checks, "human positions finite").ok is False


def test_certify_catches_non_unit_gaze(tmp_path, monkeypatch, no_motionbert):
    job_dir = _make_enriched_job(tmp_path, monkeypatch, no_motionbert)
    data = _load(job_dir)
    data["frames"][0]["humans"][0]["gaze_vec"] = [5.0, 0.0, 0.0]   # not unit
    _save(job_dir, data)
    checks = certify.certify_job(job_dir)
    assert _check(checks, "gaze vectors unit-length").ok is False


def test_certify_catches_zone_mismatch(tmp_path, monkeypatch, no_motionbert):
    job_dir = _make_enriched_job(tmp_path, monkeypatch, no_motionbert)
    data = _load(job_dir)
    # distance 1.0 is "personal"; mislabel it "public"
    data["frames"][0]["interactions"][0]["zone"] = "public"
    _save(job_dir, data)
    checks = certify.certify_job(job_dir)
    assert _check(checks, "proxemics zone matches distance").ok is False


def test_certify_catches_dangling_interaction(tmp_path, monkeypatch, no_motionbert):
    job_dir = _make_enriched_job(tmp_path, monkeypatch, no_motionbert)
    data = _load(job_dir)
    data["frames"][0]["interactions"][0]["target"] = 999   # id not in frame
    _save(job_dir, data)
    checks = certify.certify_job(job_dir)
    assert _check(checks, "interactions reference present ids").ok is False


def test_certify_catches_bad_action(tmp_path, monkeypatch, no_motionbert):
    job_dir = _make_enriched_job(tmp_path, monkeypatch, no_motionbert)
    data = _load(job_dir)
    data["frames"][0]["humans"][0]["action"] = "moonwalking"
    _save(job_dir, data)
    checks = certify.certify_job(job_dir)
    assert _check(checks, "actions in vocabulary").ok is False


def test_certify_cli_returns_nonzero_on_failure(tmp_path, monkeypatch, no_motionbert):
    job_dir = _make_enriched_job(tmp_path, monkeypatch, no_motionbert)
    data = _load(job_dir)
    data["frames"][0]["humans"][0]["action"] = "moonwalking"
    _save(job_dir, data)
    assert certify.main([job_dir]) == 1


def test_certify_cli_returns_zero_on_pass(tmp_path, monkeypatch, no_motionbert):
    job_dir = _make_enriched_job(tmp_path, monkeypatch, no_motionbert)
    assert certify.main([job_dir]) == 0
