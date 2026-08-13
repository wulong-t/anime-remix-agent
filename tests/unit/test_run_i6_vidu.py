"""Offline tests for the frozen I6 Vidu Q2 Pro experiment runner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image


def _load_runner():
    runner_path = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "i6_vidu_q2_pro"
        / "run_i6_vidu.py"
    )
    spec = importlib.util.spec_from_file_location("run_i6_vidu_under_test", runner_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


def _write_png(path: Path, color: tuple[int, int, int]) -> str:
    Image.new("RGB", (128, 72), color).save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_contract(
    tmp_path: Path,
    *,
    authorization: str = "pending",
    resolution: str = "540P",
    start_sha256: str | None = None,
) -> tuple[Path, Path]:
    start = tmp_path / "start.png"
    end = tmp_path / "end.png"
    actual_start_sha256 = _write_png(start, (10, 20, 30))
    end_sha256 = _write_png(end, (40, 50, 60))
    contract = {
        "schema_version": "i6-vidu-start-end-request-v1",
        "run_id": "I6-TEST-001",
        "region": "华北2（北京）",
        "model": "vidu/viduq2-pro_start-end2video",
        "inputs": [
            {
                "role": "start_frame",
                "path": "start.png",
                "sha256": start_sha256 or actual_start_sha256,
            },
            {
                "role": "end_frame",
                "path": "end.png",
                "sha256": end_sha256,
            },
        ],
        "prompt": "固定机位，只转动头部。",
        "parameters": {
            "resolution": resolution,
            "duration": 2,
            "seed": 0,
            "watermark": False,
        },
        "limits": {
            "maximum_tasks": 1,
            "maximum_outputs": 1,
            "automatic_retry": False,
            "content_retry": False,
        },
        "listed_price_cny": {
            "unit_price_per_second": 0.15625,
            "maximum_for_one_successful_task": 0.3125,
        },
        "authorization": {"status": authorization},
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract, ensure_ascii=False), encoding="utf-8")
    return contract_path, tmp_path


class _ExplodingTransport:
    def __getattr__(self, name: str):
        raise AssertionError(f"network seam {name} must not be used")


class _SuccessfulTransport:
    def __init__(self) -> None:
        self.uploads: list[Path] = []
        self.submissions: list[dict] = []
        self.queries: list[str] = []
        self.downloads: list[str] = []
        self._statuses = ["PENDING", "RUNNING", "SUCCEEDED"]

    def upload(self, *, path: Path, model: str, api_key: str) -> str:
        assert model == "vidu/viduq2-pro_start-end2video"
        assert api_key == "sk-test-value"
        self.uploads.append(path)
        return f"oss://dashscope-instant/test/{path.name}"

    def submit(
        self,
        *,
        workspace_id: str,
        api_key: str,
        payload: dict,
    ) -> dict:
        assert workspace_id == "workspace-test"
        assert api_key == "sk-test-value"
        self.submissions.append(payload)
        return {
            "request_id": "request-001",
            "output": {"task_id": "task-001", "task_status": "PENDING"},
        }

    def query(
        self,
        *,
        workspace_id: str,
        api_key: str,
        task_id: str,
    ) -> dict:
        assert workspace_id == "workspace-test"
        assert api_key == "sk-test-value"
        assert task_id == "task-001"
        self.queries.append(task_id)
        status = self._statuses.pop(0)
        output = {"task_id": task_id, "task_status": status}
        if status == "SUCCEEDED":
            output["video_url"] = (
                "https://vidu-output.s3.cn-northwest-1.amazonaws.com.cn/"
                "output.mp4?signature=secret"
            )
        return {
            "request_id": "query-request-001",
            "output": output,
            "usage": {
                "duration": 2,
                "size": "960*540",
                "output_video_duration": 2,
                "fps": 24,
                "video_count": 1,
                "audio": False,
                "SR": "540",
                "untrusted": "must not persist",
            },
        }

    def download(self, *, url: str) -> bytes:
        self.downloads.append(url)
        return b"\x00\x00\x00\x18ftypisom" + b"synthetic-mp4"


class _ResumeOnlyTransport(_SuccessfulTransport):
    def __init__(self) -> None:
        super().__init__()
        self._statuses = ["SUCCEEDED"]

    def upload(self, *, path: Path, model: str, api_key: str) -> str:
        raise AssertionError("resume must not upload again")

    def submit(
        self,
        *,
        workspace_id: str,
        api_key: str,
        payload: dict,
    ) -> dict:
        raise AssertionError("resume must not submit another task")


def _fake_probe(path: Path, expected_duration: int) -> dict:
    assert path.name == ".raw.mp4.tmp"
    assert expected_duration == 2
    assert path.read_bytes()[4:8] == b"ftyp"
    return {
        "container": "mp4",
        "codec": "h264",
        "width": 960,
        "height": 540,
        "fps": 24.0,
        "duration_seconds": 2.0,
        "video_streams": 1,
        "audio_streams": 0,
    }


def test_dry_run_is_local_and_redacts_prompt_and_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner,
) -> None:
    contract_path, root = _write_contract(tmp_path)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-no-persist")
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "secret-workspace")
    run_dir = tmp_path / "dry-run"

    manifest = runner.run_experiment(
        contract_path=contract_path,
        project_root=root,
        run_dir=run_dir,
        dry_run=True,
        execute_paid=False,
        transport=_ExplodingTransport(),
    )

    assert manifest["outcome"] == "dry_run"
    assert manifest["request_summary"]["parameters"] == {
        "resolution": "540P",
        "duration": 2,
        "seed": 0,
        "watermark": False,
    }
    assert manifest["request_summary"]["listed_maximum_cny"] == 0.3125
    assert manifest["execution"]["task_submit_count"] == 0
    assert not (run_dir / "provider-state.json").exists()
    serialized = (run_dir / "run-manifest.json").read_text(encoding="utf-8")
    assert "固定机位" not in serialized
    assert "sk-no-persist" not in serialized
    assert "secret-workspace" not in serialized


def test_explicit_i7_stage_is_preserved_in_dry_run(tmp_path: Path, runner) -> None:
    contract_path, root = _write_contract(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["stage"] = "I7"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    manifest = runner.run_experiment(
        contract_path=contract_path,
        project_root=root,
        run_dir=tmp_path / "i7-dry-run",
        dry_run=True,
        execute_paid=False,
        transport=_ExplodingTransport(),
    )

    assert manifest["stage"] == "I7"
    assert manifest["execution"]["task_submit_count"] == 0


def test_frozen_contract_rejects_resolution_drift(tmp_path: Path, runner) -> None:
    contract_path, root = _write_contract(tmp_path, resolution="720P")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    with pytest.raises(runner.ContractError, match="540P"):
        runner.validate_contract(contract, root)


def test_frozen_contract_rejects_input_hash_drift(tmp_path: Path, runner) -> None:
    contract_path, root = _write_contract(tmp_path, start_sha256="0" * 64)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    with pytest.raises(runner.ContractError, match="SHA256 drift"):
        runner.validate_contract(contract, root)


def test_live_mode_requires_contract_authorization_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner,
) -> None:
    contract_path, root = _write_contract(tmp_path, authorization="pending")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-value")
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "workspace-test")

    with pytest.raises(runner.ContractError, match="not granted"):
        runner.run_experiment(
            contract_path=contract_path,
            project_root=root,
            run_dir=tmp_path / "live-pending",
            dry_run=False,
            execute_paid=True,
            transport=_ExplodingTransport(),
        )


def test_live_mode_requires_beijing_workspace_id_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner,
) -> None:
    contract_path, root = _write_contract(tmp_path, authorization="granted")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-value")
    monkeypatch.delenv("DASHSCOPE_WORKSPACE_ID", raising=False)

    with pytest.raises(runner.EnvironmentError, match="DASHSCOPE_WORKSPACE_ID"):
        runner.run_experiment(
            contract_path=contract_path,
            project_root=root,
            run_dir=tmp_path / "live-no-workspace",
            dry_run=False,
            execute_paid=True,
            transport=_ExplodingTransport(),
        )


def test_mock_live_run_uploads_two_frames_submits_once_and_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner,
) -> None:
    contract_path, root = _write_contract(tmp_path, authorization="granted")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-value")
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "workspace-test")
    transport = _SuccessfulTransport()
    run_dir = tmp_path / "live-success"

    manifest = runner.run_experiment(
        contract_path=contract_path,
        project_root=root,
        run_dir=run_dir,
        dry_run=False,
        execute_paid=True,
        transport=transport,
        sleep_fn=lambda _seconds: None,
        probe_fn=_fake_probe,
    )

    assert len(transport.uploads) == 2
    assert len(transport.submissions) == 1
    payload = transport.submissions[0]
    assert payload["model"] == "vidu/viduq2-pro_start-end2video"
    assert [item["url"].rsplit("/", 1)[-1] for item in payload["input"]["media"]] == [
        "start.png",
        "end.png",
    ]
    assert payload["parameters"] == {
        "resolution": "540P",
        "duration": 2,
        "watermark": False,
        "seed": 0,
    }
    assert len(transport.queries) == 3
    assert len(transport.downloads) == 1
    assert manifest["outcome"] == "success"
    assert manifest["execution"]["task_submit_count"] == 1
    assert manifest["execution"]["output_count"] == 1
    assert manifest["execution"]["automatic_retries"] == 0
    assert manifest["execution"]["provider_usage"]["fps"] == 24
    assert "untrusted" not in manifest["execution"]["provider_usage"]
    assert (run_dir / "raw.mp4").exists()
    serialized = (run_dir / "run-manifest.json").read_text(encoding="utf-8")
    assert "oss://" not in serialized
    assert "signature=secret" not in serialized
    assert "sk-test-value" not in serialized
    assert "固定机位" not in serialized


def test_resume_queries_recorded_task_without_upload_or_resubmit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner,
) -> None:
    contract_path, root = _write_contract(tmp_path, authorization="granted")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-value")
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "workspace-test")
    run_dir = tmp_path / "resume"
    run_dir.mkdir()
    (run_dir / "provider-state.json").write_text(
        json.dumps(
            {
                "schema_version": "i6-vidu-provider-state-v1",
                "state": "running",
                "upload_count": 2,
                "task_submit_count": 1,
                "query_count": 2,
                "output_count": 0,
                "task_id": "task-001",
                "request_id": "request-001",
            }
        ),
        encoding="utf-8",
    )
    transport = _ResumeOnlyTransport()

    manifest = runner.run_experiment(
        contract_path=contract_path,
        project_root=root,
        run_dir=run_dir,
        dry_run=False,
        execute_paid=True,
        transport=transport,
        sleep_fn=lambda _seconds: None,
        probe_fn=_fake_probe,
    )

    assert manifest["execution"]["task_submit_count"] == 1
    assert manifest["execution"]["query_count"] == 3
    assert len(transport.queries) == 1
    assert len(transport.downloads) == 1


def test_closed_failed_run_cannot_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner,
) -> None:
    contract_path, root = _write_contract(tmp_path, authorization="granted")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-value")
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "workspace-test")
    run_dir = tmp_path / "closed"
    run_dir.mkdir()
    (run_dir / "provider-state.json").write_text(
        json.dumps(
            {
                "schema_version": "i6-vidu-provider-state-v1",
                "state": "failed",
                "upload_count": 2,
                "task_submit_count": 1,
                "query_count": 1,
                "output_count": 0,
                "task_id": "task-001",
                "request_id": "request-001",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(runner.ContractError, match="no-retry"):
        runner.run_experiment(
            contract_path=contract_path,
            project_root=root,
            run_dir=run_dir,
            dry_run=False,
            execute_paid=True,
            transport=_ExplodingTransport(),
        )
