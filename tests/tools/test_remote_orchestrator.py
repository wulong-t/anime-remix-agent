from __future__ import annotations

import importlib.util
import io
import json
import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "remote_orchestrator" / "orchestrator.py"
SPEC = importlib.util.spec_from_file_location(
    "remote_orchestrator_under_test", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
orchestrator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = orchestrator
SPEC.loader.exec_module(orchestrator)


class RemoteOrchestratorTests(unittest.TestCase):
    def make_pipeline(
        self,
        root: Path,
        *,
        first_id: str = "stage-01",
        second_dependencies: str = '["stage-01"]',
        push: str = "false",
        host: str = "gpu-alias",
        prompt_two: str = "prompts/two.md",
        identity_file: Path | None = None,
        allow_dirty_primary: str = "false",
    ) -> Path:
        prompts = root / "prompts"
        prompts.mkdir(parents=True)
        (prompts / "one.md").write_text("create one\n", encoding="utf-8")
        (prompts / "two.md").write_text("create two\n", encoding="utf-8")
        pipeline = root / "pipeline.toml"
        identity_line = (
            ""
            if identity_file is None
            else f"identity_file = {json.dumps(identity_file.as_posix())}"
        )
        pipeline.write_text(
            textwrap.dedent(
                f"""
                [remote]
                host = {json.dumps(host)}
                repo = "/srv/anime-remix-agent"
                worktree_root = "/srv/anime-remix-worktrees"
                {identity_line}
                allow_dirty_primary = {allow_dirty_primary}
                connect_timeout_seconds = 10
                stage_timeout_seconds = 120

                [pipeline]
                id = "smoke-test"
                base_branch = "main"
                push = {push}

                [[stages]]
                id = {json.dumps(first_id)}
                prompt = "prompts/one.md"
                depends_on = []

                [[stages]]
                id = "stage-02"
                prompt = {json.dumps(prompt_two)}
                depends_on = {second_dependencies}
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        return pipeline

    def valid_pass_result(self) -> dict[str, object]:
        return {
            "stage": "stage-01",
            "status": "pass",
            "branch": "codex/stage-01",
            "base_commit": "a" * 40,
            "head_commit": "b" * 40,
            "commit_created": True,
            "summary": "completed",
            "tests": [{"command": "test -f output.txt", "passed": True}],
            "artifacts": ["output.txt"],
            "changed_files": ["output.txt"],
            "blocking_issue": None,
            "recommended_next_action": "continue",
        }

    def test_parses_valid_toml_and_linear_stages(self) -> None:
        with TemporaryDirectory() as temporary:
            pipeline = self.make_pipeline(Path(temporary))
            config = orchestrator.load_pipeline(pipeline)

        self.assertEqual(config.id, "smoke-test")
        self.assertEqual(config.remote.host, "gpu-alias")
        self.assertEqual(
            [stage.id for stage in config.stages], ["stage-01", "stage-02"]
        )
        self.assertEqual(config.stages[1].depends_on, ("stage-01",))
        self.assertFalse(config.push)
        self.assertFalse(config.remote.allow_dirty_primary)

    def test_rejects_unsafe_stage_id(self) -> None:
        with TemporaryDirectory() as temporary:
            pipeline = self.make_pipeline(Path(temporary), first_id="../stage")
            with self.assertRaisesRegex(orchestrator.OrchestratorError, "must match"):
                orchestrator.load_pipeline(pipeline)

    def test_rejects_non_linear_dependency(self) -> None:
        with TemporaryDirectory() as temporary:
            pipeline = self.make_pipeline(Path(temporary), second_dependencies="[]")
            with self.assertRaisesRegex(
                orchestrator.OrchestratorError, "linear pipeline"
            ):
                orchestrator.load_pipeline(pipeline)

    def test_rejects_push_true_as_unsupported(self) -> None:
        with TemporaryDirectory() as temporary:
            pipeline = self.make_pipeline(Path(temporary), push="true")
            with self.assertRaisesRegex(orchestrator.OrchestratorError, "unsupported"):
                orchestrator.load_pipeline(pipeline)

    def test_rejects_prompt_escape(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside.md"
            outside.write_text("outside\n", encoding="utf-8")
            child = root / "config"
            child.mkdir()
            pipeline = self.make_pipeline(child, prompt_two="../outside.md")
            with self.assertRaisesRegex(orchestrator.OrchestratorError, "escapes"):
                orchestrator.load_pipeline(pipeline)

    def test_branch_worktree_and_stacked_base_selection(self) -> None:
        with TemporaryDirectory() as temporary:
            config = orchestrator.load_pipeline(self.make_pipeline(Path(temporary)))
            first, second = config.stages

        self.assertEqual(orchestrator.branch_for(first), "codex/stage-01")
        self.assertEqual(
            str(orchestrator.worktree_for(config, second)),
            "/srv/anime-remix-worktrees/stage-02",
        )
        self.assertEqual(orchestrator.base_ref_for(config, 0), "main")
        self.assertEqual(orchestrator.base_ref_for(config, 1), "codex/stage-01")

    def test_identity_file_is_explicit_and_hidden_from_dry_run(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = root / "dedicated-key"
            identity.write_text(
                "test fixture, not a real private key\n", encoding="utf-8"
            )
            config = orchestrator.load_pipeline(
                self.make_pipeline(root, identity_file=identity)
            )
            client = orchestrator.SSHClient(config.remote, lambda _message: None)
            arguments = client._arguments("true")
            plan = orchestrator.build_dry_run_plan(config)

        self.assertEqual(config.remote.identity_file, identity.resolve())
        self.assertIn("IdentitiesOnly=yes", arguments)
        self.assertIn(str(identity.resolve()), arguments)
        self.assertTrue(plan["identity_file_configured"])
        self.assertNotIn(str(identity.resolve()), json.dumps(plan))

    def test_reviewed_dirty_primary_policy_is_explicit_in_plan(self) -> None:
        with TemporaryDirectory() as temporary:
            config = orchestrator.load_pipeline(
                self.make_pipeline(Path(temporary), allow_dirty_primary="true")
            )
            plan = orchestrator.build_dry_run_plan(config)

        self.assertTrue(config.remote.allow_dirty_primary)
        self.assertEqual(plan["primary_checkout_policy"], "preserve_exact_snapshot")

    def test_validates_pass_result_and_pass_gate(self) -> None:
        with TemporaryDirectory() as temporary:
            config = orchestrator.load_pipeline(self.make_pipeline(Path(temporary)))
            stage = config.stages[0]
            result = orchestrator.validate_stage_result(
                self.valid_pass_result(),
                stage=stage,
                expected_branch="codex/stage-01",
                expected_base_sha="a" * 40,
            )

        self.assertTrue(orchestrator.should_continue(result))

    def test_non_pass_result_stops(self) -> None:
        with TemporaryDirectory() as temporary:
            config = orchestrator.load_pipeline(self.make_pipeline(Path(temporary)))
            stage = config.stages[0]
            payload = self.valid_pass_result()
            payload.update(
                status="blocked",
                head_commit="a" * 40,
                commit_created=False,
                tests=[],
                artifacts=[],
                changed_files=[],
                blocking_issue="missing prerequisite",
                recommended_next_action="stop",
            )
            result = orchestrator.validate_stage_result(
                payload,
                stage=stage,
                expected_branch="codex/stage-01",
                expected_base_sha="a" * 40,
            )

        self.assertFalse(orchestrator.should_continue(result))

    def test_pass_without_commit_is_invalid(self) -> None:
        with TemporaryDirectory() as temporary:
            config = orchestrator.load_pipeline(self.make_pipeline(Path(temporary)))
            stage = config.stages[0]
            payload = self.valid_pass_result()
            payload["commit_created"] = False
            with self.assertRaisesRegex(
                orchestrator.OrchestratorError, "commit_created"
            ):
                orchestrator.validate_stage_result(
                    payload,
                    stage=stage,
                    expected_branch="codex/stage-01",
                    expected_base_sha="a" * 40,
                )

    def test_terminal_pass_may_recommend_stop(self) -> None:
        with TemporaryDirectory() as temporary:
            config = orchestrator.load_pipeline(self.make_pipeline(Path(temporary)))
            stage = config.stages[-1]
            payload = self.valid_pass_result()
            payload.update(
                stage="stage-02",
                branch="codex/stage-02",
                recommended_next_action="stop",
            )

            with self.assertRaisesRegex(
                orchestrator.OrchestratorError,
                "PASS requires recommended_next_action=continue",
            ):
                orchestrator.validate_stage_result(
                    payload,
                    stage=stage,
                    expected_branch="codex/stage-02",
                    expected_base_sha="a" * 40,
                )
            result = orchestrator.validate_stage_result(
                payload,
                stage=stage,
                expected_branch="codex/stage-02",
                expected_base_sha="a" * 40,
                terminal=True,
            )

        self.assertTrue(orchestrator.should_continue(result))

    def test_result_rejects_parent_path(self) -> None:
        with TemporaryDirectory() as temporary:
            config = orchestrator.load_pipeline(self.make_pipeline(Path(temporary)))
            stage = config.stages[0]
            payload = self.valid_pass_result()
            payload["artifacts"] = ["../secret"]
            with self.assertRaisesRegex(
                orchestrator.OrchestratorError, "unsafe artifacts"
            ):
                orchestrator.validate_stage_result(
                    payload,
                    stage=stage,
                    expected_branch="codex/stage-01",
                    expected_base_sha="a" * 40,
                )

    def test_dry_run_has_no_ssh_or_state_side_effects(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = orchestrator.load_pipeline(self.make_pipeline(root))
            state_root = root / "state"

            def fail_if_called(*args: object, **kwargs: object) -> object:
                self.fail("SSH factory must not be called during dry-run")

            plan = orchestrator.run_pipeline(
                config,
                dry_run=True,
                state_root=state_root,
                ssh_factory=fail_if_called,
            )

            self.assertFalse(state_root.exists())
            self.assertFalse(plan["side_effects"])
            self.assertFalse(plan["automatic_merge"])
            self.assertEqual(plan["stages"][1]["base_ref"], "codex/stage-01")

    def test_remote_codex_command_keeps_workspace_sandbox(self) -> None:
        with TemporaryDirectory() as temporary:
            config = orchestrator.load_pipeline(self.make_pipeline(Path(temporary)))
            auto_review_command = orchestrator._remote_codex_command(
                config,
                worktree=orchestrator.PurePosixPath(
                    "/srv/anime-remix-worktrees/stage-01"
                ),
                schema_path=orchestrator.PurePosixPath("/srv/control/schema.json"),
                result_path=orchestrator.PurePosixPath("/srv/control/result.json"),
                auto_review=True,
            )
            sandbox_command = orchestrator._remote_codex_command(
                config,
                worktree=orchestrator.PurePosixPath(
                    "/srv/anime-remix-worktrees/stage-01"
                ),
                schema_path=orchestrator.PurePosixPath("/srv/control/schema.json"),
                result_path=orchestrator.PurePosixPath("/srv/control/result.json"),
                auto_review=False,
            )

        self.assertIn("--approve-for-me", auto_review_command)
        self.assertNotIn("--sandbox", auto_review_command)
        self.assertIn("workspace-write", sandbox_command)
        self.assertNotIn("--approve-for-me", sandbox_command)
        self.assertIn("--output-schema", auto_review_command)
        for command in (auto_review_command, sandbox_command):
            self.assertNotIn("danger-full-access", command)
            self.assertNotIn("dangerously-bypass", command)
            self.assertNotIn("--yolo", command)

    def test_retry_cli_accepts_one_stage_id(self) -> None:
        parser = orchestrator._build_parser()

        args = parser.parse_args(["retry", "--pipeline", "pipeline.toml", "stage-01"])

        self.assertEqual(args.command, "retry")
        self.assertEqual(args.pipeline, Path("pipeline.toml"))
        self.assertEqual(args.stage_id, "stage-01")

    def test_retry_rejects_unknown_stage_before_state_or_ssh(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = orchestrator.load_pipeline(self.make_pipeline(root))
            state_root = root / "state"

            def fail_if_called(*args: object, **kwargs: object) -> object:
                self.fail("SSH factory must not be called for an invalid retry target")

            with self.assertRaisesRegex(
                orchestrator.OrchestratorError, "not part of pipeline"
            ):
                orchestrator.run_pipeline(
                    config,
                    retry_stage_id="stage-99",
                    state_root=state_root,
                    ssh_factory=fail_if_called,
                )

            self.assertFalse(state_root.exists())

    def test_retry_requires_exact_saved_preflight_snapshot(self) -> None:
        previous = {
            "remote_primary_branch": "main",
            "remote_primary_head": "a" * 40,
            "remote_primary_status": "?? artifact.txt\n",
            "remote_primary_content_sha256": "b" * 64,
            "remote_common_git_dir": "/srv/repo/.git",
            "configured_base_sha": "a" * 40,
        }

        orchestrator._verify_retry_preflight_unchanged(previous, dict(previous))
        changed = dict(previous)
        changed["remote_primary_content_sha256"] = "c" * 64
        with self.assertRaisesRegex(
            orchestrator.OrchestratorError, "changed since the failed Stage"
        ):
            orchestrator._verify_retry_preflight_unchanged(previous, changed)

    def test_schema_has_required_closed_shape(self) -> None:
        schema_path = (
            REPO_ROOT / "tools" / "remote_orchestrator" / "stage-result.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(self.valid_pass_result()))
        self.assertEqual(
            set(schema["properties"]["status"]["enum"]),
            {"pass", "borderline", "fail", "blocked", "needs_user_review"},
        )

    def test_status_event_summary_omits_event_content(self) -> None:
        with TemporaryDirectory() as temporary:
            events = Path(temporary) / "events.jsonl"
            events.write_text(
                '{"type":"turn.started"}\n'
                '{"type":"item.completed","item":{"type":"agent_message",'
                '"status":"completed","text":"do not expose"}}\n',
                encoding="utf-8",
            )
            summary = orchestrator.summarize_last_jsonl_event(events)

        assert summary is not None
        self.assertEqual(summary["type"], "item.completed")
        self.assertEqual(summary["item_type"], "agent_message")
        self.assertEqual(summary["item_status"], "completed")
        self.assertNotIn("text", summary)

    def test_remote_codex_logs_are_streamed_with_redaction(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.stdin = io.StringIO()
                self.stdout = io.StringIO(
                    '{"type":"turn.completed","secret":"sk-abcdefgh12345678"}\n'
                )
                self.stderr = io.StringIO("OPENAI_API_KEY=do-not-log\n")
                self.returncode = 0

            def wait(self, timeout: int) -> int:
                return self.returncode

            def kill(self) -> None:
                self.returncode = -9

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            events = root / "events.jsonl"
            stderr = root / "stderr.log"
            remote = orchestrator.RemoteConfig(
                host="gpu",
                repo=orchestrator.PurePosixPath("/srv/repo"),
                worktree_root=orchestrator.PurePosixPath("/srv/worktrees"),
            )
            client = orchestrator.SSHClient(remote, lambda _message: None)
            with mock.patch.object(
                orchestrator.subprocess, "Popen", return_value=FakeProcess()
            ):
                returncode = client.run_codex(
                    "codex exec -",
                    prompt="safe prompt",
                    timeout=30,
                    events_path=events,
                    stderr_path=stderr,
                )
            event_text = events.read_text(encoding="utf-8")
            stderr_text = stderr.read_text(encoding="utf-8")

        self.assertEqual(returncode, 0)
        self.assertNotIn("sk-abcdefgh12345678", event_text)
        self.assertIn("<redacted-api-key>", event_text)
        self.assertNotIn("do-not-log", stderr_text)


if __name__ == "__main__":
    unittest.main()
