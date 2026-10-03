"""Real binary integration; isolated local mock adapters, no ASR model or provider."""
import hashlib
import importlib.util
import json
import os
import pty
import re
import shutil
from pathlib import Path
import subprocess
import tempfile
import threading
import tomllib
from http.client import HTTPConnection
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get("SHUM_TEST_BINARY", ROOT / "target/debug/shum")).resolve()
SPEC = importlib.util.spec_from_file_location("shum_headless_integration", ROOT / "scripts/shum_headless.py")
headless = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(headless)
LOGS = []


@pytest.fixture
def installation():
    if not BINARY.is_file():
        pytest.skip("build shum or set SHUM_TEST_BINARY")
    with tempfile.TemporaryDirectory(prefix="shum-api-e2e-", dir=ROOT.parent) as name:
        root = Path(name)
        home = root / "home"
        home.mkdir(mode=0o700)
        env = {**os.environ, "HOME": str(home), "XDG_CONFIG_HOME": str(home / ".config")}
        env.pop("SHUM_SETTINGS", None)
        layout = {key: str(root / "install" / dirname) for key, dirname in (
            ("installation", ""), ("bin", "bin"), ("settings", "config"),
            ("projects", "projects"), ("models", "models"),
        )}
        try:
            response, code = request(env, "installation.setup", layout)
            assert code == 0 and response["result"]["state"] == "layout_configured"
            yield root, env, layout
        finally:
            while LOGS:
                directory = LOGS.pop()
                for filename in ("stdout.log", "stderr.log"):
                    (directory / filename).unlink(missing_ok=True)
                directory.rmdir()


def request(env, method, params=None, config=None, lines=None):
    command = [str(BINARY)]
    if config:
        command += ["--config", str(config)]
    command += ["api"]
    envelope = {"version": 1, "id": "integration", "method": method, "params": params or {}}
    result = subprocess.run(command, input=lines or json.dumps(envelope) + "\n", text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=30)
    response = json.loads(result.stdout)
    def collect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "log_directory" and isinstance(item, str):
                    LOGS.append(Path(item))
                else:
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
    collect(response)
    return response, result.returncode


def mock_config(root, layout, *, llm=True, roles=("primeline", "flix", "qwen")):
    asr = root / "asr.py"
    asr.write_text("""import argparse,json,pathlib,sys
p=argparse.ArgumentParser()
for k in ('engine','model','threads'):p.add_argument('--'+k)
p.add_argument('--serve',action='store_true');a=p.parse_args()
role=pathlib.Path(a.model).name
for line in sys.stdin:
    request=json.loads(line)
    with pathlib.Path(__file__).with_name('asr-invocations.jsonl').open('a') as log:
        log.write(json.dumps({'role':role,'engine':a.engine,'request':request})+'\\n')
    response={'text':'Guten Tag '+role,'language':'de','segments':[]}
    if 'index' in request:response['index']=request['index']
    print(json.dumps(response),flush=True)
""")
    dia = root / "dia.py"
    dia.write_text("""import argparse,json,pathlib
p=argparse.ArgumentParser()
for k in ('model','input','threads','output'):p.add_argument('--'+k)
a=p.parse_args();pathlib.Path(a.output).write_text(json.dumps({'turns':[],'exclusive_turns':[]}))
""")
    review = root / "review.py"
    review.write_text("""import json,pathlib,sys
p=json.load(sys.stdin)
with pathlib.Path(__file__).with_name('review-prompts.jsonl').open('a') as log:
    log.write(json.dumps(p)+'\\n')
pathlib.Path(sys.argv[1]).write_text(json.dumps({'segments':[{'index':s['index'],'text':'Guten Tag','language':'de','gap':False} for s in p['intervals']]}))
""")
    lines = []
    for role in (*roles, "diarization"):
        model = Path(layout["models"]) / role
        model.mkdir(exist_ok=True)
        (model / "weights").write_bytes(b"mock")
        (model / ".shum-manifest.json").write_text(json.dumps({
            "repo": role, "revision": "test", "files": {"weights": hashlib.sha256(b"mock").hexdigest()},
        }))
        lines += ["[[models]]", f'role="{role}"', f'repo="{role}"', 'revision="test"',
                  f"path={json.dumps(str(model))}"]
    for role in roles:
        lines += [f"[asr.{role}]", f"program={json.dumps(os.sys.executable)}",
                  f"script={json.dumps(str(asr))}", "threads=1"]
        if role not in ("primeline", "flix", "qwen"):
            lines += ['engine="whisper"']
    lines += ["[diarization]", f"program={json.dumps(os.sys.executable)}",
              f"script={json.dumps(str(dia))}", "threads=1", "[llm]",
              f"enabled={'true' if llm else 'false'}", 'provider="custom"',
              f"program={json.dumps(os.sys.executable)}",
              f"args={json.dumps([str(review), '{output}'])}",
              "[resources]", "reserve_mib=0", "max_asr_workers=1", "supervise_asr_with_systemd=false",
              "[resources.estimated_peak_mib]"] + [f"{role}=1" for role in roles]
    config = Path(layout["settings"]) / "config.toml"
    config.write_text("\n".join(lines) + "\n")
    return config


def audio(root):
    path = root / "sample.mp3"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                    "-i", "sine=duration=2", "-y", str(path)], check=True, timeout=10)
    return path


def invocations(root, name):
    path = root / name
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def assert_review_roles(root, roles):
    prompts = invocations(root, "review-prompts.jsonl")
    assert prompts
    for prompt in prompts:
        assert prompt["intervals"]
    prompt = prompts[-1]
    assert [role for role, _ in prompt["intervals"][0]["hypotheses"]] == list(roles)
    instruction = prompt["instruction"]
    assert "no cross-model corroboration" in instruction
    assert "agreement does not prove accuracy or model independence" in instruction


def test_quality_is_offline_advisory_and_preserves_existing_outputs(installation):
    root, env, layout = installation
    config = mock_config(root, layout)
    source = audio(root)
    response, code = request(env, "workflow.start", {"project": "quality", "input": str(source)}, config)
    assert code == 0, response
    project = Path(layout["projects"]) / "quality"
    report_path = project / ".quality/report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text())
    assert report["accuracy_status"] == "unverified"
    assert report["windows"][0]["analysis"]["value"]["source_status"] == "multiple_asr_sources_unverified"
    assert report["automatic_actions"] is False
    assert report["analyzed_windows"] == 1
    assert report["execution_status"] == "complete", report["errors"]
    assert report["audio"]["value"]["partial_coverage"] is False
    assert report["policy"]["enabled"] is True
    before = {path.relative_to(project).as_posix(): path.read_bytes()
              for path in project.rglob("*") if path.is_file() and ".quality" not in path.parts}
    # Diagnostics must work without interpreters, models or LLM access.
    for name in ("asr.py", "dia.py", "review.py"):
        (root / name).unlink()
    response, code = request(env, "project.check_quality", {"project": "quality"}, config)
    assert code == 0, response.get("error", {}).get("data", response)
    assert response["result"]["accuracy_status"] == "unverified"
    assert response["result"]["automatic_actions"] is False
    after = {path.relative_to(project).as_posix(): path.read_bytes()
             for path in project.rglob("*") if path.is_file() and ".quality" not in path.parts}
    assert after == before
    response, code = request(env, "project.quality_report",
                             {"project": "quality", "offset": 0, "limit": 1}, config)
    assert code == 0, response
    assert len(response["result"]["windows"]) == 1
    assert response["result"]["next_offset"] is None
    assert response["result"]["freshness"] == "not_revalidated"
    response, code = request(env, "project.quality_report",
                             {"project": "quality", "limit": 51}, config)
    assert code == 1 and response["error"]["code"] == "invalid_params"
    response, code = request(env, "project.status", {"project": "quality"}, config)
    assert code == 0 and response["result"]["quality"]["accuracy_status"] == "unverified"
    cli = subprocess.run([str(BINARY), "--config", str(config), "project", "check-quality", "quality"],
                         env=env, text=True, capture_output=True, timeout=30)
    assert cli.returncode == 0, cli.stderr
    assert json.loads(cli.stdout)["accuracy_status"] == "unverified"

    locked = (project / ".workflow.lock").open("r+")
    import fcntl
    fcntl.flock(locked.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        response, code = request(env, "project.check_quality", {"project": "quality"}, config)
        assert code == 1 and "already in use" in response["error"]["message"]
    finally:
        locked.close()


def test_quality_partial_disabled_and_explicit_headless_check(installation):
    root, env, layout = installation
    config = mock_config(root, layout, llm=False)
    source = audio(root)
    response, code = request(env, "workflow.step",
                             {"project": "partial-quality", "input": str(source),
                              "step": "chunks", "no_quality": True}, config)
    assert code == 0, response
    project = Path(layout["projects"]) / "partial-quality"
    assert not (project / ".quality").exists()
    payload = {"version": 1, "id": "quality-headless", "method": "project.check_quality",
               "params": {"project": "partial-quality"}}
    result = subprocess.run([os.sys.executable, str(ROOT / "scripts/shum_headless.py"),
                             "--shum", str(BINARY), "--config", str(config),
                             "call", "--request", json.dumps(payload)],
                            env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["result"]["execution_status"] == "partial"
    report = json.loads((project / ".quality/report.json").read_text())
    assert set(report["windows"][0]["missing_roles"]) == {"primeline", "flix", "qwen"}
    assert report["windows"][0]["analysis"]["value"]["source_status"] == "no_sources"
    response, code = request(env, "project.check_quality",
                             {"project": "partial-quality", "unknown": True}, config)
    assert code == 1 and response["error"]["code"] == "invalid_params"


def test_quality_failure_is_explicit_without_blocking_cached_export(installation):
    root, env, layout = installation
    config = mock_config(root, layout)
    response, code = request(env, "workflow.start", {"project": "bad-quality", "input": str(audio(root))}, config)
    assert code == 0, response
    project = Path(layout["projects"]) / "bad-quality"
    exports = [project / "sample.txt", project / "sample.json"]
    before = [path.read_bytes() for path in exports]
    (project / ".quality/report.json").write_text("invalid JSON")
    response, code = request(env, "workflow.resume", {"project": "bad-quality"}, config)
    assert code == 0, response
    assert response["result"]["quality"]["execution_status"] == "failed"
    assert [path.read_bytes() for path in exports] == before
    response, code = request(env, "project.quality_report", {"project": "bad-quality"}, config)
    assert code == 1 and response["error"]["code"] == "invalid_quality_report"


def test_asr_selection_resumes_without_qwen_and_preserves_completed_asr(installation):
    root, env, layout = installation
    config = mock_config(root, layout)
    source = audio(root)
    for step in ("asr:primeline", "asr:flix"):
        response, code = request(env, "workflow.step",
                                 {"project": "selected", "input": str(source), "step": step}, config)
        assert code == 0, response
    project = Path(layout["projects"]) / "selected"
    before = {role: (project / role / "00000.json").read_bytes() for role in ("primeline", "flix")}
    # The disabled role may be completely unavailable.
    config.write_text(re.sub(r"(\[asr\.qwen\][\s\S]*?\nscript=)[^\n]+",
                            lambda match: match[1] + json.dumps(str(root / "missing-qwen.py")),
                            config.read_text(), count=1))
    shutil.rmtree(Path(layout["models"]) / "qwen")
    response, code = request(env, "workflow.resume",
                             {"project": "selected", "asr": ["primeline", "flix"]}, config)
    assert code == 0, response
    assert response["result"]["status"]["asr_roles"] == ["primeline", "flix"]
    assert response["result"]["status"]["stages"]["asr:qwen"]["status"] == "skipped"
    assert {role: (project / role / "00000.json").read_bytes() for role in before} == before
    transcript = json.loads((project / "sample.json").read_text())
    assert [role for role, _ in transcript["segments"][0]["hypotheses"]] == ["primeline", "flix"]
    assert "qwen" not in {model["role"] for model in transcript["models"]}
    assert not (project / "qwen").exists()
    quality = json.loads((project / ".quality/report.json").read_text())
    assert quality["windows"][0]["missing_roles"] == []
    assert quality["windows"][0]["skipped_roles"] == ["qwen"]
    assert quality["windows"][0]["analysis"]["value"]["source_status"] == "multiple_asr_sources_unverified"
    assert quality["asr_roles"] == ["primeline", "flix"]
    # A saved selection records history, not the default of the next invocation.
    manifest = json.loads((project / "manifest.json").read_text())
    manifest.pop("asr_roles")
    (project / "manifest.json").write_text(json.dumps(manifest))
    response, code = request(env, "workflow.resume", {"project": "selected"}, config)
    assert code == 1, response
    assert "qwen" in response["error"]["message"]
    response, code = request(env, "workflow.resume",
                             {"project": "selected", "asr": ["primeline", "flix"]}, config)
    assert code == 0, response
    assert response["result"]["status"]["asr_roles"] == ["primeline", "flix"]


@pytest.mark.parametrize("roles", [[], ["missing"], ["primeline", "primeline"]])
def test_asr_selection_rejects_invalid_api_values(installation, roles):
    root, env, layout = installation
    config = mock_config(root, layout)
    response, code = request(env, "workflow.start",
                             {"project": "invalid-asr", "input": str(audio(root)), "asr": roles}, config)
    assert code == 1 and response["error"]["code"] == "invalid_params"
    assert not (Path(layout["projects"]) / "invalid-asr").exists()


def test_single_asr_cli_and_explicit_reenable(installation):
    root, env, layout = installation
    config = mock_config(root, layout)
    result = subprocess.run([str(BINARY), "--config", str(config), "transcribe", str(audio(root)),
                             "--output", "single-asr", "--asr", "primeline"], env=env,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    project = Path(layout["projects"]) / "single-asr"
    assert not (project / "qwen").exists()
    assert not (project / "flix").exists()
    prime = (project / "primeline/00000.json").read_bytes()
    result = subprocess.run([str(BINARY), "--config", str(config), "project", "resume",
                             "single-asr", "--asr", "flix,primeline"], env=env,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert (project / "primeline/00000.json").read_bytes() == prime
    assert (project / "flix/00000.json").is_file()
    state = json.loads((project / "workflow.json").read_text())
    assert state["asr_roles"] == ["primeline", "flix"]
    response, code = request(env, "workflow.step", {
        "project": "single-asr", "step": "asr:qwen", "asr": ["primeline"],
    }, config)
    assert code == 1 and "outside" in response["error"]["message"]
    response, code = request(env, "workflow.resume", {"project": "single-asr"}, config)
    assert code == 0, response
    assert response["result"]["status"]["asr_roles"] == ["primeline", "flix", "qwen"]
    assert (project / "qwen/00000.json").is_file()
    assert (project / "primeline/00000.json").read_bytes() == prime


def test_cli_rejects_unknown_asr_without_creating_project(installation):
    root, env, layout = installation
    config = mock_config(root, layout, roles=("solo",))
    result = subprocess.run([str(BINARY), "--config", str(config), "transcribe",
                             str(audio(root)), "--output", "unknown", "--asr", "missing"],
                            env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 1
    assert "missing" in result.stderr
    assert not (Path(layout["projects"]) / "unknown").exists()


def test_doctor_checks_all_custom_models_offline_without_llm(installation):
    root, env, layout = installation
    roles = ("alpha", "beta", "gamma", "delta", "epsilon")
    config = mock_config(root, layout, roles=roles)
    config.write_text(re.sub(
        r'(\[llm\]\nenabled=true\nprovider="custom"\nprogram=)[^\n]+',
        r'\1"/nonexistent/provider-cli"', config.read_text(), count=1,
    ))
    tools = root / "local-tools"
    tools.mkdir()
    for name in ("ffmpeg", "ffprobe"):
        installed = shutil.which(name)
        assert installed is not None
        (tools / name).symlink_to(installed)
    doctor_env = {**env, "PATH": str(tools)}
    assert shutil.which("hf", path=doctor_env["PATH"]) is None
    command = [str(BINARY), "--config", str(config), "doctor"]
    healthy = subprocess.run(command, env=doctor_env, text=True,
                             capture_output=True, timeout=30)
    assert healthy.returncode == 0, healthy.stdout + healthy.stderr
    assert "model integrity verified" in healthy.stdout
    assert not (root / "review-prompts.jsonl").exists()
    (Path(layout["models"]) / "gamma/weights").write_bytes(b"damaged")
    corrupt = subprocess.run(command, env=doctor_env, text=True,
                             capture_output=True, timeout=30)
    assert corrupt.returncode == 1
    assert "gamma" in corrupt.stderr
    assert "checksum verification" in corrupt.stderr
    assert not (root / "review-prompts.jsonl").exists()


def test_completed_qwen_is_retained_across_selection_changes(installation):
    root, env, layout = installation
    config = mock_config(root, layout)
    response, code = request(env, "workflow.start",
                             {"project": "cached-asr", "input": str(audio(root))}, config)
    assert code == 0, response
    project = Path(layout["projects"]) / "cached-asr"
    before = {role: (project / role / "00000.json").read_bytes()
              for role in ("primeline", "flix", "qwen")}
    for selection in (["primeline", "flix"], ["qwen", "flix", "primeline"]):
        response, code = request(env, "workflow.resume",
                                 {"project": "cached-asr", "asr": selection}, config)
        assert code == 0, response
        transcript = json.loads((project / "sample.json").read_text())
        assert {role for role, _ in transcript["segments"][0]["hypotheses"]} == set(selection)
        assert {role: (project / role / "00000.json").read_bytes() for role in before} == before
        assert all(response["result"]["status"]["stages"][f"asr:{role}"]["attempts"] == 1
                   for role in before)

def test_custom_asr_roles_expand_and_survive_removed_configuration(installation):
    root, env, layout = installation
    roles = ("nova", "alpha", "beta", "gamma", "delta")
    config = mock_config(root, layout, roles=roles)
    source = audio(root)
    response, code = request(env, "installation.status", config=config)
    assert code == 0, response
    assert [step for step in response["result"]["steps"] if step.startswith("asr:")] == [
        "asr:alpha", "asr:beta", "asr:delta", "asr:gamma", "asr:nova",
    ]
    response, code = request(env, "workflow.start",
                             {"project": "custom", "input": str(source), "asr": ["nova"]}, config)
    assert code == 0, response
    project = Path(layout["projects"]) / "custom"
    nova = (project / "nova/00000.json").read_bytes()
    assert [role for role, _ in json.loads((project / "sample.json").read_text())["segments"][0]["hypotheses"]] == ["nova"]
    response, code = request(env, "workflow.resume",
                             {"project": "custom", "asr": list(roles)}, config)
    assert code == 0, response
    stages = response["result"]["status"]["stages"]
    assert all(stages[f"asr:{role}"]["status"] == "ready" for role in roles)
    assert stages["asr:nova"]["attempts"] == 1
    assert (project / "nova/00000.json").read_bytes() == nova
    assert len(json.loads((project / "sample.json").read_text())["segments"][0]["hypotheses"]) == 5

    mock_config(root, layout, roles=("nova",))
    response, code = request(env, "workflow.resume", {"project": "custom"}, config)
    assert code == 0, response
    assert response["result"]["status"]["asr_roles"] == ["nova"]
    assert response["result"]["status"]["stages"]["asr:alpha"]["status"] == "skipped"
    assert (project / "nova/00000.json").read_bytes() == nova
    report = json.loads((project / ".quality/report.json").read_text())
    assert report["windows"][0]["skipped_roles"] == ["alpha", "beta", "delta", "gamma"]
    assert report["windows"][0]["missing_roles"] == []
    response, code = request(env, "workflow.step",
                             {"project": "custom", "step": "asr:alpha"}, config)
    assert code == 1 and response["error"]["code"] == "invalid_params"


@pytest.mark.parametrize("roles,transport", [
    (("solo",), "cli"), (("primary", "backup"), "api"),
    (("a", "b", "c", "d", "e"), "headless"),
    (tuple("abcdefghijkl"), "api"),
])
def test_dynamic_asr_full_workflow_dispatch_and_review(installation, roles, transport):
    root, env, layout = installation
    config = mock_config(root, layout, roles=roles)
    config.write_text(config.read_text().replace(
        "max_asr_workers=1", "max_asr_workers=5\nmax_parallel_asr_models=2"))
    source = audio(root)
    if transport == "cli":
        result = subprocess.run([str(BINARY), "--config", str(config), "transcribe",
                                 str(source), "--output", "dynamic"], env=env,
                                text=True, capture_output=True, timeout=30)
        assert result.returncode == 0, result.stderr
    elif transport == "headless":
        payload = {"version": 1, "id": "dynamic", "method": "workflow.start",
                   "params": {"project": "dynamic", "input": str(source)}}
        result = subprocess.run([os.sys.executable, str(ROOT / "scripts/shum_headless.py"),
                                 "--shum", str(BINARY), "--config", str(config), "call",
                                 "--request", json.dumps(payload)], env=env, text=True,
                                capture_output=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout)["result"]["outcome"] == "complete"
    else:
        response, code = request(env, "workflow.start",
                                 {"project": "dynamic", "input": str(source)}, config)
        assert code == 0 and response["result"]["outcome"] == "complete", response
    project = Path(layout["projects"]) / "dynamic"
    selected = sorted(roles)
    assert {call["role"] for call in invocations(root, "asr-invocations.jsonl")} == set(roles)
    assert {call["engine"] for call in invocations(root, "asr-invocations.jsonl")} == {"whisper"}
    for role in roles:
        assert (project / role / "00000.json").is_file()
    assert_review_roles(root, selected)
    transcript = json.loads((project / "sample.json").read_text())
    assert [role for role, _ in transcript["segments"][0]["hypotheses"]] == selected
    assert {model["role"] for model in transcript["models"]} == {*roles, "diarization"}
    assert all(hyp["text"] == "Guten Tag " + role
               for role, hyp in transcript["segments"][0]["hypotheses"])
    report = json.loads((project / ".quality/report.json").read_text())
    assert report["asr_roles"] == selected
    assert report["accuracy_status"] == "unverified"
    assert report["windows"][0]["analysis"]["value"]["source_status"] == (
        "single_asr_source" if len(roles) == 1 else "multiple_asr_sources_unverified"
    )
    assert report["windows"][0]["missing_roles"] == []
    assert report["windows"][0]["skipped_roles"] == []
    analysis = report["windows"][0]["analysis"]["value"]
    assert [metric["role"] for metric in analysis["roles"]] == selected
    expected_pairwise_roles = selected[:8]
    assert analysis["pairwise_omitted_roles"] == selected[8:]
    assert len(analysis["pairwise"]) == len(expected_pairwise_roles) * (len(expected_pairwise_roles) - 1) // 2
    assert all(pair["left_role"] in expected_pairwise_roles
               and pair["right_role"] in expected_pairwise_roles for pair in analysis["pairwise"])
    assert ("pairwise_hypothesis_budget" in {finding["code"] for finding in analysis["findings"]}) == (len(roles) > 8)
    assert (project / "sample.txt").is_file()


def test_dynamic_stage_rerun_subset_and_current_default(installation):
    root, env, layout = installation
    config = mock_config(root, layout, roles=("echo", "delta"))
    source = audio(root)
    response, code = request(env, "workflow.step", {
        "project": "stages", "input": str(source), "step": "asr:echo", "asr": ["echo"],
    }, config)
    assert code == 0 and response["result"]["outcome"] == "stage_complete", response
    project = Path(layout["projects"]) / "stages"
    first = (project / "echo/00000.json").read_bytes()
    assert {call["role"] for call in invocations(root, "asr-invocations.jsonl")} == {"echo"}
    response, code = request(env, "workflow.resume", {"project": "stages"}, config)
    assert code == 0, response
    assert response["result"]["status"]["asr_roles"] == ["delta", "echo"]
    assert (project / "echo/00000.json").read_bytes() == first
    assert {call["role"] for call in invocations(root, "asr-invocations.jsonl")} == {"echo", "delta"}
    assert_review_roles(root, ("delta", "echo"))
    response, code = request(env, "workflow.resume",
                             {"project": "stages", "asr": ["echo"]}, config)
    assert code == 0, response
    assert_review_roles(root, ("echo",))
    subset_report = json.loads((project / ".quality/report.json").read_text())
    assert subset_report["windows"][0]["skipped_roles"] == ["delta"]
    assert subset_report["windows"][0]["analysis"]["value"]["source_status"] == "single_asr_source"
    before = len(invocations(root, "asr-invocations.jsonl"))
    response, code = request(env, "workflow.rerun", {
        "project": "stages", "step": "asr:echo", "asr": ["echo"],
    }, config)
    assert code == 0, response
    calls = invocations(root, "asr-invocations.jsonl")
    assert len(calls) == before + 1 and calls[-1]["role"] == "echo"
    assert (project / "attempts/asr-echo/1/echo/00000.json").is_file()
    response, code = request(env, "workflow.resume", {"project": "stages"}, config)
    assert code == 0, response
    assert_review_roles(root, ("delta", "echo"))
    assert len(invocations(root, "asr-invocations.jsonl")) == before + 1


def test_dynamic_config_replacement_and_inactive_memory_estimates(installation):
    root, env, layout = installation
    config = mock_config(root, layout, roles=("old", "shared"))
    source = audio(root)
    response, code = request(env, "workflow.start",
                             {"project": "replace", "input": str(source)}, config)
    assert code == 0, response
    project = Path(layout["projects"]) / "replace"
    old = (project / "old/00000.json").read_bytes()
    shared = (project / "shared/00000.json").read_bytes()
    first_review = (project / "reviews/00000.json").read_bytes()
    first_export = (project / "sample.json").read_bytes()
    mock_config(root, layout, roles=("shared", "new"))
    config.write_text(config.read_text() + "old=1\n")
    response, code = request(env, "workflow.resume", {"project": "replace"}, config)
    assert code == 0, response
    assert response["result"]["status"]["asr_roles"] == ["new", "shared"]
    assert (project / "old/00000.json").read_bytes() == old
    assert (project / "shared/00000.json").read_bytes() == shared
    assert (project / "new/00000.json").is_file()
    assert {call["role"] for call in invocations(root, "asr-invocations.jsonl")} == {
        "old", "shared", "new",
    }
    assert len(invocations(root, "asr-invocations.jsonl")) == 3
    assert_review_roles(root, ("new", "shared"))
    assert (project / "reviews/00000.json").read_bytes() == first_review
    assert (project / "sample.json").read_bytes() != first_export
    assert (project / "attempts/review/1/reviews/00000.json").is_file()
    transcript = json.loads((project / "sample.json").read_text())
    assert {role for role, _ in transcript["segments"][0]["hypotheses"]} == {"new", "shared"}
    assert {model["role"] for model in transcript["models"]} == {
        "new", "shared", "diarization",
    }
    report = json.loads((project / ".quality/report.json").read_text())
    assert report["accuracy_status"] == "unverified"
    assert report["windows"][0]["analysis"]["value"]["source_status"] == "multiple_asr_sources_unverified"
    assert report["windows"][0]["skipped_roles"] == ["old"]


def test_disabled_qwen_resource_wait_does_not_block_resume(installation):
    root, env, layout = installation
    config = mock_config(root, layout)
    source = audio(root)
    for step in ("asr:primeline", "asr:flix"):
        response, code = request(env, "workflow.step",
                                 {"project": "waiting-asr", "input": str(source), "step": step}, config)
        assert code == 0, response
    waiting = Path(layout["projects"]) / "waiting-asr/waiting-resources.json"
    waiting.write_text(json.dumps({
        "role": "qwen", "available_mib": 1 << 60,
        "settings": {"reserve_mib": 0, "max_asr_workers": 1,
                     "estimated_peak_mib": {"primeline": 1, "flix": 1, "qwen": 1}},
    }))
    saved_wait = waiting.read_bytes()
    response, code = request(env, "workflow.resume",
                             {"project": "waiting-asr", "asr": ["primeline", "flix"]}, config)
    assert code == 0, response
    assert waiting.read_bytes() == saved_wait
    response, code = request(env, "workflow.resume",
                             {"project": "waiting-asr", "asr": ["primeline", "flix", "qwen"]}, config)
    assert code == 1 and "waiting for resources after OOM in qwen" in response["error"]["message"]


def test_real_setup_is_idempotent_and_status_is_noninteractive(installation):
    root, env, layout = installation
    assert not (root / "home/.local/bin/shum").exists()
    before = (Path(layout["settings"]) / "config.toml").read_bytes()
    response, code = request(env, "installation.setup", layout)
    assert code == 0 and response["result"]["state"] == "layout_configured"
    assert before == (Path(layout["settings"]) / "config.toml").read_bytes()
    config = Path(layout["settings"]) / "config.toml"
    response, code = request(env, "installation.status", config=config)
    assert code == 0 and response["result"]["state"] == "configured"
    assert response["result"]["llm"]["access_checked"] is False
    response, code = request(env, "models.verify", {"role": "unknown"}, config)
    assert code == 1 and response["error"]["code"] == "invalid_params"
    runtimes = list((Path(layout["installation"]) / "runtime").iterdir())
    assert len(runtimes) == 1
    runtime = runtimes[0]
    for relative in ("scripts/shum_headless.py", "scripts/onboarding-smoke.sh",
                     "docs/USAGE.ru.md", "docs/API.md"):
        assert (runtime / relative).read_bytes() == (ROOT / relative).read_bytes()
    result = subprocess.run([
        os.sys.executable, str(runtime / "scripts/shum_headless.py"),
        "--shum", str(Path(layout["bin"]) / "shum"),
        "--config", str(config), "call", "--request", json.dumps({
            "version": 1, "id": "installed-client",
            "method": "installation.status", "params": {},
        }),
    ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, cwd=root, timeout=10)
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert document["id"] == "installed-client"
    assert document["result"]["state"] == "configured"


def test_reinstall_publishes_missing_runtime_and_preserves_personal_data(installation):
    root, env, layout = installation
    config = mock_config(root, layout, llm=False, roles=("primeline",))
    config.write_text(config.read_text().replace("threads=1", "threads=7"))
    before = tomllib.loads(config.read_text())
    sentinel = Path(layout["projects"]) / "existing-project"
    sentinel.mkdir()
    (sentinel / "personal.txt").write_text("preserve project")
    weights = Path(layout["models"]) / "primeline/weights"
    original_weights = weights.read_bytes()
    runtime = next((Path(layout["installation"]) / "runtime").iterdir())
    runtime.rename(root / "previous-runtime")
    response, code = request(env, "installation.status", config=config)
    assert code == 0 and response["result"]["state"] == "needs_setup_or_repair"
    response, code = request(env, "installation.setup", layout)
    assert code == 0 and response["result"]["state"] == "layout_configured", response
    after = tomllib.loads(config.read_text())
    assert after["models"] == before["models"]
    for key in ("program", "script", "threads"):
        assert after["asr"]["primeline"][key] == before["asr"]["primeline"][key]
    for key in ("program", "provider", "enabled", "args"):
        assert after["llm"][key] == before["llm"][key]
    assert (sentinel / "personal.txt").read_text() == "preserve project"
    assert weights.read_bytes() == original_weights
    assert (runtime / ".complete").is_file()
    assert list(config.parent.glob("config.toml.bak-*"))
    response, code = request(env, "installation.status", config=config)
    assert code == 0 and response["result"]["state"] == "configured"


def test_systemd_supervised_asr_finishes_and_resumes_cached_result(installation):
    if shutil.which("systemctl") is None or shutil.which("systemd-run") is None:
        pytest.skip("systemd user services are unavailable")
    available = subprocess.run(
        ["systemctl", "--user", "show-environment"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=10,
    )
    if available.returncode != 0:
        pytest.skip("systemd user manager is unavailable")
    root, env, layout = installation
    config = mock_config(root, layout, llm=False, roles=("primeline",))
    config.write_text(config.read_text().replace(
        "supervise_asr_with_systemd=false",
        "supervise_asr_with_systemd=true\nsystemd_memory_limit_mib=256\nsystemd_swap_limit_mib=0",
    ))
    params = {"project": "supervised", "step": "asr:primeline",
              "asr": ["primeline"], "input": str(audio(root)), "no_quality": True}
    response, code = request(env, "workflow.step", params, config)
    assert code == 0 and response["result"]["outcome"] == "stage_complete", response
    assert response["result"]["status"]["stages"]["asr:primeline"]["status"] == "ready"
    calls = invocations(root, "asr-invocations.jsonl")
    assert len(calls) == 1 and calls[0]["role"] == "primeline"
    response, code = request(env, "workflow.step", params, config)
    assert code == 0 and response["result"]["outcome"] == "stage_complete", response
    assert invocations(root, "asr-invocations.jsonl") == calls


def test_real_stage_full_resume_and_rerun(installation):
    root, env, layout = installation
    config = mock_config(root, layout)
    source = audio(root)
    response, code = request(env, "workflow.step", {
        "project": "partial", "step": "asr:flix", "input": str(source), "sequential_asr": True,
    }, config)
    assert code == 0 and response["result"]["outcome"] == "stage_complete"
    project = Path(layout["projects"]) / "partial"
    assert (project / "flix/00000.json").is_file()
    assert not (project / "diarization.json").exists()
    assert not (project / "sample.txt").exists()
    response, code = request(env, "workflow.resume", {"project": "partial", "sequential_asr": True}, config)
    assert code == 0 and response["result"]["outcome"] == "complete"
    assert (project / "sample.txt").is_file()
    assert (project / "sample.json").is_file()
    assert response["result"]["exports"] == {"txt": str(project / "sample.txt"),
                                            "json": str(project / "sample.json")}
    assert not (project / "transcript.txt").exists()
    response, code = request(env, "workflow.rerun", {"project": "partial", "step": "export"}, config)
    assert code == 0 and (project / "sample.txt").is_file()
    response, code = request(env, "workflow.step", {"project": "../outside", "step": "audio"}, config)
    assert code == 1 and response["error"]["code"] == "invalid_params"


def test_pending_review_is_not_success_shaped_transcript(installation):
    root, env, layout = installation
    config = mock_config(root, layout, llm=False)
    response, code = request(env, "workflow.start", {
        "project": "waiting", "input": str(audio(root)), "sequential_asr": True,
    }, config)
    assert code == 0 and response["result"]["exit_code"] == 2
    assert response["result"]["outcome"] == "pending_review"
    assert not (Path(layout["projects"]) / "waiting/sample.txt").exists()
    envelope = {"version": 1, "id": "client", "method": "project.status", "params": {"project": "waiting"}}
    result, _ = headless.call_api(envelope, str(BINARY), str(config))
    assert result["result"]["status"]["stages"]["review"]["status"] == "pending_external"


def test_stream_recovers_after_oversized_request(installation):
    _, env, _ = installation
    command = [str(BINARY), "api"]
    payload = "x" * 70000 + "\n" + json.dumps({
        "version": 1, "id": "next", "method": "installation.status", "params": {},
    }) + "\n"
    result = subprocess.run(command, input=payload, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env, timeout=10)
    lines = [json.loads(line) for line in result.stdout.splitlines()]
    assert result.returncode == 1 and len(lines) == 2
    assert lines[0]["error"]["code"] == "request_too_large"
    assert lines[1]["id"] == "next" and "result" in lines[1]


def test_authenticated_http_gateway_against_real_binary(installation):
    root, _, layout = installation
    config = Path(layout["settings"]) / "config.toml"
    token = headless._token(str(root / "http-token"))
    options = SimpleNamespace(shum=str(Path(layout["bin"]) / "shum"),
                              config=str(config), installation=None, timeout=10)
    server = headless.HeadlessServer("127.0.0.1", 0, token, options)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection(*server.server_address, timeout=15)
        connection.request("GET", "/health")
        response = connection.getresponse()
        assert response.status == 401
        response.read()
        connection.close()
        connection = HTTPConnection(*server.server_address, timeout=15)
        connection.request("POST", "/v1/requests", json.dumps({
            "version": 1, "id": "http-real", "method": "installation.status", "params": {},
        }), {"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200
        document = json.loads(response.read())
        assert document["id"] == "http-real"
        assert document["result"]["state"] == "configured"
        assert document["result"]["llm"]["access_checked"] is False
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_api_command_registration_is_explicit_and_validated(installation):
    root, env, layout = installation
    command_dir = root / "commands"
    invalid, code = request(env, "installation.setup", {**layout, "command_dir": "relative"})
    assert code == 1 and invalid["error"]["code"] == "invalid_params"
    invalid, code = request(env, "installation.setup", {**layout, "setup_shell": True})
    assert code == 1 and invalid["error"]["code"] == "invalid_params"
    assert not command_dir.exists()
    env = {**env, "PATH": str(command_dir) + ":" + env["PATH"]}
    raw_directory = root / "absent-parent/../commands"
    response, code = request(env, "installation.setup", {**layout, "command_dir": str(raw_directory)})
    assert code == 0 and response["result"]["command_path"] == str(command_dir / "shum")
    assert (command_dir / "shum").is_symlink()
    assert (command_dir / "shum").resolve() == Path(layout["bin"]) / "shum"
    result = subprocess.run(["shum", "models", "check", "--help"], cwd=root,
                            env=env, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0 and "Usage: shum models check" in result.stdout
    assert not (root / "home/.bashrc").exists()
    response, code = request({**env, "SHELL": "/bin/bash"}, "installation.setup", {
        **layout, "command_dir": str(command_dir), "setup_shell": True,
    })
    assert code == 0 and response["result"]["command_path"] == str(command_dir / "shum")
    assert (root / "home/.bashrc").is_file()
    assert (root / "home/.profile").is_file()


def test_command_repair_does_not_require_valid_runtime_or_config(installation):
    root, env, layout = installation
    runtime = Path(layout["installation"]) / "runtime"
    runtime.rename(root / "runtime-preserved")
    config = Path(layout["settings"]) / "config.toml"
    config.write_text("invalid = [\n")
    command_dir = root / "repair-commands"
    result = subprocess.run([
        str(BINARY), "--installation", layout["installation"], "setup-command",
        "--command-dir", str(command_dir),
    ], cwd=root, env=env, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert config.read_text() == "invalid = [\n"
    assert (command_dir / "shum").resolve() == Path(layout["bin"]) / "shum"
    assert not (root / "home/.bashrc").exists()


def test_interactive_onboarding_registers_command_and_prepares_new_shell():
    if not BINARY.is_file():
        pytest.skip("build shum or set SHUM_TEST_BINARY")
    with tempfile.TemporaryDirectory(prefix="shum-wizard-e2e-", dir=ROOT.parent) as name:
        root = Path(name)
        home = root / "home"
        home.mkdir()
        env = {**os.environ, "HOME": str(home), "XDG_CONFIG_HOME": str(home / ".config"),
               "SHELL": "/bin/bash", "PATH": "/usr/bin:/bin"}
        env.pop("SHUM_SETTINGS", None)
        command = [str(BINARY), "--onboarding"]
        for flag, dirname in (("storage-dir", "install"), ("bin-dir", "bin"),
                              ("settings-dir", "settings"), ("projects-dir", "projects"),
                              ("models-dir", "models")):
            command += ["--" + flag, str(root / dirname)]
        master, slave = pty.openpty()
        child = None
        try:
            child = subprocess.Popen(command, stdin=slave, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, env=env, cwd=root, text=True)
            os.close(slave)
            slave = None
            answers = ["", "", "n", "", "", "", "", "skip"]
            os.write(master, ("\n".join(answers) + "\n").encode())
            stdout, stderr = child.communicate(timeout=30)
            assert child.returncode == 0, stdout + stderr
            assert "Register the `shum` terminal command" in stdout
            assert "Shell PATH configured:" in stdout
        finally:
            if child is not None and child.poll() is None:
                child.kill()
                child.communicate()
            if slave is not None:
                os.close(slave)
            os.close(master)
        launcher = home / ".local/bin/shum"
        assert launcher.is_symlink() and launcher.resolve() == root / "bin/shum"
        result = subprocess.run([
            "/bin/bash", "--noprofile", "--rcfile", str(home / ".bashrc"),
            "-ic", "command -v shum; shum models check --help",
        ], env=env, cwd=root, stdin=subprocess.DEVNULL,
            text=True, capture_output=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines()[0] == str(launcher)
        assert "Usage: shum models check" in result.stdout
        assert not (root / "models/primeline").exists()
        assert not (root / "install/runtime/environments").exists()


@pytest.mark.parametrize("stem", ["Запись 1.v2", "workflow", "WORKFLOW", "manifest", "chunks",
                                 "diarization", "review-flags", "transcript", "."])
def test_exports_use_original_audio_name_and_preserve_internal_files(installation, stem):
    root, env, layout = installation
    config = mock_config(root, layout)
    source = audio(root)
    original = root / (stem + ".mp3")
    source.rename(original)
    response, code = request(env, "workflow.start", {
        "project": "named", "input": str(original), "sequential_asr": True,
    }, config)
    project = Path(layout["projects"]) / "named"
    assert code == 0, (
        [(key, value["status"]) for key, value in
         json.loads((project / "workflow.json").read_text())["stages"].items()
         if value["status"] != "ready"],
        response.get("error", {}).get("message"),
    )
    assert response["result"]["outcome"] == "complete"
    directory = project / "exports" if stem.lower() in {
        "workflow", "manifest", "chunks", "diarization", "review-flags",
    } else project
    expected = {"txt": str(directory / (stem + ".txt")),
                "json": str(directory / (stem + ".json"))}
    assert response["result"]["exports"] == expected
    stdout = Path(response["result"]["log_directory"]) / "stdout.log"
    assert stdout.read_text().splitlines()[-1] == expected["txt"]
    assert Path(expected["txt"]).is_file()
    assert json.loads(Path(expected["json"]).read_text())["schema_version"] == 1
    assert json.loads((project / "workflow.json").read_text())["export_stem"] == stem
    assert json.loads((project / "manifest.json").read_text())["input_sha256"]
    assert isinstance(json.loads((project / "chunks.json").read_text()), list)
    moved = root / "renamed-copy.mp3"
    original.rename(moved)
    response, code = request(env, "workflow.resume", {"project": "named", "input": str(moved)}, config)
    assert code == 0 and response["result"]["exports"] == expected
    response, code = request(env, "workflow.rerun", {"project": "named", "step": "export"}, config)
    assert code == 0 and response["result"]["exports"] == expected
    assert (project / "attempts/export/1" / (stem + ".txt")).is_file()
    response, code = request(env, "project.status", {"project": "named"}, config)
    assert code == 0 and response["result"]["exports"] == expected


def test_legacy_export_migration_preserves_cached_review_and_original_name(installation):
    root, env, layout = installation
    config = mock_config(root, layout)
    source = audio(root)
    response, code = request(env, "workflow.start", {
        "project": "legacy", "input": str(source), "sequential_asr": True,
    }, config)
    assert code == 0
    project = Path(layout["projects"]) / "legacy"
    for extension in ("txt", "json"):
        (project / ("sample." + extension)).rename(project / ("transcript." + extension))
    state = json.loads((project / "workflow.json").read_text())
    state.pop("export_stem")
    fingerprint = ["export", "export-v1", [state["stages"]["review"]["fingerprint"]]]
    state["stages"]["export"]["fingerprint"] = hashlib.sha256(
        json.dumps(fingerprint, separators=(",", ":")).encode()).hexdigest()
    output_hash = hashlib.sha256()
    for filename in ("transcript.txt", "transcript.json"):
        output_hash.update(filename.encode())
        output_hash.update(hashlib.sha256((project / filename).read_bytes()).hexdigest().encode())
    state["stages"]["export"]["output_sha256"] = output_hash.hexdigest()
    (project / "workflow.json").write_text(json.dumps(state))
    review_time = (project / "reviews/00000.json").stat().st_mtime_ns
    moved = root / "different-name.mp3"
    source.rename(moved)
    response, code = request(env, "workflow.resume", {
        "project": "legacy", "input": str(moved),
    }, config)
    assert code == 0 and response["result"]["exports"]["txt"] == str(project / "sample.txt")
    assert (project / "sample.json").is_file()
    assert not (project / "different-name.json").exists()
    assert not (project / "transcript.json").exists()
    assert (project / "attempts/export/1/transcript.json").is_file()
    assert (project / "reviews/00000.json").stat().st_mtime_ns == review_time
    assert response["result"]["status"]["stages"]["review"]["attempts"] == state["stages"]["review"]["attempts"]
