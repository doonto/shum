"""Execution integration against the real CLI and local, synthetic adapters."""

import fcntl
import json
import os
import pty
import select
import signal
import struct
import subprocess
import termios
import time
from pathlib import Path

import pytest

from test_api_integration import BINARY, installation, mock_config, request


def synthetic_audio(root, seconds=2):
    path = root / "synthetic.mp3"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", f"sine=duration={seconds}", "-y", str(path)],
        check=True, timeout=20,
    )
    return path


def controlled_adapters(root, layout, *, roles=("primeline",), llm=False, delays=None):
    config = mock_config(root, layout, roles=roles, llm=llm)
    config.write_text(config.read_text().replace("threads=1\n", "threads=1\nmax_in_flight=2\n"))
    (root / "asr-delays.json").write_text(json.dumps(delays or {}))
    (root / "asr.py").write_text(
        """import argparse,json,os,pathlib,subprocess,sys,time
p=argparse.ArgumentParser()
for key in ('engine','model','threads'): p.add_argument('--'+key)
p.add_argument('--serve',action='store_true')
a=p.parse_args()
root=pathlib.Path(__file__).parent
role=pathlib.Path(a.model).name
for line in sys.stdin:
    request=json.loads(line)
    index=request.get('index',0)
    with (root/'asr-starts.jsonl').open('a') as log:
        log.write(json.dumps({'role':role,'index':index})+'\\n')
    if (root/'spawn-child').exists() and index==0 and role=='primeline':
        (root/'owned-worker.pid').write_text(str(os.getpid()))
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],
                               stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
        (root/'owned-child.pid').write_text(str(child.pid))
    delays=json.loads((root/'asr-delays.json').read_text())
    remaining=delays.get(f'{role}:{index}',delays.get(role,0))
    while remaining>0:
        wait=min(remaining,0.25)
        time.sleep(wait)
        remaining-=wait
        with (root/'asr-heartbeats.jsonl').open('a') as log:
            log.write(json.dumps({'role':role,'index':index})+'\\n')
    response={'text':'Guten Tag '+role,'language':'de','segments':[]}
    if 'index' in request: response['index']=index
    print(json.dumps(response),flush=True)
    with (root/'asr-finishes.jsonl').open('a') as log:
        log.write(json.dumps({'role':role,'index':index})+'\\n')
"""
    )
    return config


def mock_embedding_hook(root):
    (root / "dia.py").write_text(
        """import argparse,json,pathlib,sys,time
p=argparse.ArgumentParser()
for key in ('model','input','threads','output'): p.add_argument('--'+key)
a=p.parse_args()
(pathlib.Path(__file__).parent/'dia-started').write_text('started')
for completed in (0,1,2,4):
    print(json.dumps(
        {'event':'shum.progress','version':1,'phase':'embeddings',
         'completed':completed,'total':4},
        separators=(',',':')),file=sys.stderr,flush=True)
    time.sleep(0.5)
pathlib.Path(a.output).write_text(json.dumps({'turns':[],'exclusive_turns':[]}))
"""
    )


def records(root, filename):
    path = root / filename
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def owned_child_alive(root):
    marker = root / "owned-child.pid"
    if not marker.exists():
        return False
    pid = int(marker.read_text())
    stat = Path(f"/proc/{pid}/stat")
    try:
        state = stat.read_text().split(") ", 1)[1][0]
    except FileNotFoundError:
        return False
    return state != "Z"


def owned_worker_alive(root):
    marker = root / "owned-worker.pid"
    if not marker.exists():
        return False
    pid = int(marker.read_text())
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return False
    return str(root / "asr.py").encode() in command


def await_condition(predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("timed out waiting for synthetic adapter")


def pty_until(master, text, timeout=20):
    deadline = time.monotonic() + timeout
    output = ""
    while time.monotonic() < deadline:
        ready, _, _ = select.select([master], [], [], 0.1)
        if ready:
            try:
                output += os.read(master, 65536).decode(errors="replace")
            except OSError:
                break
            if text in output:
                return output
    raise AssertionError(f"PTY did not show {text!r}: {output[-1500:]!r}")


def invoke(env, config, method, params):
    response, code = request(env, method, params, config)
    assert code == 0, response
    assert "error" not in response, response
    return response["result"]


def control_request(revision, request_id, action, *, confirmed=False, initiator=None):
    return {
        "version": 1, "request_id": request_id, "expected_revision": revision,
        "initiator": initiator or {"kind": "user"},
        "reason": "Synthetic operator test", "evidence": [{"source": "mock"}],
        "confirmed": confirmed, "action": action,
    }


def rejected(env, config, method, params, code):
    response, status = request(env, method, params, config)
    assert status == 1, response
    assert response["error"]["code"] == code, response
    return response["error"]


def start_api_workflow(env, config, project, source, step=None):
    params = {"project": project, "input": str(source)}
    if step is not None:
        params["step"] = step
    envelope = {"version": 1, "id": "background",
                "method": "workflow.step" if step else "workflow.start",
                "params": params}
    return subprocess.Popen(
        [str(BINARY), "--config", str(config), "api"],
        env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, start_new_session=True,
    ), json.dumps(envelope) + "\n"


def finish_owned(child):
    try:
        stdout, stderr = child.communicate(timeout=45)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.communicate(timeout=5)
    return stdout, stderr


def test_live_plan_read_during_parallel_out_of_order_completion(installation):
    root, env, layout = installation
    config = controlled_adapters(root, layout, delays={"primeline:0": 2.0})
    config.write_text(config.read_text().replace("max_asr_workers=1", "max_asr_workers=2"))
    with config.open("a") as file:
        file.write("[resources.asr_worker_counts]\nprimeline=2\n[resources.asr_worker_peak_mib]\nprimeline=1\n")
    source = synthetic_audio(root, seconds=56)
    child, payload = start_api_workflow(env, config, "parallel", source)
    try:
        child.stdin.write(payload)
        child.stdin.flush()
        await_condition(lambda: records(root, "asr-starts.jsonl"), timeout=20)
        live = invoke(env, config, "project.plan", {"project": "parallel"})
        assert live
        stdout, stderr = finish_owned(child)
        assert child.returncode == 0, stderr
        outcome = json.loads(stdout)
        assert outcome["result"]["outcome"] == "pending_review"
        started = records(root, "asr-starts.jsonl")
        finished = records(root, "asr-finishes.jsonl")
        assert {item["index"] for item in started} == {0, 1, 2}
        assert sorted(item["index"] for item in finished) == [0, 1, 2]
        assert [item["index"] for item in finished].index(1) < [
            item["index"] for item in finished
        ].index(0)
        assert invoke(env, config, "project.plan", {"project": "parallel"}) != live
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.communicate(timeout=5)


def test_diarization_embedding_hook_reports_actual_completed_batches(installation):
    root, env, layout = installation
    config = controlled_adapters(root, layout)
    mock_embedding_hook(root)
    child, payload = start_api_workflow(env, config, "speaker-progress",
                                        synthetic_audio(root), step="diarization")
    try:
        child.stdin.write(payload)
        child.stdin.flush()
        await_condition(lambda: (root / "dia-started").exists(), timeout=20)
        observed = []

        def intermediate_progress():
            plan = invoke(env, config, "project.plan", {"project": "speaker-progress"})["plan"]
            observed.append(plan)
            stage = plan["stages"]["diarization"]
            return stage["numeric_total"] == 4 and 0 < stage["numeric_done"] < 4

        await_condition(intermediate_progress, timeout=10)
        live = observed[-1]["stages"]["diarization"]
        assert live["status"] == "running"
        stdout, stderr = finish_owned(child)
        assert child.returncode == 0, stderr
        assert json.loads(stdout)["result"]["outcome"] == "stage_complete"
        final = invoke(env, config, "project.plan",
                       {"project": "speaker-progress"})["plan"]["stages"]["diarization"]
        assert (final["numeric_done"], final["numeric_total"]) == (4, 4)
        assert "progress" in [event["kind"] for event in invoke(
            env, config, "project.events",
            {"project": "speaker-progress", "after": 0, "limit": 500},
        )["events"]]
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.communicate(timeout=5)


def test_diarization_without_hook_does_not_invent_completed_batches(installation):
    root, env, layout = installation
    config = controlled_adapters(root, layout)
    (root / "dia.py").write_text(
        """import argparse,json,pathlib,time
p=argparse.ArgumentParser()
for key in ('model','input','threads','output'): p.add_argument('--'+key)
a=p.parse_args()
(pathlib.Path(__file__).parent/'dia-started').write_text('started')
time.sleep(1.2)
pathlib.Path(a.output).write_text(json.dumps({'turns':[],'exclusive_turns':[]}))
"""
    )
    child, payload = start_api_workflow(env, config, "unknown-embedding-progress",
                                        synthetic_audio(root), step="diarization")
    try:
        child.stdin.write(payload)
        child.stdin.flush()
        await_condition(lambda: (root / "dia-started").exists(), timeout=20)
        live = invoke(env, config, "project.plan",
                      {"project": "unknown-embedding-progress"})["plan"]["stages"][
                          "diarization"]
        assert live["status"] == "running"
        assert live["numeric_total"] is None
        stdout, stderr = finish_owned(child)
        assert child.returncode == 0, stderr
        assert json.loads(stdout)["result"]["outcome"] == "stage_complete"
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.communicate(timeout=5)


def test_plan_and_local_asr_do_not_probe_provider_before_review(installation):
    root, env, layout = installation
    config = controlled_adapters(root, layout, llm=True, delays={"primeline": 2})
    child, payload = start_api_workflow(env, config, "deferred-review", synthetic_audio(root))
    try:
        child.stdin.write(payload)
        child.stdin.flush()
        await_condition(lambda: records(root, "asr-starts.jsonl"), timeout=20)
        assert records(root, "review-prompts.jsonl") == []
        plan = invoke(env, config, "project.plan", {"project": "deferred-review"})["plan"]
        assert plan["stages"]["review"]["status"] != "ready"
        assert records(root, "review-prompts.jsonl") == []
        stdout, stderr = finish_owned(child)
        assert child.returncode == 0, stderr
        assert json.loads(stdout)["result"]["outcome"] == "complete"
        assert records(root, "review-prompts.jsonl")
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.communicate(timeout=5)


def test_interrupted_running_attempt_is_not_reported_as_complete(installation):
    root, env, layout = installation
    config = controlled_adapters(root, layout, delays={"primeline": 12})
    source = synthetic_audio(root)
    child = subprocess.Popen(
        [str(BINARY), "--config", str(config), "transcribe", str(source),
         "--output", "interrupted"],
        env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    try:
        await_condition(lambda: records(root, "asr-starts.jsonl"), timeout=20)
        await_condition(lambda: records(root, "asr-heartbeats.jsonl"), timeout=5)
        os.kill(child.pid, signal.SIGINT)
        child.communicate(timeout=5)
        plan = invoke(env, config, "project.plan", {"project": "interrupted"})["plan"]
        active = plan["stages"]["asr:primeline"]
        assert active["status"] != "ready"
        assert len(active["done"]) < len(active["units"])
        assert plan["stages"]["export"]["status"] != "ready"
        assert active["numeric_total"] is None or active["numeric_done"] < active["numeric_total"]
        (root / "asr-delays.json").write_text("{}")
        resumed = invoke(env, config, "workflow.resume", {"project": "interrupted"})
        assert resumed["outcome"] == "pending_review"
        assert invoke(env, config, "project.plan", {"project": "interrupted"}
                      )["plan"]["stages"]["asr:primeline"]["status"] == "ready"
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.communicate(timeout=5)


def test_scoped_retry_queues_during_independent_running_role(installation):
    root, env, layout = installation
    config = controlled_adapters(root, layout, roles=("primeline", "flix"),
                                 delays={"primeline": 8})
    config.write_text(config.read_text().replace("max_asr_workers=1", "max_asr_workers=2"))
    child, payload = start_api_workflow(env, config, "retry-while-running",
                                        synthetic_audio(root, seconds=56))
    try:
        child.stdin.write(payload)
        child.stdin.flush()
        await_condition(lambda: len([item for item in records(root, "asr-finishes.jsonl")
                                     if item["role"] == "flix"]) == 3, timeout=30)
        await_condition(lambda: invoke(env, config, "project.plan", {"project": "retry-while-running"})[
            "plan"]["stages"]["asr:flix"]["status"] == "ready", timeout=10)
        before = invoke(env, config, "project.plan",
                        {"project": "retry-while-running"})["plan"]
        assert before["stages"]["asr:primeline"]["status"] == "running"
        retry = control_request(before["revision"], "retry-flix-three",
                                {"action": "retry", "step": "asr:flix",
                                 "fragments": [0, 1, 2], "parameters": {}})
        preview = invoke(env, config, "workflow.control_preview",
                         {"project": "retry-while-running", "request": retry})["preview"]
        assert preview["fragments"] == [0, 1, 2]
        receipt = invoke(env, config, "workflow.control",
                         {"project": "retry-while-running", "request": retry})["receipt"]
        assert receipt["state"] == "queued"
        assert invoke(env, config, "project.plan",
                      {"project": "retry-while-running"})["plan"]["stages"][
                          "asr:primeline"]["status"] == "running"
        stdout, stderr = finish_owned(child)
        assert child.returncode == 0, stderr
        assert json.loads(stdout)["result"]["outcome"] == "pending_review"
        after = invoke(env, config, "project.plan",
                       {"project": "retry-while-running"})["plan"]
        assert after["stages"]["asr:primeline"]["status"] == "ready"
        assert any(item["action"]["action"] == "retry" for item in after["decisions"])
        assert len([item for item in records(root, "asr-starts.jsonl")
                    if item["role"] == "flix"]) >= 6
        assert len([item for item in records(root, "asr-starts.jsonl")
                    if item["role"] == "primeline"]) == 3
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.communicate(timeout=5)


def test_cancel_owned_attempt_stops_spawned_descendant(installation):
    root, env, layout = installation
    config = controlled_adapters(root, layout, delays={"primeline": 15})
    (root / "spawn-child").write_text("")
    child, payload = start_api_workflow(env, config, "owned-cancel", synthetic_audio(root))
    unrelated = subprocess.Popen(
        [os.sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=True,
    )
    try:
        child.stdin.write(payload)
        child.stdin.flush()
        await_condition(lambda: owned_child_alive(root), timeout=20)
        worker_pid = int((root / "owned-worker.pid").read_text())
        descendant_pid = int((root / "owned-child.pid").read_text())
        assert os.getpgid(descendant_pid) == worker_pid
        assert os.getpgid(unrelated.pid) != worker_pid
        plan = invoke(env, config, "project.plan", {"project": "owned-cancel"})["plan"]
        active_id = plan["stages"]["asr:primeline"]["active_attempt"]
        assert active_id
        cancel = control_request(plan["revision"], "cancel-owned-attempt", {
            "action": "cancel_attempt", "attempt_id": active_id,
        }, confirmed=True)
        invoke(env, config, "workflow.control", {"project": "owned-cancel", "request": cancel})
        await_condition(lambda: not owned_child_alive(root), timeout=10)
        result = invoke(env, config, "project.plan", {"project": "owned-cancel"})["plan"]
        assert result["attempts"][active_id]["status"] == "cancelled"
        assert "asr:primeline" not in result["skipped"]
        assert unrelated.poll() is None
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.communicate(timeout=5)
        if owned_child_alive(root):
            os.kill(int((root / "owned-child.pid").read_text()), signal.SIGTERM)
        if owned_worker_alive(root):
            os.kill(int((root / "owned-worker.pid").read_text()), signal.SIGTERM)
        if unrelated.poll() is None:
            unrelated.terminate()
        unrelated.communicate(timeout=5)


@pytest.mark.parametrize("key,action", [
    ("c", "cancel_attempt"),
    ("s", "skip"),
])
def test_pty_selection_requires_confirmation(installation, key, action):
    root, env, layout = installation
    config = controlled_adapters(root, layout, delays={"primeline": 25})
    source = synthetic_audio(root)
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
    child = None
    try:
        child = subprocess.Popen(
            [str(BINARY), "--config", str(config), "transcribe", str(source),
             "--output", "tty-control", "--asr", "primeline"],
            stdin=slave, stderr=slave, stdout=subprocess.PIPE, env={**env, "TERM": "xterm"},
            start_new_session=True,
        )
        os.close(slave)
        slave = None
        await_condition(lambda: records(root, "asr-starts.jsonl"), timeout=20)
        screen = pty_until(master, "ETA: insufficient data")
        assert "accuracy is unverified" in screen
        os.write(master, b"s")
        pty_until(master, "Choose an active attempt")
        os.write(master, b"1\r")
        pty_until(master, "Type c to cancel")
        os.write(master, key.encode() + b"\r")
        pty_until(master, "Type y then Enter to confirm")
        plan = invoke(env, config, "project.plan", {"project": "tty-control"})["plan"]
        assert not plan["decisions"]
        os.write(master, b"n\r")
        pty_until(master, "s select active attempt")
        assert not invoke(env, config, "project.plan",
                          {"project": "tty-control"})["plan"]["decisions"]
        os.write(master, b"s")
        pty_until(master, "Choose an active attempt")
        os.write(master, b"\x1b")
        pty_until(master, "s select active attempt")
        assert not invoke(env, config, "project.plan",
                          {"project": "tty-control"})["plan"]["decisions"]
        os.write(master, b"s")
        pty_until(master, "Choose an active attempt")
        os.write(master, b"1\r")
        pty_until(master, "Type c to cancel")
        os.write(master, key.encode() + b"\r")
        pty_until(master, "Type y then Enter to confirm")
        os.write(master, b"y\r")
        await_condition(lambda: any(
            entry["action"]["action"] == action for entry in
            invoke(env, config, "project.plan", {"project": "tty-control"})["plan"]["decisions"]
        ), timeout=10)
    finally:
        if child is not None:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.communicate(timeout=5)
        if slave is not None:
            os.close(slave)
        os.close(master)


@pytest.mark.parametrize("roles", [
    ("primeline",),
    ("primeline", "flix", "qwen", "meeting", "interview"),
])
def test_read_only_execution_methods_do_not_run_models(installation, roles):
    root, env, layout = installation
    config = controlled_adapters(root, layout, roles=roles)
    source = synthetic_audio(root)
    response = invoke(env, config, "workflow.step",
                      {"project": "plan-only", "step": "chunks", "input": str(source)})
    assert response["outcome"] == "stage_complete"
    plan = invoke(env, config, "project.plan", {"project": "plan-only"})
    events = invoke(env, config, "project.events",
                    {"project": "plan-only", "after": 0, "limit": 10})
    default_page = invoke(env, config, "project.events", {"project": "plan-only"})
    assert plan["plan"]["schema_version"] == 1
    assert all(f"asr:{role}" in plan["plan"]["stages"] for role in roles)
    assert plan["plan"]["stages"]["chunks"]["status"] == "ready"
    assert plan["plan"]["stages"]["chunks"]["numeric_total"] is not None
    assert events["after"] == 0 and events["events"]
    assert default_page["events"][:len(events["events"])] == events["events"]
    assert all(event["version"] == 1 and event["sequence"] > 0
               for event in events["events"])
    first = invoke(env, config, "project.events",
                   {"project": "plan-only", "after": 0, "limit": 1})
    assert len(first["events"]) == 1 and first["next_after"] == first["events"][0]["sequence"]
    if len(events["events"]) > 1:
        next_page = invoke(env, config, "project.events", {
            "project": "plan-only", "after": first["next_after"], "limit": 1,
        })
        assert next_page["events"][0]["sequence"] > first["next_after"]
    assert events["next_after"] == events["events"][-1]["sequence"]
    last = default_page["next_after"]
    replay = invoke(env, config, "project.events",
                    {"project": "plan-only", "after": last, "limit": 10})
    assert replay["events"] == [] and replay["next_after"] == last
    assert records(root, "asr-starts.jsonl") == []


def test_api_stream_is_clean_under_no_color_and_dumb_terminal(installation):
    root, env, layout = installation
    config = controlled_adapters(root, layout)
    source = synthetic_audio(root)
    invoke(env, config, "workflow.step",
           {"project": "plain-stream", "step": "chunks", "input": str(source)})
    envelopes = [
        {"version": 1, "id": "plan", "method": "project.plan",
         "params": {"project": "plain-stream"}},
        {"version": 1, "id": "events", "method": "project.events",
         "params": {"project": "plain-stream", "after": 0, "limit": 2}},
    ]
    result = subprocess.run(
        [str(BINARY), "--config", str(config), "api"],
        input="".join(json.dumps(item) + "\n" for item in envelopes),
        text=True, capture_output=True, env={**env, "NO_COLOR": "1", "TERM": "dumb"},
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "\x1b[" not in result.stdout
    lines = [json.loads(line) for line in result.stdout.splitlines()]
    assert [line["id"] for line in lines] == ["plan", "events"]
    assert all(line["version"] == 1 and "result" in line for line in lines)


@pytest.mark.parametrize("term,no_color", [("dumb", True), ("xterm", False)])
def test_noninteractive_workflow_emits_plain_text_not_terminal_escapes(
        installation, term, no_color):
    root, env, layout = installation
    config = controlled_adapters(root, layout)
    source = synthetic_audio(root)
    plain_env = {**env, "TERM": term}
    if no_color:
        plain_env["NO_COLOR"] = "1"
    else:
        plain_env.pop("NO_COLOR", None)
    result = subprocess.run(
        [str(BINARY), "--config", str(config), "project", "run", "plain-cli",
         "--step", "asr:primeline", "--input", str(source)],
        text=True, capture_output=True, env=plain_env,
        stdin=subprocess.DEVNULL, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "\x1b" not in result.stdout + result.stderr
    assert records(root, "asr-starts.jsonl")


def test_invalid_event_limits_fail_without_mutating_project(installation):
    root, env, layout = installation
    config = controlled_adapters(root, layout)
    invoke(env, config, "workflow.step",
           {"project": "bounded", "step": "chunks", "input": str(synthetic_audio(root))})
    before = invoke(env, config, "project.plan", {"project": "bounded"})["plan"]
    for limit in (0, 501):
        rejected(env, config, "project.events",
                 {"project": "bounded", "after": 0, "limit": limit}, "invalid_params")
    after = invoke(env, config, "project.plan", {"project": "bounded"})["plan"]
    assert after == before


def test_control_preview_is_nonmutating_but_submission_requires_confirmation(installation):
    root, env, layout = installation
    config = controlled_adapters(root, layout)
    invoke(env, config, "workflow.step",
           {"project": "preview", "step": "chunks", "input": str(synthetic_audio(root))})
    before = invoke(env, config, "project.plan", {"project": "preview"})["plan"]
    revision = before["revision"]
    chunks_attempt = next(id for id, attempt in before["attempts"].items()
                          if attempt["step"] == "chunks")
    unconfirmed = control_request(revision, "skip-without-confirmation",
                                  {"action": "skip", "step": "audio"})
    preview = invoke(env, config, "workflow.control_preview",
                     {"project": "preview", "request": unconfirmed})["preview"]
    assert preview["revision"] == revision and "audio" in preview["affected_steps"]
    rejected(env, config, "workflow.control",
             {"project": "preview", "request": unconfirmed}, "control_rejected")
    cases = (
        control_request(revision - 1, "stale",
                        {"action": "invalidate", "step": "chunks"}),
        control_request(revision, "duplicate-fragments",
                        {"action": "retry", "step": "asr:primeline",
                         "fragments": [0, 0], "parameters": {}}),
        control_request(revision, "unknown-fragments",
                        {"action": "retry", "step": "asr:primeline",
                         "fragments": [999], "parameters": {}}),
        control_request(revision, "bad-check-dependency",
                        {"action": "add_check", "step": "asr:primeline",
                         "dependencies": ["nonexistent"], "fragments": []}),
        control_request(revision, "cyclic-check-dependency",
                        {"action": "add_check", "step": "audio",
                         "dependencies": [chunks_attempt], "fragments": []}),
    )
    for item in cases:
        rejected(env, config, "workflow.control_preview",
                 {"project": "preview", "request": item}, "control_rejected")
    assert invoke(env, config, "project.plan", {"project": "preview"})["plan"] == before


def test_confirmed_skip_needs_explicit_restore_and_preserves_personal_config(installation):
    root, env, layout = installation
    config = controlled_adapters(root, layout)
    source = synthetic_audio(root)
    invoke(env, config, "workflow.step",
           {"project": "skip-restore", "step": "chunks", "input": str(source)})
    initial = invoke(env, config, "project.plan", {"project": "skip-restore"})["plan"]
    before_config = config.read_bytes()
    skip = control_request(initial["revision"], "operator-skip-audio",
                           {"action": "skip", "step": "audio"}, confirmed=True)
    preview = invoke(env, config, "workflow.control_preview",
                     {"project": "skip-restore", "request": skip})["preview"]
    assert preview["revision"] == initial["revision"]
    assert "audio" in preview["affected_steps"]
    assert invoke(env, config, "project.plan", {"project": "skip-restore"})["plan"] == initial
    receipt = invoke(env, config, "workflow.control",
                     {"project": "skip-restore", "request": skip})["receipt"]
    assert receipt["request_id"] == skip["request_id"]
    assert receipt["state"] == "queued"
    assert invoke(env, config, "workflow.control",
                  {"project": "skip-restore", "request": skip})["receipt"] == receipt
    response, code = request(env, "workflow.resume", {"project": "skip-restore"}, config)
    assert code != 0 or response["result"]["outcome"] != "complete"
    skipped = invoke(env, config, "project.plan", {"project": "skip-restore"})["plan"]
    assert "audio" in skipped["skipped"]
    response, code = request(env, "workflow.resume", {"project": "skip-restore"}, config)
    assert code != 0 or response["result"]["outcome"] != "complete"
    assert "audio" in invoke(env, config, "project.plan",
                             {"project": "skip-restore"})["plan"]["skipped"]
    latest_revision = invoke(env, config, "project.plan",
                             {"project": "skip-restore"})["plan"]["revision"]
    restore = control_request(latest_revision, "operator-restore-audio",
                              {"action": "restore", "step": "audio"})
    invoke(env, config, "workflow.control",
           {"project": "skip-restore", "request": restore})
    invoke(env, config, "workflow.step",
           {"project": "skip-restore", "step": "chunks"})
    assert "audio" not in invoke(env, config, "project.plan",
                                  {"project": "skip-restore"})["plan"]["skipped"]
    assert config.read_bytes() == before_config


def test_retry_preserves_cached_attempt_and_selection_history(installation):
    root, env, layout = installation
    config = controlled_adapters(root, layout)
    invoke(env, config, "workflow.step", {
        "project": "artifact-history", "step": "asr:primeline",
        "input": str(synthetic_audio(root)),
    })
    original = invoke(env, config, "project.plan", {"project": "artifact-history"})["plan"]
    initial_id = original["stages"]["asr:primeline"]["selected_attempt"]
    original_file = Path(layout["projects"]) / "artifact-history/primeline/00000.json"
    initial_bytes = original_file.read_bytes()
    retry = control_request(original["revision"], "retry-one-window",
                            {"action": "retry", "step": "asr:primeline",
                             "fragments": [0], "parameters": {}})
    invoke(env, config, "workflow.control",
           {"project": "artifact-history", "request": retry})
    invoke(env, config, "workflow.step",
           {"project": "artifact-history", "step": "asr:primeline"})
    after = invoke(env, config, "project.plan", {"project": "artifact-history"})["plan"]
    candidates = [id for id, attempt in after["attempts"].items()
                  if attempt["step"] == "asr:primeline" and id != initial_id
                  and attempt["status"] == "ready" and attempt["artifact"] is not None]
    assert candidates
    assert after["attempts"][initial_id]["artifact"] is not None
    assert original_file.read_bytes() == initial_bytes
    selected = control_request(after["revision"], "select-retry",
                               {"action": "select", "attempt_id": candidates[-1]})
    invoke(env, config, "workflow.control",
           {"project": "artifact-history", "request": selected})
    invoke(env, config, "workflow.step",
           {"project": "artifact-history", "step": "asr:primeline"})
    current = invoke(env, config, "project.plan", {"project": "artifact-history"})["plan"]
    assert current["stages"]["asr:primeline"]["selected_attempt"] == candidates[-1]
    assert current["attempts"][initial_id]["artifact"] is not None
    kinds = [event["kind"] for event in invoke(
        env, config, "project.events",
        {"project": "artifact-history", "after": 0, "limit": 500},
    )["events"]]
    assert "attempt_selected" in kinds


@pytest.mark.parametrize("step", ["asr:primeline", "review"])
def test_skipped_required_source_never_yields_verified_final_export(installation, step):
    root, env, layout = installation
    config = controlled_adapters(root, layout, llm=True)
    source = synthetic_audio(root)
    invoke(env, config, "workflow.step",
           {"project": "missing-source", "step": "chunks", "input": str(source)})
    plan = invoke(env, config, "project.plan", {"project": "missing-source"})["plan"]
    skip = control_request(plan["revision"], "skip-required", {
        "action": "skip", "step": step,
    }, confirmed=True)
    invoke(env, config, "workflow.control",
           {"project": "missing-source", "request": skip})
    response, code = request(env, "workflow.resume", {"project": "missing-source"}, config)
    assert code != 0 or response["result"]["outcome"] != "complete"
    project = Path(layout["projects"]) / "missing-source"
    assert not (project / "synthetic.txt").exists()
    result = invoke(env, config, "project.plan", {"project": "missing-source"})["plan"]
    assert step in result["skipped"]
    assert result["stages"]["export"]["status"] != "ready"
    if step == "asr:primeline":
        assert (project / "diarization.json").is_file()


def test_skipped_diarization_is_visible_as_limited_project(installation):
    root, env, layout = installation
    config = controlled_adapters(root, layout)
    invoke(env, config, "workflow.step",
           {"project": "no-speakers", "step": "chunks", "input": str(synthetic_audio(root))})
    before = invoke(env, config, "project.plan", {"project": "no-speakers"})["plan"]
    control = control_request(before["revision"], "skip-speakers", {
        "action": "skip", "step": "diarization",
    }, confirmed=True)
    invoke(env, config, "workflow.control",
           {"project": "no-speakers", "request": control})
    invoke(env, config, "workflow.step", {"project": "no-speakers", "step": "chunks"})
    plan = invoke(env, config, "project.plan", {"project": "no-speakers"})["plan"]
    assert "diarization" in plan["skipped"]
    assert plan["stages"]["diarization"]["status"] == "skipped"
    assert plan["stages"]["export"]["status"] != "ready"
    assert not (Path(layout["projects"]) / "no-speakers/synthetic.txt").exists()


def test_non_user_retry_budget_rejects_fourth_request(installation):
    root, env, layout = installation
    config = controlled_adapters(root, layout)
    invoke(env, config, "workflow.step", {
        "project": "bounded-retries", "step": "asr:primeline",
        "input": str(synthetic_audio(root)),
    })
    for index in range(3):
        revision = invoke(env, config, "project.plan",
                          {"project": "bounded-retries"})["plan"]["revision"]
        item = control_request(revision, f"rule-retry-{index}", {
            "action": "retry", "step": "asr:primeline",
            "fragments": [0], "parameters": {},
        }, initiator={"kind": "rule", "rule": "synthetic-test"})
        invoke(env, config, "workflow.control",
               {"project": "bounded-retries", "request": item})
        invoke(env, config, "workflow.step",
               {"project": "bounded-retries", "step": "asr:primeline"})
    plan = invoke(env, config, "project.plan", {"project": "bounded-retries"})["plan"]
    assert len([attempt for attempt in plan["attempts"].values()
                if attempt["step"] == "asr:primeline" and attempt["auto_retry"]]) == 3
    fourth = control_request(plan["revision"], "rule-retry-fourth", {
        "action": "retry", "step": "asr:primeline",
        "fragments": [0], "parameters": {},
    }, initiator={"kind": "rule", "rule": "synthetic-test"})
    rejected(env, config, "workflow.control_preview",
             {"project": "bounded-retries", "request": fourth}, "control_rejected")


@pytest.mark.parametrize("form", ["json", "fenced", "crlf", "mixed", "multiple", "schema", "indices"])
def test_review_wrapper_preserves_original_and_uses_existing_validation(installation, form):
    root, env, layout = installation
    config = controlled_adapters(root, layout, llm=True)
    (root / "review.py").write_text(
        """import json,pathlib,sys
prompt=json.load(sys.stdin)
form=FORM
body=json.dumps({'segments':[{'index':s['index'],'text':'Guten Tag','language':'de','gap':False}
                            for s in prompt['intervals']]})
if form=='schema': body='{"segments":null}'
if form=='indices': body=json.dumps({'segments':[{'index':999,'text':'Guten Tag','language':'de','gap':False}]})
if form in ('fenced','schema','indices'): body='```json\\n'+body+'\\n```'
if form=='crlf': body='```json\\r\\n'+body+'\\r\\n```'
if form=='mixed': body='Explanation\\n```json\\n'+body+'\\n```'
if form=='multiple': body='```json\\n'+body+'\\n```\\n```json\\n'+body+'\\n```'
pathlib.Path(__file__).with_name('original-review.txt').write_bytes(body.encode())
pathlib.Path(sys.argv[1]).write_bytes(body.encode())
""".replace("FORM", repr(form))
    )
    response, code = request(env, "workflow.start", {
        "project": "review-wrapper", "input": str(synthetic_audio(root)),
    }, config)
    project = Path(layout["projects"]) / "review-wrapper"
    originals = list(project.rglob("*.response-*.txt"))
    assert originals
    expected = (root / "original-review.txt").read_bytes()
    assert all(path.read_bytes() == expected for path in originals)
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in originals)
    valid = form in ("json", "fenced", "crlf")
    if valid:
        assert code == 0, response
        assert response["result"]["outcome"] == "complete"
        transcript = json.loads((project / "synthetic.json").read_text())
        assert transcript["segments"][0]["text"] == "Guten Tag"
    else:
        assert code != 0 or response["result"]["outcome"] != "complete"
        assert not (project / "synthetic.txt").exists()
        assert invoke(env, config, "project.plan", {"project": "review-wrapper"})["plan"]["stages"]["review"]["status"] != "ready"


@pytest.mark.parametrize("form", ["json", "fenced", "crlf", "mixed", "multiple", "schema", "indices"])
def test_resume_validates_saved_review_wrappers_without_provider_call(installation, form):
    root, env, layout = installation
    config = controlled_adapters(root, layout, llm=True)
    invoke(env, config, "workflow.step", {
        "project": "saved-review-wrapper", "step": "asr:primeline",
        "input": str(synthetic_audio(root)),
    })
    project = Path(layout["projects"]) / "saved-review-wrapper"
    asr = project / "primeline/00000.json"
    previous_asr = (asr.read_bytes(), asr.stat().st_mtime_ns)
    reviews = project / "reviews"
    reviews.mkdir(exist_ok=True)
    saved = reviews / "00000.json"
    body = json.dumps({"segments": [
        {"index": 999 if form == "indices" else 0, "text": "Saved review",
         "language": "de", "gap": False},
    ]})
    if form == "schema":
        body = '{"segments":null}'
    if form in ("fenced", "schema", "indices"):
        body = "```json\n" + body + "\n```"
    elif form == "crlf":
        body = "```json\r\n" + body + "\r\n```"
    elif form == "mixed":
        body = "Explanation\n```json\n" + body + "\n```"
    elif form == "multiple":
        body = "```json\n" + body + "\n```\n```json\n" + body + "\n```"
    original = body.encode()
    saved.write_bytes(original)
    (root / "review.py").write_text("raise AssertionError('cached review must not call provider')\n")
    response, code = request(env, "workflow.resume",
                             {"project": "saved-review-wrapper"}, config)
    assert (asr.read_bytes(), asr.stat().st_mtime_ns) == previous_asr
    if form in ("json", "fenced", "crlf"):
        assert code == 0, response
        assert response["result"]["outcome"] == "complete"
        assert json.loads(saved.read_bytes())["segments"][0]["text"] == "Saved review"
        assert json.loads((project / "synthetic.json").read_bytes())["segments"][0]["text"] == "Saved review"
        if form != "json":
            originals = list(reviews.glob("*.response-*.txt"))
            assert len(originals) == 1
            assert originals[0].read_bytes() == original
            assert originals[0].stat().st_mode & 0o777 == 0o600
    else:
        assert code != 0 or response["result"]["outcome"] != "complete"
        assert saved.read_bytes() == original
        assert not (project / "synthetic.txt").exists()


def test_graph_extension_events_reconnect_without_losing_old_attempts(installation):
    root, env, layout = installation
    config = controlled_adapters(root, layout)
    invoke(env, config, "workflow.step",
           {"project": "extended", "step": "asr:primeline", "input": str(synthetic_audio(root))})
    before = invoke(env, config, "project.plan", {"project": "extended"})["plan"]
    previous = invoke(env, config, "project.events",
                      {"project": "extended", "after": 0, "limit": 500})["next_after"]
    dependency = next(id for id, attempt in before["attempts"].items()
                      if attempt["step"] == "asr:primeline")
    check = control_request(before["revision"], "add-synthetic-check", {
        "action": "add_check", "step": "asr:primeline",
        "dependencies": [dependency], "fragments": [],
    })
    preview = invoke(env, config, "workflow.control_preview",
                     {"project": "extended", "request": check})["preview"]
    assert preview["affected_steps"] == ["asr:primeline"]
    invoke(env, config, "workflow.control", {"project": "extended", "request": check})
    invoke(env, config, "workflow.step", {"project": "extended", "step": "chunks"})
    after = invoke(env, config, "project.plan", {"project": "extended"})["plan"]
    assert set(before["attempts"]).issubset(after["attempts"])
    assert any(attempt["kind"] == "check" and
               dependency in attempt["dependencies"] for attempt in after["attempts"].values())
    page = invoke(env, config, "project.events",
                  {"project": "extended", "after": previous, "limit": 500})
    sequences = [event["sequence"] for event in page["events"]]
    assert page["after"] == previous and page["next_after"] == sequences[-1]
    assert sequences == sorted(set(sequences))
    assert all(number > previous for number in sequences)
    assert {"control_queued", "check_planned", "control_applied"}.issubset(
        {event["kind"] for event in page["events"]})
    check = next(attempt for attempt in after["attempts"].values() if attempt["kind"] == "check")
    assert check["status"] == "ready"
    assert len(check["done"]) == 1
    assert invoke(env, config, "project.events",
                  {"project": "extended", "after": sequences[-1], "limit": 500})["events"] == []
