#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=1.12.0", "httpx>=0.27.0"]
# ///
"""
Sandbox Tools — Agent-native primitives for HF Space dev-mode sandboxes.

Architecture:
  - Creates a sandbox by duplicating a template Space (runs sandbox_server.py)
  - Waits for it to come online
  - Communicates via HTTPS to the Space's API
  - Optionally deletes the Space when done

Lifecycle:
    sb = Sandbox.create(owner="burtenshaw")         # duplicate private Space, wait, connect
    sb = Sandbox.create(owner="burtenshaw",          # with options
                        hardware="t4-small",
                        private=True,
                        sleep_time=3600)
    sb = Sandbox.connect("burtenshaw/my-sandbox-abc") # attach to existing

    sb.bash("uv run train.py")
    sb.read("/app/train.py")
    sb.edit("/app/train.py", old_str="lr=1e-3", new_str="lr=1e-4")

    sb.delete()                                       # tear down when done

    # Or use as a context manager for automatic cleanup
    with Sandbox.create(owner="burtenshaw") as sb:
        sb.bash("python train.py")
    # Space deleted on exit

Tools: bash, read, write, edit, upload
"""

from __future__ import annotations

import io
import secrets as secrets_lib
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx
from huggingface_hub import CommitOperationAdd, HfApi

TEMPLATE_SPACE = "burtenshaw/sandbox"
HARDWARE_OPTIONS = [
    "cpu-basic",
    "cpu-upgrade",
    "t4-small",
    "t4-medium",
    "a10g-small",
    "a10g-large",
    "a100-large",
]
OUTPUT_LIMIT = 25000
LINE_LIMIT = 4000
DEFAULT_READ_LIMIT = 2000
DEFAULT_TIMEOUT = 240
MAX_TIMEOUT = 1200
WAIT_TIMEOUT = 600
WAIT_INTERVAL = 5
API_WAIT_TIMEOUT = 180
CPU_BASIC_HARDWARE = "cpu-basic"


def _is_transient_space_visibility_error(error: Exception) -> bool:
    """Return True when a newly duplicated Space is not queryable yet."""
    response = getattr(error, "response", None)
    if getattr(response, "status_code", None) == 404:
        return True
    message = str(error)
    return "Repository Not Found" in message or "404 Client Error" in message


_DOCKERFILE = """\
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && \\
    apt-get install -y \\
      bash git git-lfs wget curl procps \\
      htop vim nano jq tmux \\
      build-essential && \\
    rm -rf /var/lib/apt/lists/*

RUN uv pip install --system fastapi uvicorn python-multipart

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \\
    PATH=/home/user/.local/bin:$PATH \\
    PIP_USER=1 \\
    HF_HUB_DISABLE_PROGRESS_BARS=1 \\
    TQDM_DISABLE=1 \\
    HF_HUB_ENABLE_HF_TRANSFER=1 \\
    UV_NO_PROGRESS=1 \\
    PYTHONWARNINGS=ignore::DeprecationWarning

WORKDIR /app
COPY --chown=user . /app

EXPOSE 7860

CMD ["python", "sandbox_server.py"]
"""

_SANDBOX_SERVER = '''\
"""Minimal FastAPI server for sandbox operations."""
import hmac, os, subprocess, pathlib, signal, threading, re, tempfile
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
import uvicorn

_ANSI_RE = re.compile(r'\\x1b\\[[0-9;]*[a-zA-Z]|\\x1b\\].*?\\x07')

def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub('', text)

def _truncate_output(output: str, max_chars: int = 25000, head_ratio: float = 0.25) -> str:
    if len(output) <= max_chars:
        return output
    # Write full output to temp file so LLM can read specific sections
    spill_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', prefix='bash_output_', dir='/tmp', delete=False) as f:
            f.write(output)
            spill_path = f.name
    except Exception:
        pass
    head_budget = int(max_chars * head_ratio)
    tail_budget = max_chars - head_budget
    head = output[:head_budget]
    tail = output[-tail_budget:]
    total = len(output)
    omitted = total - max_chars
    meta = f"\\n\\n... ({omitted:,} of {total:,} chars omitted, showing first {head_budget:,} + last {tail_budget:,}) ...\\n"
    if spill_path:
        meta += f"Full output saved to {spill_path} — use the read tool with offset/limit to inspect specific sections.\\n"
    return head + meta + tail

def _atomic_write(path: pathlib.Path, content: str):
    """Write atomically: temp file + fsync + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        os.write(fd, content.encode("utf-8"))
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(tmp_path, str(path))
        tmp_path = None
    finally:
        if fd is not None:
            os.close(fd)
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

app = FastAPI()

def _bearer_token(header: str) -> str:
    scheme, _, supplied = header.partition(" ")
    if scheme.lower() != "bearer" or not supplied:
        return ""
    return supplied

def _require_auth(request: Request) -> None:
    sandbox_token = os.environ.get("SANDBOX_API_TOKEN") or ""
    if not sandbox_token:
        raise HTTPException(status_code=503, detail="Sandbox API token not configured")
    supplied = _bearer_token(request.headers.get("x-sandbox-authorization", ""))
    if not supplied:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if not hmac.compare_digest(supplied, sandbox_token):
        raise HTTPException(status_code=401, detail="Invalid bearer token")

_AUTH = [Depends(_require_auth)]

# Track active bash processes so they can be killed on cancel
_active_procs = {}  # pid -> subprocess.Popen
_proc_lock = threading.Lock()

class BashReq(BaseModel):
    command: str
    work_dir: str = "/app"
    timeout: int = 120

class ReadReq(BaseModel):
    path: str
    offset: Optional[int] = None
    limit: Optional[int] = 2000

class WriteReq(BaseModel):
    path: str
    content: str

class EditReq(BaseModel):
    path: str
    old_str: str
    new_str: str
    replace_all: bool = False
    mode: str = "replace"

class ExistsReq(BaseModel):
    path: str

# ── Fuzzy matching & edit utilities (embedded) ──

UNICODE_MAP = {
    "\\u2013": "-", "\\u2014": "-", "\\u2212": "-",
    "\\u2018": "'", "\\u2019": "'",
    "\\u201c": \'"\', "\\u201d": \'"\',
    "\\u00a0": " ", "\\u2003": " ", "\\u2002": " ",
    "\\u200b": "", "\\ufeff": "",
}

def _normalize_unicode(s):
    return "".join(UNICODE_MAP.get(c, c) for c in s)

def _fuzzy_find_original(content, pattern):
    """Find the original text in content that matches pattern fuzzily."""
    if pattern in content:
        return pattern, None
    # Pass 2: right-trim
    c_lines = content.split("\\n")
    c_rt = "\\n".join(l.rstrip() for l in c_lines)
    p_rt = "\\n".join(l.rstrip() for l in pattern.split("\\n"))
    if p_rt in c_rt:
        idx = c_rt.index(p_rt)
        start_line = c_rt[:idx].count("\\n")
        n_lines = p_rt.count("\\n") + 1
        matched = "\\n".join(c_lines[start_line:start_line + n_lines])
        return matched, "(matched after trimming trailing whitespace)"
    # Pass 3: both-sides trim
    c_st = "\\n".join(l.strip() for l in c_lines)
    p_st = "\\n".join(l.strip() for l in pattern.split("\\n"))
    if p_st in c_st:
        idx = c_st.index(p_st)
        start_line = c_st[:idx].count("\\n")
        n_lines = p_st.count("\\n") + 1
        matched = "\\n".join(c_lines[start_line:start_line + n_lines])
        return matched, "(matched after trimming whitespace)"
    # Pass 4: unicode normalization
    c_norm = _normalize_unicode(c_st)
    p_norm = _normalize_unicode(p_st)
    if p_norm in c_norm:
        idx = c_norm.index(p_norm)
        start_line = c_norm[:idx].count("\\n")
        n_lines = p_norm.count("\\n") + 1
        matched = "\\n".join(c_lines[start_line:start_line + n_lines])
        return matched, "(matched after unicode normalization)"
    return None, None

def _apply_edit(content, old_str, new_str, mode="replace", replace_all=False):
    """Apply edit. Returns (new_content, count, fuzzy_note) or raises ValueError."""
    if mode == "replace_all":
        replace_all = True
        mode = "replace"
    fuzzy_note = None
    if old_str not in content:
        matched, fuzzy_note = _fuzzy_find_original(content, old_str)
        if matched is None:
            raise ValueError("old_str not found in file.")
        old_str = matched
    count = content.count(old_str)
    if mode == "replace":
        if count > 1 and not replace_all:
            raise ValueError(f"old_str appears {count} times. Use replace_all=true or provide more context.")
        if replace_all:
            return content.replace(old_str, new_str), count, fuzzy_note
        return content.replace(old_str, new_str, 1), 1, fuzzy_note
    elif mode == "append_after":
        if replace_all:
            return content.replace(old_str, old_str + new_str), count, fuzzy_note
        idx = content.index(old_str) + len(old_str)
        return content[:idx] + new_str + content[idx:], 1, fuzzy_note
    elif mode == "prepend_before":
        if replace_all:
            return content.replace(old_str, new_str + old_str), count, fuzzy_note
        idx = content.index(old_str)
        return content[:idx] + new_str + content[idx:], 1, fuzzy_note
    raise ValueError(f"Unknown mode: {mode}")

def _validate_python(content, path=""):
    """Validate Python: syntax, kwargs against real installed signatures, training heuristics.

    Runs inside the sandbox where packages are pip-installed, so we can actually
    import classes and inspect their __init__ signatures to catch kwarg mismatches
    before runtime.
    """
    import ast as _ast, inspect as _inspect, importlib as _il
    warnings = []

    # 1. Syntax check
    try:
        tree = _ast.parse(content)
    except SyntaxError as e:
        warnings.append(f"Python syntax error at line {e.lineno}: {e.msg}")
        return warnings

    # 2. Build import map: name -> module path (from the script's own imports)
    import_map = {}
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ImportFrom) and node.module:
            for alias in (node.names or []):
                local_name = alias.asname or alias.name
                import_map[local_name] = (node.module, alias.name)
        elif isinstance(node, _ast.Import):
            for alias in (node.names or []):
                local_name = alias.asname or alias.name
                import_map[local_name] = (alias.name, None)

    # 3. For each Call node, resolve the callable and check kwargs against signature
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Call):
            continue
        # Skip calls with **kwargs unpacking — we can't statically know those keys
        if any(kw.arg is None for kw in node.keywords):
            continue
        call_kwargs = [kw.arg for kw in node.keywords if kw.arg]
        if not call_kwargs:
            continue

        # Resolve the callable name
        func_name = None
        if isinstance(node.func, _ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, _ast.Attribute):
            func_name = node.func.attr
        if not func_name or func_name not in import_map:
            continue

        # Try to import and inspect the real callable
        module_path, attr_name = import_map[func_name]
        try:
            mod = _il.import_module(module_path)
            obj = getattr(mod, attr_name, None) if attr_name else mod
            if obj is None:
                continue
            sig = _inspect.signature(obj)
            params = sig.parameters
            # If **kwargs is in the signature, any kwarg is valid
            if any(p.kind == _inspect.Parameter.VAR_KEYWORD for p in params.values()):
                continue
            valid_names = set(params.keys())
            for kw_name in call_kwargs:
                if kw_name not in valid_names:
                    warnings.append(
                        f"Invalid kwarg: {func_name}({kw_name}=...) at line {node.lineno} "
                        f"-- not accepted by {module_path}.{attr_name or func_name}()"
                    )
        except Exception:
            pass  # can't import/inspect — skip silently

    # 4. Training script heuristics
    if any(kw in content for kw in ("TrainingArguments", "SFTConfig", "DPOConfig", "GRPOConfig")):
        if "push_to_hub" not in content:
            warnings.append("Training script warning: no \'push_to_hub\' found")
        if "hub_model_id" not in content:
            warnings.append("Training script warning: no \'hub_model_id\' found")
    return warnings

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/bash", dependencies=_AUTH)
def bash(req: BashReq):
    try:
        proc = subprocess.Popen(
            req.command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=req.work_dir, start_new_session=True,
        )
        with _proc_lock:
            _active_procs[proc.pid] = proc
        try:
            stdout, stderr = proc.communicate(timeout=req.timeout)
            output = _strip_ansi(stdout + stderr)
            output = _truncate_output(output)
            return {"success": proc.returncode == 0, "output": output, "error": "" if proc.returncode == 0 else f"Exit code {proc.returncode}"}
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.wait()
            return {"success": False, "output": "", "error": f"Timeout after {req.timeout}s"}
        finally:
            with _proc_lock:
                _active_procs.pop(proc.pid, None)
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}

@app.post("/api/kill", dependencies=_AUTH)
def kill_all():
    """Kill all active bash processes. Called when user cancels."""
    with _proc_lock:
        pids = list(_active_procs.keys())
    killed = []
    for pid in pids:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            killed.append(pid)
        except OSError:
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except OSError:
                pass
    return {"success": True, "output": f"Killed {len(killed)} process(es): {killed}", "error": ""}

@app.post("/api/read", dependencies=_AUTH)
def read(req: ReadReq):
    try:
        p = pathlib.Path(req.path)
        if not p.exists():
            return {"success": False, "output": "", "error": f"File not found: {req.path}"}
        if p.is_dir():
            return {"success": False, "output": "", "error": f"Is a directory: {req.path}"}
        lines = p.read_text().splitlines()
        start = (req.offset or 1) - 1
        end = start + (req.limit or len(lines))
        selected = lines[start:end]
        numbered = "\\n".join(f"{start + i + 1}\\t{line}" for i, line in enumerate(selected))
        return {"success": True, "output": numbered, "error": ""}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}

@app.post("/api/write", dependencies=_AUTH)
def write(req: WriteReq):
    try:
        p = pathlib.Path(req.path)
        _atomic_write(p, req.content)
        msg = f"Wrote {len(req.content)} bytes to {req.path}"
        if p.suffix == ".py":
            warnings = _validate_python(req.content, req.path)
            if warnings:
                msg += "\\n\\nValidation warnings:\\n" + "\\n".join(f"  ! {w}" for w in warnings)
        return {"success": True, "output": msg, "error": ""}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}

@app.post("/api/edit", dependencies=_AUTH)
def edit(req: EditReq):
    try:
        p = pathlib.Path(req.path)
        if not p.exists():
            return {"success": False, "output": "", "error": f"File not found: {req.path}"}
        content = p.read_text()
        if req.old_str == req.new_str:
            return {"success": False, "output": "", "error": "old_str and new_str must differ."}
        try:
            new_content, count, fuzzy_note = _apply_edit(
                content, req.old_str, req.new_str, mode=req.mode, replace_all=req.replace_all
            )
        except ValueError as e:
            return {"success": False, "output": "", "error": str(e)}
        _atomic_write(p, new_content)
        msg = f"Edited {req.path} ({count} replacement{'s' if count > 1 else ''})"
        if fuzzy_note:
            msg += f" {fuzzy_note}"
        if p.suffix == ".py":
            warnings = _validate_python(new_content, req.path)
            if warnings:
                msg += "\\n\\nValidation warnings:\\n" + "\\n".join(f"  ! {w}" for w in warnings)
        return {"success": True, "output": msg, "error": ""}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}

@app.post("/api/exists", dependencies=_AUTH)
def exists(req: ExistsReq):
    return {"success": True, "output": str(pathlib.Path(req.path).exists()).lower(), "error": ""}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
'''


@dataclass
class ToolResult:
    success: bool
    output: str = ""
    error: str = ""

    def __str__(self):
        if self.success:
            return self.output or "(no output)"
        return f"ERROR: {self.error}"

    def to_dict(self) -> dict:
        return {"success": self.success, "output": self.output, "error": self.error}


@dataclass
class Sandbox:
    """
    A handle to an HF Space sandbox.

    Use Sandbox.create() to spin up a new one, or Sandbox.connect() to
    attach to an existing running Space.
    """

    space_id: str
    token: str | None = None
    api_token: str | None = field(default=None, repr=False)
    work_dir: str = "/app"
    timeout: int = DEFAULT_TIMEOUT
    _owns_space: bool = field(default=False, repr=False)
    _base_url: str = field(init=False, repr=False)
    _client: httpx.Client = field(init=False, repr=False)
    _hf_api: HfApi = field(init=False, repr=False)
    _files_read: set = field(init=False, repr=False, default_factory=set)

    def __post_init__(self):
        slug = self.space_id.replace("/", "-")
        # Trailing slash is critical: httpx resolves relative paths against base_url.
        # Without it, client.get("health") resolves to /health instead of /api/health.
        self._base_url = f"https://{slug}.hf.space/api/"
        self._client = httpx.Client(
            base_url=self._base_url,
            headers=self._auth_headers(),
            timeout=httpx.Timeout(MAX_TIMEOUT, connect=30),
            follow_redirects=True,
        )
        self._hf_api = HfApi(token=self.token)

    def _auth_headers(self) -> dict[str, str]:
        """Return headers for private HF Space access plus sandbox API auth.

        Private Spaces require the HF token in ``Authorization`` at the Hub
        edge. The sandbox server requires its control-plane token in the
        dedicated ``X-Sandbox-Authorization`` header.
        """
        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if self.api_token:
            headers["X-Sandbox-Authorization"] = f"Bearer {self.api_token}"
        return headers

    # ── Lifecycle ─────────────────────────────────────────────────

    class Cancelled(Exception):
        """Raised when sandbox creation is cancelled by the user."""

    @classmethod
    def create(
        cls,
        owner: str,
        *,
        name: str | None = None,
        template: str = TEMPLATE_SPACE,
        hardware: str = CPU_BASIC_HARDWARE,
        private: bool = True,
        sleep_time: int | None = None,
        token: str | None = None,
        secrets: dict[str, str] | None = None,
        wait_timeout: int = WAIT_TIMEOUT,
        log: "Callable[[str], object] | None" = None,
        cancel_event: "Any | None" = None,
    ) -> Sandbox:
        """
        Create a new sandbox by duplicating the template Space.

        Generates a unique space name, duplicates the template, waits for it
        to come online, then returns a connected Sandbox.

        Args:
            owner: HF username or org (e.g. "burtenshaw").
            name: Base name for the space. Defaults to "sandbox".
                  A unique suffix is always appended.
            template: Source Space to duplicate (default: burtenshaw/sandbox).
            hardware: Hardware tier (cpu-basic, t4-small, etc.).
            private: Whether the Space should be private. Defaults to True.
            sleep_time: Auto-sleep after N seconds of inactivity.
            token: HF API token (from user's OAuth session).
            wait_timeout: Max seconds to wait for Space to start (default: 300).
            cancel_event: A threading.Event (or compatible) checked during
                          polling loops.  When set, the Space is deleted and
                          Sandbox.Cancelled is raised.

        Returns:
            A Sandbox instance connected to the running Space.
        """
        _log = log or print
        api = HfApi(token=token)

        def _check_cancel():
            if cancel_event and cancel_event.is_set():
                _log("Sandbox creation cancelled by user, cleaning up...")
                try:
                    api.delete_repo(space_id, repo_type="space")
                    _log(f"Deleted Space {space_id}")
                except Exception:
                    pass
                raise cls.Cancelled(f"Sandbox creation cancelled: {space_id}")

        base = name or "sandbox"
        suffix = uuid.uuid4().hex[:8]
        space_id = f"{owner}/{base}-{suffix}"
        sandbox_api_token = secrets_lib.token_urlsafe(32)

        _log(f"Creating sandbox: {space_id} (from {template})...")

        kwargs = {
            "from_id": template,
            "to_id": space_id,
            "repo_type": "space",
            "private": private,
            "space_hardware": hardware,
        }
        if sleep_time is not None:
            kwargs["space_sleep_time"] = sleep_time

        api.duplicate_repo(**kwargs)
        _log(f"Space created: https://huggingface.co/spaces/{space_id}")

        _check_cancel()

        # ``duplicate_repo`` sends hardware and sleepTimeSeconds in the
        # initial create request. Avoid a second /hardware call: deployed HF
        # OAuth tokens can 401 on that endpoint for a just-created private
        # Space even though duplication itself succeeded. We rely on the
        # duplicate endpoint to honor sleepTimeSeconds for upgraded hardware;
        # cpu-basic auto-sleep is fixed by the Hub.
        _log(f"Using duplicated Space hardware: {hardware}")
        if sleep_time is not None:
            if hardware == CPU_BASIC_HARDWARE:
                _log(
                    f"Requested duplicated Space sleep time: {sleep_time}s "
                    "(cpu-basic auto-sleep is fixed by the Hub)"
                )
            else:
                _log(f"Using duplicated Space sleep time: {sleep_time}s")

        # Inject secrets BEFORE uploading server files (which triggers rebuild).
        # Secrets added after a Space is running aren't available until restart,
        # so they must be set before the build/start cycle.
        sandbox_secrets = {**(secrets or {}), "SANDBOX_API_TOKEN": sandbox_api_token}
        if sandbox_secrets:
            for key, val in sandbox_secrets.items():
                api.add_space_secret(space_id, key, val)

        # Upload sandbox server and Dockerfile (triggers rebuild)
        cls._setup_server(space_id, api, log=_log)

        _check_cancel()

        # Wait for it to come online (rebuild + start)
        _log(f"Waiting for Space to start (timeout: {wait_timeout}s)...")
        deadline = time.time() + wait_timeout
        while time.time() < deadline:
            _check_cancel()
            try:
                runtime = api.get_space_runtime(space_id)
            except Exception as e:
                if _is_transient_space_visibility_error(e):
                    _log("  Space runtime not visible yet...")
                    time.sleep(WAIT_INTERVAL)
                    continue
                raise
            if runtime.stage == "RUNNING":
                current_hardware = runtime.hardware or getattr(
                    runtime, "requested_hardware", None
                )
                if current_hardware != hardware:
                    _log(f"  RUNNING on {current_hardware}; waiting for {hardware}...")
                    time.sleep(WAIT_INTERVAL)
                    continue
                _log(f"Space is running (hardware: {runtime.hardware})")
                break
            if runtime.stage in ("RUNTIME_ERROR", "BUILD_ERROR"):
                raise RuntimeError(
                    f"Space failed to start: {runtime.stage}. "
                    f"Check https://huggingface.co/spaces/{space_id}"
                )
            _log(f"  {runtime.stage}...")
            time.sleep(WAIT_INTERVAL)
        else:
            raise TimeoutError(
                f"Space did not start within {wait_timeout}s. "
                f"Check https://huggingface.co/spaces/{space_id}"
            )

        _check_cancel()

        # Wait for the API server to be responsive (non-fatal)
        sb = cls(
            space_id=space_id,
            token=token,
            api_token=sandbox_api_token,
            _owns_space=True,
        )
        try:
            sb._wait_for_api(timeout=API_WAIT_TIMEOUT, log=_log)
        except TimeoutError as e:
            _log(
                f"Warning: API health check timed out ({e}), but Space is RUNNING. Continuing."
            )
        return sb

    @staticmethod
    def _setup_server(
        space_id: str, api: HfApi, *, log: Callable[[str], object] = print
    ) -> None:
        """Upload embedded sandbox server + Dockerfile to the Space (single commit)."""
        log(f"Uploading sandbox server to {space_id}...")
        api.create_commit(
            repo_id=space_id,
            repo_type="space",
            operations=[
                CommitOperationAdd(
                    path_in_repo="sandbox_server.py",
                    path_or_fileobj=io.BytesIO(_SANDBOX_SERVER.encode()),
                ),
                CommitOperationAdd(
                    path_in_repo="Dockerfile",
                    path_or_fileobj=io.BytesIO(_DOCKERFILE.encode()),
                ),
            ],
            commit_message="Setup sandbox server",
        )
        log("Server files uploaded, rebuild triggered.")

    @classmethod
    def connect(
        cls,
        space_id: str,
        *,
        token: str | None = None,
        api_token: str | None = None,
    ) -> Sandbox:
        """
        Connect to an existing running Space.

        Does a health check to verify the Space is reachable.
        """
        sb = cls(
            space_id=space_id,
            token=token,
            api_token=api_token,
            _owns_space=False,
        )
        sb._wait_for_api(timeout=60)
        return sb

    def _wait_for_api(
        self, timeout: int = API_WAIT_TIMEOUT, log: Callable[[str], object] = print
    ):
        """Poll the health endpoint until the server responds."""
        deadline = time.time() + timeout
        last_err = None
        last_status = None
        while time.time() < deadline:
            try:
                resp = self._client.get("health", timeout=10)
                last_status = resp.status_code
                if resp.status_code == 200:
                    log(f"API is responsive at {self._base_url}")
                    return
            except Exception as e:
                last_err = e
            time.sleep(3)
        raise TimeoutError(
            f"Sandbox API at {self._base_url} not responding after {timeout}s. "
            f"Last status: {last_status}, last error: {last_err}"
        )

    def delete(self):
        """Delete the Space. Only works if this Sandbox created it."""
        if not self._owns_space:
            raise RuntimeError(
                f"This Sandbox did not create {self.space_id}. "
                f"Use self._hf_api.delete_repo() directly if you're sure."
            )
        print(f"Deleting sandbox: {self.space_id}...")
        self._hf_api.delete_repo(self.space_id, repo_type="space")
        # Clear ownership so a second cleanup call (e.g. delete_session +
        # _run_session.finally both fire) early-returns instead of retrying
        # a 404 delete and emitting a spurious ERROR log.
        self._owns_space = False
        self._client.close()
        print("Deleted.")

    def pause(self):
        """Pause the Space (stops billing, preserves state)."""
        self._hf_api.pause_space(self.space_id)

    def restart(self):
        """Restart the Space."""
        self._hf_api.restart_space(self.space_id)
        self._wait_for_api()

    @property
    def url(self) -> str:
        """Public URL of the Space."""
        return f"https://huggingface.co/spaces/{self.space_id}"

    @property
    def status(self) -> str:
        """Current Space stage (RUNNING, BUILDING, PAUSED, etc.)."""
        return self._hf_api.get_space_runtime(self.space_id).stage

    def __enter__(self) -> Sandbox:
        return self

    def __exit__(self, *exc):
        if self._owns_space:
            try:
                self.delete()
            except Exception as e:
                print(f"Warning: failed to delete sandbox: {e}", file=sys.stderr)
        self._client.close()

    # ── HTTP plumbing ─────────────────────────────────────────────

    def _call(
        self, endpoint: str, payload: dict, timeout: float | None = None
    ) -> ToolResult:
        # Strip leading slash for correct httpx base_url resolution
        endpoint = endpoint.lstrip("/")
        effective_timeout = timeout or self.timeout
        last_error = ""

        # Retry up to 3 times for transient failures (sandbox waking from
        # sleep returns empty / non-JSON responses while it starts up).
        for attempt in range(3):
            try:
                resp = self._client.post(
                    endpoint,
                    json=payload,
                    timeout=effective_timeout,
                )
                try:
                    data = resp.json()
                except (ValueError, UnicodeDecodeError):
                    # Non-JSON response — sandbox is likely still starting up.
                    body_preview = resp.text[:200] if resp.text else "(empty)"
                    last_error = (
                        f"Sandbox returned non-JSON response (HTTP {resp.status_code}): "
                        f"{body_preview}"
                    )
                    if attempt < 2:
                        time.sleep(3 * (attempt + 1))
                        continue
                    return ToolResult(success=False, error=last_error)

                if resp.status_code == 200:
                    return ToolResult(
                        success=data.get("success", True),
                        output=data.get("output", ""),
                        error=data.get("error", ""),
                    )
                return ToolResult(
                    success=False,
                    error=data.get("error", f"HTTP {resp.status_code}"),
                )
            except httpx.TimeoutException:
                return ToolResult(
                    success=False, error=f"Timeout after {effective_timeout}s"
                )
            except httpx.ConnectError:
                last_error = (
                    f"Cannot connect to sandbox. Is {self.space_id} running? "
                    f"Status: {self.status}"
                )
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    continue
                return ToolResult(success=False, error=last_error)
            except Exception as e:
                return ToolResult(success=False, error=str(e))

        return ToolResult(success=False, error=last_error or "Unknown error")

    # ── Tools ─────────────────────────────────────────────────────

    def bash(
        self,
        command: str,
        *,
        work_dir: str | None = None,
        timeout: int | None = None,
        description: str | None = None,
    ) -> ToolResult:
        return self._call(
            "bash",
            {
                "command": command,
                "work_dir": work_dir or self.work_dir,
                "timeout": min(timeout or self.timeout, MAX_TIMEOUT),
            },
            timeout=timeout,
        )

    def read(
        self, path: str, *, offset: int | None = None, limit: int | None = None
    ) -> ToolResult:
        self._files_read.add(path)
        return self._call(
            "read",
            {
                "path": path,
                "offset": offset,
                "limit": limit or (DEFAULT_READ_LIMIT if offset is None else None),
            },
        )

    def write(self, path: str, content: str) -> ToolResult:
        if path not in self._files_read:
            check = self._call("exists", {"path": path})
            if check.success and check.output == "true":
                return ToolResult(
                    success=False,
                    error=(
                        f"File {path} exists but has not been read this session. "
                        f"Read it first, or use sandbox_edit for targeted changes."
                    ),
                )
        result = self._call("write", {"path": path, "content": content})
        if result.success:
            self._files_read.add(path)
        return result

    def edit(
        self,
        path: str,
        old_str: str,
        new_str: str,
        *,
        replace_all: bool = False,
        mode: str = "replace",
    ) -> ToolResult:
        if old_str == new_str:
            return ToolResult(success=False, error="old_str and new_str are identical.")
        if path not in self._files_read:
            return ToolResult(
                success=False,
                error=f"File {path} has not been read this session. Read it first.",
            )
        return self._call(
            "edit",
            {
                "path": path,
                "old_str": old_str,
                "new_str": new_str,
                "replace_all": replace_all,
                "mode": mode,
            },
        )

    def kill_all(self) -> ToolResult:
        """Kill all active bash processes on the sandbox. Used on cancellation."""
        return self._call("kill", {})

    # ── Tool schemas & dispatch ───────────────────────────────────

    TOOLS = {
        "bash": {
            "description": (
                "Run a shell command in the remote sandbox and return stdout/stderr.\n"
                "\n"
                "IMPORTANT: Do NOT use bash for file operations — use the dedicated tools instead:\n"
                "- To read files: use read (not cat/head/tail)\n"
                "- To edit files: use edit (not sed/awk)\n"
                "- To write files: use write (not echo/cat <<EOF)\n"
                "\n"
                "Commands run in a shell at /app. Each invocation is independent — "
                "use files in /app to persist state.\n"
                "Chain dependent commands with &&. Independent commands should be "
                "separate bash calls (they can run in parallel).\n"
                "\n"
                "For long-running commands (training, evaluation), run in the background and poll:\n"
                "  nohup <command> > /app/output.log 2>&1 & echo $!\n"
                "Then check status:\n"
                "  kill -0 <PID> 2>/dev/null && echo 'running' || echo 'done'\n"
                "  tail -n 50 /app/output.log\n"
                "\n"
                "Timeout default 240s, max 1200s."
            ),
            "parameters": {
                "type": "object",
                "required": ["command"],
                "additionalProperties": False,
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Short description (5-10 words, active voice).",
                    },
                    "work_dir": {
                        "type": "string",
                        "description": "Working directory (default: /app).",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Optional timeout in seconds (default: 240, max: 1200).",
                    },
                },
            },
        },
        "read": {
            "description": (
                "Reads a file from the sandbox filesystem. Returns contents with line "
                "numbers (cat -n format).\n"
                "\n"
                "Usage:\n"
                "- By default, reads up to 2000 lines from the beginning of the file.\n"
                "- You can optionally specify offset and limit for large files, but prefer "
                "reading the whole file first.\n"
                "- Lines longer than 4000 chars are truncated.\n"
                "- Cannot read directories — use bash with 'ls' instead.\n"
                "- You should read multiple potentially useful files in parallel when possible.\n"
                "- IMPORTANT: Always read a file before editing or overwriting it. The edit and "
                "write tools will reject operations on files you haven't read."
            ),
            "parameters": {
                "type": "object",
                "required": ["path"],
                "additionalProperties": False,
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to read.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "The line number to start reading from (1-based). Only provide if the file is too large to read at once.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "The number of lines to read. Only provide if the file is too large to read at once.",
                    },
                },
            },
        },
        "write": {
            "description": (
                "Writes a file to the sandbox filesystem. Overwrites the existing file if "
                "one exists at the path.\n"
                "\n"
                "- If this is an existing file, you MUST use the read tool first. This tool "
                "will fail if you did not read the file first.\n"
                "- ALWAYS prefer editing existing files with the edit tool over overwriting "
                "with write.\n"
                "- Creates parent directories as needed."
            ),
            "parameters": {
                "type": "object",
                "required": ["path", "content"],
                "additionalProperties": False,
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete file content to write.",
                    },
                },
            },
        },
        "edit": {
            "description": (
                "Performs string replacements in files. Supports exact matching with "
                "fuzzy fallback.\n"
                "\n"
                "Usage:\n"
                "- You must read the file at least once before editing. This tool will "
                "error if you attempt an edit without reading the file.\n"
                "- The edit will FAIL if old_str is not unique in the file. Either provide "
                "a larger string with more surrounding context to make it unique, or set "
                "replace_all to true.\n"
                "- old_str and new_str must differ.\n"
                "- Preserve indentation exactly as it appears in the file.\n"
                "- Do NOT include line number prefixes from read output in old_str or new_str.\n"
                "- To delete code, set new_str to empty string.\n"
                "- Use replace_all for renaming variables or strings across the file.\n"
                "\n"
                "Modes:\n"
                "- replace (default): replace first occurrence of old_str with new_str.\n"
                "- append_after: insert new_str immediately after old_str (old_str is kept).\n"
                "- prepend_before: insert new_str immediately before old_str (old_str is kept)."
            ),
            "parameters": {
                "type": "object",
                "required": ["path", "old_str", "new_str"],
                "additionalProperties": False,
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to edit.",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "The text to find in the file. Must match exactly (fuzzy matching is used as fallback).",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "The replacement text. For append_after/prepend_before modes, the text to insert.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences of old_str (default: false).",
                        "default": False,
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["replace", "append_after", "prepend_before"],
                        "description": "Edit mode (default: replace).",
                        "default": "replace",
                    },
                },
            },
        },
    }

    @classmethod
    def tool_definitions(cls) -> list[dict]:
        return [{"name": name, **spec} for name, spec in cls.TOOLS.items()]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        dispatch = {
            "bash": lambda a: self.bash(
                a["command"],
                work_dir=a.get("work_dir"),
                timeout=a.get("timeout"),
                description=a.get("description"),
            ),
            "read": lambda a: self.read(
                a["path"],
                offset=a.get("offset"),
                limit=a.get("limit"),
            ),
            "write": lambda a: self.write(a["path"], a["content"]),
            "edit": lambda a: self.edit(
                a["path"],
                a["old_str"],
                a["new_str"],
                replace_all=a.get("replace_all", False),
                mode=a.get("mode", "replace"),
            ),
        }
        fn = dispatch.get(name)
        if not fn:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        return fn(arguments)
