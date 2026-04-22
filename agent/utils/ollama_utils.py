"""Utilities for local Ollama server management."""

from __future__ import annotations

import os

import httpx


def get_ollama_base_url() -> str:
    """Return the Ollama base URL from env, defaulting to localhost:11434."""
    base = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
    # Strip any trailing /v1 or slash so callers get the clean base
    base = base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


def is_ollama_running() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        resp = httpx.get(f"{get_ollama_base_url()}/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


def is_model_available(model_name: str) -> bool:
    """Return True if *model_name* (with or without ollama/ prefix) is pulled locally."""
    # Strip prefix
    name = model_name.removeprefix("ollama/")
    try:
        resp = httpx.get(f"{get_ollama_base_url()}/api/tags", timeout=2.0)
        if resp.status_code != 200:
            return False
        models = resp.json().get("models", [])
        available = {m["name"] for m in models}
        # Match exact name or name without :latest tag
        return (
            name in available
            or f"{name}:latest" in available
            or any(m.split(":")[0] == name for m in available)
        )
    except Exception:
        return False


async def pull_ollama_model(model_name: str, prompt_session=None) -> bool:
    """Stream-pull *model_name* from Ollama. Returns True on success."""
    import json as _json

    name = model_name.removeprefix("ollama/")
    print(f"Pulling {name} from Ollama registry...")
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{get_ollama_base_url()}/api/pull",
                json={"name": name},
            ) as resp:
                if resp.status_code != 200:
                    print(f"Pull failed (HTTP {resp.status_code}).")
                    return False
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    status = data.get("status", "")
                    completed = data.get("completed")
                    total = data.get("total")
                    if completed and total and total > 0:
                        pct = int(100 * completed / total)
                        print(f"\r  {status}: {pct}%", end="", flush=True)
                    elif status:
                        print(f"\r  {status}          ", end="", flush=True)
                print()  # newline after progress
        return is_model_available(name)
    except Exception as e:
        print(f"Pull error: {e}")
        return False


async def ensure_ollama_readiness(model_name: str, prompt_session=None) -> bool:
    """
    For ollama/ models: verify the server is up and the model is pulled.
    Offers to pull if missing. Returns True if ready, False otherwise.
    Non-ollama models always return True immediately.
    """
    if not model_name.startswith("ollama/"):
        return True

    if not is_ollama_running():
        print(
            "\nOllama server is not running.\n"
            "Start it with:  ollama serve\n"
            "Then re-run or switch models with /model."
        )
        return False

    if is_model_available(model_name):
        return True

    name = model_name.removeprefix("ollama/")
    print(f"\nModel '{name}' is not available locally.")

    if prompt_session:
        try:
            from prompt_toolkit.formatted_text import HTML

            answer = await prompt_session.prompt_async(
                HTML(f"<b>Pull '{name}' now? (y/n): </b>")
            )
            if answer.strip().lower() in ("y", "yes"):
                return await pull_ollama_model(model_name)
        except (EOFError, KeyboardInterrupt):
            pass
    else:
        # Headless — pull automatically
        return await pull_ollama_model(model_name)

    print(f"Run:  ollama pull {name}  — then retry.")
    return False
