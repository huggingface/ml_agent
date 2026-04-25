import os
import json
import httpx
from typing import Optional
from urllib.parse import urlparse
from agent.utils.terminal_display import get_console

def get_ollama_base_url() -> str:
    """Read OLLAMA_API_BASE from environment, defaulting to localhost:11434.
    Validates that the hostname is localhost or 127.0.0.1 to mitigate SSRF.
    """
    url = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434").rstrip("/")
    
    # SSRF Mitigation: Restrict local models to localhost/loopback
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    if hostname not in ("localhost", "127.0.0.1"):
        raise ValueError(
            f"Security error: OLLAMA_API_BASE '{url}' must point to localhost or 127.0.0.1"
        )
    return url

async def is_ollama_running() -> bool:
    """Check if the Ollama server is reachable using async httpx."""
    try:
        url = f"{get_ollama_base_url()}/api/tags"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=2.0)
            return response.status_code == 200
    except (httpx.RequestError, ValueError):
        return False

async def is_model_available(model_name: str) -> bool:
    """Check if a specific model is already pulled in Ollama using async httpx."""
    try:
        url = f"{get_ollama_base_url()}/api/tags"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=2.0)
            if response.status_code != 200:
                return False
            
            tags = response.json().get("models", [])
            actual_name = model_name.replace("ollama/", "", 1)
            
            # Ollama tags can be 'name:latest' or just 'name'
            for model in tags:
                name = model.get("name", "")
                if name == actual_name or name == f"{actual_name}:latest":
                    return True
            return False
    except (httpx.RequestError, ValueError):
        return False

async def pull_ollama_model(model_name: str, prompt_session=None) -> bool:
    """Pull a model from Ollama with real-time progress tracking using async httpx."""
    actual_name = model_name.replace("ollama/", "", 1)
    url = f"{get_ollama_base_url()}/api/pull"
    
    get_console().print(f"Pulling '{actual_name}' from Ollama...")
    
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json={"name": actual_name}) as response:
                if response.status_code != 200:
                    get_console().print(f"[bold red]Error pulling model:[/bold red] {response.status_code}")
                    return False
                
                last_status = ""
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        status = data.get("status", "")
                        completed = data.get("completed")
                        total = data.get("total")
                        
                        if status != last_status:
                            if total and completed is not None:
                                percent = (completed / total) * 100
                                print(f"\r{status}: {percent:.1f}%", end="", flush=True)
                            else:
                                print(f"\r{status}", end="", flush=True)
                            last_status = status
                
                get_console().print("\n[green]Pull complete![/green]")
                return True
    except (httpx.RequestError, ValueError) as e:
        get_console().print(f"\n[bold red]Failed to pull model:[/bold red] {e}")
        return False

async def ensure_ollama_readiness(model_id: str, prompt_session) -> bool:
    """
    Check server and model availability. Prompt to pull if missing.
    Returns True if ready to proceed, False otherwise.
    """
    try:
        base_url = get_ollama_base_url()
    except ValueError as e:
        get_console().print(f"\n[bold red]Configuration Error:[/bold red] {e}")
        return False

    if not await is_ollama_running():
        get_console().print(f"\n[bold red]Error:[/bold red] Ollama server is not reachable.")
        get_console().print(f"Make sure 'ollama serve' is running at {base_url}")
        return False

    if not await is_model_available(model_id):
        get_console().print(f"\nModel '{model_id}' not found locally on Ollama.")
        
        try:
            choice = await prompt_session.prompt_async(
                f"Would you like to pull {model_id}? (y/n): "
            )
            if choice.strip().lower() in ("y", "yes"):
                return await pull_ollama_model(model_id)
            else:
                get_console().print("Model pull cancelled.")
                return False
        except (EOFError, KeyboardInterrupt):
            get_console().print("\nCancelled.")
            return False
            
    return True
