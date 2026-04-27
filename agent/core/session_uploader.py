#!/usr/bin/env python3
"""
Standalone script for uploading session trajectories to HuggingFace.
This runs as a separate process to avoid blocking the main agent.
Uses individual file uploads to avoid race conditions.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Token for session uploads. Fallback chain (least-privilege first) — matches
# backend/kpis_scheduler.py so one write-scoped token on the Space covers every
# telemetry dataset. Never hardcode tokens in source.
_SESSION_TOKEN = (
    os.environ.get("HF_SESSION_UPLOAD_TOKEN")
    or os.environ.get("HF_TOKEN")
    or os.environ.get("HF_ADMIN_TOKEN")
    or ""
)


def upload_session_as_file(
    session_file: str, repo_id: str, max_retries: int = 3
) -> bool:
    """
    Upload a single session as an individual JSONL file (no race conditions)

    Args:
        session_file: Path to local session JSON file
        repo_id: HuggingFace dataset repo ID
        max_retries: Number of retry attempts

    Returns:
        True if successful, False otherwise
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: huggingface_hub library not available", file=sys.stderr)
        return False

    try:
        # Load session data
        with open(session_file, "r") as f:
            data = json.load(f)

        # Check if already uploaded
        upload_status = data.get("upload_status")
        if upload_status == "success":
            return True

        # Use dedicated session upload token (write-only access to session dataset)
        hf_token = _SESSION_TOKEN
        if not hf_token:
            # Update status to failed
            data["upload_status"] = "failed"
            with open(session_file, "w") as f:
                json.dump(data, f, indent=2)
            return False

        # Scrub secrets (HF tokens, API keys, etc.) from messages + events
        # before they leave the local disk. Best-effort regex-based redaction —
        # see agent/core/redact.py for the patterns covered.
        try:
            from agent.core.redact import scrub  # type: ignore
        except Exception:
            # Fallback for environments where the agent package isn't importable
            # (shouldn't happen in our subprocess, but be defensive).
            import importlib.util
            _spec = importlib.util.spec_from_file_location(
                "_redact",
                Path(__file__).parent / "redact.py",
            )
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)  # type: ignore
            scrub = _mod.scrub
        scrubbed_messages = scrub(data["messages"])
        scrubbed_events = scrub(data["events"])
        scrubbed_tools = scrub(data.get("tools") or [])

        # Prepare JSONL content (single line)
        # Store messages/events/tools as JSON strings to avoid schema conflicts
        # across sessions with different tool rosters.
        session_row = {
            "session_id": data["session_id"],
            "session_start_time": data["session_start_time"],
            "session_end_time": data["session_end_time"],
            "model_name": data["model_name"],
            "messages": json.dumps(scrubbed_messages),
            "events": json.dumps(scrubbed_events),
            "tools": json.dumps(scrubbed_tools),
        }

        # Create temporary JSONL file
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tmp:
            json.dump(session_row, tmp)  # Single line JSON
            tmp_path = tmp.name

        try:
            # Generate unique path in repo: sessions/YYYY-MM-DD/session_id.jsonl
            session_id = data["session_id"]
            date_str = datetime.fromisoformat(data["session_start_time"]).strftime(
                "%Y-%m-%d"
            )
            repo_path = f"sessions/{date_str}/{session_id}.jsonl"

            # Upload with retries
            api = HfApi()
            for attempt in range(max_retries):
                try:
                    # Try to create repo if it doesn't exist (idempotent)
                    try:
                        api.create_repo(
                            repo_id=repo_id,
                            repo_type="dataset",
                            private=False,
                            token=hf_token,
                            exist_ok=True,  # Don't fail if already exists
                        )

                    except Exception:
                        # Repo might already exist, continue
                        pass

                    # Upload the session file
                    api.upload_file(
                        path_or_fileobj=tmp_path,
                        path_in_repo=repo_path,
                        repo_id=repo_id,
                        repo_type="dataset",
                        token=hf_token,
                        commit_message=f"Add session {session_id}",
                    )

                    # Update local status to success
                    data["upload_status"] = "success"
                    data["upload_url"] = f"https://huggingface.co/datasets/{repo_id}"
                    with open(session_file, "w") as f:
                        json.dump(data, f, indent=2)

                    return True

                except Exception:
                    if attempt < max_retries - 1:
                        import time

                        wait_time = 2**attempt
                        time.sleep(wait_time)
                    else:
                        # Final attempt failed
                        data["upload_status"] = "failed"
                        with open(session_file, "w") as f:
                            json.dump(data, f, indent=2)
                        return False

        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    except Exception as e:
        print(f"Error uploading session: {e}", file=sys.stderr)
        return False


def retry_failed_uploads(directory: str, repo_id: str):
    """Retry all failed/pending uploads in a directory"""
    log_dir = Path(directory)
    if not log_dir.exists():
        return

    session_files = list(log_dir.glob("session_*.json"))

    for filepath in session_files:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            upload_status = data.get("upload_status", "unknown")

            # Only retry pending or failed uploads
            if upload_status in ["pending", "failed"]:
                upload_session_as_file(str(filepath), repo_id)

        except Exception:
            pass


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: session_uploader.py <command> <args...>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "upload":
        # python session_uploader.py upload <session_file> <repo_id>
        if len(sys.argv) < 4:
            print("Usage: session_uploader.py upload <session_file> <repo_id>")
            sys.exit(1)
        session_file = sys.argv[2]
        repo_id = sys.argv[3]
        success = upload_session_as_file(session_file, repo_id)
        sys.exit(0 if success else 1)

    elif command == "retry":
        # python session_uploader.py retry <directory> <repo_id>
        if len(sys.argv) < 4:
            print("Usage: session_uploader.py retry <directory> <repo_id>")
            sys.exit(1)
        directory = sys.argv[2]
        repo_id = sys.argv[3]
        retry_failed_uploads(directory, repo_id)
        sys.exit(0)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
