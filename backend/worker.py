"""Worker mode entrypoint. Runs the session-claim loop forever."""

import asyncio

from main import worker_loop

if __name__ == "__main__":
    asyncio.run(worker_loop())
