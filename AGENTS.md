# Agent Notes

## Local Dev Servers

- Frontend: from `frontend/`, run `npm ci` if dependencies are missing, then `npm run dev`.
- Backend: from `backend/`, run `uv run uvicorn main:app --host ::1 --port 7860`.
- Frontend URL: http://localhost:5173/
- Backend health check: `curl -g http://[::1]:7860/api`
- Frontend proxy health check: `curl http://localhost:5173/api`

Notes:

- Vite proxies `/api` and `/auth` to `http://localhost:7860`.
- If `127.0.0.1:7860` is already owned by another local process, binding the backend to `::1` lets the Vite proxy resolve `localhost` cleanly.
- Prefer `npm ci` over `npm install` for setup, since `npm install` may rewrite `frontend/package-lock.json` metadata depending on npm version.
- Production defaults to the Bedrock Claude model. For local development with a personal Anthropic key, set `ANTHROPIC_API_KEY` and `ML_INTERN_CLAUDE_MODEL_ID=anthropic/claude-opus-4-6` before starting the backend. Other models are selected through the app's model switcher.

## GitHub CLI

- For multiline PR descriptions, prefer `gh pr edit <number> --body-file <file>` over inline `--body` so shell quoting, `$` env-var names, backticks, and newlines are preserved correctly.
