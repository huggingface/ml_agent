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

## Development Checks

- Before every commit, run `uv run ruff check .` and `uv run ruff format --check .`.
- If formatting fails, run `uv run ruff format .`, then re-run the Ruff checks before committing.

## GitHub CLI

- For multiline PR descriptions, prefer `gh pr edit <number> --body-file <file>` over inline `--body` so shell quoting, `$` env-var names, backticks, and newlines are preserved correctly.

## GitHub PRs

- Open code changes as GitHub PRs first. Do not push code changes directly to the Hugging Face Space deployment branch or Space remote before the PR has been opened, reviewed, and merged, unless the user explicitly asks to bypass the PR flow.

## Hugging Face Space Deploys

- The Space remote is `space` and points to `https://huggingface.co/spaces/smolagents/ml-intern`.
- Deploy GitHub `main` to the Space from the local `space-main` branch by merging `origin/main` into `space-main` with a single merge commit, then pushing `space-main:main` to the `space` remote.
- Keep the Space-only README frontmatter on `space-main`; `.gitattributes` should contain `README.md merge=ours` and the local repo config should include `merge.ours.driver=true`.
- Recommended deploy flow:

```bash
git pull --ff-only origin main
git switch space-main
git config merge.ours.driver true
git merge --no-ff origin/main -m "Deploy $(date +%Y-%m-%d)" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>"
git push space space-main:main
git switch main
```
