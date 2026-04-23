/**
 * Shared model-id constants used by session-create call sites and the
 * ClaudeCapDialog "Use a free model" escape hatch.
 *
 * Keep in sync with MODEL_OPTIONS in components/Chat/ChatInput.tsx and
 * AVAILABLE_MODELS in backend/routes/agent.py. Bare HF ids (no
 * `huggingface/` prefix) — matches upstream's auto-router.
 */

export const CLAUDE_MODEL_PATH = 'anthropic/claude-opus-4-6';
export const FIRST_FREE_MODEL_PATH = 'moonshotai/Kimi-K2.6';

export function isClaudePath(modelPath: string | undefined): boolean {
  return !!modelPath && modelPath.startsWith('anthropic/');
}
