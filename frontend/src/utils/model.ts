/**
 * Shared model-id constants used by session-create call sites and the
 * ClaudeCapDialog "Use a free model" escape hatch.
 *
 * ChatInput now loads catalog from /api/config/model, but this fallback
 * free-model constant is still used for the Claude-cap escape hatch.
 */

export const CLAUDE_MODEL_PATH = 'anthropic/claude-opus-4-6';
export const FIRST_FREE_MODEL_PATH = 'moonshotai/Kimi-K2.6';

export function isClaudePath(modelPath: string | undefined): boolean {
  return !!modelPath && modelPath.startsWith('anthropic/');
}
