/** Shared model-id constants used by the web UI. */

export const CLAUDE_MODEL_PATH = 'anthropic/claude-opus-4-6';
export const FIRST_FREE_MODEL_PATH = 'moonshotai/Kimi-K2.6';

export function isClaudePath(modelPath: string | undefined): boolean {
  return !!modelPath && modelPath.startsWith('anthropic/');
}
