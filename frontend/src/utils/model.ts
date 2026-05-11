/**
 * Shared model-id constants used by session-create call sites and the
 * premium-model cap dialog "Use a free model" escape hatch.
 *
 * Keep in sync with MODEL_OPTIONS in components/Chat/ChatInput.tsx and
 * AVAILABLE_MODELS in backend/routes/agent.py.
 */

export const CLAUDE_MODEL_PATH = 'bedrock/us.anthropic.claude-opus-4-6-v1';
export const GPT_55_MODEL_PATH = 'openai/gpt-5.5';
export const FIRST_FREE_MODEL_PATH = 'moonshotai/Kimi-K2.6';

export function isClaudePath(modelPath: string | undefined): boolean {
  return !!modelPath && modelPath.includes('anthropic');
}

export function isPremiumPath(modelPath: string | undefined): boolean {
  return modelPath === CLAUDE_MODEL_PATH || modelPath === GPT_55_MODEL_PATH;
}
