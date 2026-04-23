import { useState, useCallback, useEffect, useMemo, useRef, type KeyboardEvent, type MouseEvent } from 'react';
import { Box, TextField, IconButton, CircularProgress, Typography, Menu, MenuItem, ListItemIcon, ListItemText, Chip } from '@mui/material';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import StopIcon from '@mui/icons-material/Stop';
import { apiFetch } from '@/utils/api';
import { useUserQuota } from '@/hooks/useUserQuota';
import ClaudeCapDialog from '@/components/ClaudeCapDialog';
import { useAgentStore } from '@/store/agentStore';
import { FIRST_FREE_MODEL_PATH } from '@/utils/model';

interface ModelOption {
  id: string;
  label: string;
  description: string;
  avatarUrl: string;
  providerLabel?: string;
  recommended?: boolean;
}

interface ModelCatalogResponse {
  current?: string;
  available?: unknown;
}

interface SessionResponse {
  model?: string;
}

const FALLBACK_MODELS: ModelOption[] = [
  {
    id: 'anthropic/claude-opus-4-6',
    label: 'Claude Opus 4.6',
    description: 'Anthropic',
    avatarUrl: 'https://huggingface.co/api/avatars/Anthropic',
    providerLabel: 'Anthropic',
    recommended: true,
  },
  {
    id: 'MiniMaxAI/MiniMax-M2.7',
    label: 'MiniMax M2.7',
    description: 'HF Router',
    avatarUrl: 'https://huggingface.co/api/avatars/MiniMaxAI',
    providerLabel: 'Hugging Face Router',
    recommended: true,
  },
  {
    id: 'moonshotai/Kimi-K2.6',
    label: 'Kimi K2.6',
    description: 'HF Router',
    avatarUrl: 'https://huggingface.co/api/avatars/moonshotai',
    providerLabel: 'Hugging Face Router',
  },
  {
    id: 'zai-org/GLM-5.1',
    label: 'GLM 5.1',
    description: 'HF Router',
    avatarUrl: 'https://huggingface.co/api/avatars/zai-org',
    providerLabel: 'Hugging Face Router',
  },
];

const isRecord = (value: unknown): value is Record<string, unknown> => (
  typeof value === 'object' && value !== null
);

const toModelOption = (value: unknown): ModelOption | null => {
  if (!isRecord(value)) return null;
  if (typeof value.id !== 'string' || typeof value.label !== 'string') return null;

  const description = typeof value.description === 'string'
    ? value.description
    : typeof value.providerLabel === 'string'
      ? value.providerLabel
      : '';

  return {
    id: value.id,
    label: value.label,
    description,
    avatarUrl: typeof value.avatarUrl === 'string'
      ? value.avatarUrl
      : 'https://huggingface.co/api/avatars/huggingface',
    providerLabel: typeof value.providerLabel === 'string' ? value.providerLabel : undefined,
    recommended: Boolean(value.recommended),
  };
};

const makeUnknownModelOption = (modelId: string): ModelOption => ({
  id: modelId,
  label: modelId,
  description: 'Custom model',
  avatarUrl: 'https://huggingface.co/api/avatars/huggingface',
});

interface ChatInputProps {
  sessionId?: string;
  onSend: (text: string) => void;
  onStop?: () => void;
  isProcessing?: boolean;
  disabled?: boolean;
  placeholder?: string;
}

const isClaudeModel = (model: ModelOption) => model.id.startsWith('anthropic/');

export default function ChatInput({ sessionId, onSend, onStop, isProcessing = false, disabled = false, placeholder = 'Ask anything...' }: ChatInputProps) {
  const [input, setInput] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [modelOptions, setModelOptions] = useState<ModelOption[]>(FALLBACK_MODELS);
  const [catalogCurrent, setCatalogCurrent] = useState<string>(FALLBACK_MODELS[0].id);
  const [sessionModel, setSessionModel] = useState<string | null>(null);
  const [modelAnchorEl, setModelAnchorEl] = useState<null | HTMLElement>(null);
  const { quota, refresh: refreshQuota } = useUserQuota();
  // The daily-cap dialog is triggered from two places: (a) a 429 returned
  // from the chat transport when the user tries to send on Opus over cap —
  // surfaced via the agent-store flag — and (b) nothing else right now
  // (switching models is free). Keeping the open state in the store means
  // the hook layer can flip it without threading props through.
  const claudeQuotaExhausted = useAgentStore((s) => s.claudeQuotaExhausted);
  const setClaudeQuotaExhausted = useAgentStore((s) => s.setClaudeQuotaExhausted);
  const lastSentRef = useRef<string>('');

  useEffect(() => {
    let cancelled = false;

    apiFetch('/api/config/model')
      .then((res) => (res.ok ? res.json() as Promise<ModelCatalogResponse> : null))
      .then((data) => {
        if (cancelled || !data) return;

        const rawAvailable = Array.isArray(data.available) ? data.available : [];
        const available = rawAvailable
          .map(toModelOption)
          .filter((value: ModelOption | null): value is ModelOption => value !== null);

        if (available.length > 0) {
          setModelOptions(available);
        }
        if (typeof data.current === 'string' && data.current) {
          setCatalogCurrent(data.current);
        }
      })
      .catch(() => { /* ignore */ });

    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!sessionId) {
      setSessionModel(null);
      return;
    }

    let cancelled = false;
    apiFetch(`/api/session/${sessionId}`)
      .then((res) => (res.ok ? res.json() as Promise<SessionResponse> : null))
      .then((data) => {
        if (cancelled) return;
        setSessionModel(typeof data?.model === 'string' && data.model ? data.model : null);
      })
      .catch(() => { /* ignore */ });

    return () => { cancelled = true; };
  }, [sessionId]);

  const selectedModelPath = sessionModel ?? catalogCurrent;
  const selectedModel = useMemo(
    () => modelOptions.find((model) => model.id === selectedModelPath)
      ?? (selectedModelPath ? makeUnknownModelOption(selectedModelPath) : null)
      ?? modelOptions[0]
      ?? FALLBACK_MODELS[0],
    [modelOptions, selectedModelPath],
  );

  useEffect(() => {
    if (!disabled && !isProcessing && inputRef.current) {
      inputRef.current.focus();
    }
  }, [disabled, isProcessing]);

  const handleSend = useCallback(() => {
    if (input.trim() && !disabled) {
      lastSentRef.current = input;
      onSend(input);
      setInput('');
    }
  }, [input, disabled, onSend]);

  // When the chat transport reports a Claude-quota 429, restore the typed
  // text so the user doesn't lose their message.
  useEffect(() => {
    if (claudeQuotaExhausted && lastSentRef.current) {
      setInput(lastSentRef.current);
    }
  }, [claudeQuotaExhausted]);

  // Refresh the quota display whenever the session changes (user might
  // have started another tab that spent quota).
  useEffect(() => {
    if (sessionId) refreshQuota();
  }, [refreshQuota, sessionId]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLDivElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  const handleModelClick = (event: MouseEvent<HTMLElement>) => {
    setModelAnchorEl(event.currentTarget);
  };

  const handleModelClose = () => {
    setModelAnchorEl(null);
  };

  const handleSelectModel = async (modelPath: string) => {
    handleModelClose();
    if (!sessionId) return;

    try {
      const res = await apiFetch(`/api/session/${sessionId}/model`, {
        method: 'POST',
        body: JSON.stringify({ model: modelPath }),
      });
      if (res.ok) {
        setSessionModel(modelPath);
      }
    } catch {
      // ignore
    }
  };

  // Dialog close: just clear the flag. The typed text is already restored.
  const handleCapDialogClose = useCallback(() => {
    setClaudeQuotaExhausted(false);
  }, [setClaudeQuotaExhausted]);

  // "Use a free model" — switch the current session to Kimi (or the first
  // non-Anthropic option) and auto-retry the send that tripped the cap.
  const handleUseFreeModel = useCallback(async () => {
    setClaudeQuotaExhausted(false);
    if (!sessionId) return;
    const free = modelOptions.find((model) => model.id === FIRST_FREE_MODEL_PATH)
      ?? modelOptions.find((model) => !isClaudeModel(model))
      ?? modelOptions[0];
    if (!free) return;

    try {
      const res = await apiFetch(`/api/session/${sessionId}/model`, {
        method: 'POST',
        body: JSON.stringify({ model: free.id }),
      });
      if (res.ok) {
        setSessionModel(free.id);
        const retryText = lastSentRef.current;
        if (retryText) {
          onSend(retryText);
          setInput('');
          lastSentRef.current = '';
        }
      }
    } catch {
      // ignore
    }
  }, [modelOptions, onSend, sessionId, setClaudeQuotaExhausted]);

  // Hide the chip until the user has actually burned quota — an unused
  // Opus session shouldn't populate a counter.
  const claudeChip = (() => {
    if (!quota || quota.claudeUsedToday === 0) return null;
    if (quota.plan === 'free') {
      return quota.claudeRemaining > 0 ? 'Free today' : 'Pro only';
    }
    return `${quota.claudeUsedToday}/${quota.claudeDailyCap} today`;
  })();

  return (
    <Box
      sx={{
        pb: { xs: 2, md: 4 },
        pt: { xs: 1, md: 2 },
        position: 'relative',
        zIndex: 10,
      }}
    >
      <Box sx={{ maxWidth: '880px', mx: 'auto', width: '100%', px: { xs: 0, sm: 1, md: 2 } }}>
        <Box
          className="composer"
          sx={{
            display: 'flex',
            gap: '10px',
            alignItems: 'flex-start',
            bgcolor: 'var(--composer-bg)',
            borderRadius: 'var(--radius-md)',
            p: '12px',
            border: '1px solid var(--border)',
            transition: 'box-shadow 0.2s ease, border-color 0.2s ease',
            '&:focus-within': {
              borderColor: 'var(--accent-yellow)',
              boxShadow: 'var(--focus)',
            },
          }}
        >
          <TextField
            fullWidth
            multiline
            maxRows={6}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled || isProcessing}
            variant="standard"
            inputRef={inputRef}
            InputProps={{
              disableUnderline: true,
              sx: {
                color: 'var(--text)',
                fontSize: '15px',
                fontFamily: 'inherit',
                padding: 0,
                lineHeight: 1.5,
                minHeight: { xs: '44px', md: '56px' },
                alignItems: 'flex-start',
              },
            }}
            sx={{
              flex: 1,
              '& .MuiInputBase-root': {
                p: 0,
                backgroundColor: 'transparent',
              },
              '& textarea': {
                resize: 'none',
                padding: '0 !important',
              },
            }}
          />
          {isProcessing ? (
            <IconButton
              onClick={onStop}
              sx={{
                mt: 1,
                p: 1.5,
                borderRadius: '10px',
                color: 'var(--muted-text)',
                transition: 'all 0.2s',
                position: 'relative',
                '&:hover': {
                  bgcolor: 'var(--hover-bg)',
                  color: 'var(--accent-red)',
                },
              }}
            >
              <Box sx={{ position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <CircularProgress size={28} thickness={3} sx={{ color: 'inherit', position: 'absolute' }} />
                <StopIcon sx={{ fontSize: 16 }} />
              </Box>
            </IconButton>
          ) : (
            <IconButton
              onClick={handleSend}
              disabled={disabled || !input.trim()}
              sx={{
                mt: 1,
                p: 1,
                borderRadius: '10px',
                color: 'var(--muted-text)',
                transition: 'all 0.2s',
                '&:hover': {
                  color: 'var(--accent-yellow)',
                  bgcolor: 'var(--hover-bg)',
                },
                '&.Mui-disabled': {
                  opacity: 0.3,
                },
              }}
            >
              <ArrowUpwardIcon fontSize="small" />
            </IconButton>
          )}
        </Box>

        <Box
          onClick={handleModelClick}
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            mt: 1.5,
            gap: 0.8,
            opacity: 0.6,
            cursor: 'pointer',
            transition: 'opacity 0.2s',
            '&:hover': {
              opacity: 1,
            },
          }}
        >
          <Typography variant="caption" sx={{ fontSize: '10px', color: 'var(--muted-text)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 500 }}>
            powered by
          </Typography>
          <img
            src={selectedModel.avatarUrl}
            alt={selectedModel.label}
            style={{ height: '14px', width: '14px', objectFit: 'contain', borderRadius: '2px' }}
          />
          <Typography variant="caption" sx={{ fontSize: '10px', color: 'var(--text)', fontWeight: 600, letterSpacing: '0.02em' }}>
            {selectedModel.label}
          </Typography>
          <ArrowDropDownIcon sx={{ fontSize: '14px', color: 'var(--muted-text)' }} />
        </Box>

        <Menu
          anchorEl={modelAnchorEl}
          open={Boolean(modelAnchorEl)}
          onClose={handleModelClose}
          anchorOrigin={{
            vertical: 'top',
            horizontal: 'center',
          }}
          transformOrigin={{
            vertical: 'bottom',
            horizontal: 'center',
          }}
          slotProps={{
            paper: {
              sx: {
                bgcolor: 'var(--panel)',
                border: '1px solid var(--divider)',
                mb: 1,
                maxHeight: '400px',
              },
            },
          }}
        >
          {modelOptions.map((model) => (
            <MenuItem
              key={model.id}
              onClick={() => handleSelectModel(model.id)}
              selected={selectedModelPath === model.id}
              sx={{
                py: 1.5,
                '&.Mui-selected': {
                  bgcolor: 'rgba(255,255,255,0.05)',
                },
              }}
            >
              <ListItemIcon>
                <img
                  src={model.avatarUrl}
                  alt={model.label}
                  style={{ width: 24, height: 24, borderRadius: '4px', objectFit: 'cover' }}
                />
              </ListItemIcon>
              <ListItemText
                primary={(
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {model.label}
                    {model.recommended && (
                      <Chip
                        label="Recommended"
                        size="small"
                        sx={{
                          height: '18px',
                          fontSize: '10px',
                          bgcolor: 'var(--accent-yellow)',
                          color: '#000',
                          fontWeight: 600,
                        }}
                      />
                    )}
                    {isClaudeModel(model) && claudeChip && (
                      <Chip
                        label={claudeChip}
                        size="small"
                        sx={{
                          height: '18px',
                          fontSize: '10px',
                          bgcolor: 'rgba(255,255,255,0.08)',
                          color: 'var(--muted-text)',
                          fontWeight: 600,
                        }}
                      />
                    )}
                  </Box>
                )}
                secondary={model.description || model.providerLabel}
                secondaryTypographyProps={{
                  sx: { fontSize: '12px', color: 'var(--muted-text)' },
                }}
              />
            </MenuItem>
          ))}
        </Menu>

        <ClaudeCapDialog
          open={claudeQuotaExhausted}
          plan={quota?.plan ?? 'free'}
          cap={quota?.claudeDailyCap ?? 1}
          onClose={handleCapDialogClose}
          onUseFreeModel={handleUseFreeModel}
        />
      </Box>
    </Box>
  );
}
