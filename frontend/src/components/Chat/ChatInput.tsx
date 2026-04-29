import { useState, useCallback, useEffect, useRef, KeyboardEvent } from 'react';
import { Box, TextField, IconButton, CircularProgress, Typography, Menu, MenuItem, ListItemIcon, ListItemText, Chip } from '@mui/material';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import StopIcon from '@mui/icons-material/Stop';
import { apiFetch } from '@/utils/api';
import { useUserQuota } from '@/hooks/useUserQuota';
import ClaudeCapDialog from '@/components/ClaudeCapDialog';
import JobsUpgradeDialog from '@/components/JobsUpgradeDialog';
import { useAgentStore } from '@/store/agentStore';
import { CLAUDE_MODEL_PATH, FIRST_FREE_MODEL_PATH, isClaudePath } from '@/utils/model';

// Model configuration
interface ModelOption {
  id: string;
  name: string;
  description: string;
  modelPath: string;
  avatarUrl: string;
  recommended?: boolean;
}

interface BackendModelOption {
  id: string;
  label?: string;
  provider?: string;
  recommended?: boolean;
}

const getHfAvatarUrl = (modelId: string) => {
  const org = modelId.split('/')[0];
  return `https://huggingface.co/api/avatars/${org}`;
};

const DEFAULT_MODEL_OPTIONS: ModelOption[] = [
  {
    id: 'kimi-k2.6',
    name: 'Kimi K2.6',
    description: 'Novita',
    modelPath: 'moonshotai/Kimi-K2.6',
    avatarUrl: getHfAvatarUrl('moonshotai/Kimi-K2.6'),
    recommended: true,
  },
  {
    id: 'claude-opus',
    name: 'Claude Opus 4.6',
    description: 'Anthropic',
    modelPath: CLAUDE_MODEL_PATH,
    avatarUrl: 'https://huggingface.co/api/avatars/Anthropic',
    recommended: true,
  },
  {
    id: 'minimax-m2.7',
    name: 'MiniMax M2.7',
    description: 'Novita',
    modelPath: 'MiniMaxAI/MiniMax-M2.7',
    avatarUrl: getHfAvatarUrl('MiniMaxAI/MiniMax-M2.7'),
  },
  {
    id: 'glm-5.1',
    name: 'GLM 5.1',
    description: 'Together',
    modelPath: 'zai-org/GLM-5.1',
    avatarUrl: getHfAvatarUrl('zai-org/GLM-5.1'),
  },
];

const findModelByPath = (path: string, options: ModelOption[]): ModelOption | undefined => {
  return options.find(m => m.modelPath === path || path?.includes(m.id));
};

const optionFromBackend = (model: BackendModelOption, existing?: ModelOption): ModelOption => {
  const isClaude = model.provider === 'anthropic' || isClaudePath(model.id);
  return {
    id: existing?.id ?? model.id,
    name: model.label ?? existing?.name ?? model.id.split('/').pop() ?? model.id,
    description: existing?.description ?? (isClaude ? 'Anthropic' : 'Hugging Face'),
    modelPath: model.id,
    avatarUrl: existing?.avatarUrl ?? (isClaude
      ? 'https://huggingface.co/api/avatars/Anthropic'
      : getHfAvatarUrl(model.id)),
    recommended: model.recommended ?? existing?.recommended,
  };
};

const mergeBackendModels = (models: BackendModelOption[], fallback: ModelOption[]): ModelOption[] => {
  const next = models
    .filter((model) => !!model.id)
    .map((model) => {
      const existing = model.provider === 'anthropic'
        ? fallback.find((option) => isClaudePath(option.modelPath))
        : findModelByPath(model.id, fallback);
      return optionFromBackend(model, existing);
    });
  for (const option of fallback) {
    if (!next.some((model) => model.modelPath === option.modelPath)) {
      next.push(option);
    }
  }
  return next;
};

interface ChatInputProps {
  sessionId?: string;
  onSend: (text: string) => void;
  onStop?: () => void;
  isProcessing?: boolean;
  disabled?: boolean;
  placeholder?: string;
}

const isClaudeModel = (m: ModelOption) => isClaudePath(m.modelPath);
const firstFreeModel = (options: ModelOption[]) => options.find(m => !isClaudeModel(m)) ?? options[0];

export default function ChatInput({ sessionId, onSend, onStop, isProcessing = false, disabled = false, placeholder = 'Ask anything...' }: ChatInputProps) {
  const [input, setInput] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [modelOptions, setModelOptions] = useState<ModelOption[]>(DEFAULT_MODEL_OPTIONS);
  const [selectedModelId, setSelectedModelId] = useState<string>(DEFAULT_MODEL_OPTIONS[0].id);
  const [modelAnchorEl, setModelAnchorEl] = useState<null | HTMLElement>(null);
  const { quota, refresh: refreshQuota } = useUserQuota();
  // The daily-cap dialog is triggered from two places: (a) a 429 returned
  // from the chat transport when the user tries to send on Opus over cap —
  // surfaced via the agent-store flag — and (b) nothing else right now
  // (switching models is free). Keeping the open state in the store means
  // the hook layer can flip it without threading props through.
  const claudeQuotaExhausted = useAgentStore((s) => s.claudeQuotaExhausted);
  const setClaudeQuotaExhausted = useAgentStore((s) => s.setClaudeQuotaExhausted);
  const jobsUpgradeRequired = useAgentStore((s) => s.jobsUpgradeRequired);
  const setJobsUpgradeRequired = useAgentStore((s) => s.setJobsUpgradeRequired);
  const [awaitingTopUp, setAwaitingTopUp] = useState(false);
  const lastSentRef = useRef<string>('');

  useEffect(() => {
    let cancelled = false;
    apiFetch('/api/config/model')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (cancelled || !data?.available) return;
        const next = mergeBackendModels(data.available, DEFAULT_MODEL_OPTIONS);
        setModelOptions(next);
        if (data.current) {
          const current = findModelByPath(data.current, next);
          if (current) setSelectedModelId(current.id);
        }
      })
      .catch(() => { /* ignore */ });
    return () => { cancelled = true; };
  }, []);

  // Model is per-session: fetch this tab's current model every time the
  // session changes. Other tabs keep their own selections independently.
  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    apiFetch(`/api/session/${sessionId}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (cancelled) return;
        if (data?.model) {
          const model = findModelByPath(data.model, modelOptions);
          if (model) setSelectedModelId(model.id);
        }
      })
      .catch(() => { /* ignore */ });
    return () => { cancelled = true; };
  }, [sessionId, modelOptions]);

  const selectedModel = modelOptions.find(m => m.id === selectedModelId) || modelOptions[0];

  // Auto-focus the textarea when the session becomes ready
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLDivElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  const handleModelClick = (event: React.MouseEvent<HTMLElement>) => {
    setModelAnchorEl(event.currentTarget);
  };

  const handleModelClose = () => {
    setModelAnchorEl(null);
  };

  const handleSelectModel = async (model: ModelOption) => {
    handleModelClose();
    if (!sessionId) return;
    try {
      const res = await apiFetch(`/api/session/${sessionId}/model`, {
        method: 'POST',
        body: JSON.stringify({ model: model.modelPath }),
      });
      if (res.ok) setSelectedModelId(model.id);
    } catch { /* ignore */ }
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
    const free = modelOptions.find(m => m.modelPath === FIRST_FREE_MODEL_PATH)
      ?? firstFreeModel(modelOptions);
    try {
      const res = await apiFetch(`/api/session/${sessionId}/model`, {
        method: 'POST',
        body: JSON.stringify({ model: free.modelPath }),
      });
      if (res.ok) {
        setSelectedModelId(free.id);
        const retryText = lastSentRef.current;
        if (retryText) {
          onSend(retryText);
          setInput('');
          lastSentRef.current = '';
        }
      }
    } catch { /* ignore */ }
  }, [sessionId, onSend, setClaudeQuotaExhausted, modelOptions]);

  const handleClaudeUpgradeClick = useCallback(async () => {
    if (!sessionId) return;
    try {
      await apiFetch(`/api/pro-click/${sessionId}`, {
        method: 'POST',
        body: JSON.stringify({ source: 'claude_cap_dialog', target: 'pro_pricing' }),
      });
    } catch {
      /* tracking is best-effort */
    }
  }, [sessionId]);

  const handleJobsUpgradeClose = useCallback(() => {
    setJobsUpgradeRequired(null);
    setAwaitingTopUp(false);
  }, [setJobsUpgradeRequired]);

  const handleJobsUpgradeClick = useCallback(async () => {
    setAwaitingTopUp(true);
    if (!sessionId || !jobsUpgradeRequired) return;
    try {
      await apiFetch(`/api/pro-click/${sessionId}`, {
        method: 'POST',
        body: JSON.stringify({ source: 'hf_jobs_billing_dialog', target: 'hf_billing' }),
      });
    } catch {
      /* tracking is best-effort */
    }
  }, [sessionId, jobsUpgradeRequired]);

  const handleJobsRetry = useCallback(() => {
    const namespace = jobsUpgradeRequired?.namespace;
    setJobsUpgradeRequired(null);
    setAwaitingTopUp(false);
    const msg = namespace
      ? `I just added credits to the \`${namespace}\` namespace. Please retry the previous job.`
      : "I just added credits. Please retry the previous job.";
    onSend(msg);
  }, [jobsUpgradeRequired, setJobsUpgradeRequired, onSend]);

  // Auto-retry when the user comes back to this tab after clicking "Add credits".
  // Browsers fire visibilitychange when the tab regains focus from a sibling tab.
  useEffect(() => {
    if (!awaitingTopUp || !jobsUpgradeRequired) return;
    const onVisible = () => {
      if (document.visibilityState === 'visible') {
        handleJobsRetry();
      }
    };
    document.addEventListener('visibilitychange', onVisible);
    return () => document.removeEventListener('visibilitychange', onVisible);
  }, [awaitingTopUp, jobsUpgradeRequired, handleJobsRetry]);

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
            }
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
                }
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
                }
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

        {/* Powered By Badge */}
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
              opacity: 1
            }
          }}
        >
          <Typography variant="caption" sx={{ fontSize: '10px', color: 'var(--muted-text)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 500 }}>
            powered by
          </Typography>
          <img
            src={selectedModel.avatarUrl}
            alt={selectedModel.name}
            style={{ height: '14px', width: '14px', objectFit: 'contain', borderRadius: '2px' }}
          />
          <Typography variant="caption" sx={{ fontSize: '10px', color: 'var(--text)', fontWeight: 600, letterSpacing: '0.02em' }}>
            {selectedModel.name}
          </Typography>
          <ArrowDropDownIcon sx={{ fontSize: '14px', color: 'var(--muted-text)' }} />
        </Box>

        {/* Model Selection Menu */}
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
              }
            }
          }}
        >
          {modelOptions.map((model) => (
            <MenuItem
              key={model.id}
              onClick={() => handleSelectModel(model)}
              selected={selectedModelId === model.id}
              sx={{
                py: 1.5,
                '&.Mui-selected': {
                  bgcolor: 'rgba(255,255,255,0.05)',
                }
              }}
            >
              <ListItemIcon>
                <img
                  src={model.avatarUrl}
                  alt={model.name}
                  style={{ width: 24, height: 24, borderRadius: '4px', objectFit: 'cover' }}
                />
              </ListItemIcon>
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {model.name}
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
                }
                secondary={model.description}
                secondaryTypographyProps={{
                  sx: { fontSize: '12px', color: 'var(--muted-text)' }
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
          onUpgrade={handleClaudeUpgradeClick}
        />
        <JobsUpgradeDialog
          open={!!jobsUpgradeRequired}
          message={jobsUpgradeRequired?.message || ''}
          awaitingTopUp={awaitingTopUp}
          onClose={handleJobsUpgradeClose}
          onUpgrade={handleJobsUpgradeClick}
          onRetry={handleJobsRetry}
        />
      </Box>
    </Box>
  );
}
