import { useState, useCallback, useEffect, useRef, KeyboardEvent } from 'react';
import {
  Alert,
  Box,
  TextField,
  IconButton,
  CircularProgress,
  Typography,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Snackbar,
  Tooltip,
} from '@mui/material';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import StopIcon from '@mui/icons-material/Stop';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import { apiFetch } from '@/utils/api';
import { useUserQuota } from '@/hooks/useUserQuota';
import ClaudeCapDialog from '@/components/ClaudeCapDialog';
import JobsUpgradeDialog from '@/components/JobsUpgradeDialog';
import { useAgentStore } from '@/store/agentStore';
import { useSessionStore } from '@/store/sessionStore';
import {
  CLAUDE_MODEL_PATH,
  FIRST_FREE_MODEL_PATH,
  GPT_55_MODEL_PATH,
  isClaudePath,
  isPremiumPath,
} from '@/utils/model';

// Model configuration
interface ModelOption {
  id: string;
  name: string;
  description: string;
  modelPath: string;
  avatarUrl: string;
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
    id: 'gpt-5.5',
    name: 'GPT-5.5',
    description: 'OpenAI',
    modelPath: GPT_55_MODEL_PATH,
    avatarUrl: 'https://huggingface.co/api/avatars/openai',
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
  {
    id: 'deepseek-v4-pro',
    name: 'DeepSeek V4 Pro',
    description: 'DeepInfra',
    modelPath: 'deepseek-ai/DeepSeek-V4-Pro:deepinfra',
    avatarUrl: getHfAvatarUrl('deepseek-ai/DeepSeek-V4-Pro'),
  },
];

const findModelByPath = (path: string, options: ModelOption[]): ModelOption | undefined => {
  if (isClaudePath(path)) {
    const claude = options.find(isClaudeModel);
    if (claude) return claude;
  }
  return options.find(m => m.modelPath === path || path?.includes(m.id));
};

const readApiErrorMessage = async (res: Response, fallback: string): Promise<string> => {
  try {
    const data = await res.json();
    const detail = data?.detail;
    if (typeof detail === 'string') return detail;
    if (detail && typeof detail.message === 'string') return detail.message;
    if (detail && typeof detail.error === 'string') return detail.error;
  } catch {
    /* ignore malformed error bodies */
  }
  return fallback;
};

interface ChatInputProps {
  sessionId?: string;
  initialModelPath?: string | null;
  onSend: (text: string) => void;
  onStop?: () => void;
  onDatasetUploaded?: () => Promise<boolean> | boolean;
  isProcessing?: boolean;
  disabled?: boolean;
  placeholder?: string;
}

interface DatasetUploadResponse {
  session_id: string;
  repo_id: string;
  repo_type: 'dataset';
  private: true;
  upload_id: string;
  filename: string;
  path_in_repo: string;
  size_bytes: number;
  format: 'csv' | 'json' | 'jsonl';
  hub_url: string;
  load_dataset_snippet: string;
}

const MAX_DATASET_UPLOAD_BYTES = 100 * 1024 * 1024;
const DATASET_UPLOAD_ACCEPT = '.csv,.json,.jsonl';
const DATASET_UPLOAD_EXTENSIONS = new Set(['csv', 'json', 'jsonl']);

const isClaudeModel = (m: ModelOption) => isClaudePath(m.modelPath);
const isPremiumModel = (m: ModelOption) => isPremiumPath(m.modelPath);
const firstFreeModel = (options: ModelOption[]) => options.find(m => !isPremiumModel(m)) ?? options[0];

const formatBytes = (bytes: number) => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

export default function ChatInput({ sessionId, initialModelPath, onSend, onStop, onDatasetUploaded, isProcessing = false, disabled = false, placeholder = 'Ask anything...' }: ChatInputProps) {
  const [input, setInput] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [modelOptions, setModelOptions] = useState<ModelOption[]>(DEFAULT_MODEL_OPTIONS);
  const modelOptionsRef = useRef<ModelOption[]>(DEFAULT_MODEL_OPTIONS);
  const sessionIdRef = useRef<string | undefined>(sessionId);
  const [selectedModelId, setSelectedModelId] = useState<string>(
    () => findModelByPath(initialModelPath ?? '', DEFAULT_MODEL_OPTIONS)?.id ?? DEFAULT_MODEL_OPTIONS[0].id,
  );
  const [modelAnchorEl, setModelAnchorEl] = useState<null | HTMLElement>(null);
  const { quota, refresh: refreshQuota } = useUserQuota();
  // The daily-cap dialog is triggered from two places: (a) a 429 returned
  // from the chat transport when the user tries to send on a premium model over cap —
  // surfaced via the agent-store flag — and (b) nothing else right now
  // (switching models is free). Keeping the open state in the store means
  // the hook layer can flip it without threading props through.
  const claudeQuotaExhausted = useAgentStore((s) => s.claudeQuotaExhausted);
  const setClaudeQuotaExhausted = useAgentStore((s) => s.setClaudeQuotaExhausted);
  const jobsUpgradeRequired = useAgentStore((s) => s.jobsUpgradeRequired);
  const setJobsUpgradeRequired = useAgentStore((s) => s.setJobsUpgradeRequired);
  const updateSessionModel = useSessionStore((s) => s.updateSessionModel);
  const [awaitingTopUp, setAwaitingTopUp] = useState(false);
  const [modelSwitchError, setModelSwitchError] = useState<string | null>(null);
  const [datasetUploadError, setDatasetUploadError] = useState<string | null>(null);
  const [datasetUploadSuccess, setDatasetUploadSuccess] = useState<string | null>(null);
  const [uploadedDataset, setUploadedDataset] = useState<DatasetUploadResponse | null>(null);
  const [isUploadingDataset, setIsUploadingDataset] = useState(false);
  const lastSentRef = useRef<string>('');

  useEffect(() => {
    modelOptionsRef.current = modelOptions;
  }, [modelOptions]);

  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  useEffect(() => {
    let cancelled = false;
    apiFetch('/api/config/model')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (cancelled || !data?.available) return;
        const claude = data.available.find((m: { provider?: string; id?: string }) => (
          m.provider === 'anthropic' && m.id
        ));
        if (!claude?.id) return;

        const next = DEFAULT_MODEL_OPTIONS.map((option) => (
          isClaudeModel(option)
            ? { ...option, modelPath: claude.id, name: claude.label ?? option.name }
            : option
        ));
        modelOptionsRef.current = next;
        setModelOptions(next);
        if (!sessionIdRef.current) {
          const current = data.current ? findModelByPath(data.current, next) : null;
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
          const model = findModelByPath(data.model, modelOptionsRef.current);
          if (model) setSelectedModelId(model.id);
          updateSessionModel(sessionId, data.model);
        }
      })
      .catch(() => { /* ignore */ });
    return () => { cancelled = true; };
  }, [sessionId, updateSessionModel]);

  const selectedModel = modelOptions.find(m => m.id === selectedModelId) || modelOptions[0];

  // Auto-focus the textarea when the session becomes ready
  useEffect(() => {
    if (!disabled && !isProcessing && inputRef.current) {
      inputRef.current.focus();
    }
  }, [disabled, isProcessing]);

  const handleSend = useCallback(() => {
    if (input.trim() && !disabled && !isUploadingDataset) {
      lastSentRef.current = input;
      onSend(input);
      setInput('');
    }
  }, [input, disabled, isUploadingDataset, onSend]);

  const handleDatasetUploadClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleDatasetFileChange = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      event.target.value = '';
      if (!file) return;

      if (!sessionId) {
        setDatasetUploadError('Start a session before uploading a dataset.');
        return;
      }

      const extension = file.name.split('.').pop()?.toLowerCase() || '';
      if (!DATASET_UPLOAD_EXTENSIONS.has(extension)) {
        setDatasetUploadError('Only CSV, JSON, and JSONL dataset files are supported.');
        return;
      }
      if (file.size > MAX_DATASET_UPLOAD_BYTES) {
        setDatasetUploadError(
          `Dataset files must be 100 MB or smaller. ${file.name} is ${formatBytes(file.size)}.`
        );
        return;
      }
      if (file.size === 0) {
        setDatasetUploadError('Uploaded dataset file is empty.');
        return;
      }

      const formData = new FormData();
      formData.append('file', file);
      setIsUploadingDataset(true);
      setDatasetUploadError(null);
      setDatasetUploadSuccess(null);
      try {
        const res = await apiFetch(`/api/session/${sessionId}/datasets`, {
          method: 'POST',
          body: formData,
        });
        if (!res.ok) {
          setDatasetUploadError(await readApiErrorMessage(res, 'Dataset upload failed.'));
          return;
        }
        const payload = await res.json() as DatasetUploadResponse;
        setUploadedDataset(payload);
        setDatasetUploadSuccess(`Uploaded ${payload.filename} to ${payload.repo_id}`);
        await onDatasetUploaded?.();
      } catch (error) {
        setDatasetUploadError(
          error instanceof Error ? error.message : 'Dataset upload failed.'
        );
      } finally {
        setIsUploadingDataset(false);
      }
    },
    [sessionId, onDatasetUploaded],
  );

  // When the chat transport reports a premium-model quota 429, restore the typed
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
      if (res.ok) {
        setSelectedModelId(model.id);
        updateSessionModel(sessionId, model.modelPath);
        setModelSwitchError(null);
        return;
      }
      setModelSwitchError(await readApiErrorMessage(res, 'Could not switch model.'));
    } catch (error) {
      setModelSwitchError(error instanceof Error ? error.message : 'Could not switch model.');
    }
  };

  // Dialog close: just clear the flag. The typed text is already restored.
  const handleCapDialogClose = useCallback(() => {
    setClaudeQuotaExhausted(false);
  }, [setClaudeQuotaExhausted]);

  // "Use a free model" — switch the current session to Kimi (or the first
  // non-premium option) and auto-retry the send that tripped the cap.
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
        updateSessionModel(sessionId, free.modelPath);
        const retryText = lastSentRef.current;
        if (retryText) {
          onSend(retryText);
          setInput('');
          lastSentRef.current = '';
        }
      }
    } catch { /* ignore */ }
  }, [sessionId, onSend, setClaudeQuotaExhausted, modelOptions, updateSessionModel]);

  const handlePremiumUpgradeClick = useCallback(async () => {
    if (!sessionId) return;
    try {
      await apiFetch(`/api/pro-click/${sessionId}`, {
        method: 'POST',
        body: JSON.stringify({ source: 'premium_cap_dialog', target: 'pro_pricing' }),
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

  // Hide the chip until the user has actually burned quota; opening a
  // premium-model session without sending should not populate a counter.
  const premiumChip = (() => {
    if (!quota || quota.premiumUsedToday === 0) return null;
    if (quota.plan === 'free') {
      return quota.premiumRemaining > 0 ? 'Free today' : 'Pro only';
    }
    return `${quota.premiumUsedToday}/${quota.premiumDailyCap} today`;
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
          <input
            ref={fileInputRef}
            type="file"
            accept={DATASET_UPLOAD_ACCEPT}
            onChange={handleDatasetFileChange}
            style={{ display: 'none' }}
          />
          <Tooltip title="Upload dataset">
            <span>
              <IconButton
                onClick={handleDatasetUploadClick}
                disabled={disabled || isProcessing || isUploadingDataset || !sessionId}
                sx={{
                  mt: 1,
                  p: 1,
                  borderRadius: '10px',
                  color: uploadedDataset ? 'var(--accent-yellow)' : 'var(--muted-text)',
                  transition: 'all 0.2s',
                  '&:hover': {
                    color: 'var(--accent-yellow)',
                    bgcolor: 'var(--hover-bg)',
                  },
                  '&.Mui-disabled': {
                    opacity: 0.3,
                  },
                }}
                aria-label="Upload dataset"
              >
                {isUploadingDataset ? (
                  <CircularProgress size={18} thickness={4} sx={{ color: 'inherit' }} />
                ) : (
                  <UploadFileIcon fontSize="small" />
                )}
              </IconButton>
            </span>
          </Tooltip>
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
              disabled={disabled || isUploadingDataset || !input.trim()}
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
        {uploadedDataset && (
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1 }}>
            <Chip
              size="small"
              label={`Dataset: ${uploadedDataset.filename}`}
              component="a"
              href={uploadedDataset.hub_url}
              target="_blank"
              rel="noreferrer"
              clickable
              sx={{
                maxWidth: '100%',
                bgcolor: 'rgba(255,255,255,0.08)',
                color: 'var(--text)',
                border: '1px solid var(--divider)',
                '& .MuiChip-label': {
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                },
              }}
            />
          </Box>
        )}

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
                    {isPremiumModel(model) && premiumChip && (
                      <Chip
                        label={premiumChip}
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
          cap={quota?.premiumDailyCap ?? 1}
          onClose={handleCapDialogClose}
          onUseFreeModel={handleUseFreeModel}
          onUpgrade={handlePremiumUpgradeClick}
        />
        <JobsUpgradeDialog
          open={!!jobsUpgradeRequired}
          message={jobsUpgradeRequired?.message || ''}
          awaitingTopUp={awaitingTopUp}
          onClose={handleJobsUpgradeClose}
          onUpgrade={handleJobsUpgradeClick}
          onRetry={handleJobsRetry}
        />
        <Snackbar
          open={!!modelSwitchError}
          anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
          onClose={() => setModelSwitchError(null)}
          autoHideDuration={6000}
        >
          <Alert
            severity="error"
            variant="filled"
            onClose={() => setModelSwitchError(null)}
            sx={{ fontSize: '0.8rem', maxWidth: 480 }}
          >
            {modelSwitchError}
          </Alert>
        </Snackbar>
        <Snackbar
          open={!!datasetUploadError}
          anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
          onClose={() => setDatasetUploadError(null)}
          autoHideDuration={7000}
        >
          <Alert
            severity="error"
            variant="filled"
            onClose={() => setDatasetUploadError(null)}
            sx={{ fontSize: '0.8rem', maxWidth: 520 }}
          >
            {datasetUploadError}
          </Alert>
        </Snackbar>
        <Snackbar
          open={!!datasetUploadSuccess}
          anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
          onClose={() => setDatasetUploadSuccess(null)}
          autoHideDuration={5000}
        >
          <Alert
            severity="success"
            variant="filled"
            onClose={() => setDatasetUploadSuccess(null)}
            sx={{ fontSize: '0.8rem', maxWidth: 520 }}
          >
            {datasetUploadSuccess}
          </Alert>
        </Snackbar>
      </Box>
    </Box>
  );
}
