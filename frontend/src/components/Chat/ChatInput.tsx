import { useState, useCallback, useEffect, useRef, KeyboardEvent } from 'react';
import { Box, TextField, IconButton, CircularProgress, Typography, Menu, MenuItem, ListItemIcon, ListItemText, Chip } from '@mui/material';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import StopIcon from '@mui/icons-material/Stop';
import AttachFileIcon from '@mui/icons-material/AttachFile';
import CloseIcon from '@mui/icons-material/Close';
import StorageIcon from '@mui/icons-material/Storage';
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

interface ChatInputProps {
  sessionId?: string;
  initialModelPath?: string | null;
  onSend: (text: string, uploads?: unknown[]) => void;
  onStop?: () => void;
  isProcessing?: boolean;
  disabled?: boolean;
  placeholder?: string;
}

interface PendingRetry {
  inputText: string;
  displayText: string;
  uploads: unknown[];
}

const isClaudeModel = (m: ModelOption) => isClaudePath(m.modelPath);
const isPremiumModel = (m: ModelOption) => isPremiumPath(m.modelPath);
const firstFreeModel = (options: ModelOption[]) => options.find(m => !isPremiumModel(m)) ?? options[0];

export default function ChatInput({ sessionId, initialModelPath, onSend, onStop, isProcessing = false, disabled = false, placeholder = 'Ask anything...' }: ChatInputProps) {
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
  const [selectedFiles, setSelectedFiles] = useState<Array<{ id: string; file: File }>>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [importAsDataset, setImportAsDataset] = useState(false);
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
  const lastSentRef = useRef<PendingRetry | null>(null);

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

  const addFiles = useCallback((files: FileList | File[]) => {
    const next = Array.from(files).map((file) => ({
      id: `${file.name}-${file.lastModified}-${Math.random().toString(16).slice(2)}`,
      file,
    }));
    setSelectedFiles((current) => [...current, ...next]);
    setUploadError(null);
  }, []);

  const uploadSelectedFiles = useCallback(async (): Promise<unknown[]> => {
    if (!sessionId || selectedFiles.length === 0) return [];
    const form = new FormData();
    selectedFiles.forEach(({ file }) => form.append('files', file, file.name));
    form.append('import_as_dataset', String(importAsDataset));
    const res = await apiFetch(`/api/session/${sessionId}/uploads`, {
      method: 'POST',
      body: form,
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => 'Upload failed');
      throw new Error(detail || 'Upload failed');
    }
    const data = await res.json();
    return data?.upload ? [data.upload] : [];
  }, [sessionId, selectedFiles, importAsDataset]);

  const handleSend = useCallback(async () => {
    if ((input.trim() || selectedFiles.length > 0) && !disabled && !isUploading) {
      setUploadError(null);
      setIsUploading(true);
      const baseText = input.trim();
      try {
        const uploads = await uploadSelectedFiles();
        const placeholders = selectedFiles
          .map(({ file }, index) => `[${file.type.startsWith('image/') ? 'Image' : 'File'} #${index + 1}] ${file.name}`)
          .join('\n');
        const displayText = placeholders ? `${baseText}\n\n${placeholders}`.trim() : baseText;
        lastSentRef.current = { inputText: baseText, displayText, uploads };
        onSend(displayText, uploads);
        setInput('');
        setSelectedFiles([]);
      } catch (err) {
        setUploadError(err instanceof Error ? err.message : 'Upload failed');
      } finally {
        setIsUploading(false);
      }
    }
  }, [input, selectedFiles, disabled, isUploading, uploadSelectedFiles, onSend]);

  // When the chat transport reports a premium-model quota 429, restore the typed
  // text so the user doesn't lose their message.
  useEffect(() => {
    if (claudeQuotaExhausted && lastSentRef.current) {
      setInput(lastSentRef.current.inputText);
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
      }
    } catch { /* ignore */ }
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
        const retry = lastSentRef.current;
        if (retry) {
          onSend(retry.displayText, retry.uploads);
          setInput('');
          lastSentRef.current = null;
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
            flexDirection: 'column',
            gap: '10px',
            alignItems: 'stretch',
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
          onDragOver={(e) => {
            e.preventDefault();
          }}
          onDrop={(e) => {
            e.preventDefault();
            if (!disabled && !isProcessing && e.dataTransfer.files.length > 0) {
              addFiles(e.dataTransfer.files);
            }
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            hidden
            onChange={(e) => {
              if (e.target.files) addFiles(e.target.files);
              e.target.value = '';
            }}
          />
          {(selectedFiles.length > 0 || uploadError) && (
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, alignItems: 'center' }}>
              {selectedFiles.map(({ id, file }) => (
                <Chip
                  key={id}
                  size="small"
                  label={file.name}
                  onDelete={() => setSelectedFiles((files) => files.filter((f) => f.id !== id))}
                  deleteIcon={<CloseIcon />}
                  sx={{
                    maxWidth: '220px',
                    bgcolor: 'rgba(255,255,255,0.06)',
                    color: 'var(--text)',
                    border: '1px solid var(--divider)',
                    '& .MuiChip-label': { overflow: 'hidden', textOverflow: 'ellipsis' },
                  }}
                />
              ))}
              {selectedFiles.length > 0 && (
                <Chip
                  size="small"
                  icon={<StorageIcon sx={{ fontSize: 15 }} />}
                  label={importAsDataset ? 'Import as dataset' : 'Attach to turn'}
                  onClick={() => setImportAsDataset((v) => !v)}
                  sx={{
                    bgcolor: importAsDataset ? 'var(--accent-yellow)' : 'transparent',
                    color: importAsDataset ? '#000' : 'var(--muted-text)',
                    border: '1px solid var(--divider)',
                    fontWeight: 600,
                  }}
                />
              )}
              {uploadError && (
                <Typography variant="caption" sx={{ color: 'var(--accent-red)' }}>
                  {uploadError}
                </Typography>
              )}
            </Box>
          )}
          <Box sx={{ display: 'flex', gap: '10px', alignItems: 'flex-start' }}>
          <IconButton
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled || isProcessing || isUploading}
            sx={{
              mt: 1,
              p: 1,
              borderRadius: '10px',
              color: 'var(--muted-text)',
              '&:hover': { color: 'var(--accent-yellow)', bgcolor: 'var(--hover-bg)' },
            }}
          >
            <AttachFileIcon fontSize="small" />
          </IconButton>
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
              disabled={disabled || isUploading || (!input.trim() && selectedFiles.length === 0)}
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
      </Box>
    </Box>
  );
}
