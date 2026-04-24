import { useState, useCallback, useEffect, useRef, KeyboardEvent } from "react";
import {
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
  ListSubheader,
} from "@mui/material";
import ArrowUpwardIcon from "@mui/icons-material/ArrowUpward";
import ArrowDropDownIcon from "@mui/icons-material/ArrowDropDown";
import StopIcon from "@mui/icons-material/Stop";
import { useUserQuota } from "@/hooks/useUserQuota";
import { useModelCatalog } from "@/hooks/useModelCatalog";
import type { ModelOption, ProviderOption } from "@/hooks/useModelCatalog";
import ClaudeCapDialog from "@/components/ClaudeCapDialog";
import { useAgentStore } from "@/store/agentStore";
import { FIRST_FREE_MODEL_PATH } from "@/utils/model";
import CustomModelDialog from "@/components/Chat/CustomModelDialog";

interface ChatInputProps {
  sessionId?: string;
  onSend: (text: string) => void;
  onStop?: () => void;
  isProcessing?: boolean;
  disabled?: boolean;
  placeholder?: string;
}

const OPENAI_COMPAT_PROVIDER = "openai_compat";

const isClaudePath = (modelPath: string) => modelPath.startsWith("anthropic/");

const firstFreeModel = (models: ModelOption[]) => {
  const byPath = models.find((m) => m.id === FIRST_FREE_MODEL_PATH);
  if (byPath) return byPath;
  return models.find((m) => !isClaudePath(m.id));
};

export default function ChatInput({
  sessionId,
  onSend,
  onStop,
  isProcessing = false,
  disabled = false,
  placeholder = "Ask anything...",
}: ChatInputProps) {
  const [input, setInput] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const {
    modelOptions,
    selectedModelPath,
    selectedModel,
    switchModel,
    groups,
  } = useModelCatalog(sessionId);
  const [modelAnchorEl, setModelAnchorEl] = useState<null | HTMLElement>(null);
  const [customModalOpen, setCustomModalOpen] = useState(false);
  const [customPrefix, setCustomPrefix] = useState("openai-compat/");
  const { quota, refresh: refreshQuota } = useUserQuota();
  const claudeQuotaExhausted = useAgentStore((s) => s.claudeQuotaExhausted);
  const setClaudeQuotaExhausted = useAgentStore(
    (s) => s.setClaudeQuotaExhausted,
  );
  const lastSentRef = useRef<string>("");

  useEffect(() => {
    if (!disabled && !isProcessing && inputRef.current) {
      inputRef.current.focus();
    }
  }, [disabled, isProcessing]);

  const handleSend = useCallback(() => {
    if (input.trim() && !disabled) {
      lastSentRef.current = input;
      onSend(input);
      setInput("");
    }
  }, [input, disabled, onSend]);

  useEffect(() => {
    if (claudeQuotaExhausted && lastSentRef.current) {
      setInput(lastSentRef.current);
    }
  }, [claudeQuotaExhausted]);

  useEffect(() => {
    if (sessionId) refreshQuota();
  }, [sessionId, refreshQuota]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLDivElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  const handleModelClick = (event: React.MouseEvent<HTMLElement>) => {
    setModelAnchorEl(event.currentTarget);
  };

  const handleModelClose = () => {
    setModelAnchorEl(null);
  };

  const handleSelectModel = async (model: ModelOption) => {
    handleModelClose();
    try {
      await switchModel(model.id, model);
    } catch {
      // ignore
    }
  };

  const handleOpenCustomModal = (provider: ProviderOption) => {
    handleModelClose();
    if (!sessionId || !provider.prefix) return;
    setCustomPrefix(provider.prefix);
    setCustomModalOpen(true);
  };

  const handleCustomSubmit = async (modelId: string) => {
    const full = `${customPrefix}${modelId}`;
    const info: ModelOption = {
      id: full,
      label: modelId,
      description: "Custom OpenAI-compatible model",
      provider: OPENAI_COMPAT_PROVIDER,
      providerLabel: "OpenAI-Compatible",
    };
    await switchModel(full, info);
    setCustomModalOpen(false);
  };

  const handleCapDialogClose = useCallback(() => {
    setClaudeQuotaExhausted(false);
  }, [setClaudeQuotaExhausted]);

  const handleUseFreeModel = useCallback(async () => {
    setClaudeQuotaExhausted(false);
    if (!sessionId) return;
    const free = firstFreeModel(modelOptions);
    if (!free) return;
    try {
      await switchModel(free.id, free);
      const retryText = lastSentRef.current;
      if (retryText) {
        onSend(retryText);
        setInput("");
        lastSentRef.current = "";
      }
    } catch {
      // ignore
    }
  }, [sessionId, modelOptions, onSend, setClaudeQuotaExhausted, switchModel]);

  const claudeChip = (() => {
    if (!quota || quota.claudeUsedToday === 0) return null;
    if (quota.plan === "free") {
      return quota.claudeRemaining > 0 ? "Free today" : "Pro only";
    }
    return `${quota.claudeUsedToday}/${quota.claudeDailyCap} today`;
  })();

  return (
    <Box
      sx={{
        pb: { xs: 2, md: 4 },
        pt: { xs: 1, md: 2 },
        position: "relative",
        zIndex: 10,
      }}
    >
      <Box
        sx={{
          maxWidth: "880px",
          mx: "auto",
          width: "100%",
          px: { xs: 0, sm: 1, md: 2 },
        }}
      >
        <Box
          className="composer"
          sx={{
            display: "flex",
            gap: "10px",
            alignItems: "flex-start",
            bgcolor: "var(--composer-bg)",
            borderRadius: "var(--radius-md)",
            p: "12px",
            border: "1px solid var(--border)",
            transition: "box-shadow 0.2s ease, border-color 0.2s ease",
            "&:focus-within": {
              borderColor: "var(--accent-yellow)",
              boxShadow: "var(--focus)",
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
                color: "var(--text)",
                fontSize: "15px",
                fontFamily: "inherit",
                padding: 0,
                lineHeight: 1.5,
                minHeight: { xs: "44px", md: "56px" },
                alignItems: "flex-start",
              },
            }}
            sx={{
              flex: 1,
              "& .MuiInputBase-root": {
                p: 0,
                backgroundColor: "transparent",
              },
              "& textarea": {
                resize: "none",
                padding: "0 !important",
              },
            }}
          />
          {isProcessing ? (
            <IconButton
              onClick={onStop}
              sx={{
                mt: 1,
                p: 1.5,
                borderRadius: "10px",
                color: "var(--muted-text)",
                transition: "all 0.2s",
                position: "relative",
                "&:hover": {
                  bgcolor: "var(--hover-bg)",
                  color: "var(--accent-red)",
                },
              }}
            >
              <Box
                sx={{
                  position: "relative",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <CircularProgress
                  size={28}
                  thickness={3}
                  sx={{ color: "inherit", position: "absolute" }}
                />
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
                borderRadius: "10px",
                color: "var(--muted-text)",
                transition: "all 0.2s",
                "&:hover": {
                  color: "var(--accent-yellow)",
                  bgcolor: "var(--hover-bg)",
                },
                "&.Mui-disabled": {
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
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            mt: 1.5,
            gap: 0.8,
            opacity: 0.6,
            cursor: "pointer",
            transition: "opacity 0.2s",
            "&:hover": {
              opacity: 1,
            },
          }}
        >
          <Typography
            variant="caption"
            sx={{
              fontSize: "10px",
              color: "var(--muted-text)",
              textTransform: "uppercase",
              letterSpacing: "0.05em",
              fontWeight: 500,
            }}
          >
            powered by
          </Typography>
          {selectedModel?.avatarUrl && (
            <img
              src={selectedModel.avatarUrl}
              alt={selectedModel.label}
              style={{
                height: "14px",
                width: "14px",
                objectFit: "contain",
                borderRadius: "2px",
              }}
            />
          )}
          <Typography
            variant="caption"
            sx={{
              fontSize: "10px",
              color: "var(--text)",
              fontWeight: 600,
              letterSpacing: "0.02em",
            }}
          >
            {selectedModel?.label || "Model"}
          </Typography>
          <ArrowDropDownIcon
            sx={{ fontSize: "14px", color: "var(--muted-text)" }}
          />
        </Box>

        <Menu
          anchorEl={modelAnchorEl}
          open={Boolean(modelAnchorEl)}
          onClose={handleModelClose}
          anchorOrigin={{
            vertical: "top",
            horizontal: "center",
          }}
          transformOrigin={{
            vertical: "bottom",
            horizontal: "center",
          }}
          slotProps={{
            paper: {
              sx: {
                bgcolor: "var(--panel)",
                border: "1px solid var(--divider)",
                mb: 1,
                maxHeight: "400px",
                minWidth: 360,
              },
            },
          }}
        >
          {groups.map(({ provider, models }) => (
            <Box key={provider.id}>
              <ListSubheader
                disableSticky
                sx={{
                  bgcolor: "var(--panel)",
                  color: "var(--muted-text)",
                  fontSize: "11px",
                  lineHeight: "28px",
                  textTransform: "uppercase",
                  letterSpacing: "0.04em",
                }}
              >
                {provider.label}
              </ListSubheader>

              {models.map((model) => (
                <MenuItem
                  key={model.id}
                  onClick={() => void handleSelectModel(model)}
                  disabled={!sessionId}
                  selected={selectedModelPath === model.id}
                  sx={{
                    py: 1.5,
                    "&.Mui-selected": {
                      bgcolor: "rgba(255,255,255,0.05)",
                    },
                  }}
                >
                  <ListItemIcon>
                    {model.avatarUrl ? (
                      <img
                        src={model.avatarUrl}
                        alt={model.label}
                        style={{
                          width: 24,
                          height: 24,
                          borderRadius: "4px",
                          objectFit: "cover",
                        }}
                      />
                    ) : (
                      <Box
                        sx={{
                          width: 24,
                          height: 24,
                          borderRadius: "4px",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          bgcolor: "rgba(255,255,255,0.06)",
                          color: "var(--muted-text)",
                          fontSize: "10px",
                          fontWeight: 700,
                          textTransform: "uppercase",
                        }}
                      >
                        {provider.label.slice(0, 2)}
                      </Box>
                    )}
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box
                        sx={{ display: "flex", alignItems: "center", gap: 1 }}
                      >
                        {model.label}
                        {model.recommended && (
                          <Chip
                            label="Recommended"
                            size="small"
                            sx={{
                              height: "18px",
                              fontSize: "10px",
                              bgcolor: "var(--accent-yellow)",
                              color: "#000",
                              fontWeight: 600,
                            }}
                          />
                        )}
                        {isClaudePath(model.id) && claudeChip && (
                          <Chip
                            label={claudeChip}
                            size="small"
                            sx={{
                              height: "18px",
                              fontSize: "10px",
                              bgcolor: "rgba(255,255,255,0.08)",
                              color: "var(--muted-text)",
                              fontWeight: 600,
                            }}
                          />
                        )}
                      </Box>
                    }
                    secondary={model.description}
                    secondaryTypographyProps={{
                      sx: { fontSize: "12px", color: "var(--muted-text)" },
                    }}
                  />
                </MenuItem>
              ))}

              {provider.id === OPENAI_COMPAT_PROVIDER &&
                provider.supportsCustomModel && (
                  <MenuItem
                    onClick={() => handleOpenCustomModal(provider)}
                    disabled={!sessionId}
                    sx={{ py: 1.5 }}
                  >
                    <ListItemText
                      primary="Custom model…"
                      secondary={
                        provider.customModelHint ||
                        "Use openai-compat/<model-id>"
                      }
                      secondaryTypographyProps={{
                        sx: { fontSize: "12px", color: "var(--muted-text)" },
                      }}
                    />
                  </MenuItem>
                )}
            </Box>
          ))}
          {!sessionId && (
            <Box sx={{ px: 2, py: 1.5 }}>
              <Typography sx={{ color: "var(--muted-text)", fontSize: "12px" }}>
                Start a session to switch models.
              </Typography>
            </Box>
          )}
        </Menu>

        <CustomModelDialog
          open={customModalOpen}
          prefix={customPrefix}
          onClose={() => setCustomModalOpen(false)}
          onSubmit={handleCustomSubmit}
        />

        <ClaudeCapDialog
          open={claudeQuotaExhausted}
          plan={quota?.plan ?? "free"}
          cap={quota?.claudeDailyCap ?? 1}
          onClose={handleCapDialogClose}
          onUseFreeModel={handleUseFreeModel}
        />
      </Box>
    </Box>
  );
}
