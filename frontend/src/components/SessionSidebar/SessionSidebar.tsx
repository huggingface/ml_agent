import { useCallback, useEffect, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  IconButton,
  Typography,
  CircularProgress,
  Divider,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import ChatBubbleOutlineIcon from '@mui/icons-material/ChatBubbleOutline';
import { useSessionStore } from '@/store/sessionStore';
import { useAgentStore } from '@/store/agentStore';
import { apiFetch } from '@/utils/api';

interface SessionSidebarProps {
  onClose?: () => void;
}

export default function SessionSidebar({ onClose }: SessionSidebarProps) {
  const { sessions, activeSessionId, createSession, deleteSession, switchSession, mergeServerSessions } =
    useSessionStore();
  const { setPlan, clearPanel } =
    useAgentStore();
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [capacityError, setCapacityError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const response = await apiFetch('/api/sessions');
        if (!response.ok) return;
        const data = await response.json();
        if (!cancelled && Array.isArray(data)) {
          mergeServerSessions(data);
        }
      } catch {
        /* local sidebar metadata is still usable */
      }
    })();
    return () => { cancelled = true; };
  }, [mergeServerSessions]);

  // -- Handlers -----------------------------------------------------------

  const handleNewSession = useCallback(async () => {
    if (isCreatingSession) return;
    setIsCreatingSession(true);
    setCapacityError(null);
    try {
      const response = await apiFetch('/api/session', { method: 'POST' });
      if (response.status === 503) {
        const data = await response.json();
        setCapacityError(data.detail || 'Server is at capacity.');
        return;
      }
      const data = await response.json();
      createSession(data.session_id);
      setPlan([]);
      clearPanel();
      onClose?.();
    } catch {
      setCapacityError('Failed to create session.');
    } finally {
      setIsCreatingSession(false);
    }
  }, [isCreatingSession, createSession, setPlan, clearPanel, onClose]);

  // -- Delete with dialog confirmation ------------------------------------
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const handleDeleteClick = useCallback(
    (sessionId: string, e: React.MouseEvent) => {
      e.stopPropagation();
      setConfirmDeleteId(sessionId);
    },
    [],
  );

  const handleDeleteConfirm = useCallback(async () => {
    if (!confirmDeleteId || isDeleting) return;
    const sessionId = confirmDeleteId;
    setIsDeleting(true);

    const isLastSession = sessions.length === 1;

    useAgentStore.getState().clearSessionState(sessionId);
    try {
      await apiFetch(`/api/session/${sessionId}`, { method: 'DELETE' });
      deleteSession(sessionId);
    } catch {
      deleteSession(sessionId);
    }

    // If this was the last session, create a new one
    if (isLastSession) {
      try {
        const response = await apiFetch('/api/session', { method: 'POST' });
        if (response.ok) {
          const data = await response.json();
          createSession(data.session_id);
          setPlan([]);
          clearPanel();
        }
      } catch (error) {
        console.error('Failed to create new session after deleting last one:', error);
      }
    }

    setIsDeleting(false);
    setConfirmDeleteId(null);
  }, [deleteSession, confirmDeleteId, isDeleting, sessions, createSession, setPlan, clearPanel]);

  const handleSelect = useCallback(
    (sessionId: string) => {
      switchSession(sessionId);
      // Per-session state (plan, panel, activity) is restored automatically
      // by SessionChat's useEffect when isActive flips to true.
      onClose?.();
    },
    [switchSession, onClose],
  );

  const formatTime = (d: string) =>
    new Date(d).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  // -- Render -------------------------------------------------------------

  return (
    <Box
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        bgcolor: 'var(--panel)',
      }}
    >
      {/* -- Header -------------------------------------------------------- */}
      <Box sx={{ px: 1.75, pt: 2, pb: 0 }}>
        <Typography
          variant="caption"
          sx={{
            color: 'var(--muted-text)',
            fontSize: '0.65rem',
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
          }}
        >
          Recent chats
        </Typography>
      </Box>

      {/* -- Capacity error ------------------------------------------------ */}
      {capacityError && (
        <Alert
          severity="warning"
          variant="outlined"
          onClose={() => setCapacityError(null)}
          sx={{
            m: 1,
            fontSize: '0.7rem',
            py: 0.25,
            '& .MuiAlert-message': { py: 0 },
            borderColor: '#FF9D00',
            color: 'var(--text)',
          }}
        >
          {capacityError}
        </Alert>
      )}

      {/* -- Session list -------------------------------------------------- */}
      <Box
        sx={{
          flex: 1,
          overflow: 'auto',
          py: 1,
          '&::-webkit-scrollbar': { width: 4 },
          '&::-webkit-scrollbar-thumb': {
            bgcolor: 'var(--scrollbar-thumb)',
            borderRadius: 2,
          },
        }}
      >
        {sessions.length === 0 ? (
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              py: 8,
              px: 3,
              gap: 1.5,
            }}
          >
            <ChatBubbleOutlineIcon
              sx={{ fontSize: 28, color: 'var(--muted-text)', opacity: 0.25 }}
            />
            <Typography
              variant="caption"
              sx={{
                color: 'var(--muted-text)',
                opacity: 0.5,
                textAlign: 'center',
                lineHeight: 1.5,
                fontSize: '0.72rem',
              }}
            >
              No sessions yet
            </Typography>
          </Box>
        ) : (
          [...sessions].reverse().map((session, index) => {
            const num = sessions.length - index;
            const isSelected = session.id === activeSessionId;

            return (
              <Box
                key={session.id}
                onClick={() => handleSelect(session.id)}
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                  px: 1.5,
                  py: 0.875,
                  mx: 0.75,
                  mb: 0.2,
                  borderRadius: '10px',
                  cursor: 'pointer',
                  transition: 'background-color 0.12s ease',
                  bgcolor: isSelected
                    ? 'var(--hover-bg)'
                    : 'transparent',
                  '&:hover': {
                    bgcolor: 'var(--hover-bg)',
                  },
                  '& .delete-btn': {
                    opacity: 0,
                    transition: 'opacity 0.12s',
                  },
                  '&:hover .delete-btn': {
                    opacity: 1,
                  },
                }}
              >
                <ChatBubbleOutlineIcon
                  sx={{
                    fontSize: 15,
                    color: isSelected ? 'var(--text)' : 'var(--muted-text)',
                    opacity: isSelected ? 0.8 : 0.4,
                    flexShrink: 0,
                  }}
                />

                <Box sx={{ flex: 1, minWidth: 0 }}>
                  <Typography
                    variant="body2"
                    sx={{
                      fontWeight: isSelected ? 600 : 400,
                      color: 'var(--text)',
                      fontSize: '0.84rem',
                      lineHeight: 1.4,
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                    }}
                  >
                    {session.title.startsWith('Chat ') ? `Session ${String(num).padStart(2, '0')}` : session.title}
                  </Typography>
                  <Typography
                    variant="caption"
                    sx={{
                      color: 'var(--muted-text)',
                      fontSize: '0.65rem',
                      lineHeight: 1.2,
                    }}
                  >
                    {session.expired ? 'needs a catch-up' : formatTime(session.createdAt)}
                  </Typography>
                </Box>

                {/* Attention badge — pulsing dot when background session needs approval */}
                {session.needsAttention && !isSelected && (
                  <Box
                    sx={{
                      width: 8,
                      height: 8,
                      borderRadius: '50%',
                      bgcolor: 'var(--accent-yellow)',
                      flexShrink: 0,
                      animation: 'pulse 2s ease-in-out infinite',
                      '@keyframes pulse': {
                        '0%, 100%': { opacity: 1, transform: 'scale(1)' },
                        '50%': { opacity: 0.5, transform: 'scale(0.8)' },
                      },
                    }}
                  />
                )}

                <IconButton
                  className="delete-btn"
                  size="small"
                  onClick={(e) => handleDeleteClick(session.id, e)}
                  sx={{
                    color: 'var(--muted-text)',
                    width: 26,
                    height: 26,
                    flexShrink: 0,
                    '&:hover': { color: 'var(--accent-red)', bgcolor: 'rgba(244,67,54,0.08)' },
                  }}
                >
                  <DeleteOutlineIcon sx={{ fontSize: 15 }} />
                </IconButton>
              </Box>
            );
          })
        )}
      </Box>

      {/* -- Footer: New Task + status ------------------------------------- */}
      <Divider sx={{ opacity: 0.5 }} />
      <Box
        sx={{
          px: 1.5,
          py: 1.5,
          display: 'flex',
          flexDirection: 'column',
          gap: 1,
          flexShrink: 0,
        }}
      >
        <Box
          component="button"
          onClick={handleNewSession}
          disabled={isCreatingSession}
          sx={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 0.75,
            width: '100%',
            px: 1.5,
            py: 1.25,
            border: 'none',
            borderRadius: '10px',
            bgcolor: '#FF9D00',
            color: '#000',
            fontSize: '0.85rem',
            fontWeight: 700,
            cursor: 'pointer',
            transition: 'all 0.12s ease',
            '&:hover': {
              bgcolor: '#FFB340',
            },
            '&:disabled': {
              opacity: 0.5,
              cursor: 'not-allowed',
            },
          }}
        >
          {isCreatingSession ? (
            <>
              <CircularProgress size={12} sx={{ color: '#000' }} />
              Creating...
            </>
          ) : (
            <>
              <AddIcon sx={{ fontSize: 16 }} />
              New Task
            </>
          )}
        </Box>

      </Box>
      {/* Delete confirmation dialog */}
      <Dialog
        open={!!confirmDeleteId}
        onClose={() => !isDeleting && setConfirmDeleteId(null)}
        slotProps={{
          backdrop: { sx: { backgroundColor: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(4px)' } },
        }}
        PaperProps={{
          sx: {
            bgcolor: 'var(--panel)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)',
            boxShadow: 'var(--shadow-1)',
            maxWidth: 340,
            mx: 2,
          },
        }}
      >
        <DialogTitle
          sx={{
            color: 'var(--text)',
            fontWeight: 700,
            fontSize: '0.95rem',
            pb: 0,
            pt: 2.5,
            px: 3,
          }}
        >
          Delete conversation?
        </DialogTitle>
        <DialogContent sx={{ px: 3, pt: 1 }}>
          <DialogContentText
            sx={{
              color: 'var(--muted-text)',
              fontSize: '0.82rem',
              lineHeight: 1.6,
            }}
          >
            This will permanently remove this conversation and its history.
          </DialogContentText>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2.5, gap: 1 }}>
          <Button
            onClick={() => setConfirmDeleteId(null)}
            size="small"
            disabled={isDeleting}
            sx={{
              color: 'var(--muted-text)',
              fontSize: '0.82rem',
              px: 2,
              '&:hover': { bgcolor: 'var(--hover-bg)' },
            }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleDeleteConfirm}
            variant="contained"
            size="small"
            disabled={isDeleting}
            startIcon={isDeleting ? <CircularProgress size={16} sx={{ color: '#fff' }} /> : undefined}
            sx={{
              fontSize: '0.82rem',
              px: 2.5,
              bgcolor: 'var(--accent-red)',
              color: '#fff',
              boxShadow: 'none',
              '&:hover': {
                bgcolor: 'var(--accent-red)',
                filter: 'brightness(1.15)',
                boxShadow: 'none',
              },
              '&.Mui-disabled': {
                bgcolor: 'var(--accent-red)',
                color: '#fff',
                opacity: 0.7,
              },
            }}
          >
            {isDeleting ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
