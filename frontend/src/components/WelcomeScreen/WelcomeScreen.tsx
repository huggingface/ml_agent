import { useState, useCallback } from 'react';
import {
  Box,
  Typography,
  Button,
  CircularProgress,
  Alert,
} from '@mui/material';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import GroupAddIcon from '@mui/icons-material/GroupAdd';
import { useSessionStore } from '@/store/sessionStore';
import { useAgentStore } from '@/store/agentStore';
import { apiFetch } from '@/utils/api';
import { isInIframe, triggerLogin } from '@/hooks/useAuth';

/** HF brand orange */
const HF_ORANGE = '#FF9D00';

const ORG_JOIN_URL = 'https://huggingface.co/organizations/ml-agent-explorers/share/GzPMJUivoFPlfkvFtIqEouZKSytatKQSZT';

export default function WelcomeScreen() {
  const { createSession } = useSessionStore();
  const { setPlan, clearPanel, user } = useAgentStore();
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [iframeJoined, setIframeJoined] = useState(false);

  const inIframe = isInIframe();
  const isAuthenticated = user?.authenticated;
  const isDevUser = user?.username === 'dev';

  const tryCreateSession = useCallback(async () => {
    setIsCreating(true);
    setError(null);

    try {
      const response = await apiFetch('/api/session', { method: 'POST' });
      if (response.status === 503) {
        const data = await response.json();
        setError(data.detail || 'Server is at capacity. Please try again later.');
        return;
      }
      if (response.status === 401) {
        triggerLogin();
        return;
      }
      if (!response.ok) {
        setError('Failed to create session. Please try again.');
        return;
      }
      const data = await response.json();
      createSession(data.session_id);
      setPlan([]);
      clearPanel();
    } catch {
      // Redirect may throw — ignore
    } finally {
      setIsCreating(false);
    }
  }, [createSession, setPlan, clearPanel]);

  const handleStart = useCallback(async () => {
    if (isCreating) return;

    if (!isAuthenticated && !isDevUser) {
      if (inIframe) return;
      triggerLogin();
      return;
    }

    await tryCreateSession();
  }, [isCreating, isAuthenticated, isDevUser, inIframe, tryCreateSession]);

  // Build the direct Space URL for the "open in new tab" link
  const spaceHost = typeof window !== 'undefined'
    ? window.location.hostname.includes('.hf.space')
      ? window.location.origin
      : `https://smolagents-ml-agent.hf.space`
    : '';

  // Shared button style
  const primaryBtnSx = {
    px: 5,
    py: 1.5,
    fontSize: '1rem',
    fontWeight: 700,
    textTransform: 'none' as const,
    borderRadius: '12px',
    bgcolor: HF_ORANGE,
    color: '#000',
    boxShadow: '0 4px 24px rgba(255, 157, 0, 0.3)',
    textDecoration: 'none',
    '&:hover': {
      bgcolor: '#FFB340',
      boxShadow: '0 6px 32px rgba(255, 157, 0, 0.45)',
    },
  };

  return (
    <Box
      sx={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--body-gradient)',
        py: 8,
      }}
    >
      {/* HF Logo */}
      <Box
        component="img"
        src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg"
        alt="Hugging Face"
        sx={{ width: 96, height: 96, mb: 3, display: 'block' }}
      />

      {/* Title */}
      <Typography
        variant="h2"
        sx={{
          fontWeight: 800,
          color: 'var(--text)',
          mb: 1.5,
          letterSpacing: '-0.02em',
          fontSize: { xs: '2rem', md: '2.8rem' },
        }}
      >
        HF Agent
      </Typography>

      {/* ── Iframe: join org → then redirect to Space ────────────── */}
      {inIframe && !iframeJoined && (
        <>
          <Typography
            variant="body1"
            sx={{
              color: 'var(--muted-text)',
              maxWidth: 480,
              mb: 4,
              lineHeight: 1.8,
              fontSize: '0.95rem',
              textAlign: 'center',
              px: 2,
              '& strong': { color: 'var(--text)', fontWeight: 600 },
            }}
          >
            Under the hood, this agent uses GPUs, inference APIs, and other paid Hub goodies — but we made them all free for you. Just join <strong>ML Agent Explorers</strong> to get started!
          </Typography>

          <Button
            variant="contained"
            size="large"
            component="a"
            href={ORG_JOIN_URL}
            target="_blank"
            rel="noopener noreferrer"
            startIcon={<GroupAddIcon />}
            sx={primaryBtnSx}
          >
            Join ML Agent Explorers
          </Button>

          <Button
            variant="text"
            size="small"
            onClick={() => setIframeJoined(true)}
            sx={{
              mt: 2,
              color: 'var(--muted-text)',
              textTransform: 'none',
              fontSize: '0.85rem',
              '&:hover': { color: 'var(--text)' },
            }}
          >
            I've already joined →
          </Button>
        </>
      )}

      {/* ── Iframe: after joining → open Space ───────────────────── */}
      {inIframe && iframeJoined && (
        <>
          <Typography
            variant="body1"
            sx={{
              color: 'var(--muted-text)',
              maxWidth: 520,
              mb: 5,
              lineHeight: 1.8,
              fontSize: '0.95rem',
              textAlign: 'center',
              px: 2,
              '& strong': { color: 'var(--text)', fontWeight: 600 },
            }}
          >
            A general-purpose AI agent for <strong>machine learning engineering</strong>.
            It browses <strong>Hugging Face documentation</strong>, manages{' '}
            <strong>repositories</strong>, launches <strong>training jobs</strong>,
            and explores <strong>datasets</strong> — all through natural conversation.
          </Typography>

          <Button
            variant="contained"
            size="large"
            component="a"
            href={spaceHost}
            target="_blank"
            rel="noopener noreferrer"
            endIcon={<OpenInNewIcon />}
            sx={primaryBtnSx}
          >
            Open HF Agent
          </Button>
        </>
      )}

      {/* ── Direct: not logged in → sign in ──────────────────────── */}
      {!inIframe && !isAuthenticated && !isDevUser && (
        <>
          <Typography
            variant="body1"
            sx={{
              color: 'var(--muted-text)',
              maxWidth: 520,
              mb: 5,
              lineHeight: 1.8,
              fontSize: '0.95rem',
              textAlign: 'center',
              px: 2,
              '& strong': { color: 'var(--text)', fontWeight: 600 },
            }}
          >
            A general-purpose AI agent for <strong>machine learning engineering</strong>.
            It browses <strong>Hugging Face documentation</strong>, manages{' '}
            <strong>repositories</strong>, launches <strong>training jobs</strong>,
            and explores <strong>datasets</strong> — all through natural conversation.
          </Typography>

          <Button
            variant="contained"
            size="large"
            onClick={() => triggerLogin()}
            sx={primaryBtnSx}
          >
            Sign in with Hugging Face
          </Button>

          <Typography
            variant="caption"
            sx={{
              mt: 2.5,
              color: 'var(--muted-text)',
              fontSize: '0.78rem',
              textAlign: 'center',
              maxWidth: 360,
              lineHeight: 1.6,
            }}
          >
            Make sure to enable access to the <strong>ml-agent-explorers</strong> org when prompted.
          </Typography>
        </>
      )}

      {/* ── Direct: authenticated → start session ────────────────── */}
      {!inIframe && (isAuthenticated || isDevUser) && (
        <>
          <Typography
            variant="body1"
            sx={{
              color: 'var(--muted-text)',
              maxWidth: 520,
              mb: 5,
              lineHeight: 1.8,
              fontSize: '0.95rem',
              textAlign: 'center',
              px: 2,
              '& strong': { color: 'var(--text)', fontWeight: 600 },
            }}
          >
            A general-purpose AI agent for <strong>machine learning engineering</strong>.
            It browses <strong>Hugging Face documentation</strong>, manages{' '}
            <strong>repositories</strong>, launches <strong>training jobs</strong>,
            and explores <strong>datasets</strong> — all through natural conversation.
          </Typography>

          <Button
            variant="contained"
            size="large"
            onClick={handleStart}
            disabled={isCreating}
            startIcon={
              isCreating ? <CircularProgress size={20} color="inherit" /> : null
            }
            sx={{
              ...primaryBtnSx,
              '&.Mui-disabled': {
                bgcolor: 'rgba(255, 157, 0, 0.35)',
                color: 'rgba(0,0,0,0.45)',
              },
            }}
          >
            {isCreating ? 'Initializing...' : 'Start Session'}
          </Button>
        </>
      )}

      {/* Error */}
      {error && (
        <Alert
          severity="warning"
          variant="outlined"
          onClose={() => setError(null)}
          sx={{
            mt: 3,
            maxWidth: 400,
            fontSize: '0.8rem',
            borderColor: HF_ORANGE,
            color: 'var(--text)',
          }}
        >
          {error}
        </Alert>
      )}

      {/* Footnote */}
      <Typography
        variant="caption"
        sx={{ mt: 5, color: 'var(--muted-text)', opacity: 0.5, fontSize: '0.7rem' }}
      >
        Conversations are stored locally in your browser.
      </Typography>
    </Box>
  );
}
