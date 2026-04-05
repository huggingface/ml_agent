import { useState, useCallback, useEffect, useRef, type ReactNode } from 'react';
import {
  Box,
  Typography,
  Button,
  CircularProgress,
  Alert,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import GroupAddIcon from '@mui/icons-material/GroupAdd';
import LoginIcon from '@mui/icons-material/Login';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import { useSessionStore } from '@/store/sessionStore';
import { useAgentStore } from '@/store/agentStore';
import { apiFetch } from '@/utils/api';
import { isInIframe, triggerLogin } from '@/hooks/useAuth';
import { useOrgMembership } from '@/hooks/useOrgMembership';

const HF_ORANGE = '#FF9D00';
const ORG_JOIN_URL =
  'https://huggingface.co/organizations/ml-agent-explorers/share/GzPMJUivoFPlfkvFtIqEouZKSytatKQSZT';

// ---------------------------------------------------------------------------
// ChecklistStep sub-component
// ---------------------------------------------------------------------------

type StepStatus = 'completed' | 'active' | 'locked';

interface ChecklistStepProps {
  stepNumber: number;
  title: string;
  description: string;
  status: StepStatus;
  lockedReason?: string;
  actionLabel?: string;
  onAction?: () => void;
  actionIcon?: ReactNode;
  actionHref?: string;
  loading?: boolean;
  isLast?: boolean;
}

function StepIndicator({ status, stepNumber }: { status: StepStatus; stepNumber: number }) {
  if (status === 'completed') {
    return <CheckCircleIcon sx={{ fontSize: 28, color: 'var(--accent-green)' }} />;
  }
  return (
    <Box
      sx={{
        width: 28,
        height: 28,
        borderRadius: '50%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '0.8rem',
        fontWeight: 700,
        ...(status === 'active'
          ? { bgcolor: HF_ORANGE, color: '#000' }
          : { bgcolor: 'transparent', border: '2px solid var(--border)', color: 'var(--muted-text)' }),
      }}
    >
      {stepNumber}
    </Box>
  );
}

function ChecklistStep({
  stepNumber,
  title,
  description,
  status,
  lockedReason,
  actionLabel,
  onAction,
  actionIcon,
  actionHref,
  loading = false,
  isLast = false,
}: ChecklistStepProps) {
  const btnSx = {
    px: 3,
    py: 0.75,
    fontSize: '0.85rem',
    fontWeight: 700,
    textTransform: 'none' as const,
    borderRadius: '10px',
    whiteSpace: 'nowrap' as const,
    textDecoration: 'none',
    ...(status === 'active'
      ? {
          bgcolor: HF_ORANGE,
          color: '#000',
          boxShadow: '0 2px 12px rgba(255, 157, 0, 0.25)',
          '&:hover': { bgcolor: '#FFB340', boxShadow: '0 4px 20px rgba(255, 157, 0, 0.4)' },
        }
      : {
          bgcolor: 'rgba(255,255,255,0.04)',
          color: 'var(--muted-text)',
          '&.Mui-disabled': { bgcolor: 'rgba(255,255,255,0.04)', color: 'var(--muted-text)' },
        }),
  };

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        px: 3,
        py: 2.5,
        borderLeft: '3px solid',
        borderLeftColor:
          status === 'completed'
            ? 'var(--accent-green)'
            : status === 'active'
              ? HF_ORANGE
              : 'transparent',
        ...(!isLast && { borderBottom: '1px solid var(--border)' }),
        opacity: status === 'locked' ? 0.55 : 1,
        transition: 'opacity 0.2s, border-color 0.2s',
      }}
    >
      <StepIndicator status={status} stepNumber={stepNumber} />

      <Box sx={{ flex: 1, minWidth: 0 }}>
        <Typography
          variant="subtitle2"
          sx={{
            fontWeight: 600,
            fontSize: '0.92rem',
            color: status === 'completed' ? 'var(--muted-text)' : 'var(--text)',
            ...(status === 'completed' && { textDecoration: 'line-through', textDecorationColor: 'var(--muted-text)' }),
          }}
        >
          {title}
        </Typography>
        <Typography variant="body2" sx={{ color: 'var(--muted-text)', fontSize: '0.8rem', mt: 0.25, lineHeight: 1.5 }}>
          {status === 'locked' && lockedReason ? lockedReason : description}
        </Typography>
      </Box>

      {status === 'completed' ? (
        <Typography variant="caption" sx={{ color: 'var(--accent-green)', fontWeight: 600, fontSize: '0.78rem', whiteSpace: 'nowrap' }}>
          Done
        </Typography>
      ) : actionLabel ? (
        actionHref ? (
          <Button
            variant="contained"
            size="small"
            component="a"
            href={actionHref}
            target="_blank"
            rel="noopener noreferrer"
            disabled={status === 'locked'}
            startIcon={actionIcon}
            sx={btnSx}
            onClick={onAction}
          >
            {actionLabel}
          </Button>
        ) : (
          <Button
            variant="contained"
            size="small"
            disabled={status === 'locked' || loading}
            startIcon={loading ? <CircularProgress size={16} color="inherit" /> : actionIcon}
            onClick={onAction}
            sx={btnSx}
          >
            {loading ? 'Loading...' : actionLabel}
          </Button>
        )
      ) : null}
    </Box>
  );
}

// ---------------------------------------------------------------------------
// WelcomeScreen
// ---------------------------------------------------------------------------

export default function WelcomeScreen() {
  const { createSession } = useSessionStore();
  const { setPlan, clearPanel, user } = useAgentStore();
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const inIframe = isInIframe();
  const isAuthenticated = !!user?.authenticated;
  const isDevUser = user?.username === 'dev';

  // Iframe: localStorage-based org tracking (no auth token available)
  const [iframeOrgJoined, setIframeOrgJoined] = useState(() => {
    try { return localStorage.getItem('hf-agent-org-joined') === '1'; } catch { return false; }
  });
  const joinLinkOpened = useRef(false);

  // Auto-advance when user returns from org join link (iframe only)
  useEffect(() => {
    if (!inIframe) return;
    const handleVisibility = () => {
      if (document.visibilityState !== 'visible' || !joinLinkOpened.current) return;
      joinLinkOpened.current = false;
      try { localStorage.setItem('hf-agent-org-joined', '1'); } catch { /* ignore */ }
      setIframeOrgJoined(true);
    };
    document.addEventListener('visibilitychange', handleVisibility);
    return () => document.removeEventListener('visibilitychange', handleVisibility);
  }, [inIframe]);

  const isOrgMember = inIframe ? iframeOrgJoined : !!user?.orgMember;

  // Poll for org membership once authenticated (skipped in dev mode and iframe)
  const popupRef = useOrgMembership(isAuthenticated && !isDevUser && !inIframe && !isOrgMember);

  // ---- Actions ----

  const handleJoinOrg = useCallback(() => {
    if (inIframe) {
      // Iframe: open link, track via visibilitychange + localStorage
      joinLinkOpened.current = true;
      window.open(ORG_JOIN_URL, '_blank', 'noopener,noreferrer');
      return;
    }
    // Direct: open as popup, auto-close via polling
    const popup = window.open(ORG_JOIN_URL, 'hf-org-join', 'noopener');
    if (popup) {
      popupRef.current = popup;
    } else {
      window.open(ORG_JOIN_URL, '_blank', 'noopener,noreferrer');
    }
  }, [popupRef, inIframe]);

  const handleStartSession = useCallback(async () => {
    if (isCreating) return;
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
  }, [isCreating, createSession, setPlan, clearPanel]);

  // ---- Step status helpers ----

  const signInStatus: StepStatus = isAuthenticated ? 'completed' : 'active';
  const joinOrgStatus: StepStatus = isOrgMember ? 'completed' : isAuthenticated ? 'active' : 'locked';
  const startStatus: StepStatus = isAuthenticated && isOrgMember ? 'active' : 'locked';

  // Space URL for iframe "Open HF Agent" step
  const spaceHost =
    typeof window !== 'undefined'
      ? window.location.hostname.includes('.hf.space')
        ? window.location.origin
        : 'https://smolagents-ml-agent.hf.space'
      : '';

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
      {/* Logo */}
      <Box
        component="img"
        src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg"
        alt="Hugging Face"
        sx={{ width: 80, height: 80, mb: 2.5, display: 'block' }}
      />

      {/* Title */}
      <Typography
        variant="h2"
        sx={{
          fontWeight: 800,
          color: 'var(--text)',
          mb: 1,
          letterSpacing: '-0.02em',
          fontSize: { xs: '1.8rem', md: '2.4rem' },
        }}
      >
        HF Agent
      </Typography>

      {/* Description */}
      <Typography
        variant="body1"
        sx={{
          color: 'var(--muted-text)',
          maxWidth: 480,
          mb: 4,
          lineHeight: 1.7,
          fontSize: '0.9rem',
          textAlign: 'center',
          px: 2,
          '& strong': { color: 'var(--text)', fontWeight: 600 },
        }}
      >
        A general-purpose AI agent for <strong>machine learning engineering</strong>.
        It browses <strong>Hugging Face docs</strong>, manages <strong>repos</strong>,
        launches <strong>training jobs</strong>, and explores <strong>datasets</strong>.
      </Typography>

      {/* ── Checklist ──────────────────────────────────────────── */}
      <Box
        sx={{
          width: '100%',
          maxWidth: 520,
          bgcolor: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: '12px',
          overflow: 'hidden',
          mx: 2,
        }}
      >
        {isDevUser ? (
          /* Dev mode: single step */
          <ChecklistStep
            stepNumber={1}
            title="Start Session"
            description="Launch an AI agent session for ML engineering."
            status="active"
            actionLabel="Start Session"
            actionIcon={<RocketLaunchIcon sx={{ fontSize: 16 }} />}
            onAction={handleStartSession}
            loading={isCreating}
            isLast
          />
        ) : inIframe ? (
          /* Iframe: 2 steps */
          <>
            <ChecklistStep
              stepNumber={1}
              title="Join ML Agent Explorers"
              description="Get free access to GPUs, inference APIs, and Hub resources."
              status={isOrgMember ? 'completed' : 'active'}
              actionLabel="Join Organization"
              actionIcon={<GroupAddIcon sx={{ fontSize: 16 }} />}
              onAction={handleJoinOrg}
            />
            <ChecklistStep
              stepNumber={2}
              title="Open HF Agent"
              description="Open the agent in a full browser tab to get started."
              status={isOrgMember ? 'active' : 'locked'}
              lockedReason="Join the organization first."
              actionLabel="Open HF Agent"
              actionIcon={<OpenInNewIcon sx={{ fontSize: 16 }} />}
              actionHref={spaceHost}
              isLast
            />
          </>
        ) : (
          /* Direct access: 3 steps */
          <>
            <ChecklistStep
              stepNumber={1}
              title="Sign in with Hugging Face"
              description="Authenticate to access GPU resources and model APIs."
              status={signInStatus}
              actionLabel="Sign in"
              actionIcon={<LoginIcon sx={{ fontSize: 16 }} />}
              onAction={() => triggerLogin()}
            />
            <ChecklistStep
              stepNumber={2}
              title="Join ML Agent Explorers"
              description="Get free access to GPUs, inference APIs, and Hub resources."
              status={joinOrgStatus}
              lockedReason="Sign in first to continue."
              actionLabel="Join Organization"
              actionIcon={<GroupAddIcon sx={{ fontSize: 16 }} />}
              onAction={handleJoinOrg}
            />
            <ChecklistStep
              stepNumber={3}
              title="Start Session"
              description="Launch an AI agent session for ML engineering."
              status={startStatus}
              lockedReason="Complete the steps above to continue."
              actionLabel="Start Session"
              actionIcon={<RocketLaunchIcon sx={{ fontSize: 16 }} />}
              onAction={handleStartSession}
              loading={isCreating}
              isLast
            />
          </>
        )}
      </Box>

      {/* Polling hint when waiting for org join */}
      {isAuthenticated && !isOrgMember && !isDevUser && !inIframe && (
        <Typography
          variant="caption"
          sx={{ mt: 2, color: 'var(--muted-text)', fontSize: '0.75rem', textAlign: 'center' }}
        >
          This page updates automatically when you join the organization.
        </Typography>
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
        sx={{ mt: 4, color: 'var(--muted-text)', opacity: 0.5, fontSize: '0.7rem' }}
      >
        Conversations are stored locally in your browser.
      </Typography>
    </Box>
  );
}
