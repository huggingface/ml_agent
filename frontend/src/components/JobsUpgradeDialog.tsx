import { Box, Button, Typography } from '@mui/material';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import CreditCardIcon from '@mui/icons-material/CreditCard';
import ReplayIcon from '@mui/icons-material/Replay';
import CloseIcon from '@mui/icons-material/Close';

const HF_BILLING_URL = 'https://huggingface.co/settings/billing';

interface JobsUpgradeDialogProps {
  open: boolean;
  message: string;
  /** True after the user clicked "Add credits" — the visibility-change auto-retry
   *  in the parent uses this; it is unused inside the screen itself, which always
   *  shows both actions ("Add credits" and "I've added credits"). */
  awaitingTopUp: boolean;
  onUpgrade: () => void;
  onRetry: () => void;
  onClose: () => void;
}

export default function JobsUpgradeDialog({
  open,
  message,
  awaitingTopUp,
  onUpgrade,
  onRetry,
  onClose,
}: JobsUpgradeDialogProps) {
  if (!open) return null;

  const primarySx = {
    bgcolor: 'var(--text)',
    color: 'var(--bg)',
    fontWeight: 700,
    fontSize: '0.85rem',
    textTransform: 'none' as const,
    px: 2.5,
    py: 1,
    borderRadius: '10px',
    boxShadow: 'none',
    '&:hover': { bgcolor: 'var(--text)', opacity: 0.9, boxShadow: 'none' },
  };

  const secondarySx = {
    bgcolor: 'transparent',
    color: 'var(--text)',
    fontWeight: 600,
    fontSize: '0.85rem',
    textTransform: 'none' as const,
    px: 2.5,
    py: 1,
    borderRadius: '10px',
    border: '1px solid var(--border-hover)',
    '&:hover': { bgcolor: 'var(--hover-bg)', borderColor: 'var(--border-hover)' },
  };

  return (
    <Box
      sx={{
        position: 'fixed',
        inset: 0,
        zIndex: 1300,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: 'rgba(0,0,0,0.55)',
        backdropFilter: 'blur(8px)',
        px: 2,
      }}
      role="dialog"
      aria-modal="true"
      aria-labelledby="jobs-billing-title"
    >
      <Box
        sx={{
          position: 'relative',
          width: '100%',
          maxWidth: 480,
          bgcolor: 'var(--panel)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-md)',
          boxShadow: 'var(--shadow-1)',
          px: 4,
          py: 4,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          textAlign: 'center',
        }}
      >
        <Button
          onClick={onClose}
          aria-label="Close"
          sx={{
            position: 'absolute',
            top: 10,
            right: 10,
            minWidth: 0,
            width: 28,
            height: 28,
            borderRadius: '8px',
            color: 'var(--muted-text)',
            '&:hover': { bgcolor: 'var(--hover-bg)', color: 'var(--text)' },
          }}
        >
          <CloseIcon sx={{ fontSize: 16 }} />
        </Button>

        <Box
          sx={{
            width: 44,
            height: 44,
            borderRadius: '12px',
            bgcolor: 'var(--surface)',
            border: '1px solid var(--border)',
            color: 'var(--muted-text)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            mb: 2,
          }}
        >
          <CreditCardIcon sx={{ fontSize: 22 }} />
        </Box>

        <Typography
          id="jobs-billing-title"
          sx={{
            color: 'var(--text)',
            fontWeight: 700,
            fontSize: '1.05rem',
            letterSpacing: '-0.01em',
            mb: 1,
          }}
        >
          {awaitingTopUp ? 'Resume when you’re ready' : 'Add credits to launch this job'}
        </Typography>

        <Typography
          sx={{
            color: 'var(--muted-text)',
            fontSize: '0.85rem',
            lineHeight: 1.6,
            mb: 3,
            maxWidth: 380,
          }}
        >
          {awaitingTopUp
            ? 'Once your top-up is through, click below to resume — the agent will pick the run back up where it left off.'
            : message ||
              'Hugging Face Jobs need credits on the namespace running them. Job credits are separate from HF Pro membership. Add some, then resume.'}
        </Typography>

        <Box
          sx={{
            display: 'flex',
            flexDirection: { xs: 'column', sm: 'row' },
            gap: 1.25,
            width: '100%',
            justifyContent: 'center',
          }}
        >
          <Button
            component="a"
            href={HF_BILLING_URL}
            target="_blank"
            rel="noopener noreferrer"
            onClick={onUpgrade}
            startIcon={<OpenInNewIcon sx={{ fontSize: 16 }} />}
            variant="contained"
            sx={primarySx}
          >
            Add credits
          </Button>
          <Button
            onClick={onRetry}
            startIcon={<ReplayIcon sx={{ fontSize: 16 }} />}
            variant="outlined"
            sx={secondarySx}
          >
            I’ve added credits
          </Button>
        </Box>
      </Box>
    </Box>
  );
}
