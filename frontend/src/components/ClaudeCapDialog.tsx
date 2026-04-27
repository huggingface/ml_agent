import {
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Typography,
} from '@mui/material';
import type { PlanTier } from '@/hooks/useUserQuota';

const HF_PRICING_URL = 'https://huggingface.co/pricing';
const PRO_CAP = 20;

interface ClaudeCapDialogProps {
  open: boolean;
  plan: PlanTier;
  cap: number;
  onClose: () => void;
  onUseFreeModel: () => void;
  onUpgrade: () => void;
}

export default function ClaudeCapDialog({
  open,
  plan,
  cap,
  onClose,
  onUseFreeModel,
  onUpgrade,
}: ClaudeCapDialogProps) {
  // plan not surfaced in copy right now — Pro users see the same dialog and
  // can upgrade their org if they're also capped.
  void plan;

  return (
    <Dialog
      open={open}
      onClose={onClose}
      slotProps={{
        backdrop: { sx: { backgroundColor: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(4px)' } },
      }}
      PaperProps={{
        sx: {
          bgcolor: 'var(--panel)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-md)',
          boxShadow: 'var(--shadow-1)',
          maxWidth: 460,
          mx: 2,
        },
      }}
    >
      <DialogTitle
        sx={{ color: 'var(--text)', fontWeight: 700, fontSize: '1rem', pt: 2.5, pb: 0, px: 3 }}
      >
        You've hit your Opus limit
      </DialogTitle>
      <DialogContent sx={{ px: 3, pt: 1.25, pb: 0 }}>
        <DialogContentText
          sx={{ color: 'var(--muted-text)', fontSize: '0.85rem', lineHeight: 1.6 }}
        >
          Opus costs an arm and a leg, so we unfortunately have to cap you at {cap}{' '}
          {cap === 1 ? 'session' : 'sessions'} a day. Give Kimi, MiniMax, or GLM a spin —
          they are genuinely good and we use them all the time.
        </DialogContentText>
        <Box
          sx={{
            mt: 2,
            p: 1.5,
            borderRadius: '8px',
            bgcolor: 'var(--accent-yellow-weak)',
            border: '1px solid var(--border)',
          }}
        >
          <Typography
            variant="caption"
            sx={{
              display: 'block',
              fontWeight: 700,
              color: 'var(--text)',
              fontSize: '0.78rem',
              mb: 0.5,
              letterSpacing: '0.02em',
            }}
          >
            HF Pro ($9/mo) — more Opus, more everything
          </Typography>
          <Typography
            variant="caption"
            sx={{ display: 'block', color: 'var(--muted-text)', fontSize: '0.78rem', lineHeight: 1.55 }}
          >
            {PRO_CAP} Opus sessions/day here, 20× HF Inference credits, ZeroGPU access,
            and priority on Spaces hardware.
          </Typography>
        </Box>
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2.5, pt: 2, gap: 1 }}>
        <Button
          component="a"
          href={HF_PRICING_URL}
          target="_blank"
          rel="noopener noreferrer"
          onClick={onUpgrade}
          variant="contained"
          size="small"
          sx={{
            fontSize: '0.82rem',
            px: 2.5,
            bgcolor: 'var(--accent-yellow)',
            color: '#000',
            textTransform: 'none',
            fontWeight: 700,
            boxShadow: 'none',
            '&:hover': { bgcolor: '#FFB340', boxShadow: 'none' },
          }}
        >
          Upgrade to Pro
        </Button>
        <Button
          onClick={onUseFreeModel}
          size="small"
          sx={{
            color: 'var(--muted-text)',
            fontSize: '0.82rem',
            px: 2,
            textTransform: 'none',
            '&:hover': { bgcolor: 'var(--hover-bg)' },
          }}
        >
          Use a free model
        </Button>
      </DialogActions>
    </Dialog>
  );
}
