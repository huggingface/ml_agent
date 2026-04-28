import {
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Typography,
} from '@mui/material';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import CreditCardIcon from '@mui/icons-material/CreditCard';
import ReplayIcon from '@mui/icons-material/Replay';

const HF_BILLING_URL = 'https://huggingface.co/settings/billing';
const HF_ORANGE = '#FF9D00';

interface JobsUpgradeDialogProps {
  open: boolean;
  message: string;
  /** True after the user clicked "Add credits" — switches the dialog into retry mode. */
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
  return (
    <Dialog
      open={open}
      onClose={onClose}
      slotProps={{
        backdrop: {
          sx: { backgroundColor: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(6px)' },
        },
      }}
      PaperProps={{
        sx: {
          bgcolor: 'var(--panel)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-md)',
          boxShadow: '0 30px 80px rgba(0,0,0,0.45), var(--shadow-1)',
          maxWidth: 460,
          width: '100%',
          mx: 2,
          overflow: 'hidden',
        },
      }}
    >
      <Box
        sx={{
          height: 4,
          background: `linear-gradient(90deg, ${HF_ORANGE} 0%, #FFC560 50%, ${HF_ORANGE} 100%)`,
        }}
      />

      <DialogTitle
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1.25,
          color: 'var(--text)',
          fontWeight: 800,
          fontSize: '1.05rem',
          pt: 2.5,
          pb: 0.5,
          px: 3,
          letterSpacing: '-0.01em',
        }}
      >
        <Box
          sx={{
            width: 32,
            height: 32,
            borderRadius: '10px',
            bgcolor: 'rgba(255, 157, 0, 0.15)',
            color: HF_ORANGE,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <CreditCardIcon sx={{ fontSize: 18 }} />
        </Box>
        {awaitingTopUp ? 'Topped up?' : 'Top up to launch'}
      </DialogTitle>
      <DialogContent sx={{ px: 3, pt: 1.25, pb: 0 }}>
        <Typography
          sx={{
            color: 'var(--muted-text)',
            fontSize: '0.85rem',
            lineHeight: 1.6,
            mb: 1.5,
          }}
        >
          {awaitingTopUp
            ? "We'll auto-retry the job as soon as you switch back from the billing tab. Or hit the button below now."
            : message ||
              'Hugging Face Jobs need credits on the namespace running them. Add some, then re-run the same job — the agent will pick it back up.'}
        </Typography>
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2.5, pt: 2.5, gap: 1 }}>
        {awaitingTopUp ? (
          <Button
            onClick={onRetry}
            startIcon={<ReplayIcon sx={{ fontSize: 16 }} />}
            variant="contained"
            size="small"
            sx={{
              fontSize: '0.82rem',
              px: 2.5,
              bgcolor: HF_ORANGE,
              color: '#000',
              textTransform: 'none',
              fontWeight: 700,
              boxShadow: '0 6px 18px rgba(255, 157, 0, 0.35)',
              '&:hover': { bgcolor: '#FFB340', boxShadow: '0 8px 22px rgba(255, 157, 0, 0.45)' },
            }}
          >
            Retry now
          </Button>
        ) : (
          <Button
            component="a"
            href={HF_BILLING_URL}
            target="_blank"
            rel="noopener noreferrer"
            onClick={onUpgrade}
            startIcon={<OpenInNewIcon sx={{ fontSize: 16 }} />}
            variant="contained"
            size="small"
            sx={{
              fontSize: '0.82rem',
              px: 2.5,
              bgcolor: HF_ORANGE,
              color: '#000',
              textTransform: 'none',
              fontWeight: 700,
              boxShadow: '0 6px 18px rgba(255, 157, 0, 0.35)',
              '&:hover': { bgcolor: '#FFB340', boxShadow: '0 8px 22px rgba(255, 157, 0, 0.45)' },
            }}
          >
            Add credits
          </Button>
        )}
        <Button
          onClick={onClose}
          size="small"
          sx={{
            color: 'var(--muted-text)',
            fontSize: '0.82rem',
            px: 2,
            textTransform: 'none',
            '&:hover': { bgcolor: 'var(--hover-bg)' },
          }}
        >
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
}
