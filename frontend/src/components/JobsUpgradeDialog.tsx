import { useEffect, useState } from 'react';
import {
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  FormControl,
  MenuItem,
  Select,
  Typography,
} from '@mui/material';

const HF_PRICING_URL = 'https://huggingface.co/pricing';

interface JobsUpgradeDialogProps {
  open: boolean;
  mode: 'upgrade' | 'namespace';
  message: string;
  eligibleNamespaces: string[];
  onUpgrade: () => void;
  onDecline: () => void;
  onClose: () => void;
  onContinueWithNamespace: (namespace: string) => void;
}

export default function JobsUpgradeDialog({
  open,
  mode,
  message,
  eligibleNamespaces,
  onUpgrade,
  onDecline,
  onClose,
  onContinueWithNamespace,
}: JobsUpgradeDialogProps) {
  const [selectedNamespace, setSelectedNamespace] = useState(() => eligibleNamespaces[0] || '');

  useEffect(() => {
    if (!open) return;
    setSelectedNamespace(eligibleNamespaces[0] || '');
  }, [open, eligibleNamespaces]);

  const isNamespace = mode === 'namespace';
  const title = isNamespace ? 'Run jobs as' : 'Jobs need Pro or a paid org';

  const body = isNamespace
    ? "Pick which paid organization should pay for and own this job. We'll use the same one for the rest of this browser."
    : message;

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
        {title}
      </DialogTitle>
      <DialogContent sx={{ px: 3, pt: 1.25, pb: 0 }}>
        <DialogContentText
          sx={{ color: 'var(--muted-text)', fontSize: '0.85rem', lineHeight: 1.6 }}
        >
          {body}
        </DialogContentText>

        {isNamespace ? (
          <FormControl fullWidth size="small" sx={{ mt: 2 }}>
            <Select
              value={selectedNamespace}
              displayEmpty
              onChange={(e) => setSelectedNamespace(String(e.target.value))}
              sx={{
                bgcolor: 'var(--composer-bg)',
                color: 'var(--text)',
                fontSize: '0.88rem',
                fontWeight: 600,
                '& .MuiOutlinedInput-notchedOutline': { borderColor: 'var(--border)' },
                '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: 'var(--border)' },
                '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                  borderColor: 'var(--accent-yellow)',
                  borderWidth: 1,
                },
                '& .MuiSelect-icon': { color: 'var(--muted-text)' },
              }}
              MenuProps={{
                PaperProps: {
                  sx: {
                    bgcolor: 'var(--panel)',
                    border: '1px solid var(--border)',
                    borderRadius: '8px',
                    mt: 0.5,
                  },
                },
              }}
            >
              {eligibleNamespaces.map((namespace) => (
                <MenuItem
                  key={namespace}
                  value={namespace}
                  sx={{
                    fontSize: '0.88rem',
                    color: 'var(--text)',
                    '&.Mui-selected': { bgcolor: 'rgba(255,255,255,0.05)' },
                  }}
                >
                  {namespace}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        ) : (
          eligibleNamespaces.length > 0 && (
            <Box sx={{ mt: 1.5 }}>
              <Typography
                variant="caption"
                sx={{ color: 'var(--muted-text)', fontSize: '0.78rem', lineHeight: 1.55 }}
              >
                Eligible namespaces: {eligibleNamespaces.join(', ')}
              </Typography>
            </Box>
          )
        )}
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2.5, pt: 2.5, gap: 1 }}>
        {isNamespace ? (
          <Button
            onClick={() => onContinueWithNamespace(selectedNamespace)}
            disabled={!selectedNamespace}
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
            Continue
          </Button>
        ) : (
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
        )}
        <Button
          onClick={onDecline}
          size="small"
          sx={{
            color: 'var(--muted-text)',
            fontSize: '0.82rem',
            px: 2,
            textTransform: 'none',
            '&:hover': { bgcolor: 'var(--hover-bg)' },
          }}
        >
          {isNamespace ? 'Skip this tool call' : 'Decline tool call'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
