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
  InputLabel,
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
  const [selectedNamespace, setSelectedNamespace] = useState('');

  useEffect(() => {
    if (!open) return;
    setSelectedNamespace(eligibleNamespaces[0] || '');
  }, [open, eligibleNamespaces]);

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
          maxWidth: 500,
          mx: 2,
        },
      }}
    >
      <DialogTitle
        sx={{ color: 'var(--text)', fontWeight: 700, fontSize: '1rem', pt: 2.5, pb: 0, px: 3 }}
      >
        {mode === 'namespace' ? 'Choose the org for this job' : 'Jobs need Pro or a paid org'}
      </DialogTitle>
      <DialogContent sx={{ px: 3, pt: 1.25, pb: 0 }}>
        <DialogContentText
          sx={{ color: 'var(--muted-text)', fontSize: '0.85rem', lineHeight: 1.6 }}
        >
          {message}
        </DialogContentText>
        {eligibleNamespaces.length > 0 && (
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
                mb: 1,
                letterSpacing: '0.02em',
              }}
            >
              Eligible namespaces
            </Typography>
            {mode === 'namespace' ? (
              <FormControl fullWidth size="small">
                <InputLabel id="jobs-namespace-label">Organization</InputLabel>
                <Select
                  labelId="jobs-namespace-label"
                  value={selectedNamespace}
                  label="Organization"
                  onChange={(e) => setSelectedNamespace(String(e.target.value))}
                >
                  {eligibleNamespaces.map((namespace) => (
                    <MenuItem key={namespace} value={namespace}>
                      {namespace}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            ) : (
              <Typography
                variant="caption"
                sx={{ display: 'block', color: 'var(--muted-text)', fontSize: '0.78rem', lineHeight: 1.55 }}
              >
                {eligibleNamespaces.join(', ')}
              </Typography>
            )}
          </Box>
        )}
        <Typography
          variant="caption"
          sx={{ display: 'block', mt: 2, color: 'var(--muted-text)', fontSize: '0.78rem', lineHeight: 1.55 }}
        >
          If you decline, the agent will have to find another way forward without `hf_jobs`.
        </Typography>
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2.5, pt: 2, gap: 1 }}>
        {mode === 'namespace' ? (
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
            Run under selected org
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
          Decline tool call
        </Button>
      </DialogActions>
    </Dialog>
  );
}
