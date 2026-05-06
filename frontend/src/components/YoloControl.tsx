import { useEffect, useMemo, useState } from 'react';
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import BoltOutlinedIcon from '@mui/icons-material/BoltOutlined';
import { useSessionStore } from '@/store/sessionStore';
import { apiFetch } from '@/utils/api';

const DEFAULT_CAP_USD = 5;

function money(value: number | null | undefined): string {
  if (value === null || value === undefined) return 'uncapped';
  if (value >= 100) return `$${value.toFixed(0)}`;
  return `$${value.toFixed(2).replace(/\.00$/, '')}`;
}

export default function YoloControl() {
  const { sessions, activeSessionId, updateSessionYolo } = useSessionStore();
  const activeSession = useMemo(
    () => sessions.find((s) => s.id === activeSessionId) || null,
    [sessions, activeSessionId],
  );
  const [dialogOpen, setDialogOpen] = useState(false);
  const [capInput, setCapInput] = useState(String(DEFAULT_CAP_USD));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const enabled = Boolean(activeSession?.autoApprovalEnabled);
  const disabled = !activeSessionId || activeSession?.expired || busy;
  const remaining = activeSession?.autoApprovalRemainingUsd ?? null;
  const cap = activeSession?.autoApprovalCostCapUsd ?? null;

  useEffect(() => {
    if (!activeSession) return;
    setCapInput(String(activeSession.autoApprovalCostCapUsd ?? DEFAULT_CAP_USD));
  }, [activeSession?.id, activeSession?.autoApprovalCostCapUsd]); // eslint-disable-line react-hooks/exhaustive-deps

  async function patchPolicy(nextEnabled: boolean, nextCap?: number) {
    if (!activeSessionId) return null;
    setBusy(true);
    setError(null);
    try {
      const body: Record<string, unknown> = { enabled: nextEnabled };
      if (nextCap !== undefined) body.cost_cap_usd = nextCap;
      const response = await apiFetch(`/api/session/${activeSessionId}/yolo`, {
        method: 'PATCH',
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const data = await response.json();
      updateSessionYolo(activeSessionId, data);
      return data;
    } catch {
      setError('Could not update YOLO settings.');
      return null;
    } finally {
      setBusy(false);
    }
  }

  const handleToggle = async () => {
    if (disabled) return;
    if (enabled) {
      await patchPolicy(false);
      return;
    }
    const nextCap = cap ?? DEFAULT_CAP_USD;
    const updated = await patchPolicy(true, nextCap);
    if (updated) {
      setCapInput(String(updated.cost_cap_usd ?? nextCap));
      setDialogOpen(true);
    }
  };

  const handleSaveCap = async () => {
    const parsed = Number(capInput);
    if (!Number.isFinite(parsed) || parsed < 0) {
      setError('Enter a non-negative dollar amount.');
      return;
    }
    const updated = await patchPolicy(true, parsed);
    if (updated) setDialogOpen(false);
  };

  return (
    <>
      <Tooltip title={enabled ? 'Disable session YOLO auto-approval' : 'Enable session YOLO auto-approval'}>
        <span>
          <Button
            size="small"
            variant={enabled ? 'contained' : 'outlined'}
            disabled={disabled}
            onClick={handleToggle}
            startIcon={<BoltOutlinedIcon sx={{ fontSize: 16 }} />}
            sx={{
              minWidth: { xs: 74, md: 116 },
              height: 32,
              px: { xs: 1, md: 1.25 },
              borderRadius: '8px',
              textTransform: 'none',
              fontSize: '0.72rem',
              whiteSpace: 'nowrap',
              bgcolor: enabled ? 'var(--accent-yellow)' : 'transparent',
              color: enabled ? '#111' : 'text.secondary',
              borderColor: enabled ? 'var(--accent-yellow)' : 'divider',
              '&:hover': {
                bgcolor: enabled ? 'var(--accent-yellow)' : 'action.hover',
                borderColor: 'var(--accent-yellow)',
              },
            }}
          >
            {enabled ? `YOLO ${money(remaining)}` : 'YOLO'}
          </Button>
        </span>
      </Tooltip>

      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle sx={{ pb: 1 }}>YOLO Budget</DialogTitle>
        <DialogContent sx={{ display: 'flex', flexDirection: 'column', gap: 1.5, pt: 1 }}>
          <Typography variant="body2" color="text.secondary">
            Auto-approval is active for this session. Scheduled HF jobs still require approval.
          </Typography>
          <TextField
            autoFocus
            label="Session cap (USD)"
            type="number"
            size="small"
            value={capInput}
            onChange={(e) => setCapInput(e.target.value)}
            inputProps={{ min: 0, step: 0.5 }}
            error={Boolean(error)}
            helperText={error || `Estimated spend: ${money(activeSession?.autoApprovalEstimatedSpendUsd ?? 0)} of ${money(cap)}`}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)} sx={{ textTransform: 'none' }}>
            Close
          </Button>
          <Button onClick={handleSaveCap} disabled={busy} variant="contained" sx={{ textTransform: 'none' }}>
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}
