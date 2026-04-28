import { useEffect, useMemo, useState } from 'react';
import {
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Typography,
} from '@mui/material';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import GroupsIcon from '@mui/icons-material/Groups';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import CreditCardIcon from '@mui/icons-material/CreditCard';
import { useAgentStore } from '@/store/agentStore';

const HF_BILLING_URL = 'https://huggingface.co/settings/billing';
const HF_ORANGE = '#FF9D00';

export type JobsDialogMode = 'namespace' | 'billing';

interface JobsUpgradeDialogProps {
  open: boolean;
  mode: JobsDialogMode;
  message: string;
  eligibleNamespaces: string[];
  onUpgrade: () => void;
  onDecline: () => void;
  onClose: () => void;
  onContinueWithNamespace: (namespace: string) => void;
}

interface NamespaceOption {
  name: string;
  isPersonal: boolean;
}

function NamespaceTile({
  option,
  selected,
  onSelect,
}: {
  option: NamespaceOption;
  selected: boolean;
  onSelect: () => void;
}) {
  const Icon = option.isPersonal ? AccountCircleIcon : GroupsIcon;
  const avatarUrl = `https://huggingface.co/api/avatars/${option.name}`;

  return (
    <Box
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onSelect();
        }
      }}
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 1.5,
        px: 1.75,
        py: 1.25,
        borderRadius: '10px',
        cursor: 'pointer',
        border: '1px solid',
        borderColor: selected ? HF_ORANGE : 'var(--border)',
        bgcolor: selected ? 'rgba(255, 157, 0, 0.08)' : 'transparent',
        boxShadow: selected ? '0 0 0 3px rgba(255, 157, 0, 0.15)' : 'none',
        transition: 'border-color 0.15s, background 0.15s, box-shadow 0.15s',
        '&:hover': {
          borderColor: selected ? HF_ORANGE : 'var(--text)',
          bgcolor: selected ? 'rgba(255, 157, 0, 0.12)' : 'rgba(255,255,255,0.03)',
        },
      }}
    >
      <Box
        sx={{
          width: 32,
          height: 32,
          borderRadius: '8px',
          overflow: 'hidden',
          flexShrink: 0,
          bgcolor: 'rgba(255,255,255,0.04)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Box
          component="img"
          src={avatarUrl}
          alt={option.name}
          onError={(e) => {
            (e.currentTarget as HTMLImageElement).style.display = 'none';
          }}
          sx={{ width: '100%', height: '100%', objectFit: 'cover' }}
        />
        <Icon sx={{ fontSize: 18, color: 'var(--muted-text)', position: 'absolute' }} />
      </Box>
      <Box sx={{ flex: 1, minWidth: 0 }}>
        <Typography
          sx={{
            fontSize: '0.88rem',
            fontWeight: 700,
            color: 'var(--text)',
            lineHeight: 1.3,
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
        >
          {option.name}
        </Typography>
        <Typography
          sx={{
            fontSize: '0.72rem',
            color: 'var(--muted-text)',
            mt: 0.1,
            letterSpacing: '0.02em',
          }}
        >
          {option.isPersonal ? 'Personal account' : 'Organization'}
        </Typography>
      </Box>
      <Box
        sx={{
          width: 16,
          height: 16,
          borderRadius: '50%',
          border: '2px solid',
          borderColor: selected ? HF_ORANGE : 'var(--border)',
          bgcolor: selected ? HF_ORANGE : 'transparent',
          flexShrink: 0,
          transition: 'all 0.15s',
        }}
      />
    </Box>
  );
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
  const user = useAgentStore((s) => s.user);
  const [selectedNamespace, setSelectedNamespace] = useState(
    () => eligibleNamespaces[0] || '',
  );

  useEffect(() => {
    if (!open) return;
    setSelectedNamespace((prev) =>
      eligibleNamespaces.includes(prev) ? prev : eligibleNamespaces[0] || '',
    );
  }, [open, eligibleNamespaces]);

  const options = useMemo<NamespaceOption[]>(() => {
    const username = user?.username;
    return eligibleNamespaces.map((name) => ({
      name,
      isPersonal: !!username && name === username,
    }));
  }, [eligibleNamespaces, user?.username]);

  const isBilling = mode === 'billing';

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
          maxWidth: 480,
          width: '100%',
          mx: 2,
          overflow: 'hidden',
        },
      }}
    >
      {/* Top accent strip — adds personality without being loud */}
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
          {isBilling ? (
            <CreditCardIcon sx={{ fontSize: 18 }} />
          ) : (
            <RocketLaunchIcon sx={{ fontSize: 18 }} />
          )}
        </Box>
        {isBilling ? 'Top up to launch' : 'Pick the wallet for this run'}
      </DialogTitle>
      <DialogContent sx={{ px: 3, pt: 1.25, pb: 0 }}>
        <Typography
          sx={{
            color: 'var(--muted-text)',
            fontSize: '0.85rem',
            lineHeight: 1.6,
            mb: isBilling ? 1.5 : 2,
          }}
        >
          {isBilling
            ? message ||
              'Hugging Face Jobs need credits on the namespace running them. Add some, then re-run the same job — the agent will pick it back up.'
            : message ||
              "Hugging Face Jobs are billed against the namespace they run under. Pick where this run should be charged — we'll remember your choice for the rest of this browser."}
        </Typography>

        {!isBilling && options.length > 0 && (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, mb: 1.5 }}>
            {options.map((option) => (
              <NamespaceTile
                key={option.name}
                option={option}
                selected={selectedNamespace === option.name}
                onSelect={() => setSelectedNamespace(option.name)}
              />
            ))}
          </Box>
        )}

        {!isBilling && (
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.75,
              fontSize: '0.78rem',
              color: 'var(--muted-text)',
              mb: 1,
            }}
          >
            <CreditCardIcon sx={{ fontSize: 14 }} />
            <Typography component="span" sx={{ fontSize: 'inherit', color: 'inherit' }}>
              No credits on this namespace yet?{' '}
            </Typography>
            <Typography
              component="a"
              href={HF_BILLING_URL}
              target="_blank"
              rel="noopener noreferrer"
              sx={{
                fontSize: 'inherit',
                color: HF_ORANGE,
                fontWeight: 700,
                textDecoration: 'none',
                '&:hover': { textDecoration: 'underline' },
              }}
            >
              Manage billing
            </Typography>
          </Box>
        )}
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2.5, pt: 2.5, gap: 1 }}>
        {isBilling ? (
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
        ) : (
          <Button
            onClick={() => onContinueWithNamespace(selectedNamespace)}
            disabled={!selectedNamespace}
            startIcon={<RocketLaunchIcon sx={{ fontSize: 16 }} />}
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
              '&.Mui-disabled': {
                bgcolor: 'rgba(255,255,255,0.06)',
                color: 'var(--muted-text)',
                boxShadow: 'none',
              },
            }}
          >
            Launch the job
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
          Skip this tool call
        </Button>
      </DialogActions>
    </Dialog>
  );
}
