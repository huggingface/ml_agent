import { Box, Typography } from '@mui/material';
import { keyframes } from '@mui/system';
import { useAgentStore, type ActivityStatus } from '@/store/agentStore';

const shimmer = keyframes`
  0% { background-position: -100% center; }
  50% { background-position: 200% center; }
  100% { background-position: -100% center; }
`;

const TOOL_LABELS: Record<string, string> = {
  sandbox_create: 'Creating sandbox for code development, this might take 1-2 minutes',
  bash: 'Running command in sandbox',
  hf_jobs: 'Running a GPU job, this might take a while',
  hf_repo_files: 'Uploading file',
  hf_repo_git: 'Git operation',
  hf_inspect_dataset: 'Inspecting dataset',
  hf_search: 'Searching',
  plan_tool: 'Planning',
  research: 'Researching',
};

function statusLabel(status: ActivityStatus): string {
  switch (status.type) {
    case 'thinking': return 'Thinking';
    case 'streaming': return 'Writing';
    case 'tool': {
      const base = status.description || TOOL_LABELS[status.toolName] || `Running ${status.toolName}`;
      if (status.toolName === 'bash' && status.description && /install/i.test(status.description)) {
        return `${base} — this can take a few minutes, sit tight`;
      }
      return base;
    }
    case 'waiting-approval': return 'Waiting for approval';
    default: return '';
  }
}

export default function ActivityStatusBar() {
  const activityStatus = useAgentStore(s => s.activityStatus);

  if (activityStatus.type === 'idle') return null;

  const label = statusLabel(activityStatus);

  return (
    <Box sx={{ px: 2, py: 0.5, minHeight: 28, display: 'flex', alignItems: 'center' }}>
      <Typography
        sx={{
          fontFamily: 'monospace',
          fontSize: '0.72rem',
          fontWeight: 500,
          letterSpacing: '0.02em',
          background: 'linear-gradient(90deg, var(--muted-text) 30%, var(--text) 50%, var(--muted-text) 70%)',
          backgroundSize: '250% 100%',
          backgroundClip: 'text',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          animation: `${shimmer} 4s ease-in-out infinite`,
        }}
      >
        {label}…
      </Typography>
    </Box>
  );
}
