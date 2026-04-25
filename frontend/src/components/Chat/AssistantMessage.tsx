import { useMemo, useState } from 'react';
import { Box, IconButton, Stack, Tooltip, Typography } from '@mui/material';
import ThumbUpOutlined from '@mui/icons-material/ThumbUpOutlined';
import ThumbUp from '@mui/icons-material/ThumbUp';
import ThumbDownOutlined from '@mui/icons-material/ThumbDownOutlined';
import ThumbDown from '@mui/icons-material/ThumbDown';
import MarkdownContent from './MarkdownContent';
import ToolCallGroup from './ToolCallGroup';
import { apiFetch } from '@/utils/api';
import type { UIMessage } from 'ai';
import type { MessageMeta } from '@/types/agent';

interface AssistantMessageProps {
  message: UIMessage;
  isStreaming?: boolean;
  sessionId?: string | null;
  approveTools: (approvals: Array<{ tool_call_id: string; approved: boolean; feedback?: string | null }>) => Promise<boolean>;
}

/**
 * Groups consecutive tool parts together so they render as a single
 * ToolCallGroup (visually identical to the old segments approach).
 */
type DynamicToolPart = Extract<UIMessage['parts'][number], { type: 'dynamic-tool' }>;

function groupParts(parts: UIMessage['parts']) {
  const groups: Array<
    | { kind: 'text'; text: string; idx: number }
    | { kind: 'tools'; tools: DynamicToolPart[]; idx: number }
  > = [];

  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];

    if (part.type === 'text') {
      groups.push({ kind: 'text', text: part.text, idx: i });
    } else if (part.type === 'dynamic-tool') {
      const toolPart = part as DynamicToolPart;
      const last = groups[groups.length - 1];
      if (last?.kind === 'tools') {
        last.tools.push(toolPart);
      } else {
        groups.push({ kind: 'tools', tools: [toolPart], idx: i });
      }
    }
    // step-start, step-end, etc. are ignored visually
  }

  return groups;
}

export default function AssistantMessage({ message, isStreaming = false, sessionId, approveTools }: AssistantMessageProps) {
  const groups = useMemo(() => groupParts(message.parts), [message.parts]);
  const [feedback, setFeedback] = useState<'up' | 'down' | null>(null);
  const [feedbackBusy, setFeedbackBusy] = useState(false);

  const sendFeedback = async (rating: 'up' | 'down') => {
    if (!sessionId || feedbackBusy) return;
    setFeedbackBusy(true);
    // Optimistic toggle — feedback is observability, not a hard requirement.
    setFeedback(rating);
    try {
      await apiFetch(`/api/feedback/${sessionId}`, {
        method: 'POST',
        body: JSON.stringify({ rating, message_id: message.id }),
      });
    } catch {
      // Silently swallow — don't block chat UX on a telemetry write.
    } finally {
      setFeedbackBusy(false);
    }
  };

  // Find the last text group index for streaming cursor
  let lastTextIdx = -1;
  for (let i = groups.length - 1; i >= 0; i--) {
    if (groups[i].kind === 'text') { lastTextIdx = i; break; }
  }

  const meta = message.metadata as MessageMeta | undefined;
  const timeStr = meta?.createdAt
    ? new Date(meta.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : null;

  if (groups.length === 0) return null;

  return (
    <Box sx={{ minWidth: 0 }}>
      <Stack direction="row" alignItems="baseline" spacing={1} sx={{ mb: 0.5 }}>
        <Typography
          variant="caption"
          sx={{
            fontWeight: 700,
            fontSize: '0.72rem',
            color: 'var(--muted-text)',
            textTransform: 'uppercase',
            letterSpacing: '0.04em',
          }}
        >
          Assistant
        </Typography>
        {timeStr && (
          <Typography variant="caption" sx={{ color: 'var(--muted-text)', fontSize: '0.7rem' }}>
            {timeStr}
          </Typography>
        )}
      </Stack>

      <Box
        sx={{
          maxWidth: { xs: '95%', md: '85%' },
          bgcolor: 'var(--surface)',
          borderRadius: 1.5,
          borderTopLeftRadius: 4,
          px: { xs: 1.5, md: 2.5 },
          py: 1.5,
          border: '1px solid var(--border)',
        }}
      >
        {groups.map((group, i) => {
          if (group.kind === 'text' && group.text) {
            return (
              <MarkdownContent
                key={group.idx}
                content={group.text}
                isStreaming={isStreaming && i === lastTextIdx}
              />
            );
          }
          if (group.kind === 'tools' && group.tools.length > 0) {
            return (
              <ToolCallGroup
                key={group.idx}
                tools={group.tools}
                approveTools={approveTools}
              />
            );
          }
          return null;
        })}
      </Box>
      {!isStreaming && sessionId && (
        <Stack
          direction="row"
          spacing={0.5}
          sx={{ mt: 0.5, ml: 0.5, opacity: feedback ? 1 : 0.5, '&:hover': { opacity: 1 } }}
        >
          <Tooltip title="Helpful">
            <IconButton size="small" disabled={feedbackBusy} onClick={() => sendFeedback('up')}>
              {feedback === 'up' ? <ThumbUp fontSize="inherit" /> : <ThumbUpOutlined fontSize="inherit" />}
            </IconButton>
          </Tooltip>
          <Tooltip title="Not helpful">
            <IconButton size="small" disabled={feedbackBusy} onClick={() => sendFeedback('down')}>
              {feedback === 'down' ? <ThumbDown fontSize="inherit" /> : <ThumbDownOutlined fontSize="inherit" />}
            </IconButton>
          </Tooltip>
        </Stack>
      )}
    </Box>
  );
}
