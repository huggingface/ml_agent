import { useCallback, useEffect, useRef, useMemo } from 'react';
import { Box, Stack, Typography } from '@mui/material';
import MessageBubble from './MessageBubble';
import ActivityStatusBar from './ActivityStatusBar';
import { useAgentStore } from '@/store/agentStore';
import type { UIMessage } from 'ai';

interface MessageListProps {
  messages: UIMessage[];
  isProcessing: boolean;
  sessionId?: string | null;
  approveTools: (approvals: Array<{ tool_call_id: string; approved: boolean; feedback?: string | null }>) => Promise<boolean>;
  onUndoLastTurn: () => void | Promise<void>;
  onEditAndRegenerate?: (messageId: string, newText: string) => void | Promise<void>;
}

function getGreeting(): string {
  const h = new Date().getHours();
  if (h < 12) return 'Morning';
  if (h < 17) return 'Afternoon';
  return 'Evening';
}

function WelcomeGreeting() {
  const { user } = useAgentStore();
  const firstName = user?.name?.split(' ')[0] || user?.username;
  const greeting = firstName ? `${getGreeting()}, ${firstName}` : getGreeting();

  return (
    <Box
      sx={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        py: 8,
        gap: 1.5,
      }}
    >
      <Typography
        sx={{
          fontFamily: 'monospace',
          fontSize: '1.6rem',
          color: 'var(--text)',
          fontWeight: 600,
        }}
      >
        {greeting}
      </Typography>
      <Typography
        color="text.secondary"
        sx={{ fontFamily: 'monospace', fontSize: '0.9rem' }}
      >
        Let's build something impressive?
      </Typography>
    </Box>
  );
}

export default function MessageList({ messages, isProcessing, sessionId, approveTools, onUndoLastTurn, onEditAndRegenerate }: MessageListProps) {
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);

  const scrollToBottom = useCallback(() => {
    const el = scrollContainerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, []);

  useEffect(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const onScroll = () => {
      const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
      stickToBottom.current = distFromBottom < 80;
    };
    el.addEventListener('scroll', onScroll, { passive: true });
    return () => el.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => {
    if (stickToBottom.current) scrollToBottom();
  }, [messages, isProcessing, scrollToBottom]);

  useEffect(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const observer = new MutationObserver(() => {
      if (stickToBottom.current) el.scrollTop = el.scrollHeight;
    });
    observer.observe(el, { childList: true, subtree: true, characterData: true });
    return () => observer.disconnect();
  }, []);

  const lastUserMsgId = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') return messages[i].id;
    }
    return null;
  }, [messages]);

  // The last assistant message is "streaming" when we're processing
  const lastAssistantId = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'assistant') return messages[i].id;
    }
    return null;
  }, [messages]);

  return (
    <Box
      ref={scrollContainerRef}
      sx={{
        flex: 1,
        overflow: 'auto',
        px: { xs: 0.5, sm: 1, md: 2 },
        py: { xs: 2, md: 3 },
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <Stack
        spacing={3}
        sx={{
          maxWidth: 880,
          mx: 'auto',
          width: '100%',
          flex: messages.length === 0 && !isProcessing ? 1 : undefined,
        }}
      >
        {messages.length === 0 && !isProcessing ? (
          <WelcomeGreeting />
        ) : (
          messages.map((msg) => (
            <MessageBubble
              key={msg.id}
              message={msg}
              isLastTurn={msg.id === lastUserMsgId}
              onUndoTurn={onUndoLastTurn}
              onEditAndRegenerate={onEditAndRegenerate}
              isProcessing={isProcessing}
              isStreaming={isProcessing && msg.id === lastAssistantId}
              sessionId={sessionId}
              approveTools={approveTools}
            />
          ))
        )}

        <ActivityStatusBar />

        <div />
      </Stack>
    </Box>
  );
}
