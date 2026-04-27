/**
 * Per-session chat component.
 *
 * Each session renders its own SessionChat. The hook (useAgentChat) always
 * runs — processing events — but only the active session renders visible
 * UI (MessageList + ChatInput).
 */
import { useCallback, useEffect } from 'react';
import { useAgentChat } from '@/hooks/useAgentChat';
import { useAgentStore } from '@/store/agentStore';
import { useSessionStore } from '@/store/sessionStore';
import MessageList from '@/components/Chat/MessageList';
import ChatInput from '@/components/Chat/ChatInput';
import ExpiredBanner from '@/components/Chat/ExpiredBanner';
import { apiFetch } from '@/utils/api';
import { logger } from '@/utils/logger';

interface SessionChatProps {
  sessionId: string;
  isActive: boolean;
  onSessionDead: (sessionId: string) => void;
}

export default function SessionChat({ sessionId, isActive, onSessionDead }: SessionChatProps) {
  const { isConnected, isProcessing, activityStatus, updateSession } = useAgentStore();
  const { updateSessionTitle, sessions } = useSessionStore();
  const isExpired = sessions.find((s) => s.id === sessionId)?.expired === true;

  const { messages, sendMessage, stop, status, undoLastTurn, editAndRegenerate, approveTools, declineBlockedJobs, continueBlockedJobsWithNamespace } = useAgentChat({
    sessionId,
    isActive,
    onReady: () => logger.log(`Session ${sessionId} ready`),
    onError: (error) => logger.error(`Session ${sessionId} error:`, error),
    onSessionDead,
  });

  // When this session becomes active, restore its per-session state to the
  // global flat fields. The per-session state map is kept up-to-date by
  // side-channel callbacks even while the session is in the background.
  useEffect(() => {
    if (isActive) {
      useAgentStore.getState().switchActiveSession(sessionId);
      useAgentStore.getState().setConnected(true);
    }
  }, [isActive, sessionId]);

  // Re-sync state when the browser tab regains focus (Chrome throttles
  // timers in background tabs which can stall the AI SDK's update flushing).
  // Fires for ALL sessions so background sessions also recover after sleep.
  useEffect(() => {
    const onVisible = () => {
      if (document.visibilityState === 'visible' && isActive) {
        useAgentStore.getState().switchActiveSession(sessionId);
      }
    };
    document.addEventListener('visibilitychange', onVisible);
    return () => document.removeEventListener('visibilitychange', onVisible);
  }, [isActive, sessionId]);

  // Wrap stop to show cancelled shimmer
  const handleStop = useCallback(() => {
    stop();
    updateSession(sessionId, { activityStatus: { type: 'cancelled' } });
  }, [stop, updateSession, sessionId]);

  // SDK status is the ground truth — if it's streaming/submitted, agent is busy
  const sdkBusy = status === 'streaming' || status === 'submitted';
  const busy = isProcessing || sdkBusy;

  const handleSendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || busy) return;

      updateSession(sessionId, { isProcessing: true, activityStatus: { type: 'thinking' } });
      sendMessage({ text: text.trim(), metadata: { createdAt: new Date().toISOString() } });

      // Auto-title the session from the first user message
      const isFirstMessage = messages.filter((m) => m.role === 'user').length === 0;
      if (isFirstMessage) {
        apiFetch('/api/title', {
          method: 'POST',
          body: JSON.stringify({ session_id: sessionId, text: text.trim() }),
        })
          .then((res) => res.json())
          .then((data) => {
            if (data.title) updateSessionTitle(sessionId, data.title);
          })
          .catch(() => {
            const raw = text.trim();
            updateSessionTitle(sessionId, raw.length > 40 ? raw.slice(0, 40) + '\u2026' : raw);
          });
      }
    },
    [sessionId, sendMessage, messages, updateSessionTitle, busy, updateSession],
  );

  // Don't render UI for background sessions — hooks still run
  if (!isActive) return null;

  return (
    <>
      <MessageList
        messages={messages}
        isProcessing={busy}
        sessionId={sessionId}
        approveTools={approveTools}
        onUndoLastTurn={undoLastTurn}
        onEditAndRegenerate={editAndRegenerate}
      />
      {isExpired ? (
        <ExpiredBanner sessionId={sessionId} />
      ) : (
        <ChatInput
          sessionId={sessionId}
          onSend={handleSendMessage}
          onStop={handleStop}
          onDeclineBlockedJobs={declineBlockedJobs}
          onContinueBlockedJobsWithNamespace={continueBlockedJobsWithNamespace}
          isProcessing={busy}
          disabled={!isConnected || activityStatus.type === 'waiting-approval'}
          placeholder={
            activityStatus.type === 'waiting-approval'
              ? 'Approve or reject pending tools first...'
              : undefined
          }
        />
      )}
    </>
  );
}
