/**
 * Central hook wiring the Vercel AI SDK's useChat with our SSE-based
 * ChatTransport.
 *
 * In the per-session architecture, each session mounts its own instance
 * of this hook. Side-channel callbacks always update the session's own
 * state via `updateSession()`. If the session is currently active, the
 * store automatically mirrors updates to the flat global fields.
 */
import { useCallback, useEffect, useMemo, useRef } from 'react';
import { useChat } from '@ai-sdk/react';
import { type UIMessage, lastAssistantMessageIsCompleteWithApprovalResponses } from 'ai';
import { SSEChatTransport, type SideChannelCallbacks } from '@/lib/sse-chat-transport';
import { loadMessages, saveMessages } from '@/lib/chat-message-store';
import { llmMessagesToUIMessages } from '@/lib/convert-llm-messages';
import { apiFetch } from '@/utils/api';
import { useAgentStore } from '@/store/agentStore';
import { useSessionStore } from '@/store/sessionStore';
import { useLayoutStore } from '@/store/layoutStore';
import { logger } from '@/utils/logger';

interface UseAgentChatOptions {
  sessionId: string;
  isActive: boolean;
  onReady?: () => void;
  onError?: (error: string) => void;
  onSessionDead?: (sessionId: string) => void;
}

export function useAgentChat({ sessionId, isActive, onReady, onError, onSessionDead }: UseAgentChatOptions) {
  const callbacksRef = useRef({ onReady, onError, onSessionDead });
  callbacksRef.current = { onReady, onError, onSessionDead };

  const isActiveRef = useRef(isActive);
  isActiveRef.current = isActive;

  const { setNeedsAttention } = useSessionStore();

  // Helper: update this session's state (mirrors to globals if active)
  const updateSession = useAgentStore.getState().updateSession;

  // -- Build side-channel callbacks (stable ref) --------------------------
  const sideChannel = useMemo<SideChannelCallbacks>(
    () => ({
      onReady: () => {
        updateSession(sessionId, { isProcessing: false });
        if (isActiveRef.current) {
          useAgentStore.getState().setConnected(true);
        }
        useSessionStore.getState().setSessionActive(sessionId, true);
        callbacksRef.current.onReady?.();
      },
      onShutdown: () => {
        updateSession(sessionId, { isProcessing: false });
        if (isActiveRef.current) {
          useAgentStore.getState().setConnected(false);
        }
      },
      onError: (error: string) => {
        updateSession(sessionId, { isProcessing: false });
        if (isActiveRef.current) {
          useAgentStore.getState().setError(error);
        }
        callbacksRef.current.onError?.(error);
      },
      onProcessing: () => {
        updateSession(sessionId, {
          isProcessing: true,
          activityStatus: { type: 'thinking' },
        });
      },
      onProcessingDone: () => {
        updateSession(sessionId, { isProcessing: false });
      },
      onUndoComplete: () => {
        updateSession(sessionId, { isProcessing: false });
      },
      onCompacted: (oldTokens: number, newTokens: number) => {
        logger.log(`Context compacted: ${oldTokens} -> ${newTokens} tokens`);
      },
      onPlanUpdate: (plan) => {
        const typed = plan as Array<{ id: string; content: string; status: 'pending' | 'in_progress' | 'completed' }>;
        updateSession(sessionId, { plan: typed });
        if (isActiveRef.current && !useLayoutStore.getState().isRightPanelOpen) {
          useLayoutStore.getState().setRightPanelOpen(true);
        }
      },
      onToolLog: (tool: string, log: string) => {
        const STREAMABLE_TOOLS = new Set(['hf_jobs', 'sandbox', 'bash']);
        if (!STREAMABLE_TOOLS.has(tool)) return;

        const sessState = useAgentStore.getState().getSessionState(sessionId);
        const existingOutput = sessState.panelData?.output?.content || '';

        const newContent = existingOutput
          ? existingOutput + '\n' + log
          : log;

        if (!sessState.panelData) {
          const title = tool === 'bash' ? 'Sandbox' : tool === 'sandbox' ? 'Sandbox' : 'Job Output';
          updateSession(sessionId, {
            panelData: { title, output: { content: newContent, language: 'text' } },
            panelView: 'output',
          });
        } else {
          updateSession(sessionId, {
            panelData: { ...sessState.panelData, output: { content: newContent, language: 'text' } },
            panelView: 'output',
          });
        }

        if (isActiveRef.current && !useLayoutStore.getState().isRightPanelOpen) {
          useLayoutStore.getState().setRightPanelOpen(true);
        }
      },
      onConnectionChange: (connected: boolean) => {
        if (isActiveRef.current) useAgentStore.getState().setConnected(connected);
      },
      onSessionDead: (deadSessionId: string) => {
        logger.warn(`Session ${deadSessionId} dead, removing`);
        callbacksRef.current.onSessionDead?.(deadSessionId);
      },
      onApprovalRequired: (tools) => {
        if (!tools.length) return;
        setNeedsAttention(sessionId, true);

        updateSession(sessionId, { activityStatus: { type: 'waiting-approval' } });

        // Build panel data for this session's pending approval
        const firstTool = tools[0];
        const args = firstTool.arguments as Record<string, string | undefined>;

        let panelUpdate: Partial<import('@/store/agentStore').PerSessionState> | undefined;
        if (firstTool.tool === 'hf_jobs' && args.script) {
          panelUpdate = {
            panelData: {
              title: 'Script',
              script: { content: args.script, language: 'python' },
              parameters: firstTool.arguments as Record<string, unknown>,
            },
            panelView: 'script' as const,
            panelEditable: true,
          };
        } else if (firstTool.tool === 'hf_repo_files' && args.content) {
          const filename = args.path || 'file';
          panelUpdate = {
            panelData: {
              title: filename.split('/').pop() || 'Content',
              script: { content: args.content, language: filename.endsWith('.py') ? 'python' : 'text' },
              parameters: firstTool.arguments as Record<string, unknown>,
            },
          };
        } else {
          panelUpdate = {
            panelData: {
              title: firstTool.tool,
              output: { content: JSON.stringify(firstTool.arguments, null, 2), language: 'json' },
            },
            panelView: 'output' as const,
          };
        }
        if (panelUpdate) updateSession(sessionId, panelUpdate);

        if (isActiveRef.current) {
          useLayoutStore.getState().setRightPanelOpen(true);
          useLayoutStore.getState().setLeftSidebarOpen(false);
        }
      },
      onToolCallPanel: (toolName: string, args: Record<string, unknown>) => {
        if (toolName === 'hf_jobs' && args.operation && args.script) {
          updateSession(sessionId, {
            panelData: {
              title: 'Script',
              script: { content: String(args.script), language: 'python' },
              parameters: args,
            },
            panelView: 'script',
          });
          if (isActiveRef.current) {
            useLayoutStore.getState().setRightPanelOpen(true);
            useLayoutStore.getState().setLeftSidebarOpen(false);
          }
        } else if (toolName === 'hf_repo_files' && args.operation === 'upload' && args.content) {
          updateSession(sessionId, {
            panelData: {
              title: `File Upload: ${String(args.path || 'unnamed')}`,
              script: { content: String(args.content), language: String(args.path || '').endsWith('.py') ? 'python' : 'text' },
              parameters: args,
            },
          });
          if (isActiveRef.current) {
            useLayoutStore.getState().setRightPanelOpen(true);
            useLayoutStore.getState().setLeftSidebarOpen(false);
          }
        } else if (toolName === 'bash' && args.command) {
          updateSession(sessionId, {
            panelData: {
              title: 'Sandbox',
              script: { content: String(args.command), language: 'bash' },
            },
            panelView: 'output',
          });
        }
      },
      onToolOutputPanel: (toolName: string, _toolCallId: string, output: string, success: boolean) => {
        const sessState = useAgentStore.getState().getSessionState(sessionId);
        if (toolName === 'hf_jobs' && output) {
          updateSession(sessionId, {
            panelData: sessState.panelData
              ? { ...sessState.panelData, output: { content: output, language: 'markdown' } }
              : { title: 'Output', output: { content: output, language: 'markdown' } },
            panelView: !success ? 'output' : sessState.panelView,
          });
        } else if (toolName === 'bash') {
          if (!success) {
            updateSession(sessionId, { panelView: 'output' });
          }
        }
      },
      onStreaming: () => {
        updateSession(sessionId, { activityStatus: { type: 'streaming' } });
      },
      onToolRunning: (toolName: string, description?: string) => {
        updateSession(sessionId, { activityStatus: { type: 'tool', toolName, description } });
      },
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [sessionId],
  );

  // -- Create transport (one per session, stable for lifetime) ------------
  const transportRef = useRef<SSEChatTransport | null>(null);
  if (!transportRef.current) {
    transportRef.current = new SSEChatTransport(sessionId, sideChannel);
  }

  // Keep side-channel callbacks in sync
  useEffect(() => {
    transportRef.current?.updateSideChannel(sideChannel);
  }, [sideChannel]);

  // Destroy transport on unmount
  useEffect(() => {
    return () => {
      transportRef.current?.destroy();
      transportRef.current = null;
    };
  }, []);

  // -- Restore persisted messages for this session ------------------------
  const initialMessages = useMemo(
    () => loadMessages(sessionId),
    [sessionId],
  );

  // -- Ref for chat actions (used by sideChannel callbacks) ---------------
  const chatActionsRef = useRef<{
    setMessages: ((msgs: UIMessage[]) => void) | null;
    messages: UIMessage[];
  }>({ setMessages: null, messages: [] });

  // -- useChat from Vercel AI SDK -----------------------------------------
  const chat = useChat({
    id: sessionId,
    messages: initialMessages,
    transport: transportRef.current!,
    experimental_throttle: 80,
    // On mount, the SDK calls transport.reconnectToStream() which checks
    // is_processing and subscribes to the live event stream if the agent
    // is mid-turn.  Without this, page refresh kills live updates.
    resume: true,
    // After all approval responses are set, auto-send to continue the agent loop.
    // Without this, addToolApprovalResponse only updates the UI — it won't trigger
    // sendMessages on the transport.
    sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithApprovalResponses,
    onError: (error) => {
      logger.error('useChat error:', error);
      updateSession(sessionId, { isProcessing: false });
      if (isActiveRef.current) {
        useAgentStore.getState().setError(error.message);
      }
    },
  });

  // Keep chatActionsRef in sync every render
  chatActionsRef.current.setMessages = chat.setMessages;
  chatActionsRef.current.messages = chat.messages;

  // -- Hydrate from backend on mount (page refresh recovery) --------------
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [msgsRes, infoRes] = await Promise.all([
          apiFetch(`/api/session/${sessionId}/messages`),
          apiFetch(`/api/session/${sessionId}`),
        ]);
        if (cancelled) return;

        let pendingIds: Set<string> | undefined;
        let backendIsProcessing = false;
        if (infoRes.ok) {
          const info = await infoRes.json();
          backendIsProcessing = !!info.is_processing;
          if (info.pending_approval && Array.isArray(info.pending_approval)) {
            pendingIds = new Set(
              info.pending_approval.map((t: { tool_call_id: string }) => t.tool_call_id)
            );
            if (pendingIds.size > 0) {
              setNeedsAttention(sessionId, true);
            }
          }
        }

        if (msgsRes.ok) {
          const data = await msgsRes.json();
          if (cancelled || !Array.isArray(data) || data.length === 0) return;
          const uiMsgs = llmMessagesToUIMessages(data, pendingIds);
          if (uiMsgs.length > 0) {
            chat.setMessages(uiMsgs);
            saveMessages(sessionId, uiMsgs);
          }
        }

        // Use the backend's is_processing flag as the source of truth.
        // Message-based inference doesn't work because completed tool
        // results make tools look "done" even when the agent is still
        // mid-turn and about to call more tools.
        if (backendIsProcessing) {
          updateSession(sessionId, { isProcessing: true, activityStatus: { type: 'thinking' } });
        } else if (pendingIds && pendingIds.size > 0) {
          updateSession(sessionId, { activityStatus: { type: 'waiting-approval' } });
        }
      } catch {
        /* backend unreachable -- localStorage fallback is fine */
      }
    })();
    return () => { cancelled = true; };
  }, [sessionId]); // eslint-disable-line react-hooks/exhaustive-deps

  // -- Re-hydrate + reconnect on wake from sleep ----------------------------
  // The Vercel AI SDK only calls reconnectToStream() on mount, NOT on
  // visibility change.  So when the browser wakes from sleep and the SSE
  // stream is dead, we must manually:
  //   1. Re-hydrate messages (one-shot fetch from backend)
  //   2. Subscribe to live events via GET /api/events/{sessionId}
  //   3. Pipe those events through the side-channel callbacks for real-time UI
  //   4. Poll messages every few seconds so chat.setMessages stays in sync
  const reconnectAbortRef = useRef<AbortController | null>(null);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    /** Fetch latest messages from backend and push into the SDK. */
    const hydrateMessages = async () => {
      try {
        const [msgsRes, infoRes] = await Promise.all([
          apiFetch(`/api/session/${sessionId}/messages`),
          apiFetch(`/api/session/${sessionId}`),
        ]);
        if (!msgsRes.ok) return null;
        const data = await msgsRes.json();
        if (!Array.isArray(data) || data.length === 0) return null;

        let pendingIds: Set<string> | undefined;
        if (infoRes.ok) {
          const info = await infoRes.json();
          if (info.pending_approval && Array.isArray(info.pending_approval)) {
            pendingIds = new Set(
              info.pending_approval.map((t: { tool_call_id: string }) => t.tool_call_id)
            );
            if (pendingIds.size > 0) setNeedsAttention(sessionId, true);
          }
          return { data, pendingIds, info };
        }
        return { data, pendingIds, info: null };
      } catch {
        return null;
      }
    };

    /** Stop any running reconnection (event stream + poll). */
    const stopReconnect = () => {
      reconnectAbortRef.current?.abort();
      reconnectAbortRef.current = null;
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };

    /** Read the event stream from GET /api/events and forward to side-channel. */
    const consumeEventStream = async (signal: AbortSignal) => {
      try {
        const res = await apiFetch(`/api/events/${sessionId}`, {
          headers: { 'Accept': 'text/event-stream' },
          signal,
        });
        if (!res.ok || !res.body) return;

        const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
        let buf = '';
        while (true) {
          const { value, done } = await reader.read();
          if (done || signal.aborted) break;
          buf += value;
          const lines = buf.split('\n');
          buf = lines.pop() || '';
          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith('data: ')) continue;
            try {
              const event = JSON.parse(trimmed.slice(6));
              // Forward to side-channel for real-time UI updates
              const et = event.event_type as string;
              if (et === 'processing') sideChannel.onProcessing();
              else if (et === 'assistant_chunk') sideChannel.onStreaming();
              else if (et === 'tool_call') {
                const t = event.data?.tool as string;
                const d = event.data?.arguments?.description as string | undefined;
                sideChannel.onToolRunning(t, d);
                sideChannel.onToolCallPanel(t, (event.data?.arguments || {}) as Record<string, unknown>);
              } else if (et === 'tool_output') {
                sideChannel.onToolOutputPanel(
                  event.data?.tool as string,
                  event.data?.tool_call_id as string,
                  event.data?.output as string,
                  event.data?.success as boolean,
                );
              } else if (et === 'tool_state_change') {
                const state = event.data?.state as string;
                const toolName = event.data?.tool as string;
                if (state === 'running' && toolName) sideChannel.onToolRunning(toolName);
              } else if (et === 'turn_complete' || et === 'error' || et === 'interrupted') {
                sideChannel.onProcessingDone();
                stopReconnect();
                // Final hydration to get the complete message state
                const result = await hydrateMessages();
                if (result) {
                  const uiMsgs = llmMessagesToUIMessages(result.data, result.pendingIds);
                  if (uiMsgs.length > 0) {
                    chat.setMessages(uiMsgs);
                    saveMessages(sessionId, uiMsgs);
                  }
                }
                return;
              } else if (et === 'approval_required') {
                sideChannel.onApprovalRequired(
                  (event.data?.tools || []) as Array<{ tool: string; arguments: Record<string, unknown>; tool_call_id: string }>,
                );
                stopReconnect();
                const result = await hydrateMessages();
                if (result) {
                  const uiMsgs = llmMessagesToUIMessages(result.data, result.pendingIds);
                  if (uiMsgs.length > 0) {
                    chat.setMessages(uiMsgs);
                    saveMessages(sessionId, uiMsgs);
                  }
                }
                return;
              }
            } catch { /* ignore parse errors */ }
          }
        }
      } catch {
        /* stream ended or aborted */
      }
    };

    const onVisible = async () => {
      if (document.visibilityState !== 'visible') return;

      // Always re-hydrate messages on wake
      const result = await hydrateMessages();
      if (!result) return;

      const { data, pendingIds, info } = result;
      const uiMsgs = llmMessagesToUIMessages(data, pendingIds);
      if (uiMsgs.length > 0) {
        chat.setMessages(uiMsgs);
        saveMessages(sessionId, uiMsgs);
      }

      // If the backend is still processing, reconnect to the live event stream
      if (info?.is_processing) {
        updateSession(sessionId, { isProcessing: true, activityStatus: { type: 'thinking' } });

        // Stop any previous reconnection
        stopReconnect();

        // Start live event subscription
        const abort = new AbortController();
        reconnectAbortRef.current = abort;
        consumeEventStream(abort.signal);

        // Poll messages every 3 s so the chat message list stays up-to-date
        // (the event stream gives us real-time status but not full message diffs)
        pollTimerRef.current = setInterval(async () => {
          const fresh = await hydrateMessages();
          if (!fresh) return;
          const msgs = llmMessagesToUIMessages(fresh.data, fresh.pendingIds);
          if (msgs.length > 0) {
            chat.setMessages(msgs);
            saveMessages(sessionId, msgs);
          }
          // If backend stopped processing, clean up
          if (fresh.info && !fresh.info.is_processing) {
            updateSession(sessionId, { isProcessing: false });
            stopReconnect();
          }
        }, 3000);
      }
    };

    document.addEventListener('visibilitychange', onVisible);
    return () => {
      document.removeEventListener('visibilitychange', onVisible);
      stopReconnect();
    };
  }, [sessionId]); // eslint-disable-line react-hooks/exhaustive-deps

  // -- Persist messages ---------------------------------------------------
  const prevLenRef = useRef(initialMessages.length);
  useEffect(() => {
    if (chat.messages.length === 0) return;
    if (chat.messages.length !== prevLenRef.current) {
      prevLenRef.current = chat.messages.length;
      saveMessages(sessionId, chat.messages);
    }
  }, [sessionId, chat.messages]);

  // -- Undo last turn (REST call + client-side message removal) -----------
  // With SSE there's no persistent connection to receive the undo_complete
  // event, so we handle message removal on the frontend after a successful
  // REST call to the backend.
  const undoLastTurn = useCallback(async () => {
    try {
      const res = await apiFetch(`/api/undo/${sessionId}`, { method: 'POST' });
      if (!res.ok) {
        logger.error('Undo API returned', res.status);
        return;
      }
      // Remove the last user turn + assistant response from the UI
      const msgs = chatActionsRef.current.messages;
      const setMsgs = chatActionsRef.current.setMessages;
      if (setMsgs && msgs.length > 0) {
        let lastUserIdx = -1;
        for (let i = msgs.length - 1; i >= 0; i--) {
          if (msgs[i].role === 'user') { lastUserIdx = i; break; }
        }
        const updated = lastUserIdx > 0 ? msgs.slice(0, lastUserIdx) : [];
        setMsgs(updated);
        saveMessages(sessionId, updated);
      }
      updateSession(sessionId, { isProcessing: false });
    } catch (e) {
      logger.error('Undo failed:', e);
    }
  }, [sessionId, updateSession]);

  // -- Approve tools ------------------------------------------------------
  const approveTools = useCallback(
    async (approvals: Array<{ tool_call_id: string; approved: boolean; feedback?: string | null; edited_script?: string | null }>) => {
      // Store edited scripts so the transport can read them when sendMessages is called
      for (const a of approvals) {
        if (a.edited_script) {
          useAgentStore.getState().setEditedScript(a.tool_call_id, a.edited_script);
        }
      }

      // Update SDK tool state — this triggers sendMessages() via the transport
      for (const a of approvals) {
        chat.addToolApprovalResponse({
          id: `approval-${a.tool_call_id}`,
          approved: a.approved,
          reason: a.approved ? undefined : (a.feedback || 'Rejected by user'),
        });
      }

      setNeedsAttention(sessionId, false);
      const hasApproved = approvals.some(a => a.approved);
      if (hasApproved) {
        updateSession(sessionId, { isProcessing: true });
      }
      return true;
    },
    [sessionId, chat, updateSession, setNeedsAttention],
  );

  // -- Stop (abort SSE stream + interrupt backend agent loop) ---------------
  const stop = useCallback(() => {
    chat.stop();
    apiFetch(`/api/interrupt/${sessionId}`, { method: 'POST' }).catch(() => {});
  }, [sessionId, chat]);

  return {
    messages: chat.messages,
    sendMessage: chat.sendMessage,
    stop,
    status: chat.status,
    undoLastTurn,
    approveTools,
  };
}
