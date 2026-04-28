/**
 * SSE-based ChatTransport that bridges our backend event protocol
 * to the Vercel AI SDK's UIMessageChunk streaming interface.
 *
 * Each sendMessages() call does a POST → SSE response.
 * One request per turn phase (initial message, or approval continuation).
 */
import type { ChatTransport, UIMessage, UIMessageChunk, ChatRequestOptions } from 'ai';
import { apiFetch } from '@/utils/api';
import { logger } from '@/utils/logger';
import type { AgentEvent } from '@/types/events';
import { useAgentStore } from '@/store/agentStore';

// ---------------------------------------------------------------------------
// Side-channel callback interface (non-chat events forwarded to the store)
// ---------------------------------------------------------------------------
export interface SideChannelCallbacks {
  onReady: () => void;
  onShutdown: () => void;
  onError: (error: string) => void;
  onProcessing: () => void;
  onProcessingDone: () => void;
  onUndoComplete: () => void;
  onCompacted: (oldTokens: number, newTokens: number) => void;
  onPlanUpdate: (plan: Array<{ id: string; content: string; status: string }>) => void;
  onToolLog: (tool: string, log: string, agentId?: string, label?: string) => void;
  onConnectionChange: (connected: boolean) => void;
  onSessionDead: (sessionId: string) => void;
  onApprovalRequired: (tools: Array<{ tool: string; arguments: Record<string, unknown>; tool_call_id: string }>) => void;
  onToolCallPanel: (tool: string, args: Record<string, unknown>) => void;
  onToolOutputPanel: (tool: string, toolCallId: string, output: string, success: boolean) => void;
  onStreaming: () => void;
  onToolRunning: (toolName: string, description?: string) => void;
  onInterrupted: () => void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
let partIdCounter = 0;
function nextPartId(prefix: string): string {
  return `${prefix}-${Date.now()}-${++partIdCounter}`;
}

function lastEventKey(sessionId: string): string {
  return `hf-agent-last-event:${sessionId}`;
}

/** Parse an SSE text stream into AgentEvent objects. */
function createSSEParserStream(sessionId: string): TransformStream<string, AgentEvent> {
  let buffer = '';
  let eventId: string | null = null;
  let data = '';

  const dispatch = (controller: TransformStreamDefaultController<AgentEvent>) => {
    if (!data.trim()) {
      eventId = null;
      data = '';
      return;
    }
    try {
      const json = JSON.parse(data.trim()) as AgentEvent;
      const seq = json.seq ?? (eventId ? Number(eventId) : undefined);
      if (Number.isFinite(seq)) {
        json.seq = seq;
        localStorage.setItem(lastEventKey(sessionId), String(seq));
      }
      controller.enqueue(json);
    } catch {
      logger.warn('SSE parse error:', data.trim());
    } finally {
      eventId = null;
      data = '';
    }
  };

  return new TransformStream<string, AgentEvent>({
    transform(chunk, controller) {
      buffer += chunk;
      const lines = buffer.split('\n');
      // Keep the last (possibly incomplete) line in the buffer
      buffer = lines.pop() || '';
      for (const rawLine of lines) {
        const line = rawLine.replace(/\r$/, '');
        if (line === '') {
          dispatch(controller);
          continue;
        }
        if (line.startsWith(':')) continue;
        if (line.startsWith('id:')) {
          eventId = line.slice(3).trim();
        } else if (line.startsWith('data:')) {
          data += line.slice(5).trimStart() + '\n';
        }
      }
    },
    flush(controller) {
      const line = buffer.replace(/\r$/, '');
      if (line.startsWith('id:')) {
        eventId = line.slice(3).trim();
      } else if (line.startsWith('data:')) {
        data += line.slice(5).trimStart() + '\n';
      }
      dispatch(controller);
    },
  });
}

/** Transform AgentEvent objects into UIMessageChunk objects for the Vercel AI SDK. */
function createEventToChunkStream(sideChannel: SideChannelCallbacks): TransformStream<AgentEvent, UIMessageChunk> {
  let textPartId: string | null = null;

  function endTextPart(controller: TransformStreamDefaultController<UIMessageChunk>) {
    if (textPartId) {
      controller.enqueue({ type: 'text-end', id: textPartId });
      textPartId = null;
    }
  }

  return new TransformStream<AgentEvent, UIMessageChunk>({
    transform(event, controller) {
      switch (event.event_type) {
        // -- Side-channel only events ----------------------------------------
        case 'ready':
          sideChannel.onReady();
          break;

        case 'shutdown':
          endTextPart(controller);
          controller.enqueue({ type: 'finish-step' });
          controller.enqueue({ type: 'finish', finishReason: 'stop' });
          sideChannel.onShutdown();
          break;

        case 'interrupted':
          endTextPart(controller);
          controller.enqueue({ type: 'finish-step' });
          controller.enqueue({ type: 'finish', finishReason: 'stop' });
          sideChannel.onInterrupted();
          sideChannel.onProcessingDone();
          break;

        case 'undo_complete':
          endTextPart(controller);
          sideChannel.onUndoComplete();
          break;

        case 'compacted':
          sideChannel.onCompacted(
            (event.data?.old_tokens as number) || 0,
            (event.data?.new_tokens as number) || 0,
          );
          break;

        case 'plan_update':
          sideChannel.onPlanUpdate(
            (event.data?.plan as Array<{ id: string; content: string; status: string }>) || [],
          );
          break;

        case 'tool_log':
          sideChannel.onToolLog(
            (event.data?.tool as string) || '',
            (event.data?.log as string) || '',
            (event.data?.agent_id as string) || '',
            (event.data?.label as string) || '',
          );
          break;

        // -- Chat stream events ----------------------------------------------
        case 'processing':
          sideChannel.onProcessing();
          controller.enqueue({ type: 'start', messageMetadata: { createdAt: new Date().toISOString() } });
          controller.enqueue({ type: 'start-step' });
          break;

        case 'assistant_chunk': {
          const delta = (event.data?.content as string) || '';
          if (!delta) break;
          if (!textPartId) {
            textPartId = nextPartId('text');
            controller.enqueue({ type: 'text-start', id: textPartId });
            sideChannel.onStreaming();
          }
          controller.enqueue({ type: 'text-delta', id: textPartId, delta });
          break;
        }

        case 'assistant_stream_end':
          endTextPart(controller);
          break;

        case 'assistant_message': {
          const content = (event.data?.content as string) || '';
          if (!content) break;
          const id = nextPartId('text');
          controller.enqueue({ type: 'text-start', id });
          controller.enqueue({ type: 'text-delta', id, delta: content });
          controller.enqueue({ type: 'text-end', id });
          break;
        }

        case 'tool_call': {
          const toolName = (event.data?.tool as string) || 'unknown';
          const toolCallId = (event.data?.tool_call_id as string) || '';
          const args = (event.data?.arguments as Record<string, unknown>) || {};
          if (toolName === 'plan_tool') break;

          endTextPart(controller);
          controller.enqueue({ type: 'tool-input-start', toolCallId, toolName, dynamic: true });
          controller.enqueue({ type: 'tool-input-available', toolCallId, toolName, input: args, dynamic: true });

          sideChannel.onToolRunning(toolName, (args as Record<string, unknown>)?.description as string | undefined);
          sideChannel.onToolCallPanel(toolName, args as Record<string, unknown>);
          break;
        }

        case 'tool_output': {
          const toolCallId = (event.data?.tool_call_id as string) || '';
          const output = (event.data?.output as string) || '';
          const success = event.data?.success as boolean;
          const toolName = (event.data?.tool as string) || '';
          if (toolName === 'plan_tool' || toolCallId.startsWith('plan_tool')) break;

          if (success) {
            controller.enqueue({ type: 'tool-output-available', toolCallId, output, dynamic: true });
          } else {
            controller.enqueue({ type: 'tool-output-error', toolCallId, errorText: output, dynamic: true });
          }
          sideChannel.onToolOutputPanel(toolName, toolCallId, output, success);
          break;
        }

        case 'approval_required': {
          const tools = event.data?.tools as Array<{
            tool: string;
            arguments: Record<string, unknown>;
            tool_call_id: string;
          }>;
          if (!tools) break;

          endTextPart(controller);
          for (const t of tools) {
            controller.enqueue({ type: 'tool-input-start', toolCallId: t.tool_call_id, toolName: t.tool, dynamic: true });
            controller.enqueue({ type: 'tool-input-available', toolCallId: t.tool_call_id, toolName: t.tool, input: t.arguments, dynamic: true });
            controller.enqueue({ type: 'tool-approval-request', approvalId: `approval-${t.tool_call_id}`, toolCallId: t.tool_call_id });
          }
          sideChannel.onApprovalRequired(tools);
          // DON'T emit finish here — the stream will close naturally and the SDK
          // will see there's a pending approval. The SDK calls sendMessages again
          // after addToolApprovalResponse.
          break;
        }

        case 'tool_state_change': {
          const tcId = (event.data?.tool_call_id as string) || '';
          const state = (event.data?.state as string) || '';
          const toolName = (event.data?.tool as string) || '';
          const jobUrl = (event.data?.jobUrl as string) || undefined;
          const trackioSpaceId = (event.data?.trackioSpaceId as string) || undefined;
          const trackioProject = (event.data?.trackioProject as string) || undefined;

          if (tcId.startsWith('plan_tool')) break;

          if (jobUrl && tcId) {
            useAgentStore.getState().setJobUrl(tcId, jobUrl);
          }
          if (trackioSpaceId && tcId) {
            useAgentStore.getState().setTrackioDashboard(tcId, trackioSpaceId, trackioProject);
          }
          if (state === 'running' && toolName) {
            sideChannel.onToolRunning(toolName);
          }
          if (state === 'rejected' || state === 'abandoned') {
            controller.enqueue({ type: 'tool-output-denied', toolCallId: tcId });
          }
          if (state === 'cancelled') {
            controller.enqueue({ type: 'tool-output-error', toolCallId: tcId, errorText: 'Cancelled by user', dynamic: true });
          }
          if (state === 'billing_required') {
            const namespace = (event.data?.namespace as string) || '';
            useAgentStore.getState().setJobsUpgradeRequired({
              namespace: namespace || null,
              message: namespace
                ? `Hugging Face Jobs need credits on the "${namespace}" namespace. Add some, then re-run the same job — the agent will pick it back up.`
                : 'Hugging Face Jobs need credits on this namespace. Add some, then re-run the same job — the agent will pick it back up.',
            });
          }
          break;
        }

        case 'turn_complete':
          endTextPart(controller);
          controller.enqueue({ type: 'finish-step' });
          controller.enqueue({ type: 'finish', finishReason: 'stop' });
          sideChannel.onProcessingDone();
          break;

        case 'error': {
          const errorMsg = (event.data?.error as string) || 'Unknown error';
          endTextPart(controller);
          controller.enqueue({ type: 'finish-step' });
          controller.enqueue({ type: 'finish', finishReason: 'error' });
          sideChannel.onError(errorMsg);
          sideChannel.onProcessingDone();
          break;
        }

        default:
          logger.log('SSE transport: unknown event', event);
      }
    },
  });
}

// ---------------------------------------------------------------------------
// Transport implementation
// ---------------------------------------------------------------------------
export class SSEChatTransport implements ChatTransport<UIMessage> {
  private sessionId: string;
  private sideChannel: SideChannelCallbacks;

  constructor(sessionId: string, sideChannel: SideChannelCallbacks) {
    this.sessionId = sessionId;
    this.sideChannel = sideChannel;
    // Mark as connected immediately — no persistent connection to establish
    // Defer to avoid setState during render
    queueMicrotask(() => sideChannel.onConnectionChange(true));
  }

  updateSideChannel(sideChannel: SideChannelCallbacks): void {
    this.sideChannel = sideChannel;
  }

  destroy(): void {
    // Nothing to clean up — no persistent connections
  }

  // -- ChatTransport interface ---------------------------------------------

  async sendMessages(
    options: {
      trigger: 'submit-message' | 'regenerate-message';
      chatId: string;
      messageId: string | undefined;
      messages: UIMessage[];
      abortSignal: AbortSignal | undefined;
    } & ChatRequestOptions,
  ): Promise<ReadableStream<UIMessageChunk>> {
    const sessionId = this.sessionId;

    // Detect: is this an approval continuation or a new user message?
    // After addToolApprovalResponse, the SDK calls sendMessages again.
    // The last assistant message will have tool parts in 'approval-responded' state.
    const lastAssistant = [...options.messages].reverse().find(m => m.role === 'assistant');
    const approvedParts = lastAssistant?.parts.filter(
      (p) => p.type === 'dynamic-tool' && p.state === 'approval-responded'
    ) || [];

    let body: Record<string, unknown>;
    if (approvedParts.length > 0) {
      // Approval continuation — extract approval decisions
      const approvals = approvedParts.map((p) => {
        if (p.type !== 'dynamic-tool') return null;
        const approved = p.approval?.approved ?? true;
        const editedScript = useAgentStore.getState().getEditedScript(p.toolCallId);
        return {
          tool_call_id: p.toolCallId,
          approved,
          feedback: approved ? null : (p.approval?.reason || 'Rejected by user'),
          edited_script: editedScript ?? null,
          namespace: null,
        };
      }).filter(Boolean);
      body = { approvals };
    } else {
      // Normal user message
      const lastUserMsg = [...options.messages].reverse().find(m => m.role === 'user');
      const text = lastUserMsg
        ? lastUserMsg.parts
            .filter((p): p is Extract<typeof p, { type: 'text' }> => p.type === 'text')
            .map(p => p.text)
            .join('')
        : '';
      body = { text };
    }

    // POST to SSE endpoint
    const response = await apiFetch(`/api/chat/${sessionId}`, {
      method: 'POST',
      body: JSON.stringify(body),
      signal: options.abortSignal,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      },
    });

    if (response.status === 404) {
      // Backend lost this session (e.g. Space restart). Signal the UI so
      // it can flag the session for the catch-up banner.
      this.sideChannel.onSessionDead(sessionId);
    }
    if (response.status === 429) {
      // Claude daily-quota gate tripped. The prefix is the detection marker
      // for useAgentChat's onError handler, which surfaces the cap dialog
      // instead of a generic error banner.
      throw new Error('CLAUDE_QUOTA_EXHAUSTED');
    }
    if (!response.ok) {
      const errorText = await response.text().catch(() => 'Request failed');
      throw new Error(`Chat request failed: ${response.status} ${errorText}`);
    }

    if (!response.body) {
      throw new Error('No response body');
    }

    // Pipe: response bytes → text → SSE events → UIMessageChunks
    return response.body
      .pipeThrough(new TextDecoderStream())
      .pipeThrough(createSSEParserStream(sessionId))
      .pipeThrough(createEventToChunkStream(this.sideChannel));
  }

  async reconnectToStream(): Promise<ReadableStream<UIMessageChunk> | null> {
    // Check if the backend session is still processing a turn.
    // If so, subscribe to its event stream so we can resume live updates
    // (e.g. after page refresh or wake-from-sleep reconnection).
    try {
      const infoRes = await apiFetch(`/api/session/${this.sessionId}`);
      if (!infoRes.ok) return null;
      const info = await infoRes.json();
      if (!info.is_processing) return null;

      // Session is mid-turn — subscribe to its event broadcast.
      const lastSeq = localStorage.getItem(lastEventKey(this.sessionId));
      const qs = lastSeq ? `?after=${encodeURIComponent(lastSeq)}` : '';
      const response = await apiFetch(`/api/events/${this.sessionId}${qs}`, {
        headers: { 'Accept': 'text/event-stream' },
      });
      if (!response.ok || !response.body) return null;

      this.sideChannel.onProcessing();

      return response.body
        .pipeThrough(new TextDecoderStream())
        .pipeThrough(createSSEParserStream(this.sessionId))
        .pipeThrough(createEventToChunkStream(this.sideChannel));
    } catch {
      return null;
    }
  }
}
