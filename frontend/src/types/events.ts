/**
 * Event types from the agent backend
 */

export type EventType =
  | 'ready'
  | 'processing'
  | 'assistant_message'
  | 'assistant_chunk'
  | 'assistant_stream_end'
  | 'tool_call'
  | 'tool_output'
  | 'tool_log'
  | 'approval_required'
  | 'tool_state_change'
  | 'turn_complete'
  | 'compacted'
  | 'error'
  | 'shutdown'
  | 'interrupted'
  | 'undo_complete'
  | 'plan_update';

export interface AgentEvent {
  event_type: EventType;
  data?: Record<string, unknown>;
  seq?: number;
}

export interface ReadyEventData {
  message: string;
}

export interface ProcessingEventData {
  message: string;
}

export interface AssistantMessageEventData {
  content: string;
}

export interface ToolCallEventData {
  tool: string;
  arguments: Record<string, unknown>;
}

export interface ToolOutputEventData {
  tool: string;
  output: string;
  success: boolean;
}

export interface ToolLogEventData {
  tool: string;
  log: string;
}

export interface PlanUpdateEventData {
  plan: Array<{ id: string; content: string; status: 'pending' | 'in_progress' | 'completed' }>;
}

export interface ApprovalRequiredEventData {
  tools: ApprovalToolItem[];
  count: number;
}

export interface ApprovalToolItem {
  tool: string;
  arguments: Record<string, unknown>;
  tool_call_id: string;
}

export interface TurnCompleteEventData {
  history_size: number;
}

export interface CompactedEventData {
  old_tokens: number;
  new_tokens: number;
}

export interface ErrorEventData {
  error: string;
}
