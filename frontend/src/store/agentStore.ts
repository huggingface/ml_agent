/**
 * Agent store — manages UI state that is NOT handled by the Vercel AI SDK.
 *
 * Message state (messages, streaming, tool calls) is now managed by useChat().
 * This store only handles:
 *  - Connection / processing flags
 *  - Panel state (right panel — single-artifact pattern)
 *  - Plan state
 *  - User info / error banners
 *  - Edited scripts (for hf_jobs code editing)
 *
 * Per-session state:
 *  Each session maintains its own snapshot of processing/activity/panel/plan
 *  state in `sessionStates`. Background sessions keep updating their own
 *  snapshot via `updateSession()`. The active session's snapshot is mirrored
 *  to the flat top-level fields so the UI reads from a single place.
 */
import { create } from 'zustand';
import type { User } from '@/types/agent';

export interface PlanItem {
  id: string;
  content: string;
  status: 'pending' | 'in_progress' | 'completed';
}

export interface PanelSection {
  content: string;
  language: string;
}

export interface PanelData {
  title: string;
  script?: PanelSection;
  output?: PanelSection;
  input?: PanelSection;
  parameters?: Record<string, unknown>;
}

export type PanelView = 'script' | 'output';

export interface LLMHealthError {
  error: string;
  errorType: 'auth' | 'credits' | 'rate_limit' | 'network' | 'unknown';
  model: string;
}

export type ActivityStatus =
  | { type: 'idle' }
  | { type: 'thinking' }
  | { type: 'tool'; toolName: string; description?: string }
  | { type: 'waiting-approval' }
  | { type: 'streaming' };

/** State that is tracked per-session (each session has its own copy). */
export interface PerSessionState {
  isProcessing: boolean;
  activityStatus: ActivityStatus;
  panelData: PanelData | null;
  panelView: PanelView;
  panelEditable: boolean;
  plan: PlanItem[];
  /** Steps completed by the research sub-agent (tool_log events). */
  researchSteps: string[];
}

const defaultSessionState: PerSessionState = {
  isProcessing: false,
  activityStatus: { type: 'idle' },
  panelData: null,
  panelView: 'script',
  panelEditable: false,
  plan: [],
  researchSteps: [],
};

interface AgentStore {
  // ── Per-session state map ───────────────────────────────────────────
  sessionStates: Record<string, PerSessionState>;
  activeSessionId: string | null;

  // ── Flat state (mirrors active session — UI reads from here) ────────
  isProcessing: boolean;
  isConnected: boolean;
  activityStatus: ActivityStatus;
  user: User | null;
  error: string | null;
  llmHealthError: LLMHealthError | null;

  // Right panel (single-artifact pattern)
  panelData: PanelData | null;
  panelView: PanelView;
  panelEditable: boolean;

  // Plan
  plan: PlanItem[];

  // Edited scripts (tool_call_id -> edited content)
  editedScripts: Record<string, string>;

  // Job URLs (tool_call_id -> job URL) for HF jobs
  jobUrls: Record<string, string>;

  // ── Per-session actions ─────────────────────────────────────────────

  /** Update a session's state. If it's the active session, also update flat state. */
  updateSession: (sessionId: string, updates: Partial<PerSessionState>) => void;

  /** Get a session's current state (from map, not flat). */
  getSessionState: (sessionId: string) => PerSessionState;

  /** Switch the active session — restores its state to flat fields. */
  switchActiveSession: (sessionId: string) => void;

  /** Remove a session's state from the map. */
  clearSessionState: (sessionId: string) => void;

  // ── Global actions (not per-session) ────────────────────────────────
  setProcessing: (isProcessing: boolean) => void;
  setConnected: (isConnected: boolean) => void;
  setActivityStatus: (status: ActivityStatus) => void;
  setUser: (user: User | null) => void;
  setError: (error: string | null) => void;
  setLlmHealthError: (error: LLMHealthError | null) => void;

  setPanel: (data: PanelData, view?: PanelView, editable?: boolean) => void;
  setPanelView: (view: PanelView) => void;
  setPanelOutput: (output: PanelSection) => void;
  updatePanelScript: (content: string) => void;
  lockPanel: () => void;
  clearPanel: () => void;

  setPlan: (plan: PlanItem[]) => void;

  setEditedScript: (toolCallId: string, content: string) => void;
  getEditedScript: (toolCallId: string) => string | undefined;
  clearEditedScripts: () => void;

  setJobUrl: (toolCallId: string, jobUrl: string) => void;
  getJobUrl: (toolCallId: string) => string | undefined;
}

/**
 * Helper: patch the active session's snapshot with partial per-session fields.
 * Returns the `sessionStates` slice to spread into a `set()` call, or `{}`
 * if there's no active session snapshot to update.
 */
function syncSnapshot(
  state: AgentStore,
  patch: Partial<PerSessionState>,
): { sessionStates: Record<string, PerSessionState> } | Record<string, never> {
  const { activeSessionId, sessionStates } = state;
  if (!activeSessionId || !sessionStates[activeSessionId]) return {};
  return {
    sessionStates: {
      ...sessionStates,
      [activeSessionId]: { ...sessionStates[activeSessionId], ...patch },
    },
  };
}

export const useAgentStore = create<AgentStore>()((set, get) => ({
  sessionStates: {},
  activeSessionId: null,

  isProcessing: false,
  isConnected: false,
  activityStatus: { type: 'idle' },
  user: null,
  error: null,
  llmHealthError: null,

  panelData: null,
  panelView: 'script',
  panelEditable: false,

  plan: [],

  editedScripts: {},
  jobUrls: {},

  // ── Per-session state management ──────────────────────────────────

  updateSession: (sessionId, updates) => {
    const state = get();
    const current = state.sessionStates[sessionId] || { ...defaultSessionState };
    const updated = { ...current, ...updates };

    // Apply the processing→idle side effect
    const processingCleared = 'isProcessing' in updates && !updates.isProcessing;
    if (processingCleared) {
      if (updated.activityStatus.type !== 'waiting-approval') {
        updated.activityStatus = { type: 'idle' };
      }
    }

    const isActive = state.activeSessionId === sessionId;

    // Build flat-state mirror: only the fields explicitly in `updates`
    // (plus activityStatus when the processing→idle side-effect fires).
    // This prevents overwriting flat fields changed by global setters
    // (e.g. setPanelView called from CodePanel) with stale snapshot values.
    let flatMirror: Record<string, unknown> = {};
    if (isActive) {
      for (const key of Object.keys(updates)) {
        flatMirror[key] = updated[key as keyof PerSessionState];
      }
      // Side-effect may have changed activityStatus even if it wasn't in updates
      if (processingCleared) {
        flatMirror.activityStatus = updated.activityStatus;
      }
    }

    set({
      sessionStates: { ...state.sessionStates, [sessionId]: updated },
      ...flatMirror,
    });
  },

  getSessionState: (sessionId) => {
    return get().sessionStates[sessionId] || { ...defaultSessionState };
  },

  switchActiveSession: (sessionId) => {
    const state = get();

    // Build a new sessionStates map (never mutate the existing object)
    const updatedStates = { ...state.sessionStates };

    // Save current active session's flat state back to its snapshot
    if (state.activeSessionId && state.activeSessionId !== sessionId) {
      updatedStates[state.activeSessionId] = {
        isProcessing: state.isProcessing,
        activityStatus: state.activityStatus,
        panelData: state.panelData,
        panelView: state.panelView,
        panelEditable: state.panelEditable,
        plan: state.plan,
        researchSteps: state.sessionStates[state.activeSessionId]?.researchSteps ?? [],
      };
    }

    // Restore the new session's state
    const incoming = updatedStates[sessionId] || { ...defaultSessionState };
    set({
      activeSessionId: sessionId,
      sessionStates: updatedStates,
      isProcessing: incoming.isProcessing,
      activityStatus: incoming.activityStatus,
      panelData: incoming.panelData,
      panelView: incoming.panelView,
      panelEditable: incoming.panelEditable,
      plan: incoming.plan,
      // Clear transient error on switch
      error: null,
    });
  },

  clearSessionState: (sessionId) => {
    set((state) => {
      const { [sessionId]: _, ...rest } = state.sessionStates;
      return { sessionStates: rest };
    });
  },

  // ── Global flags ──────────────────────────────────────────────────

  setProcessing: (isProcessing) => {
    const current = get().activityStatus;
    const preserveStatus = current.type === 'waiting-approval';
    set({ isProcessing, ...(!isProcessing && !preserveStatus ? { activityStatus: { type: 'idle' } } : {}) });
  },
  setConnected: (isConnected) => set({ isConnected }),
  setActivityStatus: (status) => set({ activityStatus: status }),
  setUser: (user) => set({ user }),
  setError: (error) => set({ error }),
  setLlmHealthError: (error) => set({ llmHealthError: error }),

  // ── Panel (single-artifact) ───────────────────────────────────────
  // Each setter also patches the active session's snapshot so that
  // getSessionState() stays consistent with flat state.

  setPanel: (data, view, editable) => set((state) => {
    const patch: Partial<PerSessionState> = {
      panelData: data,
      panelView: view ?? (data.script ? 'script' : 'output'),
      panelEditable: editable ?? false,
    };
    return { ...patch, ...syncSnapshot(state, patch) };
  }),

  setPanelView: (view) => set((state) => {
    const patch: Partial<PerSessionState> = { panelView: view };
    return { ...patch, ...syncSnapshot(state, patch) };
  }),

  setPanelOutput: (output) => set((state) => {
    const panelData = state.panelData
      ? { ...state.panelData, output }
      : { title: 'Output', output };
    const patch: Partial<PerSessionState> = { panelData, panelView: 'output' };
    return { ...patch, ...syncSnapshot(state, patch) };
  }),

  updatePanelScript: (content) => set((state) => {
    const panelData = state.panelData?.script
      ? { ...state.panelData, script: { ...state.panelData.script, content } }
      : state.panelData;
    if (!panelData) return {};
    const patch: Partial<PerSessionState> = { panelData };
    return { ...patch, ...syncSnapshot(state, patch) };
  }),

  lockPanel: () => set((state) => {
    const patch: Partial<PerSessionState> = { panelEditable: false };
    return { ...patch, ...syncSnapshot(state, patch) };
  }),

  clearPanel: () => set((state) => {
    const patch: Partial<PerSessionState> = { panelData: null, panelView: 'script', panelEditable: false };
    return { ...patch, ...syncSnapshot(state, patch) };
  }),

  // ── Plan ──────────────────────────────────────────────────────────

  setPlan: (plan) => set((state) => {
    const patch: Partial<PerSessionState> = { plan };
    return { ...patch, ...syncSnapshot(state, patch) };
  }),

  // ── Edited scripts ────────────────────────────────────────────────

  setEditedScript: (toolCallId, content) => {
    set((state) => ({
      editedScripts: { ...state.editedScripts, [toolCallId]: content },
    }));
  },

  getEditedScript: (toolCallId) => get().editedScripts[toolCallId],

  clearEditedScripts: () => set({ editedScripts: {} }),

  // ── Job URLs ────────────────────────────────────────────────────────

  setJobUrl: (toolCallId, jobUrl) => {
    set((state) => ({
      jobUrls: { ...state.jobUrls, [toolCallId]: jobUrl },
    }));
  },

  getJobUrl: (toolCallId) => get().jobUrls[toolCallId],
}));
