import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { SessionMeta } from '@/types/agent';
import { deleteMessages, moveMessages } from '@/lib/chat-message-store';
import { moveBackendMessages, deleteBackendMessages } from '@/lib/backend-message-store';

interface SessionStore {
  sessions: SessionMeta[];
  activeSessionId: string | null;

  // Actions
  createSession: (id: string, model?: string | null) => void;
  deleteSession: (id: string) => void;
  switchSession: (id: string) => void;
  setSessionActive: (id: string, isActive: boolean) => void;
  updateSessionTitle: (id: string, title: string) => void;
  updateSessionModel: (id: string, model: string | null) => void;
  setNeedsAttention: (id: string, needs: boolean) => void;
  /** Mark a session as expired (backend no longer has it). The UI shows a
   *  recovery banner and disables input. */
  markExpired: (id: string) => void;
  /** Clear the expired flag (used after restore-with-summary succeeds). */
  clearExpired: (id: string) => void;
  /** Merge durable server-side sessions into local sidebar metadata. */
  mergeServerSessions: (sessions: Array<{
    session_id: string;
    title?: string | null;
    created_at: string;
    is_active?: boolean;
    model?: string | null;
    pending_approval?: unknown[] | null;
    auto_approval?: {
      enabled?: boolean;
      cost_cap_usd?: number | null;
      estimated_spend_usd?: number;
      remaining_usd?: number | null;
    } | null;
  }>) => void;
  updateSessionYolo: (id: string, policy: {
    enabled: boolean;
    cost_cap_usd?: number | null;
    estimated_spend_usd?: number;
    remaining_usd?: number | null;
  }) => void;
  /** Atomically swap a session's id in the list + both localStorage caches.
   *  Used when we rehydrate an expired session into a freshly-created backend
   *  session — preserves title, timestamps, and messages. */
  renameSession: (oldId: string, newId: string) => void;
}

export const useSessionStore = create<SessionStore>()(
  persist(
    (set, get) => ({
      sessions: [],
      activeSessionId: null,

      createSession: (id: string, model?: string | null) => {
        const newSession: SessionMeta = {
          id,
          title: `Chat ${get().sessions.length + 1}`,
          createdAt: new Date().toISOString(),
          isActive: true,
          needsAttention: false,
          model: model ?? null,
          autoApprovalEnabled: false,
          autoApprovalCostCapUsd: null,
          autoApprovalEstimatedSpendUsd: 0,
          autoApprovalRemainingUsd: null,
        };
        set((state) => ({
          sessions: [...state.sessions, newSession],
          activeSessionId: id,
        }));
      },

      deleteSession: (id: string) => {
        deleteMessages(id);
        deleteBackendMessages(id);
        set((state) => {
          const newSessions = state.sessions.filter((s) => s.id !== id);
          const newActiveId =
            state.activeSessionId === id
              ? newSessions[newSessions.length - 1]?.id || null
              : state.activeSessionId;
          return {
            sessions: newSessions,
            activeSessionId: newActiveId,
          };
        });
      },

      markExpired: (id: string) => {
        set((state) => ({
          sessions: state.sessions.map((s) => (s.id === id ? { ...s, expired: true } : s)),
        }));
      },

      clearExpired: (id: string) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, expired: false } : s,
          ),
        }));
      },

      mergeServerSessions: (serverSessions) => {
        set((state) => {
          const byId = new Map(state.sessions.map((s) => [s.id, s]));
          const merged = [...state.sessions];
          for (const server of serverSessions) {
            const id = server.session_id;
            if (!id) continue;
            const existing = byId.get(id);
            if (existing) {
              const auto = server.auto_approval;
              const updated = {
                ...existing,
                title: server.title || existing.title,
                isActive: server.is_active ?? existing.isActive,
                model: server.model ?? existing.model ?? null,
                needsAttention: Boolean(server.pending_approval?.length) || existing.needsAttention,
                expired: false,
                ...(auto
                  ? {
                      autoApprovalEnabled: Boolean(auto.enabled),
                      autoApprovalCostCapUsd: auto.cost_cap_usd ?? null,
                      autoApprovalEstimatedSpendUsd: auto.estimated_spend_usd ?? 0,
                      autoApprovalRemainingUsd: auto.remaining_usd ?? null,
                    }
                  : {}),
              };
              const idx = merged.findIndex((s) => s.id === id);
              if (idx >= 0) merged[idx] = updated;
              byId.set(id, updated);
              continue;
            }
            const newSession: SessionMeta = {
              id,
              title: server.title || `Chat ${merged.length + 1}`,
              createdAt: server.created_at || new Date().toISOString(),
              isActive: server.is_active ?? true,
              needsAttention: Boolean(server.pending_approval?.length),
              model: server.model ?? null,
              expired: false,
              autoApprovalEnabled: Boolean(server.auto_approval?.enabled),
              autoApprovalCostCapUsd: server.auto_approval?.cost_cap_usd ?? null,
              autoApprovalEstimatedSpendUsd: server.auto_approval?.estimated_spend_usd ?? 0,
              autoApprovalRemainingUsd: server.auto_approval?.remaining_usd ?? null,
            };
            merged.push(newSession);
            byId.set(id, newSession);
          }
          return {
            sessions: merged,
            activeSessionId: state.activeSessionId || merged[merged.length - 1]?.id || null,
          };
        });
      },

      updateSessionYolo: (id, policy) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id
              ? {
                  ...s,
                  autoApprovalEnabled: policy.enabled,
                  autoApprovalCostCapUsd: policy.cost_cap_usd ?? null,
                  autoApprovalEstimatedSpendUsd: policy.estimated_spend_usd ?? 0,
                  autoApprovalRemainingUsd: policy.remaining_usd ?? null,
                }
              : s,
          ),
        }));
      },

      renameSession: (oldId: string, newId: string) => {
        if (oldId === newId) return;
        moveMessages(oldId, newId);
        moveBackendMessages(oldId, newId);
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === oldId ? { ...s, id: newId, expired: false } : s,
          ),
          activeSessionId: state.activeSessionId === oldId ? newId : state.activeSessionId,
        }));
      },

      switchSession: (id: string) => {
        set((state) => ({
          activeSessionId: id,
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, needsAttention: false } : s
          ),
        }));
      },

      setSessionActive: (id: string, isActive: boolean) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, isActive } : s
          ),
        }));
      },

      updateSessionTitle: (id: string, title: string) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, title } : s
          ),
        }));
      },

      updateSessionModel: (id: string, model: string | null) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, model } : s
          ),
        }));
      },

      setNeedsAttention: (id: string, needs: boolean) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, needsAttention: needs } : s
          ),
        }));
      },
    }),
    {
      name: 'hf-agent-sessions',
      partialize: (state) => ({
        sessions: state.sessions,
        activeSessionId: state.activeSessionId,
      }),
    }
  )
);
