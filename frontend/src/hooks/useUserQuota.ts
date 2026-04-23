/**
 * Reads the current user's Claude daily quota + plan tier from the backend.
 *
 * Fetches once when the user becomes authenticated, and exposes a `refresh()`
 * that callers invoke after a successful session-create / model-switch so the
 * chip reflects the new count without a full page reload.
 */
import { useCallback, useEffect, useState } from 'react';
import { useAgentStore } from '@/store/agentStore';
import { apiFetch } from '@/utils/api';

export type PlanTier = 'free' | 'pro' | 'org';

export interface UserQuota {
  plan: PlanTier;
  claudeUsedToday: number;
  claudeDailyCap: number;
  claudeRemaining: number;
}

export function useUserQuota() {
  const user = useAgentStore((s) => s.user);
  const [quota, setQuota] = useState<UserQuota | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    if (!user?.authenticated) return;
    setLoading(true);
    try {
      const res = await apiFetch('/api/user/quota');
      if (!res.ok) return;
      const data = await res.json();
      setQuota({
        plan: (data.plan ?? 'free') as PlanTier,
        claudeUsedToday: data.claude_used_today ?? 0,
        claudeDailyCap: data.claude_daily_cap ?? 1,
        claudeRemaining: data.claude_remaining ?? 0,
      });
    } catch {
      /* backend unreachable — leave previous value */
    } finally {
      setLoading(false);
    }
  }, [user?.authenticated]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { quota, loading, refresh };
}
