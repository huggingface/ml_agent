import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Box, Stack, Typography, Chip, Button, TextField, IconButton, Link, CircularProgress } from '@mui/material';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import LaunchIcon from '@mui/icons-material/Launch';
import SendIcon from '@mui/icons-material/Send';
import BlockIcon from '@mui/icons-material/Block';
import { useAgentStore } from '@/store/agentStore';
import { useLayoutStore } from '@/store/layoutStore';
import { logger } from '@/utils/logger';
import { RESEARCH_MAX_STEPS } from '@/lib/research-store';
import type { UIMessage } from 'ai';

// ---------------------------------------------------------------------------
// Type helpers — extract the dynamic-tool part type from UIMessage
// ---------------------------------------------------------------------------
type DynamicToolPart = Extract<UIMessage['parts'][number], { type: 'dynamic-tool' }>;

type ToolPartState = DynamicToolPart['state'];

/** Check if a tool part was cancelled (output-error with cancellation message). */
function isCancelledTool(tool: DynamicToolPart): boolean {
  return tool.state === 'output-error' &&
    typeof (tool as Record<string, unknown>).errorText === 'string' &&
    ((tool as Record<string, unknown>).errorText as string).includes('Cancelled by user');
}

interface ToolCallGroupProps {
  tools: DynamicToolPart[];
  approveTools: (approvals: Array<{ tool_call_id: string; approved: boolean; feedback?: string | null; edited_script?: string | null }>) => Promise<boolean>;
}

// ---------------------------------------------------------------------------
// Research sub-steps (inline under the research tool row)
// ---------------------------------------------------------------------------

/** Hook that ticks every second while startedAt is set, returning elapsed seconds. */
function useElapsed(startedAt: number | null): number | null {
  const [elapsed, setElapsed] = useState<number | null>(null);
  useEffect(() => {
    if (startedAt === null) { setElapsed(null); return; }
    setElapsed(Math.round((Date.now() - startedAt) / 1000));
    const id = setInterval(() => setElapsed(Math.round((Date.now() - startedAt) / 1000)), 1000);
    return () => clearInterval(id);
  }, [startedAt]);
  return elapsed;
}

/** Format token count like the CLI: "12.4k" or "800". */
function formatTokens(tokens: number): string {
  return tokens >= 1000 ? `${(tokens / 1000).toFixed(1)}k` : String(tokens);
}

/** Format elapsed seconds like the CLI: "18s" or "2m 5s". */
function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

/** Build the research stats chip label. */
function researchChipLabel(
  stats: { toolCount: number; tokenCount: number; startedAt: number | null; finalElapsed: number | null },
  liveElapsed: number | null,
): string | null {
  const elapsed = stats.finalElapsed ?? liveElapsed;
  if (elapsed === null && stats.toolCount === 0) return null;
  const parts: string[] = [];
  if (stats.startedAt !== null) parts.push('running');
  if (stats.toolCount > 0) parts.push(`${stats.toolCount} tools`);
  if (stats.tokenCount > 0) parts.push(`${formatTokens(stats.tokenCount)} tokens`);
  if (elapsed !== null) parts.push(formatElapsed(elapsed));
  return parts.join(' \u00B7 ');
}

/** Parse JSON args from a step string like "tool_name  {json}" (may be truncated at 80 chars). */
function parseStepArgs(step: string): Record<string, string> {
  const jsonStart = step.indexOf('{');
  if (jsonStart < 0) return {};
  const jsonStr = step.slice(jsonStart);
  try {
    const parsed = JSON.parse(jsonStr);
    const result: Record<string, string> = {};
    for (const [k, v] of Object.entries(parsed)) {
      if (typeof v === 'string') result[k] = v;
    }
    return result;
  } catch {
    // JSON likely truncated — extract key-value pairs via regex
    const result: Record<string, string> = {};
    // Match complete "key": "value" pairs
    for (const m of jsonStr.matchAll(/"(\w+)":\s*"([^"]*)"/g)) {
      result[m[1]] = m[2];
    }
    // Match truncated trailing value: "key": "value... (no closing quote)
    if (Object.keys(result).length === 0 || !result.query) {
      const trunc = jsonStr.match(/"(\w+)":\s*"([^"]+)$/);
      if (trunc && !result[trunc[1]]) {
        result[trunc[1]] = trunc[2];
      }
    }
    return result;
  }
}

/** Pretty labels for research sub-agent tool calls */
function formatResearchStep(raw: string): { label: string } {
  // Backend sends logs like "▸ tool_name  {args}" — strip the prefix
  const step = raw.replace(/^▸\s*/, '');
  const args = parseStepArgs(step);

  if (step.startsWith('github_find_examples')) {
    const detail = (args.keyword) || (args.repo);
    return { label: detail ? `Finding examples: ${detail}` : 'Finding examples' };
  }
  if (step.startsWith('github_read_file')) {
    const path = (args.path) || '';
    const filename = path.split('/').pop() || path;
    return { label: filename ? `Reading ${filename}` : 'Reading file' };
  }
  if (step.startsWith('explore_hf_docs')) {
    const endpoint = (args.endpoint) || (args.query);
    return { label: endpoint ? `Exploring docs: ${endpoint}` : 'Exploring docs' };
  }
  if (step.startsWith('fetch_hf_docs')) {
    const url = (args.url) || '';
    const page = url.split('/').pop()?.replace(/\.md$/, '');
    return { label: page ? `Reading docs: ${page}` : 'Fetching docs' };
  }
  if (step.startsWith('hf_inspect_dataset')) {
    const dataset = (args.dataset);
    return { label: dataset ? `Inspecting dataset: ${dataset}` : 'Inspecting dataset' };
  }
  if (step.startsWith('hf_papers')) {
    const op = args.operation as string;
    const detail = (args.query) || (args.arxiv_id);
    const opLabels: Record<string, string> = {
      trending: 'Browsing trending papers',
      search: 'Searching papers',
      paper_details: 'Reading paper details',
      read_paper: 'Reading paper',
      citation_graph: 'Tracing citations',
      snippet_search: 'Searching paper snippets',
      recommend: 'Finding related papers',
      find_datasets: 'Finding paper datasets',
      find_models: 'Finding paper models',
      find_collections: 'Finding paper collections',
      find_all_resources: 'Finding paper resources',
    };
    const base = (op && opLabels[op]) || 'Searching papers';
    return { label: detail ? `${base}: ${detail}` : base };
  }
  if (step.startsWith('find_hf_api')) {
    const detail = (args.query) || (args.tag);
    return { label: detail ? `Finding API: ${detail}` : 'Finding API endpoints' };
  }
  if (step.startsWith('hf_repo_files')) {
    const repo = (args.repo_id) || (args.repo);
    return { label: repo ? `Reading ${repo} files` : 'Reading repo files' };
  }
  if (step.startsWith('read')) {
    const path = (args.path) || '';
    const filename = path.split('/').pop();
    return { label: filename ? `Reading ${filename}` : 'Reading file' };
  }
  if (step.startsWith('bash')) {
    const cmd = args.command as string;
    const short = cmd && cmd.length > 40 ? cmd.slice(0, 40) + '...' : cmd;
    return { label: short ? `Running: ${short}` : 'Running command' };
  }
  return { label: step.replace(/^▸\s*/, '') };
}

/** Rolling display of research sub-tool calls per agent — hidden when complete. */
function ResearchSteps({ agents }: { agents: Record<string, { label: string; steps: string[]; stats: { startedAt: number | null } }> }) {
  const running = Object.entries(agents).filter(([, a]) => a.stats.startedAt !== null);
  if (running.length === 0) return null;

  return (
    <Box sx={{ pl: 4.5, pr: 1.5, pb: 1, pt: 0.25 }}>
      {running.map(([agentId, agent]) => {
        const visible = agent.steps.slice(-RESEARCH_MAX_STEPS);
        return (
          <Box key={agentId} sx={{ mb: running.length > 1 ? 0.75 : 0 }}>
            {running.length > 1 && (
              <Typography
                sx={{
                  fontFamily: '"JetBrains Mono", ui-monospace, SFMono-Regular, monospace',
                  fontSize: '0.62rem',
                  color: 'var(--accent-yellow)',
                  opacity: 0.7,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  mb: 0.25,
                }}
              >
                {agent.label}
              </Typography>
            )}
            {visible.map((step, i) => {
              const { label } = formatResearchStep(step);
              const isLast = i === visible.length - 1;
              return (
                <Stack
                  key={i}
                  direction="row"
                  alignItems="center"
                  spacing={0.75}
                  sx={{ py: 0.2 }}
                >
                  {isLast ? (
                    <CircularProgress size={10} thickness={5} sx={{ color: 'var(--accent-yellow)', flexShrink: 0 }} />
                  ) : (
                    <CheckCircleOutlineIcon sx={{ fontSize: 12, color: 'var(--muted-text)', flexShrink: 0 }} />
                  )}
                  <Typography
                    sx={{
                      fontFamily: '"JetBrains Mono", ui-monospace, SFMono-Regular, monospace',
                      fontSize: '0.68rem',
                      color: isLast ? 'var(--text)' : 'var(--muted-text)',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {label}
                  </Typography>
                </Stack>
              );
            })}
          </Box>
        );
      })}
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Hardware pricing ($/hr) — from HF Spaces & Jobs pricing
// ---------------------------------------------------------------------------
const HARDWARE_PRICING: Record<string, string> = {
  'cpu-basic': 'free',
  'cpu-upgrade': '$0.03/hr',
  't4-small': '$0.60/hr',
  't4-medium': '$1.00/hr',
  'a10g-small': '$1.05/hr',
  'a10g-large': '$3.15/hr',
  'a10g-largex2': '$6.30/hr',
  'a10g-largex4': '$12.60/hr',
  'a100-large': '$4.13/hr',
  'a100x4': '$16.52/hr',
  'a100x8': '$33.04/hr',
  'l4x1': '$0.80/hr',
  'l4x4': '$3.20/hr',
  'l40sx1': '$1.80/hr',
  'l40sx4': '$7.20/hr',
  'l40sx8': '$14.40/hr',
};

function costLabel(hardware: string): string | null {
  return HARDWARE_PRICING[hardware] || null;
}

// ---------------------------------------------------------------------------
// Visual helpers
// ---------------------------------------------------------------------------

function StatusIcon({ state, cancelled, isRejected }: { state: ToolPartState; cancelled?: boolean; isRejected?: boolean }) {
  if (cancelled || isRejected) {
    return <BlockIcon sx={{ fontSize: 16, color: 'var(--muted-text)' }} />;
  }
  switch (state) {
    case 'approval-requested':
      return <HourglassEmptyIcon sx={{ fontSize: 16, color: 'var(--accent-yellow)' }} />;
    case 'approval-responded':
      return <CircularProgress size={14} thickness={5} sx={{ color: 'var(--accent-green)' }} />;
    case 'output-available':
      return <CheckCircleOutlineIcon sx={{ fontSize: 16, color: 'success.main' }} />;
    case 'output-error':
      return <ErrorOutlineIcon sx={{ fontSize: 16, color: 'error.main' }} />;
    case 'output-denied':
      return <BlockIcon sx={{ fontSize: 16, color: 'var(--muted-text)' }} />;
    case 'input-streaming':
    case 'input-available':
    default:
      return <CircularProgress size={14} thickness={5} sx={{ color: 'var(--accent-yellow)' }} />;
  }
}

function statusLabel(state: ToolPartState): string | null {
  switch (state) {
    case 'approval-requested': return 'awaiting approval';
    case 'approval-responded': return 'approved';
    case 'input-streaming':
    case 'input-available': return 'running';
    case 'output-denied': return 'denied';
    case 'output-error': return 'error';
    default: return null;
  }
}

function statusColor(state: ToolPartState): string {
  switch (state) {
    case 'approval-requested': return 'var(--accent-yellow)';
    case 'approval-responded': return 'var(--accent-green)';
    case 'output-available': return 'var(--accent-green)';
    case 'output-error': return 'var(--accent-red)';
    case 'output-denied': return 'var(--muted-text)';
    default: return 'var(--accent-yellow)';
  }
}

// ---------------------------------------------------------------------------
// Inline approval UI (per-tool)
// ---------------------------------------------------------------------------

function InlineApproval({
  toolCallId,
  toolName,
  input,
  scriptLabel,
  onResolve,
}: {
  toolCallId: string;
  toolName: string;
  input: unknown;
  scriptLabel: string;
  onResolve: (toolCallId: string, approved: boolean, feedback?: string) => void;
}) {
  const [feedback, setFeedback] = useState('');
  const args = input as Record<string, unknown> | undefined;
  const { setPanel, getEditedScript } = useAgentStore();
  const { setRightPanelOpen, setLeftSidebarOpen } = useLayoutStore();
  const hasEditedScript = !!getEditedScript(toolCallId);

  const handleScriptClick = useCallback(() => {
    if (toolName === 'hf_jobs' && args?.script) {
      const scriptContent = getEditedScript(toolCallId) || String(args.script);
      setPanel(
        { title: scriptLabel, script: { content: scriptContent, language: 'python' }, parameters: { tool_call_id: toolCallId } },
        'script',
        true,
      );
      setRightPanelOpen(true);
      setLeftSidebarOpen(false);
    }
  }, [toolCallId, toolName, args, scriptLabel, setPanel, getEditedScript, setRightPanelOpen, setLeftSidebarOpen]);

  return (
    <Box sx={{ px: 1.5, py: 1.5, borderTop: '1px solid var(--tool-border)' }}>
      {toolName === 'sandbox_create' && args && (() => {
        const hw = String(args.hardware || 'cpu-basic');
        const cost = costLabel(hw);
        return (
          <Box sx={{ mb: 1.5 }}>
            <Typography variant="body2" sx={{ color: 'var(--muted-text)', fontSize: '0.75rem', mb: 0.5 }}>
              Create a remote dev environment on{' '}
              <Box component="span" sx={{ fontWeight: 500, color: 'var(--text)' }}>
                {hw}
              </Box>
              {cost && (
                <Box component="span" sx={{ color: cost === 'free' ? 'var(--accent-green)' : 'var(--accent-yellow)', fontWeight: 500 }}>
                  {' '}({cost})
                </Box>
              )}
              {!!args.private && (
                <Box component="span" sx={{ color: 'var(--muted-text)' }}>{' (private)'}</Box>
              )}
            </Typography>
            <Typography variant="body2" sx={{ color: 'var(--muted-text)', fontSize: '0.7rem', opacity: 0.7 }}>
              Creates a temporary HF Space to develop and test scripts before running jobs. Takes 1-2 min to start.
            </Typography>
          </Box>
        );
      })()}

      {toolName === 'hf_jobs' && args && (() => {
        const hw = String(args.hardware_flavor || 'cpu-basic');
        const cost = costLabel(hw);
        return (
        <Box sx={{ mb: 1.5 }}>
          <Typography variant="body2" sx={{ color: 'var(--muted-text)', fontSize: '0.75rem', mb: 1 }}>
            Execute <Box component="span" sx={{ color: 'var(--accent-yellow)', fontWeight: 500 }}>{scriptLabel.replace('Script', 'Job')}</Box> on{' '}
            <Box component="span" sx={{ fontWeight: 500, color: 'var(--text)' }}>
              {hw}
            </Box>
            {cost && (
              <Box component="span" sx={{ color: cost === 'free' ? 'var(--accent-green)' : 'var(--accent-yellow)', fontWeight: 500 }}>
                {' '}({cost})
              </Box>
            )}
            {!!args.timeout && (
              <> for up to <Box component="span" sx={{ fontWeight: 500, color: 'var(--text)' }}>
                {String(args.timeout)}
              </Box></>
            )}
          </Typography>
          {typeof args.script === 'string' && args.script && (
            <Box
              onClick={handleScriptClick}
              sx={{
                mt: 0.5,
                p: 1.5,
                bgcolor: 'var(--code-panel-bg)',
                border: '1px solid var(--tool-border)',
                borderRadius: '8px',
                cursor: 'pointer',
                transition: 'border-color 0.15s ease',
                '&:hover': { borderColor: 'var(--accent-yellow)' },
              }}
            >
              <Box
                component="pre"
                sx={{
                  m: 0,
                  fontFamily: '"JetBrains Mono", ui-monospace, SFMono-Regular, monospace',
                  fontSize: '0.7rem',
                  lineHeight: 1.5,
                  color: 'var(--text)',
                  overflow: 'hidden',
                  display: '-webkit-box',
                  WebkitLineClamp: 3,
                  WebkitBoxOrient: 'vertical',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-all',
                }}
              >
                {String(args.script).trim()}
              </Box>
              <Typography
                variant="caption"
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 0.5,
                  mt: 1,
                  fontSize: '0.65rem',
                  color: 'var(--muted-text)',
                  '&:hover': { color: 'var(--accent-yellow)' },
                }}
              >
                Click to view & edit
              </Typography>
            </Box>
          )}
        </Box>
        );
      })()}

      <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
        <TextField
          fullWidth
          size="small"
          placeholder="Feedback (optional)"
          value={feedback}
          onChange={(e) => setFeedback(e.target.value)}
          variant="outlined"
          sx={{
            '& .MuiOutlinedInput-root': {
              bgcolor: 'var(--hover-bg)',
              fontFamily: 'inherit',
              fontSize: '0.8rem',
              '& fieldset': { borderColor: 'var(--tool-border)' },
              '&:hover fieldset': { borderColor: 'var(--border-hover)' },
              '&.Mui-focused fieldset': { borderColor: 'var(--accent-yellow)' },
            },
            '& .MuiOutlinedInput-input': {
              color: 'var(--text)',
              '&::placeholder': { color: 'var(--muted-text)', opacity: 0.7 },
            },
          }}
        />
        <IconButton
          onClick={() => onResolve(toolCallId, false, feedback || 'Rejected by user')}
          disabled={!feedback}
          size="small"
          sx={{
            color: 'var(--accent-red)',
            border: '1px solid var(--tool-border)',
            borderRadius: '6px',
            '&:hover': { bgcolor: 'rgba(224,90,79,0.1)', borderColor: 'var(--accent-red)' },
            '&.Mui-disabled': { color: 'var(--muted-text)', opacity: 0.3 },
          }}
        >
          <SendIcon sx={{ fontSize: 14 }} />
        </IconButton>
      </Box>

      <Box sx={{ display: 'flex', gap: 1 }}>
        <Button
          size="small"
          onClick={() => onResolve(toolCallId, false, feedback || 'Rejected by user')}
          sx={{
            flex: 1,
            textTransform: 'none',
            border: '1px solid rgba(255,255,255,0.05)',
            color: 'var(--accent-red)',
            fontSize: '0.75rem',
            py: 0.75,
            borderRadius: '8px',
            '&:hover': { bgcolor: 'rgba(224,90,79,0.05)', borderColor: 'var(--accent-red)' },
          }}
        >
          Reject
        </Button>
        <Button
          size="small"
          onClick={() => onResolve(toolCallId, true)}
          sx={{
            flex: 1,
            textTransform: 'none',
            border: hasEditedScript ? '1px solid var(--accent-green)' : '1px solid rgba(255,255,255,0.05)',
            color: 'var(--accent-green)',
            fontSize: '0.75rem',
            py: 0.75,
            borderRadius: '8px',
            bgcolor: hasEditedScript ? 'rgba(47,204,113,0.08)' : 'transparent',
            '&:hover': { bgcolor: 'rgba(47,204,113,0.05)', borderColor: 'var(--accent-green)' },
          }}
        >
          {hasEditedScript ? 'Approve (edited)' : 'Approve'}
        </Button>
      </Box>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function ToolCallGroup({ tools, approveTools }: ToolCallGroupProps) {
  const { setPanel, lockPanel, getJobUrl, getEditedScript, setJobStatus, getJobStatus, setToolError, getToolError, setToolRejected, getToolRejected } = useAgentStore();
  const researchAgents = useAgentStore(s => {
    const activeId = s.activeSessionId;
    return activeId ? (s.sessionStates[activeId]?.researchAgents) : undefined;
  }) ?? {};
  // Aggregate stats across all agents for the chip label
  const aggregateStats = useMemo(() => {
    const agents = Object.values(researchAgents);
    if (agents.length === 0) return { toolCount: 0, tokenCount: 0, startedAt: null, finalElapsed: null };
    const running = agents.filter(a => a.stats.startedAt !== null);
    return {
      toolCount: agents.reduce((sum, a) => sum + a.stats.toolCount, 0),
      tokenCount: agents.reduce((sum, a) => sum + a.stats.tokenCount, 0),
      startedAt: running.length > 0 ? Math.min(...running.map(a => a.stats.startedAt!)) : null,
      finalElapsed: running.length === 0 && agents.length > 0
        ? Math.max(...agents.map(a => a.stats.finalElapsed ?? 0))
        : null,
    };
  }, [researchAgents]);
  const researchStats = aggregateStats;
  const liveElapsed = useElapsed(researchStats.startedAt);
  const isProcessing = useAgentStore(s => s.isProcessing);
  const { setRightPanelOpen, setLeftSidebarOpen } = useLayoutStore();

  // ── Batch approval state ──────────────────────────────────────────
  const pendingTools = useMemo(
    () => tools.filter(t => t.state === 'approval-requested'),
    [tools],
  );

  const [decisions, setDecisions] = useState<Record<string, { approved: boolean; feedback?: string }>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const submittingRef = useRef(false);

  // Track which toolCallIds we've already submitted so we can detect new approval rounds
  const submittedIdsRef = useRef<Set<string>>(new Set());

  // ── Panel lock state (for auto-follow vs user-selected) ───────────
  const [lockedToolId, setLockedToolId] = useState<string | null>(null);

  // Reset submission state when new (unseen) pending tools arrive — e.g. second approval round
  useEffect(() => {
    if (!isSubmitting || pendingTools.length === 0) return;
    const hasNewPending = pendingTools.some(t => !submittedIdsRef.current.has(t.toolCallId));
    if (hasNewPending) {
      submittingRef.current = false;
      setIsSubmitting(false);
      setDecisions({});
    }
  }, [pendingTools, isSubmitting]);

  // Clean up stale decisions for tools that are no longer pending
  useEffect(() => {
    const pendingIds = new Set(pendingTools.map(t => t.toolCallId));
    const decisionIds = Object.keys(decisions);
    const hasStale = decisionIds.some(id => !pendingIds.has(id));
    if (hasStale) {
      setDecisions(prev => {
        const cleaned = { ...prev };
        for (const id of decisionIds) {
          if (!pendingIds.has(id)) delete cleaned[id];
        }
        return cleaned;
      });
    }
  }, [pendingTools, decisions]);

  // Persist error states when tools error
  useEffect(() => {
    for (const tool of tools) {
      const currentlyHasError = tool.state === 'output-error';
      const persistedError = getToolError(tool.toolCallId);

      // Persist error state if we detect it and haven't already
      if (currentlyHasError && !persistedError) {
        setToolError(tool.toolCallId, true);
      }
    }
  }, [tools, setToolError, getToolError]);

  const { scriptLabelMap, toolDisplayMap } = useMemo(() => {
    const hfJobs = tools.filter(t => t.toolName === 'hf_jobs' && (t.input as Record<string, unknown>)?.script);
    const scriptMap: Record<string, string> = {};
    const displayMap: Record<string, string> = {};
    for (let i = 0; i < hfJobs.length; i++) {
      const id = hfJobs[i].toolCallId;
      if (hfJobs.length > 1) {
        scriptMap[id] = `Script ${i + 1}`;
        displayMap[id] = `hf_jobs #${i + 1}`;
      } else {
        scriptMap[id] = 'Script';
        displayMap[id] = 'hf_jobs';
      }
    }
    // Pretty name for research tool
    for (const t of tools) {
      if (t.toolName === 'research') {
        displayMap[t.toolCallId] = 'research';
      }
    }
    return { scriptLabelMap: scriptMap, toolDisplayMap: displayMap };
  }, [tools]);

  // ── Send all decisions as a single batch ──────────────────────────
  const sendBatch = useCallback(
    async (batch: Record<string, { approved: boolean; feedback?: string }>) => {
      if (submittingRef.current) return;
      submittingRef.current = true;
      setIsSubmitting(true);

      const approvals = Object.entries(batch).map(([toolCallId, d]) => {
        const editedScript = d.approved ? (getEditedScript(toolCallId) ?? null) : null;
        if (editedScript) {
          logger.log(`Sending edited script for ${toolCallId} (${editedScript.length} chars)`);
        }
        // Mark tool as rejected if not approved
        if (!d.approved) {
          setToolRejected(toolCallId, true);
        }
        return {
          tool_call_id: toolCallId,
          approved: d.approved,
          feedback: d.approved ? null : (d.feedback || 'Rejected by user'),
          edited_script: editedScript,
        };
      });

      const ok = await approveTools(approvals);
      if (ok) {
        // Track which tool IDs were submitted so we can detect new approval rounds
        for (const a of approvals) submittedIdsRef.current.add(a.tool_call_id);
        lockPanel();
      } else {
        logger.error('Batch approval failed');
        submittingRef.current = false;
        setIsSubmitting(false);
      }
    },
    [approveTools, lockPanel, getEditedScript, setToolRejected],
  );

  const handleApproveAll = useCallback(() => {
    const batch: Record<string, { approved: boolean }> = {};
    for (const t of pendingTools) batch[t.toolCallId] = { approved: true };
    sendBatch(batch);
  }, [pendingTools, sendBatch]);

  const handleRejectAll = useCallback(() => {
    const batch: Record<string, { approved: boolean }> = {};
    for (const t of pendingTools) batch[t.toolCallId] = { approved: false };
    sendBatch(batch);
  }, [pendingTools, sendBatch]);

  const handleIndividualDecision = useCallback(
    (toolCallId: string, approved: boolean, feedback?: string) => {
      setDecisions(prev => {
        const next = { ...prev, [toolCallId]: { approved, feedback } };
        if (pendingTools.every(t => next[t.toolCallId])) {
          queueMicrotask(() => sendBatch(next));
        }
        return next;
      });
    },
    [pendingTools, sendBatch],
  );

  const undoDecision = useCallback((toolCallId: string) => {
    setDecisions(prev => {
      const next = { ...prev };
      delete next[toolCallId];
      return next;
    });
  }, []);

  // ── Show tool panel (shared logic) ────────────────────────────────
  const showToolPanel = useCallback(
    (tool: DynamicToolPart) => {
      const args = tool.input as Record<string, unknown> | undefined;
      const displayName = toolDisplayMap[tool.toolCallId] || tool.toolName;

      if (tool.toolName === 'hf_jobs' && args?.script) {
        const jobOutput = tool.output ?? (tool.state === 'output-error' ? (tool as Record<string, unknown>).errorText : undefined);
        const hasOutput = (tool.state === 'output-available' || tool.state === 'output-error') && jobOutput;
        const scriptContent = getEditedScript(tool.toolCallId) || String(args.script);
        setPanel(
          {
            title: displayName,
            script: { content: scriptContent, language: 'python' },
            ...(hasOutput ? { output: { content: String(jobOutput), language: 'markdown' } } : {}),
            parameters: { tool_call_id: tool.toolCallId },
          },
          hasOutput ? 'output' : 'script',
        );
        setRightPanelOpen(true);
        setLeftSidebarOpen(false);
        return;
      }

      const inputSection = args ? { content: JSON.stringify(args, null, 2), language: 'json' } : undefined;

      const outputText = tool.output ?? (tool.state === 'output-error' ? (tool as Record<string, unknown>).errorText : undefined);

      const hasCompleted = tool.state === 'output-available' || tool.state === 'output-error' || tool.state === 'output-denied';

      if (outputText) {
        // Tool has output - show it (regardless of state)
        let language = 'text';
        const content = String(outputText);
        if (content.trim().startsWith('{') || content.trim().startsWith('[')) language = 'json';
        else if (content.includes('```')) language = 'markdown';

        setPanel({ title: displayName, output: { content, language }, input: inputSection }, 'output');
        setRightPanelOpen(true);
      } else if (tool.state === 'output-error') {
        const content = `Tool \`${tool.toolName}\` returned an error with no output message.`;
        setPanel({ title: displayName, output: { content, language: 'markdown' }, input: inputSection }, 'output');
        setRightPanelOpen(true);
      } else if (hasCompleted && args) {
        // Tool completed but has no output - show input as fallback
        setPanel({ title: displayName, output: { content: JSON.stringify(args, null, 2), language: 'json' }, input: inputSection }, 'output');
        setRightPanelOpen(true);
      } else if (args) {
        const runningMessages = [
          'Crunching numbers and herding tensors...',
          'Teaching the model some new tricks...',
          'Consulting the GPU oracle...',
          'Wrangling data into submission...',
          'Brewing a fresh batch of predictions...',
          'Negotiating with the transformer heads...',
          'Polishing the attention weights...',
          'Aligning the embedding stars...',
        ];
        const funMsg = runningMessages[Math.floor(Math.random() * runningMessages.length)];
        setPanel({ title: displayName, output: { content: funMsg, language: 'text' }, input: inputSection }, 'output');
        setRightPanelOpen(true);
      }
    },
    [toolDisplayMap, setPanel, getEditedScript, setRightPanelOpen, setLeftSidebarOpen],
  );

  // ── Panel click handler ───────────────────────────────────────────
  const handleClick = useCallback(
    (tool: DynamicToolPart) => {
      // Toggle lock: if clicking the same tool that's already locked, unlock it
      if (lockedToolId === tool.toolCallId) {
        setLockedToolId(null);
        return;
      }

      // Lock this tool
      setLockedToolId(tool.toolCallId);

      // Show the panel
      showToolPanel(tool);
    },
    [lockedToolId, showToolPanel],
  );

  // ── Auto-follow currently active tool when not locked ─────────────
  const activeToolIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (lockedToolId !== null) return; // User has locked a tool, don't auto-follow

    // Find the currently running tool (latest tool that's in progress)
    const runningTool = tools.slice().reverse().find(t =>
      t.state === 'input-available' ||
      t.state === 'input-streaming' ||
      t.state === 'approval-responded'
    );

    if (runningTool) {
      // Track this as the active tool and show its panel
      activeToolIdRef.current = runningTool.toolCallId;
      showToolPanel(runningTool);
    } else if (activeToolIdRef.current) {
      // No running tool, but we were following one - check if it completed
      const completedTool = tools.find(t => t.toolCallId === activeToolIdRef.current);
      if (completedTool && (completedTool.state === 'output-available' || completedTool.state === 'output-error')) {
        // The tool we were following has completed - update its panel
        showToolPanel(completedTool);
      }
    }
  }, [tools, lockedToolId, showToolPanel]);

  // ── Parse hf_jobs metadata from output ────────────────────────────
  function parseJobMeta(output: unknown): { jobUrl?: string; jobStatus?: string } {
    if (typeof output !== 'string') return {};
    const urlMatch = output.match(/\*\*View at:\*\*\s*(https:\/\/[^\s\n]+)/);
    const statusMatch = output.match(/\*\*Final Status:\*\*\s*([^\n]+)/);
    return {
      jobUrl: urlMatch?.[1],
      jobStatus: statusMatch?.[1]?.trim(),
    };
  }

  // ── Render ────────────────────────────────────────────────────────
  const decidedCount = pendingTools.filter(t => decisions[t.toolCallId]).length;

  return (
    <Box
      sx={{
        borderRadius: 2,
        border: '1px solid var(--tool-border)',
        bgcolor: 'var(--tool-bg)',
        overflow: 'hidden',
        my: 1,
      }}
    >
      {/* Batch approval header — hidden once user starts deciding individually */}
      {pendingTools.length > 1 && !isSubmitting && decidedCount === 0 && (
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            px: 1.5,
            py: 1,
            borderBottom: '1px solid var(--tool-border)',
          }}
        >
          <Typography
            variant="body2"
            sx={{ fontSize: '0.72rem', color: 'var(--muted-text)', mr: 'auto', whiteSpace: 'nowrap' }}
          >
            {`${pendingTools.length} tool${pendingTools.length > 1 ? 's' : ''} pending`}
          </Typography>
          <Button
            size="small"
            onClick={handleRejectAll}
            sx={{
              textTransform: 'none',
              color: 'var(--accent-red)',
              border: '1px solid rgba(255,255,255,0.05)',
              fontSize: '0.72rem',
              py: 0.5,
              px: 1.5,
              borderRadius: '8px',
              '&:hover': { bgcolor: 'rgba(224,90,79,0.05)', borderColor: 'var(--accent-red)' },
            }}
          >
            Reject all
          </Button>
          <Button
            size="small"
            onClick={handleApproveAll}
            sx={{
              textTransform: 'none',
              color: 'var(--accent-green)',
              border: '1px solid var(--accent-green)',
              fontSize: '0.72rem',
              fontWeight: 600,
              py: 0.5,
              px: 1.5,
              borderRadius: '8px',
              '&:hover': { bgcolor: 'rgba(47,204,113,0.1)' },
            }}
          >
            Approve all{pendingTools.length > 1 ? ` (${pendingTools.length})` : ''}
          </Button>
        </Box>
      )}

      {/* Tool list */}
      <Stack divider={<Box sx={{ borderBottom: '1px solid var(--tool-border)' }} />}>
        {tools.map((tool) => {
          const state = tool.state;
          const isPending = state === 'approval-requested';
          const clickable =
            state === 'output-available' ||
            state === 'output-error' ||
            !!tool.input ||
            (!isProcessing && (state === 'input-available' || state === 'input-streaming'));
          const localDecision = decisions[tool.toolCallId];

          const cancelled = isCancelledTool(tool);
          const currentlyHasError = state === 'output-error';
          const persistedError = getToolError(tool.toolCallId);
          const persistedRejection = getToolRejected(tool.toolCallId);

          // Stale in-progress tools after page reload: treat as completed
          const stale = !isProcessing && (state === 'input-available' || state === 'input-streaming');
          const displayState = stale ? 'output-available'
            : isPending && localDecision
              ? (localDecision.approved ? 'input-available' : 'output-denied')
              : state;
          const isRejected = displayState === 'output-denied' || persistedRejection;
          const hasError = (persistedError || currentlyHasError) && !isRejected;
          const label = cancelled ? 'cancelled'
            : isRejected ? 'rejected'
            : hasError ? 'error'
            : statusLabel(displayState as ToolPartState);

          // Parse job metadata from hf_jobs output and store
          const jobUrlFromStore = tool.toolName === 'hf_jobs' ? getJobUrl(tool.toolCallId) : undefined;
          const jobStatusFromStore = tool.toolName === 'hf_jobs' ? getJobStatus(tool.toolCallId) : undefined;

          const jobMetaFromOutput = tool.toolName === 'hf_jobs' && (tool.output || (tool as Record<string, unknown>).errorText)
            ? parseJobMeta(tool.output ?? (tool as Record<string, unknown>).errorText)
            : {};

          // Store job status if we just parsed it and don't have it stored yet
          if (tool.toolName === 'hf_jobs' && jobMetaFromOutput.jobStatus && !jobStatusFromStore) {
            setJobStatus(tool.toolCallId, jobMetaFromOutput.jobStatus);
          }

          // Combine job URL and status from store (persisted) with output metadata (freshly parsed)
          // Prefer stored values to ensure they persist across renders
          const jobMeta = {
            jobUrl: jobUrlFromStore || jobMetaFromOutput.jobUrl,
            jobStatus: jobStatusFromStore || jobMetaFromOutput.jobStatus,
          };

          return (
            <Box key={tool.toolCallId}>
              {/* Main tool row */}
              <Stack
                direction="row"
                alignItems="center"
                spacing={1}
                onClick={() => !isPending && handleClick(tool)}
                sx={{
                  px: 1.5,
                  py: 1,
                  cursor: isPending ? 'default' : clickable ? 'pointer' : 'default',
                  transition: 'background-color 0.15s',
                  bgcolor: lockedToolId === tool.toolCallId ? 'var(--hover-bg)' : 'transparent',
                  borderLeft: lockedToolId === tool.toolCallId ? '3px solid var(--accent-yellow)' : '3px solid transparent',
                  '&:hover': clickable && !isPending ? { bgcolor: 'var(--hover-bg)' } : {},
                }}
              >
                <StatusIcon
                  cancelled={cancelled}
                  isRejected={isRejected}
                  state={
                    hasError
                      ? 'output-error'
                      : ((tool.toolName === 'hf_jobs' && jobMeta.jobStatus && ['ERROR', 'FAILED', 'CANCELLED'].includes(jobMeta.jobStatus))
                        ? 'output-error'
                        : displayState as ToolPartState)
                  }
                />

                <Typography
                  variant="body2"
                  sx={{
                    fontFamily: '"JetBrains Mono", ui-monospace, SFMono-Regular, monospace',
                    fontWeight: 600,
                    fontSize: '0.78rem',
                    color: 'var(--text)',
                    flex: 1,
                    minWidth: 0,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {toolDisplayMap[tool.toolCallId] || tool.toolName}
                </Typography>

                {/* Status chip (non hf_jobs, or hf_jobs without final status) */}
                {(() => {
                  // Research tool: override chip label with live stats (but not if cancelled/done)
                  const researchDone = cancelled || state === 'output-available' || state === 'output-error' || state === 'output-denied';
                  const researchLabel = tool.toolName === 'research' && !researchDone
                    ? researchChipLabel(researchStats, liveElapsed)
                    : (tool.toolName === 'research' && researchDone && researchStats.finalElapsed !== null)
                      ? researchChipLabel({ ...researchStats, startedAt: null }, null)
                      : null;
                  const chipLabel = researchLabel || label;
                  if (!chipLabel || (tool.toolName === 'hf_jobs' && jobMeta.jobStatus)) return null;

                  return (
                    <Chip
                      label={chipLabel}
                      size="small"
                      sx={{
                        height: 20,
                        fontSize: '0.65rem',
                        fontWeight: 600,
                        bgcolor: (cancelled || isRejected) ? 'rgba(255,255,255,0.05)'
                          : hasError ? 'rgba(224,90,79,0.12)'
                          : (researchLabel && displayState === 'output-available') ? 'rgba(47,204,113,0.12)'
                          : 'var(--accent-yellow-weak)',
                        color: (cancelled || isRejected) ? 'var(--muted-text)'
                          : hasError ? 'var(--accent-red)'
                          : statusColor(displayState as ToolPartState),
                        letterSpacing: '0.03em',
                      }}
                    />
                  );
                })()}

                {/* HF Jobs: final status chip from job metadata */}
                {tool.toolName === 'hf_jobs' && jobMeta.jobStatus && (
                  <Chip
                    label={jobMeta.jobStatus}
                    size="small"
                    sx={{
                      height: 20,
                      fontSize: '0.65rem',
                      fontWeight: 600,
                      bgcolor: jobMeta.jobStatus === 'COMPLETED'
                        ? 'rgba(47,204,113,0.12)'
                        : ['ERROR', 'FAILED', 'CANCELLED'].includes(jobMeta.jobStatus!)
                          ? 'rgba(224,90,79,0.12)'
                          : 'rgba(255,193,59,0.12)',
                      color: jobMeta.jobStatus === 'COMPLETED'
                        ? 'var(--accent-green)'
                        : ['ERROR', 'FAILED', 'CANCELLED'].includes(jobMeta.jobStatus!)
                          ? 'var(--accent-red)'
                          : 'var(--accent-yellow)',
                      letterSpacing: '0.03em',
                    }}
                  />
                )}

                {/* View on HF link — single place, shown whenever URL is available */}
                {tool.toolName === 'hf_jobs' && jobMeta.jobUrl && (
                  <Link
                    href={jobMeta.jobUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                    sx={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 0.5,
                      color: 'var(--accent-yellow)',
                      fontSize: '0.68rem',
                      textDecoration: 'none',
                      ml: 0.5,
                      '&:hover': { textDecoration: 'underline' },
                    }}
                  >
                    <LaunchIcon sx={{ fontSize: 12 }} />
                    View on HF
                  </Link>
                )}

                {clickable && !isPending && (
                  <OpenInNewIcon sx={{ fontSize: 14, color: 'var(--muted-text)', opacity: 0.6 }} />
                )}
              </Stack>

              {/* Research sub-agent rolling steps (visible only while running) */}
              {tool.toolName === 'research' && !cancelled && state !== 'output-available' && state !== 'output-error' && state !== 'output-denied' && (
                <ResearchSteps agents={researchAgents} />
              )}

              {/* Per-tool approval: undecided */}
              {isPending && !localDecision && !isSubmitting && (
                <InlineApproval
                  toolCallId={tool.toolCallId}
                  toolName={tool.toolName}
                  input={tool.input}
                  scriptLabel={scriptLabelMap[tool.toolCallId] || 'Script'}
                  onResolve={handleIndividualDecision}
                />
              )}

              {/* Per-tool approval: locally decided (undo available) */}
              {isPending && localDecision && !isSubmitting && (
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    px: 1.5,
                    py: 0.75,
                    borderTop: '1px solid var(--tool-border)',
                  }}
                >
                  <Typography variant="body2" sx={{ fontSize: '0.72rem', color: 'var(--muted-text)' }}>
                    {localDecision.approved
                      ? 'Marked for approval'
                      : `Marked for rejection${localDecision.feedback ? `: ${localDecision.feedback}` : ''}`}
                  </Typography>
                  <Button
                    size="small"
                    onClick={() => undoDecision(tool.toolCallId)}
                    sx={{
                      textTransform: 'none',
                      fontSize: '0.7rem',
                      color: 'var(--muted-text)',
                      minWidth: 'auto',
                      px: 1,
                      '&:hover': { color: 'var(--text)' },
                    }}
                  >
                    Undo
                  </Button>
                </Box>
              )}
            </Box>
          );
        })}
      </Stack>
    </Box>
  );
}
