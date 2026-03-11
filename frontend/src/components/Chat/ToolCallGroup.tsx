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
import type { UIMessage } from 'ai';

// ---------------------------------------------------------------------------
// Type helpers — extract the dynamic-tool part type from UIMessage
// ---------------------------------------------------------------------------
type DynamicToolPart = Extract<UIMessage['parts'][number], { type: 'dynamic-tool' }>;

type ToolPartState = DynamicToolPart['state'];

interface ToolCallGroupProps {
  tools: DynamicToolPart[];
  approveTools: (approvals: Array<{ tool_call_id: string; approved: boolean; feedback?: string | null; edited_script?: string | null }>) => Promise<boolean>;
}

// ---------------------------------------------------------------------------
// Visual helpers
// ---------------------------------------------------------------------------

function StatusIcon({ state }: { state: ToolPartState }) {
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
      {toolName === 'sandbox_create' && args && (
        <Box sx={{ mb: 1.5 }}>
          <Typography variant="body2" sx={{ color: 'var(--muted-text)', fontSize: '0.75rem', mb: 1 }}>
            Create sandbox on{' '}
            <Box component="span" sx={{ fontWeight: 500, color: 'var(--text)' }}>
              {String(args.hardware || 'cpu-basic')}
            </Box>
            {!!args.private && (
              <Box component="span" sx={{ color: 'var(--muted-text)' }}>{' (private)'}</Box>
            )}
          </Typography>
        </Box>
      )}

      {toolName === 'hf_jobs' && args && (
        <Box sx={{ mb: 1.5 }}>
          <Typography variant="body2" sx={{ color: 'var(--muted-text)', fontSize: '0.75rem', mb: 1 }}>
            Execute <Box component="span" sx={{ color: 'var(--accent-yellow)', fontWeight: 500 }}>{scriptLabel.replace('Script', 'Job')}</Box> on{' '}
            <Box component="span" sx={{ fontWeight: 500, color: 'var(--text)' }}>
              {String(args.hardware_flavor || 'default')}
            </Box>
            {!!args.timeout && (
              <> with timeout <Box component="span" sx={{ fontWeight: 500, color: 'var(--text)' }}>
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
      )}

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
  const { setPanel, lockPanel, getJobUrl, getEditedScript } = useAgentStore();
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
    [approveTools, lockPanel, getEditedScript],
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

  // ── Panel click handler ───────────────────────────────────────────
  const handleClick = useCallback(
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
      if ((tool.state === 'output-available' || tool.state === 'output-error') && outputText) {
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
      } else if (args) {
        setPanel({ title: displayName, output: { content: JSON.stringify(args, null, 2), language: 'json' }, input: inputSection }, 'output');
        setRightPanelOpen(true);
      }
    },
    [toolDisplayMap, setPanel, getEditedScript, setRightPanelOpen, setLeftSidebarOpen],
  );

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
            !!tool.input;
          const localDecision = decisions[tool.toolCallId];

          const displayState = isPending && localDecision
            ? (localDecision.approved ? 'input-available' : 'output-denied')
            : state;
          const label = statusLabel(displayState as ToolPartState);

          // Parse job metadata from hf_jobs output and store
          const jobUrlFromStore = tool.toolName === 'hf_jobs' ? getJobUrl(tool.toolCallId) : undefined;
          const jobMetaFromOutput = tool.toolName === 'hf_jobs' && tool.state === 'output-available'
            ? parseJobMeta(tool.output)
            : {};
          
          // Combine job URL from store (available immediately) with output metadata (available at completion)
          const jobMeta = {
            jobUrl: jobUrlFromStore || jobMetaFromOutput.jobUrl,
            jobStatus: jobMetaFromOutput.jobStatus,
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
                  '&:hover': clickable && !isPending ? { bgcolor: 'var(--hover-bg)' } : {},
                }}
              >
                <StatusIcon state={
                  (tool.toolName === 'hf_jobs' && jobMeta.jobStatus && ['ERROR', 'FAILED', 'CANCELLED'].includes(jobMeta.jobStatus) && displayState === 'output-available')
                    ? 'output-error'
                    : displayState as ToolPartState
                } />

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
                {label && !(tool.toolName === 'hf_jobs' && jobMeta.jobStatus) && (
                  <Chip
                    label={label}
                    size="small"
                    sx={{
                      height: 20,
                      fontSize: '0.65rem',
                      fontWeight: 600,
                      bgcolor: displayState === 'output-error' ? 'rgba(224,90,79,0.12)'
                        : displayState === 'output-denied' ? 'rgba(255,255,255,0.05)'
                        : 'var(--accent-yellow-weak)',
                      color: statusColor(displayState as ToolPartState),
                      letterSpacing: '0.03em',
                    }}
                  />
                )}

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
