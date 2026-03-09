import { useRef, useEffect, useMemo, useState, useCallback } from 'react';
import { Box, Stack, Typography, IconButton, Button, Tooltip } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import RadioButtonUncheckedIcon from '@mui/icons-material/RadioButtonUnchecked';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import PlayCircleOutlineIcon from '@mui/icons-material/PlayCircleOutline';
import CodeIcon from '@mui/icons-material/Code';
import ArticleIcon from '@mui/icons-material/Article';
import EditIcon from '@mui/icons-material/Edit';
import UndoIcon from '@mui/icons-material/Undo';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import CheckIcon from '@mui/icons-material/Check';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus, vs } from 'react-syntax-highlighter/dist/esm/styles/prism';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useAgentStore } from '@/store/agentStore';
import { useLayoutStore } from '@/store/layoutStore';
import { processLogs } from '@/utils/logProcessor';
import type { PanelView } from '@/store/agentStore';

// ── Helpers ──────────────────────────────────────────────────────

function PlanStatusIcon({ status }: { status: string }) {
  if (status === 'completed') return <CheckCircleIcon sx={{ fontSize: 16, color: 'var(--accent-green)' }} />;
  if (status === 'in_progress') return <PlayCircleOutlineIcon sx={{ fontSize: 16, color: 'var(--accent-yellow)' }} />;
  return <RadioButtonUncheckedIcon sx={{ fontSize: 16, color: 'var(--muted-text)', opacity: 0.5 }} />;
}

// ── Markdown styles (adapts via CSS vars) ────────────────────────
const markdownSx = {
  color: 'var(--text)',
  fontSize: '13px',
  lineHeight: 1.6,
  '& p': { m: 0, mb: 1.5, '&:last-child': { mb: 0 } },
  '& pre': {
    bgcolor: 'var(--code-bg)',
    p: 1.5,
    borderRadius: 1,
    overflow: 'auto',
    fontSize: '12px',
    border: '1px solid var(--tool-border)',
  },
  '& code': {
    bgcolor: 'var(--hover-bg)',
    px: 0.5,
    py: 0.25,
    borderRadius: 0.5,
    fontSize: '12px',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, monospace',
  },
  '& pre code': { bgcolor: 'transparent', p: 0 },
  '& a': {
    color: 'var(--accent-yellow)',
    textDecoration: 'none',
    '&:hover': { textDecoration: 'underline' },
  },
  '& ul, & ol': { pl: 2.5, my: 1 },
  '& li': { mb: 0.5 },
  '& table': {
    borderCollapse: 'collapse',
    width: '100%',
    my: 2,
    fontSize: '12px',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, monospace',
  },
  '& th': {
    borderBottom: '2px solid var(--border-hover)',
    textAlign: 'left',
    p: 1,
    fontWeight: 600,
  },
  '& td': {
    borderBottom: '1px solid var(--tool-border)',
    p: 1,
  },
  '& h1, & h2, & h3, & h4': { mt: 2, mb: 1, fontWeight: 600 },
  '& h1': { fontSize: '1.25rem' },
  '& h2': { fontSize: '1.1rem' },
  '& h3': { fontSize: '1rem' },
  '& blockquote': {
    borderLeft: '3px solid var(--accent-yellow)',
    pl: 2,
    ml: 0,
    color: 'var(--muted-text)',
  },
} as const;

// ── View toggle button ──────────────────────────────────────────

function ViewToggle({ view, icon, label, isActive, onClick }: {
  view: PanelView;
  icon: React.ReactNode;
  label: string;
  isActive: boolean;
  onClick: (v: PanelView) => void;
}) {
  return (
    <Box
      onClick={() => onClick(view)}
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 0.5,
        px: 1.5,
        py: 0.75,
        borderRadius: 1,
        cursor: 'pointer',
        fontSize: '0.7rem',
        fontWeight: 600,
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
        whiteSpace: 'nowrap',
        color: isActive ? 'var(--text)' : 'var(--muted-text)',
        bgcolor: isActive ? 'var(--tab-active-bg)' : 'transparent',
        border: '1px solid',
        borderColor: isActive ? 'var(--tab-active-border)' : 'transparent',
        transition: 'all 0.15s ease',
        '&:hover': { bgcolor: 'var(--tab-hover-bg)' },
      }}
    >
      {icon}
      <span>{label}</span>
    </Box>
  );
}

// ── Component ────────────────────────────────────────────────────

export default function CodePanel() {
  const { panelData, panelView, panelEditable, setPanelView, updatePanelScript, setEditedScript, plan } =
    useAgentStore();
  const { setRightPanelOpen, themeMode } = useLayoutStore();
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editedContent, setEditedContent] = useState('');
  const [originalContent, setOriginalContent] = useState('');
  const [copied, setCopied] = useState(false);
  const [showInput, setShowInput] = useState(false);

  const isDark = themeMode === 'dark';
  const syntaxTheme = isDark ? vscDarkPlus : vs;

  const activeSection = panelView === 'script' ? panelData?.script : panelData?.output;
  const hasScript = !!panelData?.script;
  const hasOutput = !!panelData?.output;
  const hasBothViews = hasScript && hasOutput;

  const isEditableScript = panelView === 'script' && panelEditable;
  const hasUnsavedChanges = isEditing && editedContent !== originalContent;

  // Reset input toggle when panel data changes
  useEffect(() => {
    setShowInput(false);
  }, [panelData]);

  // Sync edited content when panel data changes
  useEffect(() => {
    if (panelData?.script?.content && panelView === 'script' && panelEditable) {
      setOriginalContent(panelData.script.content);
      if (!isEditing) {
        setEditedContent(panelData.script.content);
      }
    }
  }, [panelData?.script?.content, panelView, panelEditable, isEditing]);

  // Exit editing when switching away from script view or losing editable
  useEffect(() => {
    if (!isEditableScript && isEditing) {
      setIsEditing(false);
    }
  }, [isEditableScript, isEditing]);

  const handleStartEdit = useCallback(() => {
    if (panelData?.script?.content) {
      setEditedContent(panelData.script.content);
      setOriginalContent(panelData.script.content);
      setIsEditing(true);
      setTimeout(() => textareaRef.current?.focus(), 0);
    }
  }, [panelData?.script?.content]);

  const handleCancelEdit = useCallback(() => {
    setEditedContent(originalContent);
    setIsEditing(false);
  }, [originalContent]);

  const handleSaveEdit = useCallback(() => {
    if (editedContent !== originalContent) {
      updatePanelScript(editedContent);
      const toolCallId = panelData?.parameters?.tool_call_id as string | undefined;
      if (toolCallId) {
        setEditedScript(toolCallId, editedContent);
      }
      setOriginalContent(editedContent);
    }
    setIsEditing(false);
  }, [panelData?.parameters?.tool_call_id, editedContent, originalContent, updatePanelScript, setEditedScript]);

  const handleCopy = useCallback(async () => {
    const contentToCopy = isEditing ? editedContent : (activeSection?.content || '');
    if (contentToCopy) {
      try {
        await navigator.clipboard.writeText(contentToCopy);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch (err) {
        console.error('Failed to copy:', err);
      }
    }
  }, [isEditing, editedContent, activeSection?.content]);

  const visibleSection = (showInput && panelData?.input) ? panelData.input : activeSection;

  const displayContent = useMemo(() => {
    if (!visibleSection?.content) return '';
    if (!visibleSection.language || visibleSection.language === 'text') {
      return processLogs(visibleSection.content);
    }
    return visibleSection.content;
  }, [visibleSection?.content, visibleSection?.language]);

  // Auto-scroll only for live log streaming, not when opening panel
  const hasAutoScrolled = useRef(false);
  useEffect(() => {
    hasAutoScrolled.current = false;
  }, [panelData]);
  useEffect(() => {
    if (scrollRef.current && panelView === 'output' && hasAutoScrolled.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
    hasAutoScrolled.current = true;
  }, [displayContent, panelView]);

  // ── Syntax-highlighted code block (DRY) ────────────────────────
  const renderSyntaxBlock = (language: string) => (
    <SyntaxHighlighter
      language={language}
      style={syntaxTheme}
      customStyle={{
        margin: 0,
        padding: 0,
        background: 'transparent',
        fontSize: '13px',
        fontFamily: 'inherit',
      }}
      wrapLines
      wrapLongLines
    >
      {displayContent}
    </SyntaxHighlighter>
  );

  // ── Content renderer ───────────────────────────────────────────
  const renderContent = () => {
    if (!visibleSection?.content) {
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', opacity: 0.5 }}>
          <Typography variant="caption">NO CONTENT TO DISPLAY</Typography>
        </Box>
      );
    }

    if (!showInput && isEditing && isEditableScript) {
      return (
        <Box sx={{ position: 'relative', width: '100%', height: '100%' }}>
          <SyntaxHighlighter
            language={activeSection?.language === 'python' ? 'python' : activeSection?.language === 'json' ? 'json' : 'text'}
            style={syntaxTheme}
            customStyle={{
              margin: 0,
              padding: 0,
              background: 'transparent',
              fontSize: '13px',
              fontFamily: '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, monospace',
              lineHeight: 1.55,
              pointerEvents: 'none',
            }}
            wrapLines
            wrapLongLines
          >
            {editedContent || ' '}
          </SyntaxHighlighter>
          <textarea
            ref={textareaRef}
            value={editedContent}
            onChange={(e) => setEditedContent(e.target.value)}
            spellCheck={false}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              background: 'transparent',
              border: 'none',
              outline: 'none',
              resize: 'none',
              color: 'transparent',
              caretColor: 'var(--text)',
              fontFamily: '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, monospace',
              fontSize: '13px',
              lineHeight: 1.55,
              overflow: 'hidden',
            }}
          />
        </Box>
      );
    }

    const lang = visibleSection.language;
    if (lang === 'python') return renderSyntaxBlock('python');
    if (lang === 'json') return renderSyntaxBlock('json');

    if (lang === 'markdown') {
      return (
        <Box sx={markdownSx}>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{displayContent}</ReactMarkdown>
        </Box>
      );
    }

    return (
      <Box
        component="pre"
        sx={{ m: 0, fontFamily: 'inherit', color: 'var(--text)', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}
      >
        <code>{displayContent}</code>
      </Box>
    );
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', bgcolor: 'var(--panel)' }}>
      {/* ── Header ─────────────────────────────────────────────── */}
      <Box
        sx={{
          height: 60,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          px: 2,
          borderBottom: '1px solid var(--border)',
          flexShrink: 0,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flex: 1, minWidth: 0 }}>
          {panelData ? (
            <>
              <Typography
                variant="caption"
                sx={{
                  fontWeight: 600,
                  color: 'var(--muted-text)',
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                  fontSize: '0.7rem',
                  flexShrink: 0,
                }}
              >
                {panelData.title}
              </Typography>
              {hasBothViews && (
                <Box sx={{ display: 'flex', gap: 0.5, ml: 1 }}>
                  <ViewToggle
                    view="script"
                    icon={<CodeIcon sx={{ fontSize: 14 }} />}
                    label="Script"
                    isActive={panelView === 'script'}
                    onClick={setPanelView}
                  />
                  <ViewToggle
                    view="output"
                    icon={<ArticleIcon sx={{ fontSize: 14 }} />}
                    label="Result"
                    isActive={panelView === 'output'}
                    onClick={setPanelView}
                  />
                </Box>
              )}
            </>
          ) : (
            <Typography
              variant="caption"
              sx={{ fontWeight: 600, color: 'var(--muted-text)', textTransform: 'uppercase', letterSpacing: '0.05em' }}
            >
              Code Panel
            </Typography>
          )}
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {activeSection?.content && (
            <Tooltip title={copied ? 'Copied!' : 'Copy'} placement="top">
              <IconButton
                size="small"
                onClick={handleCopy}
                sx={{
                  color: copied ? 'var(--accent-green)' : 'var(--muted-text)',
                  '&:hover': { color: 'var(--accent-yellow)', bgcolor: 'var(--hover-bg)' },
                }}
              >
                {copied ? <CheckIcon sx={{ fontSize: 18 }} /> : <ContentCopyIcon sx={{ fontSize: 18 }} />}
              </IconButton>
            </Tooltip>
          )}
          {isEditableScript && !isEditing && (
            <Button
              size="small"
              startIcon={<EditIcon sx={{ fontSize: 14 }} />}
              onClick={handleStartEdit}
              sx={{
                textTransform: 'none',
                color: 'var(--muted-text)',
                fontSize: '0.75rem',
                py: 0.5,
                '&:hover': { color: 'var(--accent-yellow)', bgcolor: 'var(--hover-bg)' },
              }}
            >
              Edit
            </Button>
          )}
          {isEditing && (
            <>
              <Button
                size="small"
                startIcon={<UndoIcon sx={{ fontSize: 14 }} />}
                onClick={handleCancelEdit}
                sx={{
                  textTransform: 'none',
                  color: 'var(--muted-text)',
                  fontSize: '0.75rem',
                  py: 0.5,
                  '&:hover': { color: 'var(--accent-red)', bgcolor: 'var(--hover-bg)' },
                }}
              >
                Cancel
              </Button>
              <Button
                size="small"
                variant="contained"
                onClick={handleSaveEdit}
                disabled={!hasUnsavedChanges}
                sx={{
                  textTransform: 'none',
                  fontSize: '0.75rem',
                  py: 0.5,
                  bgcolor: hasUnsavedChanges ? 'var(--accent-yellow)' : 'var(--hover-bg)',
                  color: hasUnsavedChanges ? '#000' : 'var(--muted-text)',
                  '&:hover': {
                    bgcolor: hasUnsavedChanges ? 'var(--accent-yellow)' : 'var(--hover-bg)',
                    opacity: 0.9,
                  },
                  '&.Mui-disabled': {
                    bgcolor: 'var(--hover-bg)',
                    color: 'var(--muted-text)',
                    opacity: 0.5,
                  },
                }}
              >
                Save
              </Button>
            </>
          )}
          <IconButton size="small" onClick={() => setRightPanelOpen(false)} sx={{ color: 'var(--muted-text)' }}>
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>
      </Box>

      {/* ── Main content area ─────────────────────────────────── */}
      <Box sx={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {!panelData ? (
          <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', p: 4 }}>
            <Typography variant="body2" color="text.secondary" sx={{ opacity: 0.5 }}>
              NO DATA LOADED
            </Typography>
          </Box>
        ) : (
          <Box sx={{ flex: 1, overflow: 'hidden', p: 2 }}>
            <Box
              ref={scrollRef}
              className="code-panel"
              sx={{
                bgcolor: 'var(--code-panel-bg)',
                borderRadius: 'var(--radius-md)',
                p: '18px',
                border: '1px solid var(--border)',
                fontFamily: '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, monospace',
                fontSize: '13px',
                lineHeight: 1.55,
                height: '100%',
                overflow: 'auto',
              }}
            >
              {/* Input / Output toggle */}
              {panelData?.input && panelView === 'output' && (
                <Box sx={{ display: 'flex', gap: 0.5, mb: 1.5 }}>
                  {['input', 'output'].map((tab) => (
                    <Typography
                      key={tab}
                      onClick={() => setShowInput(tab === 'input')}
                      variant="caption"
                      sx={{
                        fontSize: '0.65rem',
                        fontWeight: 600,
                        textTransform: 'uppercase',
                        letterSpacing: '0.05em',
                        cursor: 'pointer',
                        px: 1,
                        py: 0.25,
                        borderRadius: 0.5,
                        color: (tab === 'input') === showInput ? 'var(--text)' : 'var(--muted-text)',
                        bgcolor: (tab === 'input') === showInput ? 'var(--hover-bg)' : 'transparent',
                        transition: 'all 0.12s ease',
                        '&:hover': { color: 'var(--text)' },
                      }}
                    >
                      {tab}
                    </Typography>
                  ))}
                </Box>
              )}
              {renderContent()}
            </Box>
          </Box>
        )}
      </Box>

      {/* ── Plan display (bottom) ─────────────────────────────── */}
      {plan && plan.length > 0 && (
        <Box
          sx={{
            borderTop: '1px solid var(--border)',
            bgcolor: 'var(--plan-bg)',
            maxHeight: '30%',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <Box
            sx={{
              p: 1.5,
              borderBottom: '1px solid var(--border)',
              display: 'flex',
              alignItems: 'center',
              gap: 1,
            }}
          >
            <Typography
              variant="caption"
              sx={{ fontWeight: 600, color: 'var(--muted-text)', textTransform: 'uppercase', letterSpacing: '0.05em' }}
            >
              CURRENT PLAN
            </Typography>
          </Box>

          <Stack spacing={1} sx={{ p: 2, overflow: 'auto' }}>
            {plan.map((item) => (
              <Stack key={item.id} direction="row" alignItems="flex-start" spacing={1.5}>
                <Box sx={{ mt: 0.2 }}>
                  <PlanStatusIcon status={item.status} />
                </Box>
                <Typography
                  variant="body2"
                  sx={{
                    fontSize: '13px',
                    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, monospace',
                    color: item.status === 'completed' ? 'var(--muted-text)' : 'var(--text)',
                    textDecoration: item.status === 'completed' ? 'line-through' : 'none',
                    opacity: item.status === 'pending' ? 0.7 : 1,
                  }}
                >
                  {item.content}
                </Typography>
              </Stack>
            ))}
          </Stack>
        </Box>
      )}
    </Box>
  );
}
