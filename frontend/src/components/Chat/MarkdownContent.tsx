import { useMemo, useRef, useState, useEffect, type ComponentPropsWithoutRef } from 'react';
import { Box } from '@mui/material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { SxProps, Theme } from '@mui/material/styles';

interface MarkdownContentProps {
  content: string;
  sx?: SxProps<Theme>;
  /** When true, shows a blinking cursor and throttles renders. */
  isStreaming?: boolean;
}

/** Shared markdown styles — adapts to light/dark via CSS variables. */
const markdownSx: SxProps<Theme> = {
  fontSize: '0.925rem',
  lineHeight: 1.7,
  color: 'var(--text)',
  wordBreak: 'break-word',

  '& p': { m: 0, mb: 1.5, '&:last-child': { mb: 0 } },

  '& h1, & h2, & h3, & h4': { mt: 2.5, mb: 1, fontWeight: 600, lineHeight: 1.3 },
  '& h1': { fontSize: '1.35rem' },
  '& h2': { fontSize: '1.15rem' },
  '& h3': { fontSize: '1.05rem' },

  '& pre': {
    bgcolor: 'var(--code-bg)',
    p: 2,
    borderRadius: 2,
    overflow: 'auto',
    fontSize: '0.82rem',
    lineHeight: 1.6,
    border: '1px solid var(--tool-border)',
    my: 2,
  },
  '& code': {
    bgcolor: 'var(--hover-bg)',
    px: 0.75,
    py: 0.25,
    borderRadius: 0.5,
    fontSize: '0.84rem',
    fontFamily: '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, monospace',
  },
  '& pre code': { bgcolor: 'transparent', p: 0 },

  '& a': {
    color: 'var(--accent-yellow)',
    textDecoration: 'none',
    fontWeight: 500,
    '&:hover': { textDecoration: 'underline' },
  },

  '& ul, & ol': { pl: 3, my: 1 },
  '& li': { mb: 0.5 },
  '& li::marker': { color: 'var(--muted-text)' },

  '& blockquote': {
    borderLeft: '3px solid var(--accent-yellow)',
    pl: 2,
    ml: 0,
    my: 1.5,
    color: 'var(--muted-text)',
    fontStyle: 'italic',
  },

  '& table': {
    borderCollapse: 'collapse',
    width: '100%',
    my: 2,
    fontSize: '0.85rem',
    display: 'block',
    overflowX: 'auto',
    WebkitOverflowScrolling: 'touch',
  },
  '& thead': {
    position: 'sticky',
    top: 0,
  },
  '& th': {
    borderBottom: '2px solid var(--border-hover)',
    bgcolor: 'var(--hover-bg)',
    textAlign: 'left',
    px: 1.5,
    py: 0.75,
    fontWeight: 600,
    whiteSpace: 'nowrap',
  },
  '& td': {
    borderBottom: '1px solid var(--tool-border)',
    px: 1.5,
    py: 0.75,
  },
  '& tr:nth-of-type(even) td': {
    bgcolor: 'color-mix(in srgb, var(--hover-bg) 50%, transparent)',
  },

  '& hr': {
    border: 'none',
    borderTop: '1px solid var(--border)',
    my: 2,
  },

  '& img': {
    maxWidth: '100%',
    borderRadius: 2,
  },
};

/**
 * Throttled content for streaming: render the full markdown through
 * ReactMarkdown but only re-parse every ~80ms to avoid layout thrashing.
 * This is the Claude approach — always render as markdown, never split
 * into raw text. The parser handles incomplete tables gracefully.
 */
function useThrottledValue(value: string, isStreaming: boolean, intervalMs = 80): string {
  const [throttled, setThrottled] = useState(value);
  const lastUpdate = useRef(0);
  const pending = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latestValue = useRef(value);
  latestValue.current = value;

  useEffect(() => {
    if (!isStreaming) {
      // Not streaming — always use latest value immediately
      setThrottled(value);
      return;
    }

    const now = Date.now();
    const elapsed = now - lastUpdate.current;

    if (elapsed >= intervalMs) {
      // Enough time passed — update immediately
      setThrottled(value);
      lastUpdate.current = now;
    } else {
      // Schedule an update for the remaining time
      if (pending.current) clearTimeout(pending.current);
      pending.current = setTimeout(() => {
        setThrottled(latestValue.current);
        lastUpdate.current = Date.now();
        pending.current = null;
      }, intervalMs - elapsed);
    }

    return () => {
      if (pending.current) clearTimeout(pending.current);
    };
  }, [value, isStreaming, intervalMs]);

  // When streaming ends, flush immediately
  useEffect(() => {
    if (!isStreaming) {
      setThrottled(latestValue.current);
    }
  }, [isStreaming]);

  return throttled;
}

export default function MarkdownContent({ content, sx, isStreaming = false }: MarkdownContentProps) {
  // Throttle re-parses during streaming to ~12fps (every 80ms)
  const displayContent = useThrottledValue(content, isStreaming);

  const remarkPlugins = useMemo(() => [remarkGfm], []);

  const components = useMemo(() => ({
    a: ({ href, children, ...props }: ComponentPropsWithoutRef<'a'>) => (
      <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
        {children}
      </a>
    ),
  }), []);

  return (
    <Box sx={[markdownSx, ...(Array.isArray(sx) ? sx : sx ? [sx] : [])]}>
      <ReactMarkdown remarkPlugins={remarkPlugins} components={components}>{displayContent}</ReactMarkdown>
    </Box>
  );
}
