"""
Shared utilities for file editing tools — fuzzy matching, syntax validation,
and richer edit operations.

Used by both local_tools.py and the embedded sandbox server.
"""

from __future__ import annotations

# ── Unicode normalization map ────────────────────────────────────────────

UNICODE_MAP = {
    "\u2013": "-",  # en-dash
    "\u2014": "-",  # em-dash
    "\u2212": "-",  # minus sign
    "\u2018": "'",  # left single quote
    "\u2019": "'",  # right single quote
    "\u201c": '"',  # left double quote
    "\u201d": '"',  # right double quote
    "\u00a0": " ",  # non-breaking space
    "\u2003": " ",  # em space
    "\u2002": " ",  # en space
    "\u200b": "",  # zero-width space
    "\ufeff": "",  # BOM
}


def _normalize_unicode(s: str) -> str:
    return "".join(UNICODE_MAP.get(c, c) for c in s)


# ── 4-pass fuzzy matching ────────────────────────────────────────────────


def fuzzy_find(content: str, pattern: str) -> tuple[int | None, str | None]:
    """Find *pattern* in *content* with increasingly relaxed matching.

    Returns (start_index_in_original_content, match_note) or (None, None).
    The index always refers to the *original* content string so callers can
    use ``content[idx : idx + len(matched_text)]`` for replacement.

    Strategy (mirrors Codex):
      1. Exact match
      2. Right-trim each line (trailing whitespace)
      3. Both-sides trim (all surrounding whitespace per line)
      4. Unicode normalization on top of both-sides trim
    """
    # Pass 1 — exact
    if pattern in content:
        return content.index(pattern), None

    # Helper: build a line-stripped version *and* a mapping from stripped
    # positions back to original positions.  We need this so callers can
    # apply the replacement on the original content, not the stripped copy.

    def _build_stripped(text: str, strip_fn):
        """Return (stripped_text, line_start_map).

        line_start_map[i] = original byte offset of the start of line i.
        """
        orig_lines = text.split("\n")
        stripped_lines = [strip_fn(line) for line in orig_lines]
        return "\n".join(stripped_lines), orig_lines, stripped_lines

    # Pass 2 — right-trim
    c_rt, c_orig_lines, c_rt_lines = _build_stripped(content, str.rstrip)
    p_rt = "\n".join(line.rstrip() for line in pattern.split("\n"))
    idx = c_rt.find(p_rt)
    if idx != -1:
        orig_idx = _map_back(idx, c_orig_lines, c_rt_lines)
        return orig_idx, "(matched after trimming trailing whitespace)"

    # Pass 3 — both-sides trim
    c_st, _, c_st_lines = _build_stripped(content, str.strip)
    p_st = "\n".join(line.strip() for line in pattern.split("\n"))
    idx = c_st.find(p_st)
    if idx != -1:
        orig_idx = _map_back(idx, c_orig_lines, c_st_lines)
        return orig_idx, "(matched after trimming whitespace)"

    # Pass 4 — unicode normalization + both-sides trim
    c_norm = _normalize_unicode(c_st)
    p_norm = _normalize_unicode(p_st)
    idx = c_norm.find(p_norm)
    if idx != -1:
        orig_idx = _map_back(idx, c_orig_lines, c_st_lines)
        return orig_idx, "(matched after unicode normalization)"

    return None, None


def _map_back(
    stripped_idx: int,
    orig_lines: list[str],
    stripped_lines: list[str],
) -> int:
    """Map a character index in the stripped/joined text back to the original text."""
    # Walk through stripped lines to find which line the index falls on
    pos = 0
    for i, sl in enumerate(stripped_lines):
        line_end = pos + len(sl)
        if stripped_idx <= line_end:
            col_in_stripped = stripped_idx - pos
            # Find where this stripped line's content starts in the original line
            ol = orig_lines[i]
            # The stripped line is a subset of the original line; find its offset
            lstripped = len(ol) - len(ol.lstrip())
            orig_col = lstripped + col_in_stripped
            # Compute absolute position in original text
            orig_pos = sum(len(orig_lines[j]) + 1 for j in range(i)) + orig_col
            return orig_pos
        pos = line_end + 1  # +1 for the \n
    # Fallback: return 0 (shouldn't happen if idx is valid)
    return 0


def fuzzy_find_original_match(
    content: str, pattern: str
) -> tuple[str | None, str | None]:
    """Find the *original* text in content that matches pattern fuzzily.

    Returns (original_matched_text, match_note) or (None, None).
    This extracts the exact substring from the original content that
    corresponds to the fuzzy match, preserving its original whitespace/unicode.
    """
    if pattern in content:
        return pattern, None

    idx, note = fuzzy_find(content, pattern)
    if idx is None:
        return None, None

    # We need to find the original text span that corresponds to the match.
    # The match covers len(pattern) worth of *logical* content.
    # Count how many original lines the pattern spans.
    pattern_lines = pattern.split("\n")
    n_lines = len(pattern_lines)

    # Find which original line the match starts on
    orig_lines = content.split("\n")
    char_pos = 0
    start_line = 0
    for i, ol in enumerate(orig_lines):
        if char_pos + len(ol) >= idx:
            start_line = i
            break
        char_pos += len(ol) + 1

    end_line = min(start_line + n_lines, len(orig_lines))
    # Extract the original lines that were matched
    matched_lines = orig_lines[start_line:end_line]
    original_text = "\n".join(matched_lines)
    return original_text, note


# ── Richer edit operations ───────────────────────────────────────────────


def apply_edit(
    content: str,
    old_str: str,
    new_str: str,
    mode: str = "replace",
    replace_all: bool = False,
) -> tuple[str, int, str | None]:
    """Apply an edit operation to content.

    Modes:
      - replace: replace first occurrence (or all if replace_all=True)
      - replace_all: replace all occurrences (alias)
      - append_after: insert new_str after old_str
      - prepend_before: insert new_str before old_str

    Returns (new_content, num_replacements, fuzzy_note).
    Raises ValueError if old_str not found.
    """
    if mode == "replace_all":
        replace_all = True
        mode = "replace"

    # Try exact match first, then fuzzy
    fuzzy_note = None
    if old_str not in content:
        original_match, fuzzy_note = fuzzy_find_original_match(content, old_str)
        if original_match is None:
            raise ValueError(
                "old_str was not found in the file. Make sure old_str matches "
                "the file contents exactly, including whitespace and indentation. "
                "Use the read tool to verify the current file contents before retrying."
            )
        old_str = original_match

    count = content.count(old_str)

    if mode == "replace":
        if count > 1 and not replace_all:
            raise ValueError(
                f"Found {count} matches of old_str in the file, but replace_all is "
                f"false. To replace all occurrences, set replace_all to true. To "
                f"replace only one, provide a larger old_str with more surrounding "
                f"context to uniquely identify the instance."
            )
        if replace_all:
            new_content = content.replace(old_str, new_str)
            return new_content, count, fuzzy_note
        else:
            new_content = content.replace(old_str, new_str, 1)
            return new_content, 1, fuzzy_note

    elif mode == "append_after":
        if replace_all:
            new_content = content.replace(old_str, old_str + new_str)
            return new_content, count, fuzzy_note
        else:
            idx = content.index(old_str) + len(old_str)
            new_content = content[:idx] + new_str + content[idx:]
            return new_content, 1, fuzzy_note

    elif mode == "prepend_before":
        if replace_all:
            new_content = content.replace(old_str, new_str + old_str)
            return new_content, count, fuzzy_note
        else:
            idx = content.index(old_str)
            new_content = content[:idx] + new_str + content[idx:]
            return new_content, 1, fuzzy_note

    else:
        raise ValueError(
            f"Unknown edit mode: {mode}. Use replace, append_after, or prepend_before."
        )


# ── Syntax validation (Python) ───────────────────────────────────────────


def validate_python(content: str, path: str = "") -> list[str]:
    """Lightweight post-write validation for Python files.

    Checks syntax and training script conventions. This runs on the host
    (not in the sandbox), so it only does static checks — no import resolution
    or signature inspection since packages are installed in the sandbox, not here.

    The sandbox server has its own richer version that does real signature
    inspection against installed packages.

    Returns a list of warning strings (empty = all good).
    Never raises — validation failures are advisory only.
    """
    import ast

    warnings = []

    # 1. Syntax check via ast.parse
    try:
        ast.parse(content)
    except SyntaxError as e:
        warnings.append(f"Python syntax error at line {e.lineno}: {e.msg}")
        return warnings

    # 2. Training script heuristics
    if any(
        kw in content
        for kw in ("TrainingArguments", "SFTConfig", "DPOConfig", "GRPOConfig")
    ):
        if "push_to_hub" not in content:
            warnings.append(
                "Training script warning: no 'push_to_hub' found — model may be lost when job ends"
            )
        if "hub_model_id" not in content:
            warnings.append("Training script warning: no 'hub_model_id' found")

    return warnings
