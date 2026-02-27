"""
HF Papers Tool — Discover papers, read their contents, and find linked resources.

Operations: trending, search, paper_details, read_paper,
            find_datasets, find_models, find_collections, find_all_resources
"""

import asyncio
import re
from typing import Any

import httpx
from bs4 import BeautifulSoup, Tag

from agent.tools.types import ToolResult

HF_API = "https://huggingface.co/api"
ARXIV_HTML = "https://arxiv.org/html"
AR5IV_HTML = "https://ar5iv.labs.arxiv.org/html"

DEFAULT_LIMIT = 10
MAX_LIMIT = 50
MAX_SUMMARY_LEN = 300
MAX_SECTION_PREVIEW_LEN = 280
MAX_SECTION_TEXT_LEN = 8000

SORT_MAP = {
    "downloads": "downloads",
    "likes": "likes",
    "trending": "trendingScore",
}


# ---------------------------------------------------------------------------
# HTML paper parsing
# ---------------------------------------------------------------------------


def _parse_paper_html(html: str) -> dict[str, Any]:
    """Parse arxiv HTML into structured sections.

    Returns:
        {
            "title": str,
            "abstract": str,
            "sections": [{"id": str, "title": str, "level": int, "text": str}],
        }
    """
    soup = BeautifulSoup(html, "html.parser")

    # Title
    title_el = soup.find("h1", class_="ltx_title")
    title = title_el.get_text(strip=True).removeprefix("Title:") if title_el else ""

    # Abstract
    abstract_el = soup.find("div", class_="ltx_abstract")
    abstract = ""
    if abstract_el:
        # Skip the "Abstract" heading itself
        for child in abstract_el.children:
            if isinstance(child, Tag) and child.name in ("h6", "h2", "h3", "p", "span"):
                if child.get_text(strip=True).lower() == "abstract":
                    continue
            if isinstance(child, Tag) and child.name == "p":
                abstract += child.get_text(separator=" ", strip=True) + " "
        abstract = abstract.strip()

    # Sections — collect h2/h3 headings and text between them
    sections: list[dict[str, Any]] = []
    headings = soup.find_all(["h2", "h3"], class_=lambda c: c and "ltx_title" in c)

    for heading in headings:
        level = 2 if heading.name == "h2" else 3
        heading_text = heading.get_text(separator=" ", strip=True)

        # Collect text from siblings until next heading of same or higher level
        text_parts: list[str] = []
        sibling = heading.find_next_sibling()
        while sibling:
            if isinstance(sibling, Tag):
                if sibling.name in ("h2", "h3") and "ltx_title" in (
                    sibling.get("class") or []
                ):
                    break
                # Also stop at h2 if we're collecting h3 content
                if sibling.name == "h2" and level == 3:
                    break
                text_parts.append(sibling.get_text(separator=" ", strip=True))
            sibling = sibling.find_next_sibling()

        # Also check parent section element for contained paragraphs
        parent_section = heading.find_parent("section")
        if parent_section and not text_parts:
            for p in parent_section.find_all("p", recursive=False):
                text_parts.append(p.get_text(separator=" ", strip=True))

        section_text = "\n\n".join(t for t in text_parts if t)

        # Extract section number from heading text (e.g., "4 Experiments" → "4")
        num_match = re.match(r"^([A-Z]?\d+(?:\.\d+)*)\s", heading_text)
        section_id = num_match.group(1) if num_match else ""

        sections.append(
            {
                "id": section_id,
                "title": heading_text,
                "level": level,
                "text": section_text,
            }
        )

    return {"title": title, "abstract": abstract, "sections": sections}


def _find_section(sections: list[dict], query: str) -> dict | None:
    """Find a section by number or name (fuzzy)."""
    query_lower = query.lower().strip()

    # Exact match on section number
    for s in sections:
        if s["id"] == query_lower or s["id"] == query:
            return s

    # Exact match on title
    for s in sections:
        if query_lower == s["title"].lower():
            return s

    # Substring match on title
    for s in sections:
        if query_lower in s["title"].lower():
            return s

    # Number prefix match (e.g., "4" matches "4.1", "4.2", etc. — return parent)
    for s in sections:
        if s["id"].startswith(query_lower + ".") or s["id"] == query_lower:
            return s

    return None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _clean_description(text: str) -> str:
    """Strip HTML card artifacts and collapse whitespace from HF API descriptions."""
    text = re.sub(r"[\t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _format_paper_list(
    papers: list, title: str, date: str | None = None, query: str | None = None
) -> str:
    lines = [f"# {title}"]
    if date:
        lines[0] += f" ({date})"
    if query:
        lines.append(f"Filtered by: '{query}'")
    lines.append(f"Showing {len(papers)} paper(s)\n")

    for i, item in enumerate(papers, 1):
        paper = item.get("paper", item)
        arxiv_id = paper.get("id", "")
        paper_title = paper.get("title", "Unknown")
        upvotes = paper.get("upvotes", 0)
        summary = paper.get("ai_summary") or _truncate(
            paper.get("summary", ""), MAX_SUMMARY_LEN
        )
        keywords = paper.get("ai_keywords") or []
        github = paper.get("githubRepo") or ""
        stars = paper.get("githubStars") or 0

        lines.append(f"## {i}. {paper_title}")
        lines.append(f"**arxiv_id:** {arxiv_id} | **upvotes:** {upvotes}")
        lines.append(f"https://huggingface.co/papers/{arxiv_id}")
        if keywords:
            lines.append(f"**Keywords:** {', '.join(keywords[:5])}")
        if github:
            lines.append(f"**GitHub:** {github} ({stars} stars)")
        if summary:
            lines.append(f"**Summary:** {_truncate(summary, MAX_SUMMARY_LEN)}")
        lines.append("")

    return "\n".join(lines)


def _format_paper_detail(paper: dict) -> str:
    arxiv_id = paper.get("id", "")
    title = paper.get("title", "Unknown")
    upvotes = paper.get("upvotes", 0)
    ai_summary = paper.get("ai_summary") or ""
    summary = paper.get("summary", "")
    keywords = paper.get("ai_keywords") or []
    github = paper.get("githubRepo") or ""
    stars = paper.get("githubStars") or 0
    authors = paper.get("authors") or []

    lines = [f"# {title}"]
    lines.append(f"**arxiv_id:** {arxiv_id} | **upvotes:** {upvotes}")
    lines.append(f"https://huggingface.co/papers/{arxiv_id}")
    lines.append(f"https://arxiv.org/abs/{arxiv_id}")

    if authors:
        names = [a.get("name", "") for a in authors[:10]]
        author_str = ", ".join(n for n in names if n)
        if len(authors) > 10:
            author_str += f" (+{len(authors) - 10} more)"
        lines.append(f"**Authors:** {author_str}")

    if keywords:
        lines.append(f"**Keywords:** {', '.join(keywords)}")
    if github:
        lines.append(f"**GitHub:** {github} ({stars} stars)")

    if ai_summary:
        lines.append(f"\n## AI Summary\n{ai_summary}")
    if summary:
        lines.append(f"\n## Abstract\n{_truncate(summary, 500)}")

    lines.append(
        "\n**Next:** Use read_paper to read specific sections, or find_all_resources to discover linked datasets/models."
    )
    return "\n".join(lines)


def _format_read_paper_toc(parsed: dict[str, Any], arxiv_id: str) -> str:
    """Format TOC view: abstract + section list with previews."""
    lines = [f"# {parsed['title']}"]
    lines.append(f"https://arxiv.org/abs/{arxiv_id}\n")

    if parsed["abstract"]:
        lines.append(f"## Abstract\n{parsed['abstract']}\n")

    lines.append("## Sections")
    for s in parsed["sections"]:
        prefix = "  " if s["level"] == 3 else ""
        preview = (
            _truncate(s["text"], MAX_SECTION_PREVIEW_LEN) if s["text"] else "(empty)"
        )
        lines.append(f"{prefix}- **{s['title']}**: {preview}")

    lines.append(
        '\n**Tip:** Call read_paper with section parameter (e.g. section="4" or section="Experiments") to read a specific section.'
    )
    return "\n".join(lines)


def _format_read_paper_section(section: dict, arxiv_id: str) -> str:
    """Format a single section's full text."""
    lines = [f"# {section['title']}"]
    lines.append(f"https://arxiv.org/abs/{arxiv_id}\n")

    text = section["text"]
    if len(text) > MAX_SECTION_TEXT_LEN:
        text = (
            text[:MAX_SECTION_TEXT_LEN]
            + f"\n\n... (truncated at {MAX_SECTION_TEXT_LEN} chars)"
        )

    lines.append(text if text else "(This section has no extractable text content.)")
    return "\n".join(lines)


def _format_datasets(datasets: list, arxiv_id: str, sort: str) -> str:
    lines = [f"# Datasets linked to paper {arxiv_id}"]
    lines.append(f"https://huggingface.co/papers/{arxiv_id}")
    lines.append(f"Showing {len(datasets)} dataset(s), sorted by {sort}\n")

    for i, ds in enumerate(datasets, 1):
        ds_id = ds.get("id", "unknown")
        downloads = ds.get("downloads", 0)
        likes = ds.get("likes", 0)
        desc = _truncate(_clean_description(ds.get("description") or ""), MAX_SUMMARY_LEN)
        tags = ds.get("tags") or []
        interesting = [t for t in tags if not t.startswith(("arxiv:", "region:"))][:5]

        lines.append(f"**{i}. [{ds_id}](https://huggingface.co/datasets/{ds_id})**")
        lines.append(f"   Downloads: {downloads:,} | Likes: {likes}")
        if interesting:
            lines.append(f"   Tags: {', '.join(interesting)}")
        if desc:
            lines.append(f"   {desc}")
        lines.append("")

    if datasets:
        top = datasets[0].get("id", "")
        lines.append(f'**Inspect top dataset:** hf_inspect_dataset(dataset="{top}")')
    return "\n".join(lines)


def _format_datasets_compact(datasets: list) -> str:
    if not datasets:
        return "## Datasets\nNone found"
    lines = [f"## Datasets ({len(datasets)})"]
    for ds in datasets:
        lines.append(
            f"- **{ds.get('id', '?')}** ({ds.get('downloads', 0):,} downloads)"
        )
    return "\n".join(lines)


def _format_models(models: list, arxiv_id: str, sort: str) -> str:
    lines = [f"# Models linked to paper {arxiv_id}"]
    lines.append(f"https://huggingface.co/papers/{arxiv_id}")
    lines.append(f"Showing {len(models)} model(s), sorted by {sort}\n")

    for i, m in enumerate(models, 1):
        model_id = m.get("id", "unknown")
        downloads = m.get("downloads", 0)
        likes = m.get("likes", 0)
        pipeline = m.get("pipeline_tag") or ""
        library = m.get("library_name") or ""

        lines.append(f"**{i}. [{model_id}](https://huggingface.co/{model_id})**")
        meta = f"   Downloads: {downloads:,} | Likes: {likes}"
        if pipeline:
            meta += f" | Task: {pipeline}"
        if library:
            meta += f" | Library: {library}"
        lines.append(meta)
        lines.append("")

    return "\n".join(lines)


def _format_models_compact(models: list) -> str:
    if not models:
        return "## Models\nNone found"
    lines = [f"## Models ({len(models)})"]
    for m in models:
        pipeline = m.get("pipeline_tag") or ""
        suffix = f" ({pipeline})" if pipeline else ""
        lines.append(
            f"- **{m.get('id', '?')}** ({m.get('downloads', 0):,} downloads){suffix}"
        )
    return "\n".join(lines)


def _format_collections(collections: list, arxiv_id: str) -> str:
    lines = [f"# Collections containing paper {arxiv_id}"]
    lines.append(f"Showing {len(collections)} collection(s)\n")

    for i, c in enumerate(collections, 1):
        slug = c.get("slug", "")
        title = c.get("title", "Untitled")
        upvotes = c.get("upvotes", 0)
        owner = c.get("owner", {}).get("name", "")
        desc = _truncate(c.get("description") or "", MAX_SUMMARY_LEN)
        num_items = len(c.get("items", []))

        lines.append(f"**{i}. {title}**")
        lines.append(f"   By: {owner} | Upvotes: {upvotes} | Items: {num_items}")
        lines.append(f"   https://huggingface.co/collections/{slug}")
        if desc:
            lines.append(f"   {desc}")
        lines.append("")

    return "\n".join(lines)


def _format_collections_compact(collections: list) -> str:
    if not collections:
        return "## Collections\nNone found"
    lines = [f"## Collections ({len(collections)})"]
    for c in collections:
        title = c.get("title", "Untitled")
        owner = c.get("owner", {}).get("name", "")
        upvotes = c.get("upvotes", 0)
        lines.append(f"- **{title}** by {owner} ({upvotes} upvotes)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Operation handlers
# ---------------------------------------------------------------------------


def _error(message: str) -> ToolResult:
    return {
        "formatted": message,
        "totalResults": 0,
        "resultsShared": 0,
        "isError": True,
    }


def _validate_arxiv_id(args: dict) -> str | None:
    """Return arxiv_id or None if missing."""
    return args.get("arxiv_id")


async def _op_trending(args: dict[str, Any], limit: int) -> ToolResult:
    date = args.get("date")
    query = args.get("query")

    params: dict[str, Any] = {"limit": limit if not query else max(limit * 3, 30)}
    if date:
        params["date"] = date

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{HF_API}/daily_papers", params=params)
        resp.raise_for_status()
        papers = resp.json()

    if query:
        q = query.lower()
        papers = [
            p
            for p in papers
            if q in p.get("title", "").lower()
            or q in p.get("paper", {}).get("title", "").lower()
            or q in p.get("paper", {}).get("summary", "").lower()
            or any(
                q in kw.lower() for kw in (p.get("paper", {}).get("ai_keywords") or [])
            )
        ]

    papers = papers[:limit]
    if not papers:
        msg = "No trending papers found"
        if query:
            msg += f" matching '{query}'"
        if date:
            msg += f" for {date}"
        return {"formatted": msg, "totalResults": 0, "resultsShared": 0}

    formatted = _format_paper_list(papers, "Trending Papers", date=date, query=query)
    return {
        "formatted": formatted,
        "totalResults": len(papers),
        "resultsShared": len(papers),
    }


async def _op_search(args: dict[str, Any], limit: int) -> ToolResult:
    query = args.get("query")
    if not query:
        return _error("'query' is required for search operation.")

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{HF_API}/papers/search", params={"q": query, "limit": limit}
        )
        resp.raise_for_status()
        papers = resp.json()

    if not papers:
        return {
            "formatted": f"No papers found for '{query}'",
            "totalResults": 0,
            "resultsShared": 0,
        }

    formatted = _format_paper_list(papers, f"Papers matching '{query}'")
    return {
        "formatted": formatted,
        "totalResults": len(papers),
        "resultsShared": len(papers),
    }


async def _op_paper_details(args: dict[str, Any], limit: int) -> ToolResult:
    arxiv_id = _validate_arxiv_id(args)
    if not arxiv_id:
        return _error("'arxiv_id' is required for paper_details.")

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{HF_API}/papers/{arxiv_id}")
        resp.raise_for_status()
        paper = resp.json()

    return {
        "formatted": _format_paper_detail(paper),
        "totalResults": 1,
        "resultsShared": 1,
    }


async def _op_read_paper(args: dict[str, Any], limit: int) -> ToolResult:
    arxiv_id = _validate_arxiv_id(args)
    if not arxiv_id:
        return _error("'arxiv_id' is required for read_paper.")

    section_query = args.get("section")

    # Try fetching HTML from arxiv, then ar5iv, then fallback to abstract
    parsed = None
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        for base_url in [ARXIV_HTML, AR5IV_HTML]:
            try:
                resp = await client.get(f"{base_url}/{arxiv_id}")
                if resp.status_code == 200:
                    parsed = _parse_paper_html(resp.text)
                    if parsed["sections"]:  # Only use if we got real sections
                        break
                    parsed = None
            except httpx.RequestError:
                continue

    # Fallback: return abstract from HF API
    if not parsed or not parsed["sections"]:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(f"{HF_API}/papers/{arxiv_id}")
                resp.raise_for_status()
                paper = resp.json()
            abstract = paper.get("summary", "")
            title = paper.get("title", "")
            msg = f"# {title}\nhttps://arxiv.org/abs/{arxiv_id}\n\n"
            msg += f"## Abstract\n{abstract}\n\n"
            msg += "HTML version not available for this paper. Only abstract shown.\n"
            msg += f"PDF: https://arxiv.org/pdf/{arxiv_id}"
            return {"formatted": msg, "totalResults": 1, "resultsShared": 1}
        except Exception:
            return _error(
                f"Could not fetch paper {arxiv_id}. Check the arxiv ID is correct."
            )

    # Return TOC or specific section
    if not section_query:
        formatted = _format_read_paper_toc(parsed, arxiv_id)
        return {
            "formatted": formatted,
            "totalResults": len(parsed["sections"]),
            "resultsShared": len(parsed["sections"]),
        }

    section = _find_section(parsed["sections"], section_query)
    if not section:
        available = "\n".join(f"- {s['title']}" for s in parsed["sections"])
        return _error(
            f"Section '{section_query}' not found. Available sections:\n{available}"
        )

    formatted = _format_read_paper_section(section, arxiv_id)
    return {"formatted": formatted, "totalResults": 1, "resultsShared": 1}


async def _op_find_datasets(args: dict[str, Any], limit: int) -> ToolResult:
    arxiv_id = _validate_arxiv_id(args)
    if not arxiv_id:
        return _error("'arxiv_id' is required for find_datasets.")

    sort = args.get("sort", "downloads")
    sort_key = SORT_MAP.get(sort, "downloads")

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{HF_API}/datasets",
            params={
                "filter": f"arxiv:{arxiv_id}",
                "limit": limit,
                "sort": sort_key,
                "direction": -1,
            },
        )
        resp.raise_for_status()
        datasets = resp.json()

    if not datasets:
        return {
            "formatted": f"No datasets found linked to paper {arxiv_id}.\nhttps://huggingface.co/papers/{arxiv_id}",
            "totalResults": 0,
            "resultsShared": 0,
        }

    return {
        "formatted": _format_datasets(datasets, arxiv_id, sort),
        "totalResults": len(datasets),
        "resultsShared": len(datasets),
    }


async def _op_find_models(args: dict[str, Any], limit: int) -> ToolResult:
    arxiv_id = _validate_arxiv_id(args)
    if not arxiv_id:
        return _error("'arxiv_id' is required for find_models.")

    sort = args.get("sort", "downloads")
    sort_key = SORT_MAP.get(sort, "downloads")

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{HF_API}/models",
            params={
                "filter": f"arxiv:{arxiv_id}",
                "limit": limit,
                "sort": sort_key,
                "direction": -1,
            },
        )
        resp.raise_for_status()
        models = resp.json()

    if not models:
        return {
            "formatted": f"No models found linked to paper {arxiv_id}.\nhttps://huggingface.co/papers/{arxiv_id}",
            "totalResults": 0,
            "resultsShared": 0,
        }

    return {
        "formatted": _format_models(models, arxiv_id, sort),
        "totalResults": len(models),
        "resultsShared": len(models),
    }


async def _op_find_collections(args: dict[str, Any], limit: int) -> ToolResult:
    arxiv_id = _validate_arxiv_id(args)
    if not arxiv_id:
        return _error("'arxiv_id' is required for find_collections.")

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{HF_API}/collections", params={"paper": arxiv_id})
        resp.raise_for_status()
        collections = resp.json()

    if not collections:
        return {
            "formatted": f"No collections found containing paper {arxiv_id}.\nhttps://huggingface.co/papers/{arxiv_id}",
            "totalResults": 0,
            "resultsShared": 0,
        }

    collections = collections[:limit]
    return {
        "formatted": _format_collections(collections, arxiv_id),
        "totalResults": len(collections),
        "resultsShared": len(collections),
    }


async def _op_find_all_resources(args: dict[str, Any], limit: int) -> ToolResult:
    arxiv_id = _validate_arxiv_id(args)
    if not arxiv_id:
        return _error("'arxiv_id' is required for find_all_resources.")

    per_cat = min(limit, 10)

    async with httpx.AsyncClient(timeout=15) as client:
        results = await asyncio.gather(
            client.get(
                f"{HF_API}/datasets",
                params={
                    "filter": f"arxiv:{arxiv_id}",
                    "limit": per_cat,
                    "sort": "downloads",
                    "direction": -1,
                },
            ),
            client.get(
                f"{HF_API}/models",
                params={
                    "filter": f"arxiv:{arxiv_id}",
                    "limit": per_cat,
                    "sort": "downloads",
                    "direction": -1,
                },
            ),
            client.get(f"{HF_API}/collections", params={"paper": arxiv_id}),
            return_exceptions=True,
        )

    sections = []
    total = 0

    # Datasets
    if isinstance(results[0], Exception):
        sections.append(f"## Datasets\nError: {results[0]}")
    else:
        datasets = results[0].json()
        total += len(datasets)
        sections.append(_format_datasets_compact(datasets[:per_cat]))

    # Models
    if isinstance(results[1], Exception):
        sections.append(f"## Models\nError: {results[1]}")
    else:
        models = results[1].json()
        total += len(models)
        sections.append(_format_models_compact(models[:per_cat]))

    # Collections
    if isinstance(results[2], Exception):
        sections.append(f"## Collections\nError: {results[2]}")
    else:
        collections = results[2].json()
        total += len(collections)
        sections.append(_format_collections_compact(collections[:per_cat]))

    header = f"# Resources linked to paper {arxiv_id}\nhttps://huggingface.co/papers/{arxiv_id}\n"
    formatted = header + "\n\n".join(sections)
    return {"formatted": formatted, "totalResults": total, "resultsShared": total}


# ---------------------------------------------------------------------------
# Operation dispatch
# ---------------------------------------------------------------------------

_OPERATIONS = {
    "trending": _op_trending,
    "search": _op_search,
    "paper_details": _op_paper_details,
    "read_paper": _op_read_paper,
    "find_datasets": _op_find_datasets,
    "find_models": _op_find_models,
    "find_collections": _op_find_collections,
    "find_all_resources": _op_find_all_resources,
}


# ---------------------------------------------------------------------------
# Tool spec + handler
# ---------------------------------------------------------------------------

HF_PAPERS_TOOL_SPEC = {
    "name": "hf_papers",
    "description": (
        "Discover ML research papers, find their linked resources (datasets, models, collections), "
        "and read paper contents on HuggingFace Hub and arXiv.\n\n"
        "Use this when exploring a research area, looking for datasets for a task, "
        "implementing a paper's approach, or trying to improve performance on something. "
        "Typical flow:\n"
        "  hf_papers(search/trending) → hf_papers(read_paper) → hf_papers(find_all_resources) → hf_inspect_dataset\n\n"
        "Operations:\n"
        "- trending: Get trending daily papers, optionally filter by topic keyword\n"
        "- search: Full-text search for papers by query\n"
        "- paper_details: Get metadata, abstract, AI summary, and github link for a paper\n"
        "- read_paper: Read paper contents — without section: returns abstract + table of contents; "
        "with section: returns full section text\n"
        "- find_datasets: Find datasets linked to a paper\n"
        "- find_models: Find models linked to a paper\n"
        "- find_collections: Find collections that include a paper\n"
        "- find_all_resources: Parallel fetch of datasets + models + collections for a paper (unified view)"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": list(_OPERATIONS.keys()),
                "description": "Operation to execute.",
            },
            "query": {
                "type": "string",
                "description": (
                    "Search query. Required for: search. "
                    "Optional for: trending (filters results by keyword match on title, summary, and AI-generated keywords)."
                ),
            },
            "arxiv_id": {
                "type": "string",
                "description": (
                    "ArXiv paper ID (e.g. '2305.18290'). "
                    "Required for: paper_details, read_paper, find_datasets, find_models, find_collections, find_all_resources. "
                    "Get IDs from trending or search results first."
                ),
            },
            "section": {
                "type": "string",
                "description": (
                    "Section name or number to read (e.g. '3', 'Experiments', '4.2'). "
                    "Optional for: read_paper. Without this, read_paper returns the abstract + table of contents "
                    "so you can choose which section to read."
                ),
            },
            "date": {
                "type": "string",
                "description": "Date in YYYY-MM-DD format. Optional for: trending (defaults to recent papers).",
            },
            "sort": {
                "type": "string",
                "enum": ["downloads", "likes", "trending"],
                "description": (
                    "Sort order for find_datasets and find_models. Default: downloads. "
                    "Use 'downloads' for most-used, 'likes' for community favorites, 'trending' for recently popular."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results to return (default: 10, max: 50).",
            },
        },
        "required": ["operation"],
    },
}


async def hf_papers_handler(arguments: dict[str, Any]) -> tuple[str, bool]:
    """Handler for agent tool router."""
    operation = arguments.get("operation")
    if not operation:
        return "'operation' parameter is required.", False

    handler = _OPERATIONS.get(operation)
    if not handler:
        valid = ", ".join(_OPERATIONS.keys())
        return f"Unknown operation: '{operation}'. Valid: {valid}", False

    limit = min(arguments.get("limit", DEFAULT_LIMIT), MAX_LIMIT)

    try:
        result = await handler(arguments, limit)
        return result["formatted"], not result.get("isError", False)
    except httpx.HTTPStatusError as e:
        return f"API error: {e.response.status_code} — {e.response.text[:200]}", False
    except httpx.RequestError as e:
        return f"Request error: {e}", False
    except Exception as e:
        return f"Error in {operation}: {e}", False
