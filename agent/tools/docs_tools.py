"""
Documentation search tools for exploring HuggingFace and Gradio documentation.
"""

import asyncio
import json
import os
from typing import Any

import httpx
from bs4 import BeautifulSoup
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, TEXT, Schema
from whoosh.filedb.filestore import RamStorage
from whoosh.qparser import MultifieldParser, OrGroup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MAX_RESULTS = 20
MAX_RESULTS_CAP = 50

GRADIO_LLMS_TXT_URL = "https://gradio.app/llms.txt"
GRADIO_SEARCH_URL = "https://playground-worker.pages.dev/api/prompt"

COMPOSITE_ENDPOINTS: dict[str, list[str]] = {
    "optimum": [
        "optimum",
        "optimum-habana",
        "optimum-neuron",
        "optimum-intel",
        "optimum-executorch",
        "optimum-tpu",
    ],
    "courses": [
        "llm-course",
        "robotics-course",
        "mcp-course",
        "smol-course",
        "agents-course",
        "deep-rl-course",
        "computer-vision-course",
        "audio-course",
        "ml-games-course",
        "diffusion-course",
        "ml-for-3d-course",
        "cookbook",
    ],
}

# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------

_docs_cache: dict[str, list[dict[str, str]]] = {}
_index_cache: dict[str, tuple[Any, MultifieldParser]] = {}
_cache_lock = asyncio.Lock()
_openapi_cache: dict[str, Any] | None = None
_openapi_index_cache: tuple[Any, MultifieldParser, list[dict[str, Any]]] | None = None

# ---------------------------------------------------------------------------
# Gradio Documentation
# ---------------------------------------------------------------------------


async def _fetch_gradio_docs(query: str | None = None) -> str:
    """
    Fetch Gradio documentation.
    Without query: Get full documentation from llms.txt
    With query: Run embedding search on guides/demos for relevant content
    """
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        if not query:
            resp = await client.get(GRADIO_LLMS_TXT_URL)
            resp.raise_for_status()
            return resp.text

        resp = await client.post(
            GRADIO_SEARCH_URL,
            headers={
                "Content-Type": "application/json",
                "Origin": "https://gradio-docs-mcp.up.railway.app",
            },
            json={
                "prompt_to_embed": query,
                "SYSTEM_PROMPT": "$INSERT_GUIDES_DOCS_DEMOS",
                "FALLBACK_PROMPT": "No results found",
            },
        )
        resp.raise_for_status()
        return resp.json().get("SYS_PROMPT", "No results found")


# ---------------------------------------------------------------------------
# HF Documentation - Fetching
# ---------------------------------------------------------------------------


async def _fetch_endpoint_docs(hf_token: str, endpoint: str) -> list[dict[str, str]]:
    """Fetch all docs for an endpoint by parsing sidebar and fetching each page."""
    url = f"https://huggingface.co/docs/{endpoint}"
    headers = {"Authorization": f"Bearer {hf_token}"}

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        sidebar = soup.find("nav", class_=lambda x: x and "flex-auto" in x)
        if not sidebar:
            raise ValueError(f"Could not find navigation sidebar for '{endpoint}'")

        nav_items = []
        for link in sidebar.find_all("a", href=True):
            href = link["href"]
            page_url = f"https://huggingface.co{href}" if href.startswith("/") else href
            nav_items.append({"title": link.get_text(strip=True), "url": page_url})

        if not nav_items:
            raise ValueError(f"No navigation links found for '{endpoint}'")

        async def fetch_page(item: dict[str, str]) -> dict[str, str]:
            md_url = f"{item['url']}.md"
            try:
                r = await client.get(md_url, headers=headers)
                r.raise_for_status()
                content = r.text.strip()
                glimpse = content[:200] + "..." if len(content) > 200 else content
            except Exception as e:
                content, glimpse = "", f"[Could not fetch: {str(e)[:50]}]"
            return {
                "title": item["title"],
                "url": item["url"],
                "md_url": md_url,
                "glimpse": glimpse,
                "content": content,
                "section": endpoint,
            }

        return list(await asyncio.gather(*[fetch_page(item) for item in nav_items]))


async def _get_docs(hf_token: str, endpoint: str) -> list[dict[str, str]]:
    """Get docs for endpoint with caching. Expands composite endpoints."""
    async with _cache_lock:
        if endpoint in _docs_cache:
            return _docs_cache[endpoint]

    sub_endpoints = COMPOSITE_ENDPOINTS.get(endpoint, [endpoint])
    all_docs: list[dict[str, str]] = []

    for sub in sub_endpoints:
        async with _cache_lock:
            if sub in _docs_cache:
                all_docs.extend(_docs_cache[sub])
                continue

        docs = await _fetch_endpoint_docs(hf_token, sub)
        async with _cache_lock:
            _docs_cache[sub] = docs
        all_docs.extend(docs)

    async with _cache_lock:
        _docs_cache[endpoint] = all_docs
    return all_docs


# ---------------------------------------------------------------------------
# HF Documentation - Search
# ---------------------------------------------------------------------------


async def _build_search_index(
    endpoint: str, docs: list[dict[str, str]]
) -> tuple[Any, MultifieldParser]:
    """Build or retrieve cached Whoosh search index."""
    async with _cache_lock:
        if endpoint in _index_cache:
            return _index_cache[endpoint]

    analyzer = StemmingAnalyzer()
    schema = Schema(
        title=TEXT(stored=True, analyzer=analyzer),
        url=ID(stored=True, unique=True),
        md_url=ID(stored=True),
        section=ID(stored=True),
        glimpse=TEXT(stored=True, analyzer=analyzer),
        content=TEXT(stored=False, analyzer=analyzer),
    )
    storage = RamStorage()
    index = storage.create_index(schema)
    writer = index.writer()
    for doc in docs:
        writer.add_document(
            title=doc.get("title", ""),
            url=doc.get("url", ""),
            md_url=doc.get("md_url", ""),
            section=doc.get("section", endpoint),
            glimpse=doc.get("glimpse", ""),
            content=doc.get("content", ""),
        )
    writer.commit()

    parser = MultifieldParser(
        ["title", "content"],
        schema=schema,
        fieldboosts={"title": 2.0, "content": 1.0},
        group=OrGroup,
    )

    async with _cache_lock:
        _index_cache[endpoint] = (index, parser)
    return index, parser


async def _search_docs(
    endpoint: str, docs: list[dict[str, str]], query: str, limit: int
) -> tuple[list[dict[str, Any]], str | None]:
    """Search docs using Whoosh. Returns (results, fallback_message)."""
    index, parser = await _build_search_index(endpoint, docs)

    try:
        query_obj = parser.parse(query)
    except Exception:
        return [], "Query contained unsupported syntax; showing default ordering."

    with index.searcher() as searcher:
        results = searcher.search(query_obj, limit=limit)
        matches = [
            {
                "title": hit["title"],
                "url": hit["url"],
                "md_url": hit.get("md_url", ""),
                "section": hit.get("section", endpoint),
                "glimpse": hit["glimpse"],
                "score": round(hit.score, 2),
            }
            for hit in results
        ]

    if not matches:
        return [], "No strong matches found; showing default ordering."
    return matches, None


# ---------------------------------------------------------------------------
# HF Documentation - Formatting
# ---------------------------------------------------------------------------


def _format_results(
    endpoint: str,
    items: list[dict[str, Any]],
    total: int,
    query: str | None = None,
    note: str | None = None,
) -> str:
    """Format search results as readable text."""
    base_url = f"https://huggingface.co/docs/{endpoint}"
    out = f"Documentation structure for: {base_url}\n\n"

    if query:
        out += f"Query: '{query}' → showing {len(items)} result(s) out of {total} pages"
        if note:
            out += f" ({note})"
        out += "\n\n"
    else:
        out += f"Found {len(items)} page(s) (total available: {total}).\n"
        if note:
            out += f"({note})\n"
        out += "\n"

    for i, item in enumerate(items, 1):
        out += f"{i}. **{item['title']}**\n"
        out += f"   URL: {item['url']}\n"
        out += f"   Section: {item.get('section', endpoint)}\n"
        if query and "score" in item:
            out += f"   Relevance score: {item['score']:.2f}\n"
        out += f"   Glimpse: {item['glimpse']}\n\n"

    return out


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def explore_hf_docs_handler(arguments: dict[str, Any]) -> tuple[str, bool]:
    """Explore documentation structure with optional search query."""
    endpoint = arguments.get("endpoint", "").lstrip("/")
    query = arguments.get("query")
    max_results = arguments.get("max_results")

    if not endpoint:
        return "Error: No endpoint provided", False

    # Gradio uses its own API
    if endpoint.lower() == "gradio":
        try:
            clean_query = (
                query.strip() if isinstance(query, str) and query.strip() else None
            )
            content = await _fetch_gradio_docs(clean_query)
            header = "# Gradio Documentation\n\n"
            if clean_query:
                header += f"Query: '{clean_query}'\n\n"
            header += "Source: https://gradio.app/docs\n\n---\n\n"
            return header + content, True
        except httpx.HTTPStatusError as e:
            return f"HTTP error fetching Gradio docs: {e.response.status_code}", False
        except httpx.RequestError as e:
            return f"Request error fetching Gradio docs: {str(e)}", False
        except Exception as e:
            return f"Error fetching Gradio docs: {str(e)}", False

    # HF docs
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return "Error: HF_TOKEN environment variable not set", False

    try:
        max_results_int = int(max_results) if max_results is not None else None
    except (TypeError, ValueError):
        return "Error: max_results must be an integer", False

    if max_results_int is not None and max_results_int <= 0:
        return "Error: max_results must be greater than zero", False

    try:
        docs = await _get_docs(hf_token, endpoint)
        total = len(docs)

        # Determine limit
        if max_results_int is None:
            limit = DEFAULT_MAX_RESULTS
            limit_note = f"Showing top {DEFAULT_MAX_RESULTS} results (set max_results to adjust)."
        elif max_results_int > MAX_RESULTS_CAP:
            limit = MAX_RESULTS_CAP
            limit_note = f"Requested {max_results_int} but showing top {MAX_RESULTS_CAP} (maximum)."
        else:
            limit = max_results_int
            limit_note = None

        # Search or paginate
        clean_query = (
            query.strip() if isinstance(query, str) and query.strip() else None
        )
        fallback_msg = None

        if clean_query:
            results, fallback_msg = await _search_docs(
                endpoint, docs, clean_query, limit
            )
            if not results:
                results = docs[:limit]
        else:
            results = docs[:limit]

        # Combine notes
        notes = []
        if fallback_msg:
            notes.append(fallback_msg)
        if limit_note:
            notes.append(limit_note)
        note = "; ".join(notes) if notes else None

        return _format_results(endpoint, results, total, clean_query, note), True

    except httpx.HTTPStatusError as e:
        return f"HTTP error: {e.response.status_code} - {e.response.text[:200]}", False
    except httpx.RequestError as e:
        return f"Request error: {str(e)}", False
    except ValueError as e:
        return f"Error: {str(e)}", False
    except Exception as e:
        return f"Unexpected error: {str(e)}", False


async def hf_docs_fetch_handler(arguments: dict[str, Any]) -> tuple[str, bool]:
    """Fetch full markdown content of a documentation page."""
    url = arguments.get("url", "")
    if not url:
        return "Error: No URL provided", False

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return "Error: HF_TOKEN environment variable not set", False

    if not url.endswith(".md"):
        url = f"{url}.md"

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(
                url, headers={"Authorization": f"Bearer {hf_token}"}
            )
            resp.raise_for_status()
        return f"Documentation from: {url}\n\n{resp.text}", True
    except httpx.HTTPStatusError as e:
        return (
            f"HTTP error fetching {url}: {e.response.status_code} - {e.response.text[:200]}",
            False,
        )
    except httpx.RequestError as e:
        return f"Request error fetching {url}: {str(e)}", False
    except Exception as e:
        return f"Error fetching documentation: {str(e)}", False


# ---------------------------------------------------------------------------
# OpenAPI Search
# ---------------------------------------------------------------------------


async def _fetch_openapi_spec() -> dict[str, Any]:
    """Fetch and cache HuggingFace OpenAPI specification."""
    global _openapi_cache
    if _openapi_cache is not None:
        return _openapi_cache

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.get("https://huggingface.co/.well-known/openapi.json")
        resp.raise_for_status()

    _openapi_cache = resp.json()
    return _openapi_cache


def _extract_all_tags(spec: dict[str, Any]) -> list[str]:
    """Extract all unique tags from OpenAPI spec."""
    tags = set()
    for tag_obj in spec.get("tags", []):
        if "name" in tag_obj:
            tags.add(tag_obj["name"])
    for path_item in spec.get("paths", {}).values():
        for method, op in path_item.items():
            if method in ["get", "post", "put", "delete", "patch", "head", "options"]:
                for tag in op.get("tags", []):
                    tags.add(tag)
    return sorted(tags)


def _extract_all_endpoints(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract all endpoints from OpenAPI spec."""
    servers = spec.get("servers", [])
    base_url = (
        servers[0].get("url", "https://huggingface.co")
        if servers
        else "https://huggingface.co"
    )

    endpoints = []
    for path, path_item in spec.get("paths", {}).items():
        for method, op in path_item.items():
            if method not in ["get", "post", "put", "delete", "patch", "head", "options"]:
                continue
            endpoints.append({
                "path": path,
                "method": method.upper(),
                "operationId": op.get("operationId", ""),
                "summary": op.get("summary", ""),
                "description": op.get("description", ""),
                "tags": " ".join(op.get("tags", [])),
                "parameters": op.get("parameters", []),
                "request_body": op.get("requestBody", {}),
                "responses": op.get("responses", {}),
                "base_url": base_url,
            })
    return endpoints


async def _build_openapi_index() -> tuple[Any, MultifieldParser, list[dict[str, Any]]]:
    """Build or retrieve cached Whoosh index for OpenAPI endpoints."""
    global _openapi_index_cache
    async with _cache_lock:
        if _openapi_index_cache is not None:
            return _openapi_index_cache

    spec = await _fetch_openapi_spec()
    endpoints = _extract_all_endpoints(spec)

    analyzer = StemmingAnalyzer()
    schema = Schema(
        path=ID(stored=True, unique=True),
        method=ID(stored=True),
        operationId=TEXT(stored=True, analyzer=analyzer),
        summary=TEXT(stored=True, analyzer=analyzer),
        description=TEXT(stored=True, analyzer=analyzer),
        tags=TEXT(stored=True, analyzer=analyzer),
        param_names=TEXT(stored=False, analyzer=analyzer),
    )
    storage = RamStorage()
    index = storage.create_index(schema)
    writer = index.writer()

    for ep in endpoints:
        param_names = " ".join(p.get("name", "") for p in ep.get("parameters", []))
        writer.add_document(
            path=ep["path"],
            method=ep["method"],
            operationId=ep.get("operationId", ""),
            summary=ep.get("summary", ""),
            description=ep.get("description", ""),
            tags=ep.get("tags", ""),
            param_names=param_names,
        )
    writer.commit()

    parser = MultifieldParser(
        ["summary", "description", "operationId", "tags", "param_names"],
        schema=schema,
        fieldboosts={"summary": 3.0, "operationId": 2.0, "description": 1.0, "tags": 1.5},
        group=OrGroup,
    )

    async with _cache_lock:
        _openapi_index_cache = (index, parser, endpoints)
    return index, parser, endpoints


async def _search_openapi(
    query: str, tag: str | None, limit: int = 20
) -> tuple[list[dict[str, Any]], str | None]:
    """Search OpenAPI endpoints using Whoosh. Returns (results, fallback_message)."""
    index, parser, endpoints = await _build_openapi_index()

    try:
        query_obj = parser.parse(query)
    except Exception:
        return [], "Query contained unsupported syntax."

    with index.searcher() as searcher:
        results = searcher.search(query_obj, limit=limit * 2)  # Get extra for tag filtering
        matches = []
        for hit in results:
            # Find full endpoint data
            ep = next((e for e in endpoints if e["path"] == hit["path"] and e["method"] == hit["method"]), None)
            if ep is None:
                continue
            # Filter by tag if provided
            if tag and tag not in ep.get("tags", ""):
                continue
            matches.append({**ep, "score": round(hit.score, 2)})
            if len(matches) >= limit:
                break

    return matches, None if matches else "No matches found for query."


def _generate_curl_example(endpoint: dict[str, Any]) -> str:
    """Generate curl command example for an endpoint."""
    method = endpoint["method"]
    path = endpoint["path"]
    base_url = endpoint["base_url"]

    # Build URL with path parameters
    full_path = path
    for param in endpoint.get("parameters", []):
        if param.get("in") == "path" and param.get("required"):
            name = param["name"]
            example = param.get(
                "example", param.get("schema", {}).get("example", f"<{name}>")
            )
            full_path = full_path.replace(f"{{{name}}}", str(example))

    curl = f"curl -X {method} \\\n  '{base_url}{full_path}'"

    # Add query parameters
    query_params = [p for p in endpoint.get("parameters", []) if p.get("in") == "query"]
    if query_params and query_params[0].get("required"):
        param = query_params[0]
        example = param.get("example", param.get("schema", {}).get("example", "value"))
        curl += f"?{param['name']}={example}"

    curl += " \\\n  -H 'Authorization: Bearer $HF_TOKEN'"

    # Add request body
    if method in ["POST", "PUT", "PATCH"] and endpoint.get("request_body"):
        content = endpoint["request_body"].get("content", {})
        if "application/json" in content:
            curl += " \\\n  -H 'Content-Type: application/json'"
            schema = content["application/json"].get("schema", {})
            example = schema.get("example", "{}")
            if isinstance(example, dict):
                example = json.dumps(example, indent=2)
            curl += f" \\\n  -d '{example}'"

    return curl


def _format_parameters(parameters: list[dict[str, Any]]) -> str:
    """Format parameter information from OpenAPI spec."""
    if not parameters:
        return ""

    path_params = [p for p in parameters if p.get("in") == "path"]
    query_params = [p for p in parameters if p.get("in") == "query"]
    header_params = [p for p in parameters if p.get("in") == "header"]

    output = []

    for label, params in [
        ("Path Parameters", path_params),
        ("Query Parameters", query_params),
        ("Header Parameters", header_params),
    ]:
        if not params:
            continue
        if output:
            output.append("")
        output.append(f"**{label}:**")
        for p in params:
            name = p.get("name", "")
            required = " (required)" if p.get("required") else " (optional)"
            desc = p.get("description", "")
            ptype = p.get("schema", {}).get("type", "string")
            example = p.get("example") or p.get("schema", {}).get("example", "")

            output.append(f"- `{name}` ({ptype}){required}: {desc}")
            if example:
                output.append(f"  Example: `{example}`")

    return "\n".join(output)


def _format_response_info(responses: dict[str, Any]) -> str:
    """Format response information from OpenAPI spec."""
    if not responses:
        return "No response information available"

    output = []
    for status, resp_obj in list(responses.items())[:3]:
        desc = resp_obj.get("description", "")
        output.append(f"- **{status}**: {desc}")
        content = resp_obj.get("content", {})
        if "application/json" in content:
            schema = content["application/json"].get("schema", {})
            if "type" in schema:
                output.append(f"  Returns: {schema.get('type', 'object')}")

    return "\n".join(output)


def _format_openapi_results(
    results: list[dict[str, Any]],
    tag: str | None = None,
    query: str | None = None,
    note: str | None = None,
) -> str:
    """Format OpenAPI search results with curl examples."""
    if not results:
        if query and tag:
            return f"No API endpoints found matching '{query}' in tag '{tag}'"
        elif query:
            return f"No API endpoints found matching '{query}'"
        elif tag:
            return f"No API endpoints found with tag '{tag}'"
        return "No API endpoints found"

    # Build header
    if query and tag:
        out = f"# API Endpoints matching '{query}' (tag: `{tag}`)\n\n"
    elif query:
        out = f"# API Endpoints matching '{query}'\n\n"
    elif tag:
        out = f"# API Endpoints for tag: `{tag}`\n\n"
    else:
        out = "# API Endpoints\n\n"

    out += f"Found {len(results)} endpoint(s)"
    if note:
        out += f" ({note})"
    out += "\n\n---\n\n"

    for i, ep in enumerate(results, 1):
        out += f"## {i}. {ep['method']} {ep['path']}\n\n"

        if query and "score" in ep:
            out += f"**Relevance:** {ep['score']:.2f}\n\n"

        if ep.get("summary"):
            out += f"**Summary:** {ep['summary']}\n\n"

        if ep.get("description"):
            desc = ep["description"][:300]
            if len(ep["description"]) > 300:
                desc += "..."
            out += f"**Description:** {desc}\n\n"

        if ep.get("tags"):
            out += f"**Tags:** {ep['tags']}\n\n"

        params_info = _format_parameters(ep.get("parameters", []))
        if params_info:
            out += params_info + "\n\n"

        out += "**Usage:**\n```bash\n"
        out += _generate_curl_example(ep)
        out += "\n```\n\n"

        out += "**Returns:**\n"
        out += _format_response_info(ep["responses"])
        out += "\n\n---\n\n"

    return out


async def search_openapi_handler(arguments: dict[str, Any]) -> tuple[str, bool]:
    """Search HuggingFace OpenAPI specification by query and/or tag."""
    tag = arguments.get("tag", "").strip() or None
    query = arguments.get("query", "").strip() or None

    if not tag and not query:
        return "Error: Provide either 'query' (keyword search) or 'tag' (category filter), or both.", False

    try:
        note = None

        # If query provided, try Whoosh search first
        if query:
            results, search_note = await _search_openapi(query, tag, limit=20)

            # If Whoosh found results, return them
            if results:
                return _format_openapi_results(results, tag=tag, query=query, note=search_note), True

            # Whoosh found nothing - fall back to tag-based if tag provided
            if tag:
                note = f"No matches for '{query}'; showing all endpoints in tag '{tag}'"
            else:
                # No tag to fall back to
                return _format_openapi_results([], query=query), True

        # Tag-based search (either as fallback or primary)
        if tag:
            _, _, endpoints = await _build_openapi_index()
            results = [ep for ep in endpoints if tag in ep.get("tags", "")]
            return _format_openapi_results(results, tag=tag, query=None, note=note), True

        return "Error: No results found", False

    except httpx.HTTPStatusError as e:
        return f"HTTP error fetching OpenAPI spec: {e.response.status_code}", False
    except httpx.RequestError as e:
        return f"Request error: {str(e)}", False
    except Exception as e:
        return f"Error searching OpenAPI spec: {str(e)}", False


async def _get_api_search_tool_spec() -> dict[str, Any]:
    """Generate OpenAPI tool spec with tags populated at runtime."""
    spec = await _fetch_openapi_spec()
    tags = _extract_all_tags(spec)

    return {
        "name": "find_hf_api",
        "description": (
            "Find HuggingFace Hub REST API endpoints to make HTTP requests. Returns curl examples with authentication. "
            "⚠️ USE THIS TOOL when you need to call the HF Hub API directly - for operations like: "
            "uploading/downloading files, managing repos, listing models/datasets, getting user info, "
            "managing webhooks, collections, discussions, or any Hub interaction not covered by other tools. "
            "**Use cases:** (1) 'Stream Space logs' → query='space logs', "
            "(2) 'Get Space metrics/Zero-GPU usage' → query='space metrics', "
            "(3) 'List organization members' → query='organization members', "
            "(4) 'Generate repo access token' → query='jwt token', "
            "(5) 'Check repo security scan' → query='security scan'. "
            "**Search modes:** Use 'query' for keyword search, 'tag' to browse a category, or both. "
            "If query finds no results, falls back to showing all endpoints in the tag. "
            "**Output:** Full endpoint details with method, path, parameters, curl command, and response schema."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Keyword search across endpoint summaries, descriptions, and operation IDs. "
                        "Examples: 'upload file', 'create repository', 'list user models', 'delete branch', "
                        "'webhook', 'collection', 'discussion comments'. Supports stemming (upload/uploading both work)."
                    ),
                },
                "tag": {
                    "type": "string",
                    "enum": tags,
                    "description": (
                        "Filter by API category. Use alone to browse all endpoints in a category, "
                        "or combine with 'query' to search within a category."
                    ),
                },
            },
            "required": [],
        },
    }


# ---------------------------------------------------------------------------
# Tool Specifications
# ---------------------------------------------------------------------------

DOC_ENDPOINTS = [
    "hub",
    "transformers",
    "diffusers",
    "datasets",
    "gradio",
    "trackio",
    "smolagents",
    "huggingface_hub",
    "huggingface.js",
    "transformers.js",
    "inference-providers",
    "inference-endpoints",
    "peft",
    "accelerate",
    "optimum",
    "tokenizers",
    "courses",
    "evaluate",
    "tasks",
    "dataset-viewer",
    "trl",
    "simulate",
    "sagemaker",
    "timm",
    "safetensors",
    "tgi",
    "setfit",
    "lerobot",
    "autotrain",
    "tei",
    "bitsandbytes",
    "sentence_transformers",
    "chat-ui",
    "leaderboards",
    "lighteval",
    "argilla",
    "distilabel",
    "microsoft-azure",
    "kernels",
    "google-cloud",
]

EXPLORE_HF_DOCS_TOOL_SPEC = {
    "name": "explore_hf_docs",
    "description": (
        "Explore Hugging Face documentation structure and discover available pages with 200-character previews. "
        "⚠️ MANDATORY: ALWAYS use this BEFORE implementing any ML task (training, fine-tuning, data processing, inference). "
        "Your training data may be outdated - current documentation is the source of truth. "
        "**Use when:** (1) Starting any implementation task, (2) User asks 'how to' questions, "
        "(3) Before writing training/processing code, (4) Researching library capabilities, "
        "(5) Verifying API syntax and parameters. "
        "**Pattern:** explore (discover structure) → fetch_hf_docs (get details) → implement with researched approach. "
        "Returns: Sidebar navigation with titles, URLs, and glimpses of all pages in the selected documentation. "
        "**Then:** Use fetch_hf_docs with specific URLs from results to get full content. "
        "**Critical for reliability:** Never implement based on internal knowledge without checking current docs first - APIs change frequently."
        " By default returns the top 20 results; set max_results (max 50) to adjust."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "endpoint": {
                "type": "string",
                "enum": DOC_ENDPOINTS,
                "description": (
                    "The documentation endpoint to explore. Each endpoint corresponds to a major section of the Hugging Face documentation:\n\n"
                    "• courses — All Hugging Face courses (LLM, robotics, MCP, smol (llm training), agents, deep RL, computer vision, games, diffusion, 3D, audio) and the cookbook recipes. Probably the best place for examples.\n"
                    "• hub — Find answers to questions about models/datasets/spaces, auth, versioning, metadata.\n"
                    "• transformers — Core model library: architectures, configs, tokenizers, training & inference APIs.\n"
                    "• diffusers — Diffusion pipelines, schedulers, fine-tuning, training, and deployment patterns.\n"
                    "• datasets — Dataset loading, streaming, processing, Arrow format, Hub integration.\n"
                    "• gradio — UI components and demos for ML models. Uses Gradio's native API: without query returns full docs (llms.txt), with query uses embedding search for precise results.\n"
                    "• trackio — Experiment tracking, metrics logging, and run comparison.\n"
                    "• smolagents — Lightweight agent abstractions and tool-using patterns.\n"
                    "• huggingface_hub — Python client for Hub operations (auth, upload/download, repo management).\n"
                    "• huggingface.js — JS/TS client for Hub APIs in browser and Node.\n"
                    "• transformers.js — Run Transformer models in browser/Node via WebGPU/WASM.\n"
                    "• inference-providers — Unified interface for third-party inference backends.\n"
                    "• inference-endpoints — Managed, scalable model deployments on HF infrastructure.\n"
                    "• peft — Parameter-efficient fine-tuning methods (LoRA, adapters, etc.).\n"
                    "• accelerate — Hardware-agnostic, distributed and mixed-precision training orchestration.\n"
                    "• optimum — Hardware-aware optimization and model export tooling, including Habana, Neuron, Intel, ExecuTorch, and TPU variants.\n"
                    "• tokenizers — Fast tokenizer internals, training, and low-level APIs.\n"
                    "• evaluate — Metrics, evaluation workflows, and training-loop integration.\n"
                    "• tasks — Canonical task definitions and model categorization.\n"
                    "• dataset-viewer — Dataset preview, streaming views, and viewer internals.\n"
                    "• trl — RLHF, DPO, PPO, and SFT utilities for LLMs.\n"
                    "• simulate — Experimental simulation tools and workflows.\n"
                    "• sagemaker — Deploying Hugging Face models on AWS SageMaker.\n"
                    "• timm — Image model zoo and utilities via HF integrations.\n"
                    "• safetensors — Safe, fast tensor serialization format.\n"
                    "• tgi — High-throughput text generation server for LLMs.\n"
                    "• setfit — Few-shot text classification via sentence embeddings.\n"
                    "• lerobot — Robotics datasets, policies, and learning workflows.\n"
                    "• autotrain — No/low-code model training on Hugging Face.\n"
                    "• tei — Optimized inference server for embedding workloads.\n"
                    "• bitsandbytes — Quantization and memory-efficient optimizers.\n"
                    "• sentence_transformers — Embedding models, training recipes, similarity/search workflows.\n"
                    "• chat-ui — Reference chat interfaces for LLM deployment.\n"
                    "• leaderboards — Evaluation leaderboards and submission mechanics.\n"
                    "• lighteval — Lightweight, reproducible LLM evaluation framework.\n"
                    "• argilla — Data annotation, feedback, and human-in-the-loop workflows.\n"
                    "• distilabel — Synthetic data generation and distillation pipelines.\n"
                    "• microsoft-azure — Azure deployment and integration guides.\n"
                    "• kernels — Lightweight execution environments and notebook-style workflows.\n"
                    "• google-cloud — GCP deployment and serving workflows.\n"
                ),
            },
            "query": {
                "type": "string",
                "description": (
                    "Optional keyword query to rank and filter documentation pages. "
                    "For Gradio, use concise queries like 'how to use the image component' or 'audio component demo'."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Max results (default 20, max 50). Ignored for Gradio.",
                "minimum": 1,
                "maximum": 50,
            },
        },
        "required": ["endpoint"],
    },
}

HF_DOCS_FETCH_TOOL_SPEC = {
    "name": "fetch_hf_docs",
    "description": (
        "Fetch full markdown content of a specific HF documentation page. "
        "⚠️ CRITICAL: Use this after explore_hf_docs to get detailed implementation guidance. "
        "**Use when:** (1) Found relevant page in explore_hf_docs results, (2) Need complete API documentation, "
        "(3) Need training method details (SFT/DPO/GRPO), (4) Need configuration examples, "
        "(5) Need parameter descriptions and usage patterns. "
        "**Pattern:** explore_hf_docs (find relevant page) → fetch_hf_docs (get full content) → implement using documented approach. "
        "Provide full URL from explore_hf_docs results (e.g., 'https://huggingface.co/docs/trl/sft_trainer'). "
        "Returns: Complete markdown documentation with examples, parameters, and usage patterns. "
        "**For training tasks:** ALWAYS fetch trainer docs (SFTConfig, DPOConfig, etc.) before creating training scripts. "
        "**Critical for reliability:** This ensures you use current APIs and best practices."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": (
                    "The full URL to the documentation page. "
                    "Example: 'https://huggingface.co/docs/trl/dpo_trainer' "
                    "The .md extension will be added automatically if not present."
                ),
            },
        },
        "required": ["url"],
    },
}
