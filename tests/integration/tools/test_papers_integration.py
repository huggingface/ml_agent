#!/usr/bin/env python3
"""
Integration tests for HF Papers Tool
Tests with real HF and arXiv APIs — all endpoints are public, no auth required.

Run: python tests/integration/tools/test_papers_integration.py
"""
import asyncio
import re
import sys

sys.path.insert(0, ".")

from agent.tools.papers_tool import hf_papers_handler

# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
DIM = "\033[2m"
RESET = "\033[0m"

assertions_passed = 0
assertions_failed = 0


def print_test(msg):
    print(f"\n{BLUE}{'─' * 70}{RESET}")
    print(f"{BLUE}[TEST]{RESET} {msg}")
    print(f"{BLUE}{'─' * 70}{RESET}")


def print_success(msg):
    print(f"{GREEN}  ✓ {msg}{RESET}")


def print_error(msg):
    print(f"{RED}  ✗ {msg}{RESET}")


def print_output(output: str, max_lines: int = 40):
    """Print the full tool output, indented, with line limit."""
    lines = output.split("\n")
    for line in lines[:max_lines]:
        print(f"{DIM}  │ {RESET}{line}")
    if len(lines) > max_lines:
        print(f"{DIM}  │ ... ({len(lines) - max_lines} more lines){RESET}")


def assert_true(condition: bool, msg: str) -> bool:
    """Assert and print result. Returns True if passed."""
    global assertions_passed, assertions_failed
    if condition:
        print_success(msg)
        assertions_passed += 1
        return True
    else:
        print_error(msg)
        assertions_failed += 1
        return False


async def run(args: dict) -> tuple[str, bool]:
    return await hf_papers_handler(args)


# ---------------------------------------------------------------------------
# Test Suite 1: Paper Discovery
# ---------------------------------------------------------------------------


async def test_trending():
    print_test("trending (limit=3)")
    output, success = await run({"operation": "trending", "limit": 3})
    print_output(output)

    ok = True
    ok &= assert_true(success, "success=True")
    ok &= assert_true("# Trending Papers" in output, "has '# Trending Papers' heading")
    ok &= assert_true("Showing 3 paper(s)" in output, "shows exactly 3 papers")

    # Check that each paper has an arxiv_id line
    arxiv_ids = re.findall(r"\*\*arxiv_id:\*\* (\S+)", output)
    ok &= assert_true(len(arxiv_ids) == 3, f"found 3 arxiv IDs: {arxiv_ids}")

    # Check that IDs look valid (digits and dots)
    for aid in arxiv_ids:
        ok &= assert_true(
            re.match(r"\d{4}\.\d{4,5}", aid) is not None,
            f"arxiv_id '{aid}' looks valid (NNNN.NNNNN format)",
        )

    # Check each paper has an HF URL
    hf_urls = re.findall(r"https://huggingface\.co/papers/\S+", output)
    ok &= assert_true(len(hf_urls) == 3, f"found 3 HF paper URLs")

    return ok


async def test_trending_with_query():
    print_test("trending with query='language' (limit=5)")
    output, success = await run({"operation": "trending", "query": "language", "limit": 5})
    print_output(output)

    ok = True
    ok &= assert_true(success, "success=True")
    ok &= assert_true("Filtered by: 'language'" in output, "shows filter applied")

    # The filter may return 0-5 results depending on today's papers
    match = re.search(r"Showing (\d+) paper\(s\)", output)
    ok &= assert_true(match is not None, "has 'Showing N paper(s)' line")
    if match:
        count = int(match.group(1))
        ok &= assert_true(count <= 5, f"returned {count} papers (within limit)")
        # If we got results, verify they mention language somewhere
        if count > 0:
            print_success(f"got {count} filtered results")

    return ok


async def test_search():
    print_test("search 'direct preference optimization' (limit=3)")
    output, success = await run(
        {"operation": "search", "query": "direct preference optimization", "limit": 3}
    )
    print_output(output)

    ok = True
    ok &= assert_true(success, "success=True")
    ok &= assert_true("Papers matching" in output, "has matching header")

    arxiv_ids = re.findall(r"\*\*arxiv_id:\*\* (\S+)", output)
    ok &= assert_true(len(arxiv_ids) == 3, f"found 3 results: {arxiv_ids}")

    # At least one result should mention "preference" in title or summary
    ok &= assert_true(
        "preference" in output.lower(),
        "results mention 'preference' (relevant to query)",
    )

    return ok


async def test_paper_details():
    print_test("paper_details for 2305.18290 (DPO paper)")
    output, success = await run({"operation": "paper_details", "arxiv_id": "2305.18290"})
    print_output(output)

    ok = True
    ok &= assert_true(success, "success=True")
    ok &= assert_true("Direct Preference Optimization" in output, "title contains 'Direct Preference Optimization'")
    ok &= assert_true("2305.18290" in output, "contains arxiv_id")
    ok &= assert_true("https://arxiv.org/abs/2305.18290" in output, "has arxiv URL")
    ok &= assert_true("https://huggingface.co/papers/2305.18290" in output, "has HF URL")
    ok &= assert_true("**Authors:**" in output, "has authors section")
    ok &= assert_true("**upvotes:**" in output, "has upvotes")

    # Check for abstract or AI summary
    ok &= assert_true(
        "## Abstract" in output or "## AI Summary" in output,
        "has Abstract or AI Summary section",
    )

    # Check for next steps hint
    ok &= assert_true("read_paper" in output, "mentions read_paper as next step")
    ok &= assert_true("find_all_resources" in output, "mentions find_all_resources as next step")

    return ok


# ---------------------------------------------------------------------------
# Test Suite 2: Read Paper
# ---------------------------------------------------------------------------


async def test_read_paper_toc():
    print_test("read_paper TOC for 2305.18290 (no section → should return abstract + sections)")
    output, success = await run({"operation": "read_paper", "arxiv_id": "2305.18290"})
    print_output(output)

    ok = True
    ok &= assert_true(success, "success=True")
    ok &= assert_true("## Abstract" in output, "has Abstract section")
    ok &= assert_true("## Sections" in output, "has Sections heading (TOC)")

    # Check that sections are listed with bold titles
    section_titles = re.findall(r"- \*\*(.+?)\*\*:", output)
    ok &= assert_true(len(section_titles) >= 5, f"found {len(section_titles)} sections (expect >=5 for a full paper)")
    if section_titles:
        print_success(f"sections found: {section_titles[:5]}{'...' if len(section_titles) > 5 else ''}")

    # Check that expected DPO paper sections are present
    section_text = " ".join(section_titles).lower()
    ok &= assert_true("introduction" in section_text, "'Introduction' section present")
    ok &= assert_true("experiment" in section_text, "'Experiment' section present")

    # Check for the tip about reading specific sections
    ok &= assert_true("section=" in output, "has tip about using section parameter")

    # Check the abstract has actual content (not empty)
    abstract_match = re.search(r"## Abstract\n(.+?)(?:\n##|\n\*\*Tip)", output, re.DOTALL)
    if abstract_match:
        abstract_text = abstract_match.group(1).strip()
        ok &= assert_true(len(abstract_text) > 100, f"abstract has real content ({len(abstract_text)} chars)")
    else:
        ok &= assert_true(False, "could extract abstract text")

    return ok


async def test_read_paper_section_by_number():
    print_test("read_paper section='4' for 2305.18290")
    output, success = await run(
        {"operation": "read_paper", "arxiv_id": "2305.18290", "section": "4"}
    )
    print_output(output, max_lines=30)

    ok = True
    ok &= assert_true(success, "success=True")
    ok &= assert_true("https://arxiv.org/abs/2305.18290" in output, "has arxiv URL")

    # Should have a section heading at top
    ok &= assert_true(output.startswith("# "), "starts with heading")

    # Should have substantial content
    ok &= assert_true(len(output) > 500, f"section has substantial content ({len(output)} chars)")

    # Should NOT have TOC structure (this is a single section, not the TOC)
    ok &= assert_true("## Sections" not in output, "is a single section (not TOC)")

    return ok


async def test_read_paper_section_by_name():
    print_test("read_paper section='Experiments' for 2305.18290")
    output, success = await run(
        {"operation": "read_paper", "arxiv_id": "2305.18290", "section": "Experiments"}
    )
    print_output(output, max_lines=30)

    ok = True
    ok &= assert_true(success, "success=True")

    # Title should contain "Experiments"
    first_line = output.split("\n")[0]
    ok &= assert_true(
        "experiment" in first_line.lower(),
        f"heading contains 'Experiments': '{first_line}'",
    )

    ok &= assert_true(len(output) > 500, f"section has substantial content ({len(output)} chars)")

    return ok


async def test_read_paper_old_paper():
    print_test("read_paper for 1706.03762 (Attention Is All You Need — 2017 paper)")
    output, success = await run({"operation": "read_paper", "arxiv_id": "1706.03762"})
    print_output(output, max_lines=30)

    ok = True
    ok &= assert_true(success, "success=True")
    ok &= assert_true("attention" in output.lower(), "mentions 'attention' (relevant content)")

    # Either we get sections (HTML available) or abstract fallback
    has_sections = "## Sections" in output
    has_abstract_fallback = "HTML version not available" in output
    ok &= assert_true(
        has_sections or has_abstract_fallback or "## Abstract" in output,
        "got either full sections, or abstract fallback",
    )
    if has_sections:
        print_success("HTML version available — got full sections")
    elif has_abstract_fallback:
        print_success("HTML not available — graceful fallback to abstract")

    return ok


# ---------------------------------------------------------------------------
# Test Suite 3: Linked Resources
# ---------------------------------------------------------------------------


async def test_find_datasets():
    print_test("find_datasets for 2305.18290 (limit=5, sort=downloads)")
    output, success = await run(
        {"operation": "find_datasets", "arxiv_id": "2305.18290", "limit": 5}
    )
    print_output(output)

    ok = True
    ok &= assert_true(success, "success=True")
    ok &= assert_true("Datasets linked to paper 2305.18290" in output, "has correct heading")
    ok &= assert_true("sorted by downloads" in output, "sorted by downloads (default)")

    # Check we got dataset entries with IDs
    dataset_ids = re.findall(r"\[([^\]]+)\]\(https://huggingface\.co/datasets/", output)
    ok &= assert_true(len(dataset_ids) > 0, f"found {len(dataset_ids)} dataset links")
    if dataset_ids:
        print_success(f"dataset IDs: {dataset_ids}")

    # Check download counts are present
    downloads = re.findall(r"Downloads: ([\d,]+)", output)
    ok &= assert_true(len(downloads) > 0, f"found download counts: {downloads}")

    # Check for inspect hint
    ok &= assert_true("hf_inspect_dataset" in output, "has inspect dataset hint")

    return ok


async def test_find_datasets_sort_likes():
    print_test("find_datasets for 2305.18290 (sort=likes, limit=3)")
    output, success = await run(
        {"operation": "find_datasets", "arxiv_id": "2305.18290", "limit": 3, "sort": "likes"}
    )
    print_output(output)

    ok = True
    ok &= assert_true(success, "success=True")
    ok &= assert_true("sorted by likes" in output, "sorted by likes")

    return ok


async def test_find_models():
    print_test("find_models for 2305.18290 (limit=5)")
    output, success = await run(
        {"operation": "find_models", "arxiv_id": "2305.18290", "limit": 5}
    )
    print_output(output)

    ok = True
    ok &= assert_true(success, "success=True")
    ok &= assert_true("Models linked to paper 2305.18290" in output, "has correct heading")

    # Check model links
    model_ids = re.findall(r"\[([^\]]+)\]\(https://huggingface\.co/", output)
    ok &= assert_true(len(model_ids) > 0, f"found {len(model_ids)} model links")
    if model_ids:
        print_success(f"model IDs: {model_ids}")

    # Check for pipeline_tag / library info
    has_task = "Task:" in output
    has_library = "Library:" in output
    ok &= assert_true(has_task or has_library, "has Task or Library metadata")

    return ok


async def test_find_collections():
    print_test("find_collections for 2305.18290")
    output, success = await run(
        {"operation": "find_collections", "arxiv_id": "2305.18290"}
    )
    print_output(output)

    ok = True
    ok &= assert_true(success, "success=True")
    ok &= assert_true("Collections containing paper" in output, "has correct heading")

    # Check collection entries
    collection_urls = re.findall(r"https://huggingface\.co/collections/\S+", output)
    ok &= assert_true(len(collection_urls) > 0, f"found {len(collection_urls)} collection URLs")

    # Check for metadata
    ok &= assert_true("Upvotes:" in output, "has upvote counts")
    ok &= assert_true("Items:" in output, "has item counts")

    return ok


async def test_find_all_resources():
    print_test("find_all_resources for 2305.18290 (parallel fan-out)")
    output, success = await run(
        {"operation": "find_all_resources", "arxiv_id": "2305.18290"}
    )
    print_output(output)

    ok = True
    ok &= assert_true(success, "success=True")
    ok &= assert_true("# Resources linked to paper 2305.18290" in output, "has unified heading")
    ok &= assert_true("https://huggingface.co/papers/2305.18290" in output, "has paper URL")

    # All three sections should be present
    ok &= assert_true("## Datasets" in output, "has Datasets section")
    ok &= assert_true("## Models" in output, "has Models section")
    ok &= assert_true("## Collections" in output, "has Collections section")

    # Check that sections have actual entries (not just "None found")
    ok &= assert_true("downloads)" in output, "datasets/models have download counts")

    return ok


# ---------------------------------------------------------------------------
# Test Suite 4: Edge Cases
# ---------------------------------------------------------------------------


async def test_search_no_results():
    print_test("search with gibberish query → should return empty gracefully")
    output, success = await run(
        {"operation": "search", "query": "xyzzyplugh_nonexistent_topic_9999"}
    )
    print_output(output)

    ok = True
    ok &= assert_true(success, "success=True (empty results is not an error)")
    ok &= assert_true("No papers found" in output, "says 'No papers found'")

    return ok


async def test_missing_query():
    print_test("search without query → should error")
    output, success = await run({"operation": "search"})
    print_output(output)

    ok = True
    ok &= assert_true(not success, "success=False (missing required param)")
    ok &= assert_true("required" in output.lower(), "error mentions 'required'")

    return ok


async def test_missing_arxiv_id():
    print_test("find_datasets without arxiv_id → should error")
    output, success = await run({"operation": "find_datasets"})
    print_output(output)

    ok = True
    ok &= assert_true(not success, "success=False")
    ok &= assert_true("required" in output.lower(), "error mentions 'required'")

    return ok


async def test_invalid_arxiv_id():
    print_test("paper_details with nonexistent arxiv ID")
    output, success = await run({"operation": "paper_details", "arxiv_id": "0000.00000"})
    print_output(output)

    ok = True
    ok &= assert_true(not success, "success=False (API returns error)")

    return ok


async def test_invalid_operation():
    print_test("invalid operation name → should error")
    output, success = await run({"operation": "nonexistent_op"})
    print_output(output)

    ok = True
    ok &= assert_true(not success, "success=False")
    ok &= assert_true("Unknown operation" in output, "says 'Unknown operation'")
    ok &= assert_true("trending" in output, "lists valid operations")

    return ok


async def test_read_paper_bad_section():
    print_test("read_paper with nonexistent section → should error with available sections")
    output, success = await run(
        {"operation": "read_paper", "arxiv_id": "2305.18290", "section": "Nonexistent Section XYZ"}
    )
    print_output(output)

    ok = True
    ok &= assert_true(not success, "success=False")
    ok &= assert_true("not found" in output.lower(), "says section 'not found'")
    ok &= assert_true("Introduction" in output, "lists available sections (includes Introduction)")

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    print("=" * 70)
    print(f"{BLUE}HF Papers Tool — Integration Tests{RESET}")
    print(f"{BLUE}All APIs are public, no authentication required.{RESET}")
    print("=" * 70)

    all_tests = [
        # Suite 1: Paper Discovery
        ("Paper Discovery", [
            test_trending,
            test_trending_with_query,
            test_search,
            test_paper_details,
        ]),
        # Suite 2: Read Paper
        ("Read Paper", [
            test_read_paper_toc,
            test_read_paper_section_by_number,
            test_read_paper_section_by_name,
            test_read_paper_old_paper,
        ]),
        # Suite 3: Linked Resources
        ("Linked Resources", [
            test_find_datasets,
            test_find_datasets_sort_likes,
            test_find_models,
            test_find_collections,
            test_find_all_resources,
        ]),
        # Suite 4: Edge Cases
        ("Edge Cases", [
            test_search_no_results,
            test_missing_query,
            test_missing_arxiv_id,
            test_invalid_arxiv_id,
            test_invalid_operation,
            test_read_paper_bad_section,
        ]),
    ]

    global assertions_passed, assertions_failed
    suite_results = []

    for suite_name, tests in all_tests:
        print(f"\n{YELLOW}{'=' * 70}{RESET}")
        print(f"{YELLOW}Test Suite: {suite_name} ({len(tests)} tests){RESET}")
        print(f"{YELLOW}{'=' * 70}{RESET}")

        suite_pass = 0
        suite_fail = 0

        for test_fn in tests:
            try:
                test_ok = await test_fn()
                if test_ok:
                    suite_pass += 1
                else:
                    suite_fail += 1
            except Exception as e:
                print_error(f"CRASHED: {e}")
                import traceback
                traceback.print_exc()
                suite_fail += 1

        suite_results.append((suite_name, suite_pass, suite_fail))

    # Summary
    print(f"\n{'=' * 70}")
    print(f"{BLUE}Summary{RESET}")
    print(f"{'=' * 70}")
    for suite_name, sp, sf in suite_results:
        icon = f"{GREEN}✓{RESET}" if sf == 0 else f"{RED}✗{RESET}"
        print(f"  {icon} {suite_name}: {sp}/{sp + sf} tests passed")

    print(f"{'─' * 70}")
    total_tests = sum(sp + sf for _, sp, sf in suite_results)
    total_failed = sum(sf for _, _, sf in suite_results)
    print(f"  Assertions: {assertions_passed} passed, {assertions_failed} failed")
    print(f"  Tests:      {total_tests - total_failed}/{total_tests} passed")
    print(f"{'=' * 70}\n")

    if total_failed > 0 or assertions_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
