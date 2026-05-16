"""Tests for convert_mcp_content_to_llm_content — MCP block → LiteLLM format."""

import base64

import pytest
from mcp.types import EmbeddedResource, ImageContent, TextContent

from agent.core.tools import convert_mcp_content_to_llm_content


# ── helpers ─────────────────────────────────────────────────────────────────


def _text(t: str) -> TextContent:
    return TextContent(type="text", text=t)


def _image(data: str = "aGVsbG8=", mime: str = "image/png") -> ImageContent:
    return ImageContent(type="image", data=data, mimeType=mime)


def _embedded_text(text: str, uri: str = "file:///test") -> EmbeddedResource:
    from mcp.types import TextResourceContents

    return EmbeddedResource(
        type="resource",
        resource=TextResourceContents(uri=uri, text=text),
    )


def _embedded_blob(
    blob: str = "aGVsbG8=", mime: str = "application/octet-stream", uri: str = "file:///test"
) -> EmbeddedResource:
    from mcp.types import BlobResourceContents

    return EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(uri=uri, blob=blob, mimeType=mime),
    )


# ── empty / trivial ──────────────────────────────────────────────────────────


def test_empty_list_returns_empty_string():
    assert convert_mcp_content_to_llm_content([]) == ""


# ── text-only: must return str ───────────────────────────────────────────────


def test_single_text_block_returns_str():
    result = convert_mcp_content_to_llm_content([_text("hello")])
    assert result == "hello"
    assert isinstance(result, str)


def test_multiple_text_blocks_joined_with_newline():
    result = convert_mcp_content_to_llm_content([_text("line one"), _text("line two")])
    assert result == "line one\nline two"
    assert isinstance(result, str)


def test_embedded_text_resource_returns_str():
    result = convert_mcp_content_to_llm_content([_embedded_text("resource text")])
    assert result == "resource text"
    assert isinstance(result, str)


def test_embedded_blob_resource_returns_binary_placeholder():
    result = convert_mcp_content_to_llm_content([_embedded_blob(mime="image/jpeg")])
    assert isinstance(result, str)
    assert "[Binary data: image/jpeg]" in result


def test_embedded_resource_without_text_or_blob_returns_uri_placeholder():
    from mcp.types import TextResourceContents

    resource = EmbeddedResource(
        type="resource",
        resource=TextResourceContents(uri="file:///mystery", text=""),
    )
    result = convert_mcp_content_to_llm_content([resource])
    assert isinstance(result, str)


def test_text_and_embedded_text_returns_str():
    result = convert_mcp_content_to_llm_content(
        [_text("intro"), _embedded_text("body")]
    )
    assert result == "intro\nbody"
    assert isinstance(result, str)


# ── image present: must return list[dict] ───────────────────────────────────


def test_single_image_returns_list():
    result = convert_mcp_content_to_llm_content([_image()])
    assert isinstance(result, list)
    assert len(result) == 1


def test_image_block_has_image_url_type():
    result = convert_mcp_content_to_llm_content([_image()])
    assert result[0]["type"] == "image_url"


def test_image_block_url_encodes_mime_and_data():
    data = base64.b64encode(b"fake-png-bytes").decode()
    result = convert_mcp_content_to_llm_content([_image(data=data, mime="image/png")])
    url = result[0]["image_url"]["url"]
    assert url == f"data:image/png;base64,{data}"


def test_jpeg_mime_type_is_preserved():
    data = base64.b64encode(b"fake-jpeg").decode()
    result = convert_mcp_content_to_llm_content([_image(data=data, mime="image/jpeg")])
    url = result[0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")


def test_mixed_text_and_image_returns_list():
    result = convert_mcp_content_to_llm_content([_text("caption"), _image()])
    assert isinstance(result, list)
    assert len(result) == 2


def test_mixed_text_block_has_correct_fields():
    result = convert_mcp_content_to_llm_content([_text("caption"), _image()])
    text_block = result[0]
    assert text_block["type"] == "text"
    assert text_block["text"] == "caption"


def test_mixed_order_preserved():
    data = base64.b64encode(b"px").decode()
    result = convert_mcp_content_to_llm_content(
        [_text("before"), _image(data=data), _text("after")]
    )
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0] == {"type": "text", "text": "before"}
    assert result[1]["type"] == "image_url"
    assert result[2] == {"type": "text", "text": "after"}


def test_multiple_images_all_appear_in_list():
    data1 = base64.b64encode(b"img1").decode()
    data2 = base64.b64encode(b"img2").decode()
    result = convert_mcp_content_to_llm_content(
        [_image(data=data1, mime="image/png"), _image(data=data2, mime="image/webp")]
    )
    assert isinstance(result, list)
    assert len(result) == 2
    assert f"data:image/png;base64,{data1}" in result[0]["image_url"]["url"]
    assert f"data:image/webp;base64,{data2}" in result[1]["image_url"]["url"]


def test_image_with_embedded_blob_returns_list():
    result = convert_mcp_content_to_llm_content([_image(), _embedded_blob()])
    assert isinstance(result, list)
    text_block = next(b for b in result if b["type"] == "text")
    assert "[Binary data:" in text_block["text"]


# ── unknown content type fallback ───────────────────────────────────────────


def test_unknown_content_type_falls_back_to_str_representation():
    class _Unknown:
        def __str__(self):
            return "mystery-content"

    result = convert_mcp_content_to_llm_content([_Unknown()])
    assert result == "mystery-content"
    assert isinstance(result, str)


def test_unknown_content_alongside_image_becomes_text_block():
    class _Unknown:
        def __str__(self):
            return "mystery"

    result = convert_mcp_content_to_llm_content([_Unknown(), _image()])
    assert isinstance(result, list)
    text_block = result[0]
    assert text_block == {"type": "text", "text": "mystery"}
