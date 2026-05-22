"""Tests for Canvas payload extraction from ADK events."""

from __future__ import annotations

import json

import pytest
from app.canvas.artifact_extract import (
    CANVAS_TOOL_NAME,
    coerce_canvas_payload,
    iter_canvas_artifacts_from_events,
)
from google.adk.events.event import Event
from google.genai import types


def test_coerce_accepts_direct_canvas_dict():
    payload = {
        "ok": True,
        "mime": "text/markdown",
        "output_kind": "markdown_report",
        "artifact": "# Title\n",
    }
    assert coerce_canvas_payload(payload) == payload


def test_coerce_unwraps_output_json_string():
    inner = {
        "ok": True,
        "mime": "text/html",
        "output_kind": "html_report",
        "artifact": "<!DOCTYPE html><html><body><p>x</p></body></html>",
    }
    wrapped = {"output": json.dumps(inner, ensure_ascii=False)}
    assert coerce_canvas_payload(wrapped) == inner


def test_coerce_single_key_dict():
    inner = {"ok": False, "error": "canvas_validation:x", "title": "t"}
    wrapped = {"output": json.dumps(inner)}
    assert coerce_canvas_payload(wrapped) == inner


def test_iter_extracts_from_function_response_events():
    canvas_json = {
        "ok": True,
        "mime": "text/markdown",
        "output_kind": "code_snippet",
        "artifact": "```py\n1\n```",
        "programming_language": "py",
    }
    fr = types.FunctionResponse(
        name=CANVAS_TOOL_NAME,
        response={"output": json.dumps(canvas_json)},
    )
    ev = Event(author="model", content=types.Content(parts=[types.Part(function_response=fr)]))
    assert iter_canvas_artifacts_from_events([ev]) == [canvas_json]


def test_iter_ignores_other_tools():
    fr = types.FunctionResponse(name="other_tool", response={"output": "{}"})
    ev = Event(author="model", content=types.Content(parts=[types.Part(function_response=fr)]))
    assert iter_canvas_artifacts_from_events([ev]) == []


def test_iter_surfaces_bare_tool_error_dict():
    fr = types.FunctionResponse(
        name=CANVAS_TOOL_NAME,
        response={"error": "missing args"},
    )
    ev = Event(author="model", content=types.Content(parts=[types.Part(function_response=fr)]))
    out = iter_canvas_artifacts_from_events([ev])
    assert len(out) == 1
    assert out[0]["ok"] is False
    assert out[0]["output_kind"] == "tool_error"
    assert "missing" in out[0]["error"]


@pytest.mark.parametrize("raw", [None, "", "not-json", 42, []])
def test_coerce_rejects_non_objects(raw):
    assert coerce_canvas_payload(raw) is None
