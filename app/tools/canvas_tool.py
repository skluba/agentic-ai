"""ADK **`produce_structured_canvas`** — Jinja2 + Markdown artefacts for planners."""

from __future__ import annotations

import json

import markdown as md_lib
from jinja2 import Environment, StrictUndefined, select_autoescape
from markupsafe import Markup
from pydantic import ValidationError

from app.canvas.models import CanvasProduceInput


def make_canvas_delivery_tool():  # noqa: ANN201
    """Factory returning a synchronous callable for structured canvas outputs."""

    jinja = Environment(
        autoescape=select_autoescape(enabled_extensions=("html", "xml")),
        undefined=StrictUndefined,
    )
    html_template = jinja.from_string(
        '<!DOCTYPE html>\n<html lang="en"><head>'
        '<meta charset="utf-8"><title>{{ title }}</title>'
        "</head><body><article>"
        "<header><h1>{{ title }}</h1></header>"
        '<section class="content">{{ body_html }}</section>'
        "</article></body></html>",
    )

    def produce_structured_canvas(
        output_kind: str,
        title: str,
        markdown_body: str,
        programming_language: str = "",
    ) -> str:
        """Render a Canvas artefact (Markdown, HTML wrapping, or code fence).

        Use after substantive research (**corpus / MCP / web / remote news**) when the caller
        wants a **standalone deliverable** (report memo, stakeholder summary, reproducible snippet).
        Arguments should reflect **validated** citations from tool payloads only.
        """
        try:
            payload = CanvasProduceInput(
                output_kind=output_kind.strip().lower(),
                title=title,
                markdown_body=markdown_body,
                programming_language=programming_language,
            )
        except ValidationError as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": f"canvas_validation:{exc!s}",
                    "title": title,
                },
                ensure_ascii=False,
            )

        if payload.output_kind == "markdown_report":
            lines = [
                f"# {payload.title}".strip(),
                "",
                payload.markdown_body,
                "",
                "---",
                "_Canvas artefact (`markdown_report`) — generated from grounded tool payloads._",
            ]
            artefact = "\n".join(lines)
            return json.dumps(
                {
                    "ok": True,
                    "mime": "text/markdown",
                    "output_kind": payload.output_kind,
                    "artifact": artefact,
                },
                ensure_ascii=False,
            )

        if payload.output_kind == "html_report":
            body_html = Markup(
                md_lib.markdown(
                    payload.markdown_body,
                    extensions=["tables", "fenced_code"],
                )
            )
            html_doc = html_template.render(title=payload.title, body_html=body_html)
            return json.dumps(
                {
                    "ok": True,
                    "mime": "text/html",
                    "output_kind": payload.output_kind,
                    "artifact": html_doc,
                },
                ensure_ascii=False,
            )

        lang = payload.programming_language
        fenced = (
            f"# {payload.title}\n\n"
            f"```{lang}\n{payload.markdown_body.rstrip()}\n```\n\n"
            "_Canvas artefact (`code_snippet`)._\n"
        )
        return json.dumps(
            {
                "ok": True,
                "mime": "text/markdown",
                "output_kind": payload.output_kind,
                "artifact": fenced,
                "programming_language": lang,
            },
            ensure_ascii=False,
        )

    return produce_structured_canvas
