"""Pydantic contracts for **`produce_structured_canvas`** tool arguments."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

CanvasOutputKind = Literal["markdown_report", "html_report", "code_snippet"]


class CanvasProduceInput(BaseModel):
    """Validated payload authored by planner/synthesiser before rendering."""

    output_kind: CanvasOutputKind = Field(
        ...,
        description="markdown_report · html_report wraps markdown into a simple article shell; "
        "code_snippet emits a fenced snippet.",
    )
    title: str = Field(..., min_length=1, description="Displayed heading.")
    markdown_body: str = Field(
        ...,
        min_length=1,
        description="Primary narrative Markdown (bullet lists, headings, cites to tool payloads).",
    )
    programming_language: str = Field(
        default="",
        max_length=64,
        description="Required when ``output_kind == 'code_snippet'``.",
    )

    model_config = {"extra": "forbid"}

    @field_validator("output_kind", mode="before")
    @classmethod
    def _normalize_kind(cls, value: Any) -> str:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator(
        "title",
        "markdown_body",
        "programming_language",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def _code_requires_language(self) -> CanvasProduceInput:
        if self.output_kind == "code_snippet" and not self.programming_language.strip():
            msg = "`programming_language` is required when `output_kind` is `code_snippet`."
            raise ValueError(msg)
        return self
