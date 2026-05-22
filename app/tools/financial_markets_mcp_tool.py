"""Yahoo Finance market pages via MCP **fetch** (structured HTML→markdown snapshots)."""

from __future__ import annotations

import json
from typing import Literal

from app.config import Settings
from app.mcp.fetch_client import fetch_keyed_urls_via_mcp
from app.mcp.stdio_params import stdio_parameters_for_fetch_server

# Hard-coded retail finance endpoints (assignment spec).
YAHOO_FINANCE_MCP_PAGES: dict[str, str] = {
    "us_stocks_most_active": "https://finance.yahoo.com/markets/stocks/most-active/",
    "crypto_all": "https://finance.yahoo.com/markets/crypto/all/",
    "currencies": "https://finance.yahoo.com/markets/currencies/",
}

SegmentChoice = Literal["stocks", "crypto", "currencies", "all"]


def urls_for_segments(segments: SegmentChoice) -> dict[str, str]:
    """Subset of :data:`YAHOO_FINANCE_MCP_PAGES` to retrieve."""
    if segments == "stocks":
        return {"us_stocks_most_active": YAHOO_FINANCE_MCP_PAGES["us_stocks_most_active"]}
    if segments == "crypto":
        return {"crypto_all": YAHOO_FINANCE_MCP_PAGES["crypto_all"]}
    if segments == "currencies":
        return {"currencies": YAHOO_FINANCE_MCP_PAGES["currencies"]}
    return dict(YAHOO_FINANCE_MCP_PAGES)


def make_financial_markets_mcp_tool(settings: Settings):
    """Return an async ADK callable that batches MCP ``fetch`` for Yahoo market tables."""

    server = stdio_parameters_for_fetch_server(settings)

    async def fetch_yahoo_finance_markets_via_mcp(
        segments: SegmentChoice = "all",
        max_length_per_url: int = 6000,
    ) -> str:
        """Pull latest Yahoo Finance listings via MCP fetch (not general web search).

        Use when the user asks about **stocks**, **cryptocurrencies**, or **currency/FX** markets.
        Data comes only from predefined Yahoo URLs; cite ``source_key`` headers in summaries.
        """
        bounded = max(500, min(int(max_length_per_url), 20_000))
        keyed = urls_for_segments(segments)
        try:
            payloads = await fetch_keyed_urls_via_mcp(
                server,
                keyed,
                max_length_per_url=bounded,
            )
        except OSError as exc:
            return json.dumps(
                {"ok": False, "error": f"spawn_failed:{exc}", "segments": segments},
                ensure_ascii=False,
            )
        except Exception as exc:  # noqa: BLE001 — surface to model for fallback
            return json.dumps(
                {"ok": False, "error": str(exc), "segments": segments},
                ensure_ascii=False,
            )

        ok_any = any(p.get("ok") for p in payloads.values())
        return json.dumps(
            {
                "ok": ok_any,
                "segments_requested": segments,
                "canonical_urls": keyed,
                "mcp_fetch_by_source": payloads,
            },
            ensure_ascii=False,
        )

    return fetch_yahoo_finance_markets_via_mcp
