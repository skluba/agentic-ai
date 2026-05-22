"""A2A `AgentExecutor` handling News Agent requests for Phase 5."""

from __future__ import annotations

from a2a.helpers.proto_helpers import new_text_message, new_text_status_update_event
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue_v2 import EventQueue
from a2a.types.a2a_pb2 import Role, TaskState

from app.agents.news_focused import run_news_kb_turn
from app.config import Settings
from app.knowledge.store import KnowledgeCorpus


class NewsKbA2AExecutor(AgentExecutor):
    """Runs one ADK news turn and returns a single A2A `Message` response."""

    def __init__(self, settings: Settings, corpus: KnowledgeCorpus) -> None:
        self._settings = settings
        self._corpus = corpus

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        topic = context.get_user_input().strip()
        if not topic:
            msg = new_text_message(
                text=(
                    "**NEWS_AGENT error:** Provide a topical question or keywords "
                    "when calling this specialist."
                ),
                role=Role.ROLE_AGENT,
                task_id=context.task_id or "",
                context_id=context.context_id or "",
            )
            await event_queue.enqueue_event(msg)
            return

        briefing_request = (
            "Produce the latest substantive news briefing grounded in uploads (if present) "
            f"plus web coverage for:\n---\n{topic}\n---\nPrioritise timelines, movers, "
            "and attributable outlets."
        )
        text_reply, _events = await run_news_kb_turn(
            settings=self._settings,
            corpus=self._corpus,
            payload=briefing_request,
            user_id="news-agent-remote",
            session_id=context.context_id or "news-remote",
            app_label="news_kb_a2a_turn",
        )
        outbound = (
            text_reply.strip()
            or "**NEWS_AGENT:** Model produced no textual content — widen the topic "
            "or verify Gemini credentials."
        )
        reply_message = new_text_message(
            text=outbound,
            role=Role.ROLE_AGENT,
            task_id=context.task_id or "",
            context_id=context.context_id or "",
        )
        await event_queue.enqueue_event(reply_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(
            new_text_status_update_event(
                task_id=context.task_id or "",
                context_id=context.context_id or "",
                state=TaskState.TASK_STATE_CANCELED,
                text="News specialist cancellation acknowledged.",
            )
        )
