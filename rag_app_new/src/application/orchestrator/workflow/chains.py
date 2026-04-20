"""Chains for the router, the RAG agent, and the simple-response path."""

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from src.application.orchestrator.workflow.tools import get_tools
from src.domain.prompts import (
    AGENT_SYSTEM_PROMPT,
    ROUTER_PROMPT,
    SIMPLE_RESPONSE_PROMPT,
)
from src.infrastructure.model import get_model


def _escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _cached_system(text: str) -> SystemMessage:
    # Bedrock Converse caches the system prompt when we append a cachePoint
    # block. Identical prefixes across ReAct turns hit the cache at 10% of
    # input-token price instead of full rate.
    return SystemMessage(
        content=[
            {"type": "text", "text": text},
            {"cachePoint": {"type": "default"}},
        ]
    )


def get_agent_chain(customer_name: str = "Guest") -> Runnable:
    model = get_model(temperature=0.3).bind_tools(get_tools())
    system = AGENT_SYSTEM_PROMPT.replace("{customer_name}", customer_name)
    prompt = ChatPromptTemplate.from_messages(
        [_cached_system(system), MessagesPlaceholder(variable_name="messages")]
    )
    return prompt | model


def get_finalize_chain(customer_name: str = "Guest") -> Runnable:
    """Agent chain WITHOUT tools bound — used to force a text answer
    when the ReAct tool-call budget is exhausted."""
    model = get_model(temperature=0.3)
    system = AGENT_SYSTEM_PROMPT.replace("{customer_name}", customer_name) + (
        "\n\nYou have already gathered research via tool calls and your tool "
        "budget is now exhausted. Do NOT attempt any more tool calls. Produce "
        "the best final answer you can from the tool results already in the "
        "conversation history. If the information is insufficient, say so "
        "clearly and explain what is missing."
    )
    prompt = ChatPromptTemplate.from_messages(
        [_cached_system(system), MessagesPlaceholder(variable_name="messages")]
    )
    return prompt | model


def get_router_chain() -> Runnable:
    model = get_model(temperature=0.0, router=True)
    prompt = ChatPromptTemplate.from_messages(
        [("system", ROUTER_PROMPT), MessagesPlaceholder(variable_name="messages")]
    )
    return prompt | model


def get_simple_response_chain(customer_name: str = "Guest") -> Runnable:
    model = get_model(temperature=0.7)
    system = SIMPLE_RESPONSE_PROMPT.replace(
        "{customer_name}", _escape_braces(customer_name)
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), MessagesPlaceholder(variable_name="messages")]
    )
    return prompt | model
