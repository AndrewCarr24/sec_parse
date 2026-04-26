from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


IntentType = Literal["rag_query", "simple", "off_topic"]


class AgentState(TypedDict):
    """State for the RAG agent graph (router + ReAct)."""

    messages: Annotated[list[BaseMessage], add_messages]
    customer_name: str
    intent: IntentType
    tool_call_count: int
    cache_hit: bool
