"""Short-term memory + long-term memory strategies via AgentCore Memory."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from bedrock_agentcore.memory import MemoryClient
from langgraph_checkpoint_aws import AgentCoreMemorySaver
from loguru import logger

from src.config import settings

_memory_instance: "ShortTermMemory | None" = None


def get_memory_instance() -> "ShortTermMemory":
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ShortTermMemory()
    return _memory_instance


class ShortTermMemory:
    """Wraps AgentCore Memory for checkpointing + preference/fact/summary strategies."""

    def __init__(self) -> None:
        if not settings.MEMORY_ID:
            raise RuntimeError(
                "MEMORY_ID is required. Set it from the CDK stack output."
            )
        self._memory_id = settings.MEMORY_ID
        self._client = MemoryClient(region_name=settings.AWS_REGION)
        logger.info(f"Using MEMORY_ID: {self._memory_id}")

    @property
    def memory_id(self) -> str:
        return self._memory_id

    def get_memory(self) -> AgentCoreMemorySaver:
        """Return a LangGraph checkpointer for thread persistence."""
        return AgentCoreMemorySaver(
            memory_id=self._memory_id,
            region_name=settings.AWS_REGION,
        )

    def _retrieve_from_namespace(
        self,
        namespace: str,
        query: str,
        actor_id: str,
        top_k: int,
        category: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        try:
            results = self._client.retrieve_memories(
                memory_id=self._memory_id,
                namespace=namespace,
                query=query,
                actor_id=actor_id,
                top_k=top_k,
            )
            return category, results
        except Exception as e:
            logger.warning(f"Failed to retrieve {category}: {e}")
            return category, []

    def retrieve_specific_memories(
        self,
        query: str,
        actor_id: str,
        session_id: str,
        memory_types: list[str],
        top_k: int = 5,
    ) -> dict[str, list[dict[str, Any]]]:
        type_to_namespace = {
            "preferences": f"/users/{actor_id}/preferences",
            "facts": f"/conversations/{actor_id}/facts",
            "summaries": f"/conversations/{session_id}/summaries",
        }
        tasks = [
            (type_to_namespace[t], t) for t in memory_types if t in type_to_namespace
        ]
        if not tasks:
            return {}

        retrieved: dict[str, list[dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {
                executor.submit(
                    self._retrieve_from_namespace,
                    ns, query, actor_id, top_k, cat,
                ): cat
                for ns, cat in tasks
            }
            for future in as_completed(futures):
                try:
                    cat, results = future.result()
                    retrieved[cat] = results
                except Exception as e:
                    cat = futures[future]
                    logger.warning(f"Parallel retrieval failed for {cat}: {e}")
                    retrieved[cat] = []
        return retrieved

    def process_turn(
        self,
        actor_id: str,
        session_id: str,
        user_input: str,
        agent_response: str,
    ) -> dict[str, Any]:
        try:
            retrieved_memories, event_info = self._client.process_turn(
                memory_id=self._memory_id,
                actor_id=actor_id,
                session_id=session_id,
                user_input=user_input,
                agent_response=agent_response,
            )
            logger.info(f"Saved turn to memory (actor={actor_id}, session={session_id})")
            return {
                "success": True,
                "retrieved_memories": retrieved_memories,
                "event_info": event_info,
            }
        except Exception as e:
            logger.error(f"Failed to process conversation turn: {e}")
            return {"success": False, "error": str(e)}
