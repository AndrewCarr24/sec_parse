"""AgentCore entrypoint."""

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from loguru import logger

from src.infrastructure.streaming import stream_response

app = BedrockAgentCoreApp()


@app.entrypoint
async def invoke(payload: dict):
    """
    Main entrypoint.

    Expected payload:
    {
        "prompt": "<user input>",              # Required
        "customer_name": "<name>",             # Optional
        "conversation_id": "<conversation id>" # Optional (used as thread_id)
    }
    """
    user_input = payload.get("prompt", "")
    if not user_input:
        return {"error": "No prompt provided in the payload."}

    customer_name = payload.get("customer_name", "Guest")
    conversation_id = payload.get("conversation_id")

    logger.info(
        f"Invoking RAG agent (customer={customer_name}, "
        f"conversation_id={conversation_id})"
    )
    return stream_response(
        user_input=user_input,
        customer_name=customer_name,
        conversation_id=conversation_id,
    )


if __name__ == "__main__":
    app.run()
