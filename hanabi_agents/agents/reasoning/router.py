from typing import Dict, Any, Literal
import logging

# Set up logging
logger = logging.getLogger(__name__)


def should_execute_tools(state: Dict[str, Any]) -> Literal["execute_tools", "end"]:
    """
    Determine whether to execute tools or end the reasoning process.

    Args:
        state: Current state of the reasoning graph

    Returns:
        "execute_tools" if tools should be executed, "end" otherwise
    """
    # Get the messages from the state
    messages = state.get("messages", [])

    # Check if we have any messages
    if not messages:
        logger.info("No messages in state, ending reasoning process")
        return "end"

    # Get the last message
    last_message = messages[-1]

    # Check if the last message has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info(
            f"Found tool calls in last message: {last_message.tool_calls}")
        return "execute_tools"

    # If no tool calls, end the reasoning process
    logger.info("No tool calls in last message, ending reasoning process")
    return "end"
