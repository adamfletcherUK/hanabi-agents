from typing import Dict, Any
import logging
from langchain_core.messages import ToolMessage

# Set up logging
logger = logging.getLogger(__name__)


def handle_tool_error(state: Dict[str, Any], agent_id: int) -> Dict[str, Any]:
    """
    Handle errors that occur during tool execution.

    This method is called when a tool execution fails. It logs the error,
    creates appropriate error messages, and ensures the graph can continue
    execution rather than terminating.

    Args:
        state: The current state of the agent
        agent_id: The ID of the agent

    Returns:
        Updated state with error information
    """
    error = state.get("error")

    # Log the error
    logger.error(f"Agent {agent_id} tool execution error: {error}")

    # Check if we're in the action phase (messages will be present)
    if "messages" not in state or not state["messages"]:
        logger.warning(
            "Tool error occurred outside action phase or with no messages")
        return state

    # Get the last message which should contain tool calls
    last_message = state["messages"][-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        logger.warning(
            "Tool error occurred but no tool calls found in the last message")
        return state

    # Create error messages for each tool call
    tool_calls = last_message.tool_calls
    error_messages = []

    for tc in tool_calls:
        tool_call_id = tc.get("id") if isinstance(
            tc, dict) and "id" in tc else "unknown"
        tool_name = tc.get("name") if isinstance(
            tc, dict) and "name" in tc else "unknown tool"

        error_message = ToolMessage(
            content=f"Error executing {tool_name}: {repr(error)}. Please try a different approach.",
            tool_call_id=tool_call_id
        )
        error_messages.append(error_message)

    # Return updated state with error messages
    return {
        **state,
        "messages": state["messages"] + error_messages,
        "tool_error": str(error)  # Store the error for reference
    }
