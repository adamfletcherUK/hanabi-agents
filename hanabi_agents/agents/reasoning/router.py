from typing import Dict, Any, Literal
import logging
from .nodes import _normalize_tool_name

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
    # Check if we have proposed tool calls directly in the state
    if "proposed_tool_calls" in state and state["proposed_tool_calls"]:
        logger.info(
            f"Found proposed_tool_calls in state: {state['proposed_tool_calls']}")

        # Normalize tool names in the proposed tool calls
        for tool_call in state["proposed_tool_calls"]:
            if isinstance(tool_call, dict) and "name" in tool_call:
                original_tool_name = tool_call["name"]
                normalized_tool_name = _normalize_tool_name(original_tool_name)
                if normalized_tool_name != original_tool_name:
                    logger.info(
                        f"Normalized tool name from '{original_tool_name}' to '{normalized_tool_name}'")
                    tool_call["name"] = normalized_tool_name

        return "execute_tools"

    # Get the messages from the state
    messages = state.get("messages", [])

    # Check if we have any messages
    if not messages:
        logger.info("No messages in state, ending reasoning process")
        return "end"

    # Check the last few messages for tool calls (in case the last message doesn't have them)
    for message in reversed(messages[-3:]):  # Check the last 3 messages
        if hasattr(message, "tool_calls") and message.tool_calls:
            logger.info(
                f"Found tool calls in message: {message.tool_calls}")

            # Normalize tool names in the tool calls
            normalized_tool_calls = []
            for tool_call in message.tool_calls:
                if isinstance(tool_call, dict) and "name" in tool_call:
                    original_tool_name = tool_call["name"]
                    normalized_tool_name = _normalize_tool_name(
                        original_tool_name)
                    if normalized_tool_name != original_tool_name:
                        logger.info(
                            f"Normalized tool name from '{original_tool_name}' to '{normalized_tool_name}'")
                        tool_call_copy = tool_call.copy()
                        tool_call_copy["name"] = normalized_tool_name
                        normalized_tool_calls.append(tool_call_copy)
                    else:
                        normalized_tool_calls.append(tool_call)
                else:
                    normalized_tool_calls.append(tool_call)

            # Store the normalized tool calls in the state for later use
            state["proposed_tool_calls"] = normalized_tool_calls

            return "execute_tools"

    # Check if we have tool calls in agent memory
    agent_memory = state.get("agent_memory", None)
    if agent_memory and hasattr(agent_memory, "get_memory"):
        tool_calls = agent_memory.get_memory("proposed_tool_calls")
        if tool_calls:
            logger.info(
                f"Found tool calls in agent memory: {tool_calls}")

            # Normalize tool names in the tool calls
            normalized_tool_calls = []
            for tool_call in tool_calls:
                if isinstance(tool_call, dict) and "name" in tool_call:
                    original_tool_name = tool_call["name"]
                    normalized_tool_name = _normalize_tool_name(
                        original_tool_name)
                    if normalized_tool_name != original_tool_name:
                        logger.info(
                            f"Normalized tool name from '{original_tool_name}' to '{normalized_tool_name}'")
                        tool_call_copy = tool_call.copy()
                        tool_call_copy["name"] = normalized_tool_name
                        normalized_tool_calls.append(tool_call_copy)
                    else:
                        normalized_tool_calls.append(tool_call)
                else:
                    normalized_tool_calls.append(tool_call)

            # Store the normalized tool calls in the state for later use
            state["proposed_tool_calls"] = normalized_tool_calls

            return "execute_tools"

    # If no tool calls found anywhere, end the reasoning process
    logger.info(
        "No tool calls found in state, messages, or agent memory, ending reasoning process")
    return "end"
