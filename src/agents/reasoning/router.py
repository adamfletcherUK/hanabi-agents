from typing import Dict, Any


def should_execute_tools(state: Dict[str, Any]) -> str:
    """
    Determine if tools should be executed based on the state.

    Args:
        state: The current state of the agent

    Returns:
        str: "execute_tools" if tools should be executed, "end" otherwise
    """
    # Check if we're in the action phase (messages will be present)
    is_action_phase = "messages" in state

    # If we're not in the action phase, we don't execute tools
    if not is_action_phase:
        return "end"

    # Check if the last message has tool calls
    if not state.get("messages"):
        return "end"

    last_message = state["messages"][-1]
    has_tool_calls = hasattr(
        last_message, "tool_calls") and last_message.tool_calls

    # Only execute tools if we have tool calls
    if has_tool_calls:
        return "execute_tools"

    # Otherwise, we're done
    return "end"
