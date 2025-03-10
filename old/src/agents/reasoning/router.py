from typing import Dict, Any, Literal
import logging

logger = logging.getLogger(__name__)


def should_execute_tools(state: Dict[str, Any]) -> Literal["execute_tools", "end"]:
    """
    Determine whether to execute tools or end the reasoning process.

    Args:
        state: Current state of the reasoning graph

    Returns:
        "execute_tools" if tools should be executed, "end" otherwise
    """
    # Add logging to help diagnose issues
    logger.info(f"Router checking state keys: {list(state.keys())}")

    # Check if we're in the action phase
    is_action_phase = state.get("is_action_phase", False)
    is_active_player = state.get(
        "agent_id", -1) == state.get("game_state", {}).current_player

    logger.info(
        f"Is action phase: {is_action_phase}, Is active player: {is_active_player}")

    # If we're not in the action phase and not the active player, we don't need to execute tools
    if not is_action_phase and not is_active_player:
        logger.info(
            "Not in action phase and not active player, returning 'end'")
        return "end"

    # Check if we have proposed tool calls directly in the state
    if "proposed_tool_calls" in state and state["proposed_tool_calls"]:
        logger.info(
            f"Found proposed_tool_calls in state: {state['proposed_tool_calls']}")
        return "execute_tools"

    # Check if we have messages
    if "messages" in state and state["messages"]:
        # Check the last message for tool calls
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.info(
                f"Found tool_calls in last message: {last_message.tool_calls}")
            return "execute_tools"
        else:
            logger.info("Last message has no tool calls, returning 'end'")
            return "end"
    else:
        logger.info("No messages in state, returning 'end'")
        return "end"
