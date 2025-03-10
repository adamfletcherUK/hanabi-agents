from typing import Dict, Any, Optional
import logging

# Set up logging
logger = logging.getLogger(__name__)


def handle_tool_error(state: Dict[str, Any], agent_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Handle errors that occur during tool execution.

    This function is called when a tool execution fails. It updates the state
    with error information and provides guidance for the agent.

    Args:
        state: Current state of the reasoning graph
        agent_id: ID of the agent (if available)

    Returns:
        Updated state with error information
    """
    # Log the error
    logger.error(f"Tool execution error occurred for agent {agent_id}")

    # Create a copy of the state to avoid modifying the original
    new_state = state.copy()

    # Get the proposed action if available
    proposed_action = new_state.get("proposed_action", {})
    action_type = proposed_action.get("action_type", "unknown")

    # Create an error record
    error_record = {
        "agent_id": agent_id,
        "action_type": action_type,
        "proposed_action": proposed_action,
        "error": "Tool execution failed",
        "guidance": _generate_error_guidance(action_type)
    }

    # Add the error to the state
    errors = new_state.get("errors", [])
    errors.append(error_record)
    new_state["errors"] = errors

    # Add a message about the error
    messages = new_state.get("messages", [])
    messages.append({
        "role": "system",
        "content": f"Error executing {action_type} action. {error_record['guidance']}"
    })
    new_state["messages"] = messages

    return new_state


def _generate_error_guidance(action_type: str) -> str:
    """
    Generate guidance for the agent based on the type of action that failed.

    Args:
        action_type: Type of action that failed

    Returns:
        Guidance message for the agent
    """
    if action_type == "play":
        return (
            "When playing a card, make sure the card index is valid and within the range of your hand. "
            "Consider whether this is a good card to play based on the current firework piles."
        )
    elif action_type == "clue":
        return (
            "When giving a clue, make sure you have enough clue tokens, the target player is valid, "
            "and the clue type and value are valid. Color clues must be one of: red, yellow, green, blue, white. "
            "Number clues must be between 1 and 5."
        )
    elif action_type == "discard":
        return (
            "When discarding a card, make sure the card index is valid and within the range of your hand. "
            "Consider whether this is a good card to discard based on what you know about it."
        )
    else:
        return (
            "Please check that your action is valid and try again. "
            "Valid actions are: play (with a valid card index), "
            "clue (with a valid target player, clue type, and clue value), "
            "and discard (with a valid card index)."
        )
