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
    # Check if we have a proposed action
    proposed_action = state.get("proposed_action")
    if proposed_action is None:
        logger.info("No proposed action, ending reasoning process")
        return "end"

    # Check if the proposed action has an action_type
    action_type = proposed_action.get("action_type")
    if action_type is None:
        logger.info(
            "Proposed action has no action_type, ending reasoning process")
        return "end"

    # Check if the action type is valid
    if action_type not in ["play", "clue", "discard"]:
        logger.info(
            f"Invalid action type: {action_type}, ending reasoning process")
        return "end"

    # Check if the action has the required parameters
    if action_type == "play" or action_type == "discard":
        if "card_index" not in proposed_action:
            logger.info(
                f"Missing card_index for {action_type} action, ending reasoning process")
            return "end"
    elif action_type == "clue":
        if "target_id" not in proposed_action or "clue_type" not in proposed_action or "clue_value" not in proposed_action:
            logger.info(
                "Missing parameters for clue action, ending reasoning process")
            return "end"

    # If we have a valid proposed action, execute tools
    logger.info(f"Valid proposed action: {proposed_action}, executing tools")
    return "execute_tools"
