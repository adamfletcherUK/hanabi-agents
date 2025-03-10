from typing import Dict, Any
import logging
from ...game.state import GameState, Color

# Set up logging
logger = logging.getLogger(__name__)


def validate_action_format(action: Dict[str, Any]) -> bool:
    """Validate the format of an action."""
    if not action or not isinstance(action, dict):
        return False

    # Check if action has a type
    if "type" not in action:
        return False

    action_type = action["type"]

    # Validate play_card action
    if action_type == "play_card":
        return "card_index" in action

    # Validate give_clue action
    elif action_type == "give_clue":
        if "target_id" not in action:
            return False

        if "clue" not in action or not isinstance(action["clue"], dict):
            return False

        clue = action["clue"]
        if "type" not in clue or "value" not in clue:
            return False

        if clue["type"] not in ["color", "number"]:
            return False

        return True

    # Validate discard action
    elif action_type == "discard":
        return "card_index" in action

    # Unknown action type
    return False


def validate_action_before_submission(game_state: GameState, action: Dict[str, Any], agent_id: int) -> bool:
    """Validate an action before submitting it to the game engine."""
    if not action:
        logger.warning(f"Agent {agent_id}: Empty action")
        return False

    try:
        # Validate action format
        if not validate_action_format(action):
            logger.error(
                f"Agent {agent_id}: Invalid action format: {action}")
            return False

        action_type = action.get("type")

        # Validate play_card action
        if action_type == "play_card":
            card_index = action.get("card_index")
            if not isinstance(card_index, int) or card_index < 0 or card_index >= len(game_state.hands[agent_id]):
                logger.error(
                    f"Agent {agent_id}: Invalid card index: {card_index}")
                return False
            return True

        # Validate give_clue action
        elif action_type == "give_clue":
            # Check if we have clue tokens
            if game_state.clue_tokens <= 0:
                logger.error(
                    f"Agent {agent_id}: No clue tokens available")
                return False

            # Check target player
            target_id = action.get("target_id")
            if target_id == agent_id:
                logger.error(
                    f"Agent {agent_id}: Cannot give clue to yourself")
                return False
            if not isinstance(target_id, int) or target_id < 0 or target_id >= len(game_state.hands):
                logger.error(
                    f"Agent {agent_id}: Invalid target ID: {target_id}")
                return False

            # Check clue format
            clue = action.get("clue")
            if not isinstance(clue, dict):
                logger.error(
                    f"Agent {agent_id}: Invalid clue format: {clue}")
                return False

            # Check clue type and value
            clue_type = clue.get("type")
            clue_value = clue.get("value")
            if not clue_type or not clue_value:
                logger.error(
                    f"Agent {agent_id}: Missing clue type or value: {clue}")
                return False

            # Validate clue type
            if clue_type not in ["color", "number"]:
                logger.error(
                    f"Agent {agent_id}: Invalid clue type: {clue_type}")
                return False

            # Validate clue value based on type
            if clue_type == "color":
                if clue_value not in [c.value for c in Color]:
                    logger.error(
                        f"Agent {agent_id}: Invalid color value: {clue_value}")
                    return False
            elif clue_type == "number":
                try:
                    num_value = int(clue_value)
                    if num_value < 1 or num_value > 5:
                        logger.error(
                            f"Agent {agent_id}: Invalid number value (out of range): {clue_value}")
                        return False
                except ValueError:
                    logger.error(
                        f"Agent {agent_id}: Invalid number value (not convertible to int): {clue_value}")
                    return False

            # Check if the clue matches any cards in the target player's hand
            matches = False
            for card in game_state.hands[target_id]:
                if clue_type == "color" and card.color.value == clue_value:
                    matches = True
                    break
                elif clue_type == "number" and card.number == int(clue_value):
                    matches = True
                    break

            if not matches:
                logger.error(
                    f"Agent {agent_id}: No {clue_type} {clue_value} cards in player {target_id}'s hand")
                return False

            return True

        # Validate discard action
        elif action_type == "discard":
            # Check if we need more clue tokens
            if game_state.clue_tokens >= 8:
                logger.error(
                    f"Agent {agent_id}: Cannot discard when clue tokens are at maximum (8)")
                return False

            # Check card index
            card_index = action.get("card_index")
            if not isinstance(card_index, int) or card_index < 0 or card_index >= len(game_state.hands[agent_id]):
                logger.error(
                    f"Agent {agent_id}: Invalid card index: {card_index}")
                return False

            return True

        # Unknown action type
        else:
            logger.error(
                f"Agent {agent_id}: Unknown action type: {action_type}")
            return False

    except Exception as e:
        logger.error(
            f"Agent {agent_id}: Error validating action: {e}")
        return False
