from typing import Dict, Any
import json
import re
import logging
from ...game.state import Color

# Set up logging
logger = logging.getLogger(__name__)


def parse_action_response(response: str, agent_id: int) -> Dict[str, Any]:
    """Parse the action response from the LLM."""
    try:
        # Store the raw response for debugging
        # self._last_raw_response = response

        # Log the raw response for debugging
        logger.debug(
            f"Agent {agent_id}: Raw action response: {response}")

        # Extract JSON from the response
        json_match = re.search(
            r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)

        if not json_match:
            # Try to find any JSON-like structure without code blocks
            json_match = re.search(r'({[\s\S]*?})', response)

        if json_match:
            json_str = json_match.group(1)
            logger.debug(
                f"Agent {agent_id}: Extracted JSON: {json_str}")

            try:
                # Try to parse the JSON
                action = json.loads(json_str)

                # Ensure the action has the required fields
                if "type" not in action:
                    logger.warning(
                        f"Agent {agent_id}: Missing 'type' in action: {action}")
                    return {}

                # Fix common issues with action format
                action_type = action["type"].lower().strip()

                # Normalize action type
                if "play" in action_type:
                    action["type"] = "play_card"
                elif "clue" in action_type or "hint" in action_type:
                    action["type"] = "give_clue"
                elif "discard" in action_type:
                    action["type"] = "discard"

                # Fix play_card actions
                if action["type"] == "play_card":
                    # Ensure card_index is present and an integer
                    if "card_index" not in action:
                        if "index" in action:
                            action["card_index"] = action["index"]
                        elif "position" in action:
                            action["card_index"] = action["position"]
                        elif "card" in action and isinstance(action["card"], int):
                            action["card_index"] = action["card"]
                        else:
                            # Default to first card
                            action["card_index"] = 0

                    # Convert card_index to int if it's a string
                    if isinstance(action["card_index"], str):
                        try:
                            action["card_index"] = int(
                                action["card_index"])
                        except ValueError:
                            action["card_index"] = 0

                # Fix give_clue actions
                elif action["type"] == "give_clue":
                    # Ensure target_id is present
                    if "target_id" not in action:
                        if "target" in action:
                            action["target_id"] = action["target"]
                        elif "player" in action:
                            action["target_id"] = action["player"]
                        else:
                            # Default to next player
                            action["target_id"] = (agent_id + 1) % 5

                    # Convert target_id to int if it's a string
                    if isinstance(action["target_id"], str):
                        try:
                            action["target_id"] = int(action["target_id"])
                        except ValueError:
                            action["target_id"] = (agent_id + 1) % 5

                    # Ensure clue is present and properly formatted
                    if "clue" not in action or not isinstance(action["clue"], dict):
                        clue_type = None
                        clue_value = None

                        # Try to extract clue type and value from action
                        if "color" in action:
                            clue_type = "color"
                            clue_value = action["color"]
                        elif "number" in action:
                            clue_type = "number"
                            clue_value = action["number"]

                        # Create clue dict
                        if clue_type and clue_value:
                            action["clue"] = {
                                "type": clue_type,
                                "value": clue_value
                            }
                        else:
                            # Default clue
                            action["clue"] = {
                                "type": "color",
                                "value": "red"
                            }

                    # Ensure clue has type and value
                    if "type" not in action["clue"]:
                        action["clue"]["type"] = "color"
                    if "value" not in action["clue"]:
                        action["clue"]["value"] = "red"

                    # Normalize clue type
                    clue_type = action["clue"]["type"].lower().strip()
                    if "color" in clue_type:
                        action["clue"]["type"] = "color"
                    elif "number" in clue_type or "value" in clue_type:
                        action["clue"]["type"] = "number"

                    # Normalize color values
                    if action["clue"]["type"] == "color":
                        color_value = str(
                            action["clue"]["value"]).lower().strip()
                        valid_colors = [c.value for c in Color]

                        # Find closest match
                        for valid_color in valid_colors:
                            if valid_color in color_value:
                                action["clue"]["value"] = valid_color
                                break

                    # Normalize number values
                    if action["clue"]["type"] == "number":
                        try:
                            num_value = int(action["clue"]["value"])
                            if 1 <= num_value <= 5:
                                action["clue"]["value"] = num_value
                            else:
                                action["clue"]["value"] = max(
                                    1, min(5, num_value))
                        except (ValueError, TypeError):
                            action["clue"]["value"] = 1

                # Fix discard actions
                elif action["type"] == "discard":
                    # Ensure card_index is present and an integer
                    if "card_index" not in action:
                        if "index" in action:
                            action["card_index"] = action["index"]
                        elif "position" in action:
                            action["card_index"] = action["position"]
                        elif "card" in action and isinstance(action["card"], int):
                            action["card_index"] = action["card"]
                        else:
                            # Default to first card
                            action["card_index"] = 0

                    # Convert card_index to int if it's a string
                    if isinstance(action["card_index"], str):
                        try:
                            action["card_index"] = int(
                                action["card_index"])
                        except ValueError:
                            action["card_index"] = 0

                logger.debug(
                    f"Agent {agent_id}: Normalized action: {action}")
                return action

            except json.JSONDecodeError as e:
                logger.error(
                    f"Agent {agent_id}: JSON decode error: {e}")
                logger.debug(
                    f"Agent {agent_id}: Problematic JSON: {json_str}")

        # If we couldn't extract or parse JSON, try to infer the action from the text
        return infer_action_from_text(response, agent_id)

    except Exception as e:
        logger.error(
            f"Agent {agent_id}: Error parsing action response: {e}")
        return {}


def infer_action_from_text(text: str, agent_id: int) -> Dict[str, Any]:
    """Infer an action from plain text when JSON parsing fails."""
    text = text.lower()

    # Check for play card action
    play_match = re.search(
        r'play(?:\s+card)?(?:\s+at)?\s+(?:index|position|card)?\s*(\d+)', text)
    if play_match:
        try:
            card_index = int(play_match.group(1))
            return {
                "type": "play_card",
                "card_index": card_index
            }
        except (ValueError, IndexError):
            pass

    # Check for discard action
    discard_match = re.search(
        r'discard(?:\s+card)?(?:\s+at)?\s+(?:index|position|card)?\s*(\d+)', text)
    if discard_match:
        try:
            card_index = int(discard_match.group(1))
            return {
                "type": "discard",
                "card_index": card_index
            }
        except (ValueError, IndexError):
            pass

    # Check for give clue action
    clue_match = re.search(
        r'(?:give|hint)(?:\s+a)?\s+(?:clue|hint)(?:\s+to)?\s+(?:player|agent)?\s*(\d+)', text)
    color_match = re.search(r'(red|blue|green|yellow|white)', text)
    number_match = re.search(r'number\s*(\d+)', text)

    if clue_match:
        try:
            target_id = int(clue_match.group(1))

            # Determine clue type and value
            if color_match:
                return {
                    "type": "give_clue",
                    "target_id": target_id,
                    "clue": {
                        "type": "color",
                        "value": color_match.group(1)
                    }
                }
            elif number_match:
                return {
                    "type": "give_clue",
                    "target_id": target_id,
                    "clue": {
                        "type": "number",
                        "value": int(number_match.group(1))
                    }
                }
            else:
                # Default to red if no color/number found
                return {
                    "type": "give_clue",
                    "target_id": target_id,
                    "clue": {
                        "type": "color",
                        "value": "red"
                    }
                }
        except (ValueError, IndexError):
            pass

    # If we couldn't infer anything, return empty dict
    logger.warning(
        f"Agent {agent_id}: Could not infer action from text: {text[:100]}...")
    return {}
