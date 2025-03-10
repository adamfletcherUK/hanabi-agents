from typing import Dict, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)


def extract_action_from_state(final_state: Dict[str, Any], agent_id: int) -> Dict[str, Any]:
    """
    Extract the action from the final state of the reasoning graph.

    Args:
        final_state: Final state from the reasoning graph
        agent_id: ID of the agent

    Returns:
        Action dictionary

    Raises:
        ValueError: If no valid action could be extracted from the state
    """
    # Log the state keys to help diagnose issues
    logger.info(
        f"Agent {agent_id}: Final state keys: {list(final_state.keys())}")

    # First check if we have proposed tool calls
    if "proposed_tool_calls" in final_state and final_state["proposed_tool_calls"]:
        logger.info(
            f"Agent {agent_id}: Found proposed_tool_calls in final state: {final_state['proposed_tool_calls']}")
        tool_call = final_state["proposed_tool_calls"][0]

        # Extract the tool call information
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})

        logger.info(
            f"Agent {agent_id}: Extracted tool call - name: {tool_name}, args: {tool_args}")

        # Map tool names to action types
        if tool_name == "play_card":
            return {
                "type": "play_card",
                "card_index": tool_args.get("card_index", 0)
            }
        elif tool_name == "give_clue":
            return {
                "type": "give_clue",
                "target_id": tool_args.get("target_id", 0),
                "clue": {
                    "type": tool_args.get("clue_type", "color"),
                    "value": tool_args.get("clue_value", "")
                }
            }
        elif tool_name == "discard":
            return {
                "type": "discard",
                "card_index": tool_args.get("card_index", 0)
            }

    # Check if we have a tool result
    if "tool_result" in final_state and final_state["tool_result"]:
        logger.info(
            f"Agent {agent_id}: Found tool_result in final state: {final_state['tool_result']}")
        return final_state["tool_result"]

    # If we don't have a tool result but have messages, check for tool calls
    if "messages" in final_state and final_state["messages"]:
        logger.info(f"Agent {agent_id}: Checking messages for tool calls")
        last_message = final_state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            # Extract action from tool call
            logger.info(
                f"Agent {agent_id}: Found tool_calls in last message: {last_message.tool_calls}")
            tool_call = last_message.tool_calls[0]
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})

            logger.info(
                f"Agent {agent_id}: Extracted tool call from message - name: {tool_name}, args: {tool_args}")

            # Map tool names to action types
            if tool_name == "play_card":
                return {
                    "type": "play_card",
                    "card_index": tool_args.get("card_index", 0)
                }
            elif tool_name == "give_clue":
                return {
                    "type": "give_clue",
                    "target_id": tool_args.get("target_id", 0),
                    "clue": {
                        "type": tool_args.get("clue_type", "color"),
                        "value": tool_args.get("clue_value", "")
                    }
                }
            elif tool_name == "discard":
                return {
                    "type": "discard",
                    "card_index": tool_args.get("card_index", 0)
                }
        else:
            logger.warning(f"Agent {agent_id}: Last message has no tool calls")
            # Try to parse the content for an action
            if hasattr(last_message, "content") and last_message.content:
                logger.info(
                    f"Agent {agent_id}: Attempting to parse message content for action")
                from ..actions.parser import parse_action_response
                action = parse_action_response(last_message.content, agent_id)
                if action and "type" in action:
                    logger.info(
                        f"Agent {agent_id}: Successfully parsed action from content: {action}")
                    return action

            logger.debug(
                f"Agent {agent_id}: Last message content: {last_message.content if hasattr(last_message, 'content') else 'No content'}")
    else:
        logger.warning(f"Agent {agent_id}: No messages found in final state")

    # If we still don't have an action, try to extract from current_thoughts
    if "current_thoughts" in final_state and final_state["current_thoughts"]:
        logger.info(
            f"Agent {agent_id}: Attempting to extract action from current_thoughts")
        # Try the last thought first
        last_thought = final_state["current_thoughts"][-1]
        if last_thought:
            from ..actions.parser import parse_action_response
            action = parse_action_response(last_thought, agent_id)
            if action and "type" in action:
                logger.info(
                    f"Agent {agent_id}: Successfully parsed action from thought: {action}")
                return action

    # If we still don't have an action, create a smart fallback action based on the game state
    logger.warning(
        f"Agent {agent_id}: Creating smart fallback action based on game state")

    # Get the game state
    game_state = final_state.get("game_state")
    if not game_state:
        logger.error(
            f"Agent {agent_id}: No game state found in final state, using simple fallback")
        return {
            "type": "discard",
            "card_index": 0  # Discard the first card as a last resort
        }

    # Check if we have clue tokens available
    if game_state.clue_tokens > 0:
        # Find a valid target player
        target_id = (agent_id + 1) % len(game_state.hands)  # Next player

        # Check what cards they have
        target_hand = game_state.hands.get(target_id, [])
        if not target_hand:
            logger.warning(
                f"Agent {agent_id}: Target player {target_id} has no cards, using discard fallback")
            return {
                "type": "discard",
                "card_index": 0
            }

        # Try to find a valid color clue
        valid_colors = set()
        for card in target_hand:
            valid_colors.add(card.color.value)

        if valid_colors:
            # Use the first valid color
            color_value = list(valid_colors)[0]
            logger.info(
                f"Agent {agent_id}: Using valid color clue: {color_value} for player {target_id}")
            return {
                "type": "give_clue",
                "target_id": target_id,
                "clue": {
                    "type": "color",
                    "value": color_value
                }
            }

        # If no valid colors (shouldn't happen), try a number clue
        valid_numbers = set()
        for card in target_hand:
            valid_numbers.add(card.number)

        if valid_numbers:
            # Use the first valid number
            number_value = list(valid_numbers)[0]
            logger.info(
                f"Agent {agent_id}: Using valid number clue: {number_value} for player {target_id}")
            return {
                "type": "give_clue",
                "target_id": target_id,
                "clue": {
                    "type": "number",
                    "value": number_value
                }
            }

    # If we can't give a clue, discard a card
    logger.info(
        f"Agent {agent_id}: No valid clue possible, using discard fallback")
    return {
        "type": "discard",
        "card_index": 0  # Discard the first card
    }
