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

    # Check if we have a tool result
    if "tool_result" in final_state and final_state["tool_result"]:
        logger.info(
            f"Agent {agent_id}: Found tool_result in final state: {final_state['tool_result']}")
        return final_state["tool_result"]

    # Check for proposed tool calls from discussion phase
    if "proposed_tool_calls" in final_state and final_state["proposed_tool_calls"]:
        logger.info(
            f"Agent {agent_id}: Found proposed_tool_calls in final state: {final_state['proposed_tool_calls']}")
        tool_call = final_state["proposed_tool_calls"][0]
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
            logger.debug(
                f"Agent {agent_id}: Last message content: {last_message.content if hasattr(last_message, 'content') else 'No content'}")
    else:
        logger.warning(f"Agent {agent_id}: No messages found in final state")

    # If we still don't have an action, raise an error instead of using a default action
    error_message = f"Agent {agent_id}: Failed to extract any action from the final state"
    logger.error(error_message)

    # Include state information in the error for debugging
    state_info = ""
    if "current_thoughts" in final_state:
        state_info += f"\nCurrent thoughts: {final_state['current_thoughts']}"
    if "discussion_history" in final_state:
        state_info += f"\nDiscussion history: {final_state['discussion_history']}"

    # Raise an exception with detailed information
    raise ValueError(f"{error_message}. {state_info}")
