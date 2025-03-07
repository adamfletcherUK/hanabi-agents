from typing import Dict, Any
import logging
from langchain_core.messages import HumanMessage, AIMessage
from ..prompts.state_analysis import create_state_analysis_prompt
from ..prompts.thought_generation import create_thought_generation_prompt
from ..prompts.action_proposal import create_action_proposal_prompt

# Set up logging
logger = logging.getLogger(__name__)


def analyze_game_state(state: Dict[str, Any], model: Any, agent_id: int, memory: Dict[str, Any] = None) -> Dict[str, Any]:
    """Analyze the game state to understand the current situation."""
    try:
        # Extract state components
        game_state = state["game_state"]

        # Create the prompt
        prompt = create_state_analysis_prompt(game_state, agent_id, memory)

        # Generate analysis using the LLM
        response = model.invoke([HumanMessage(content=prompt)])

        # Process the response
        if response:
            # Add the analysis to current_thoughts
            current_thoughts = state.get("current_thoughts", [])
            current_thoughts.append(response.content)

            # Return updated state
            return {
                **state,
                "current_thoughts": current_thoughts
            }

        return state
    except Exception as e:
        logger.error(f"Error analyzing game state: {e}")
        return state


def generate_thoughts(state: Dict[str, Any], model: Any, agent_id: int, memory: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generate thoughts about the game state."""
    try:
        # Extract state components
        game_state = state["game_state"]
        discussion_history = state["discussion_history"]
        current_thoughts = state.get("current_thoughts", [])

        # Create the prompt
        prompt = create_thought_generation_prompt(
            game_state, discussion_history, current_thoughts, agent_id, memory)

        # Generate thoughts using the LLM
        response = model.invoke([HumanMessage(content=prompt)])

        # Process the response
        if response:
            # For thought generation, we want natural language, not JSON
            # Clean up the response
            cleaned_response = response.content.strip()

            # Remove any JSON formatting that might have been included
            cleaned_response = cleaned_response.replace(
                "```json", "").replace("```", "")

            # Add the new thought to the list
            current_thoughts.append(cleaned_response)

        # Return updated state
        return {
            **state,
            "current_thoughts": current_thoughts
        }
    except Exception as e:
        logger.error(f"Error generating thoughts: {e}")
        return state


def propose_action(state: Dict[str, Any], model: Any, agent_id: int, memory: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Propose an action based on the game state and thoughts.

    This method handles both the discussion phase and the action phase:
    - In discussion phase: Generates thoughts about potential actions
    - In action phase: Generates actual tool calls for execution

    Args:
        state: The current state of the agent
        model: The LLM model to use
        agent_id: The ID of the agent
        memory: The agent's memory dictionary

    Returns:
        Updated state with proposed action or thoughts
    """
    try:
        # Extract state components
        game_state = state["game_state"]
        discussion_history = state["discussion_history"]
        current_thoughts = state.get("current_thoughts", [])

        # Determine if we're in the action phase (messages will be present)
        is_action_phase = "messages" in state

        # Create the prompt based on the phase
        prompt = create_action_proposal_prompt(
            game_state, discussion_history, current_thoughts, agent_id, memory)

        # Generate response using the LLM with tools
        response = model.invoke([HumanMessage(content=prompt)])

        # Log if tool calls were generated
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info(
                f"Agent {agent_id} generated tool calls: {response.tool_calls}")

            # If in discussion phase but tool calls were generated, extract the content
            # but also preserve the tool calls for later use
            if not is_action_phase:
                content = response.content if response.content else "I'm considering my options."
                logger.info(
                    f"Tool calls generated during discussion phase, using content: {content}")
                current_thoughts.append(f"I'm considering: {content}")

                # Store the tool calls in the state for later use
                state["proposed_tool_calls"] = response.tool_calls

                # Also store in memory if available
                if memory is not None:
                    memory["proposed_tool_calls"] = response.tool_calls
                    logger.info(
                        f"Stored tool calls in memory: {response.tool_calls}")

                return {
                    **state,
                    "current_thoughts": current_thoughts,
                    "proposed_tool_calls": response.tool_calls
                }

        # Process the response based on the phase
        if is_action_phase:
            messages = state.get("messages", [])

            # In action phase, check if we have stored tool calls from discussion phase
            if not (hasattr(response, "tool_calls") and response.tool_calls) and "proposed_tool_calls" in state:
                logger.info(
                    f"Using tool calls from discussion phase: {state['proposed_tool_calls']}")

                # If the response doesn't have tool calls but we have stored ones,
                # create a new response with the stored tool calls
                from langchain_core.messages import AIMessage
                tool_response = AIMessage(
                    content=response.content if hasattr(
                        response, "content") and response.content else "I'll take this action.",
                    tool_calls=state["proposed_tool_calls"]
                )
                messages.append(tool_response)

                # Also add a tool result to make it easier for the extractor
                state["tool_result"] = {
                    "type": state["proposed_tool_calls"][0].get("name", ""),
                    **state["proposed_tool_calls"][0].get("args", {})
                }

                # Map tool names to action types for the tool result
                if state["tool_result"]["type"] == "play_card":
                    state["tool_result"] = {
                        "type": "play_card",
                        "card_index": state["tool_result"].get("card_index", 0)
                    }
                elif state["tool_result"]["type"] == "give_clue":
                    state["tool_result"] = {
                        "type": "give_clue",
                        "target_id": state["tool_result"].get("target_id", 0),
                        "clue": {
                            "type": state["tool_result"].get("clue_type", "color"),
                            "value": state["tool_result"].get("clue_value", "")
                        }
                    }
                elif state["tool_result"]["type"] == "discard":
                    state["tool_result"] = {
                        "type": "discard",
                        "card_index": state["tool_result"].get("card_index", 0)
                    }

                return {
                    **state,
                    "messages": messages,
                    "tool_result": state["tool_result"]
                }
            else:
                # If the response has tool calls or we don't have stored ones,
                # just add the response to messages
                messages.append(response)
                return {
                    **state,
                    "messages": messages
                }
        else:
            # In discussion phase, add the content to thoughts
            content = response.content.strip()
            current_thoughts.append(content)

            return {
                **state,
                "current_thoughts": current_thoughts
            }

    except Exception as e:
        logger.error(f"Error proposing action: {e}")
        # Add error to thoughts instead of re-raising to avoid terminating the graph
        if "messages" in state:
            # In action phase, add error message
            messages = state.get("messages", [])
            messages.append(
                AIMessage(content=f"Error proposing action: {e}"))
            return {
                **state,
                "messages": messages
            }
        else:
            # In discussion phase, add error to thoughts
            current_thoughts = state.get("current_thoughts", [])
            current_thoughts.append(f"Error proposing action: {e}")
            return {
                **state,
                "current_thoughts": current_thoughts
            }
