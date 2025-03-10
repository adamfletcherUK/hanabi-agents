from typing import Dict, Any
import logging
from langchain_core.messages import HumanMessage, AIMessage
from ..prompts.state_analysis import create_state_analysis_prompt
from ..prompts.thought_generation import create_thought_generation_prompt
from ..prompts.action_proposal import create_action_proposal_prompt
from ..utils.memory_utils import get_action_history, get_game_history

# Set up logging
logger = logging.getLogger(__name__)


def analyze_game_state(state: Dict[str, Any], model: Any, agent_id: int, memory: Dict[str, Any] = None, *, store=None, config=None) -> Dict[str, Any]:
    """Analyze the game state to understand the current situation."""
    try:
        # Track execution path
        execution_path = state.get("execution_path", [])
        execution_path.append("analyze_state")

        # Extract state components
        game_state = state["game_state"]

        # Get learning history if available
        learning_history = ""
        if store and config:
            try:
                # Get the agent instance from the config if available
                agent = config.get("agent_instance")
                if agent:
                    # Get action history
                    action_history = get_action_history(agent, limit=5)
                    if action_history:
                        learning_history += "\nRecent Action History:\n"
                        for action in action_history:
                            action_type = action.get("action_type", "unknown")
                            result = "succeeded" if action.get(
                                "result", {}).get("success", False) else "failed"
                            learning_history += f"- {action_type} {result}\n"

                    # Get game history
                    game_history = get_game_history(agent, limit=3)
                    if game_history:
                        learning_history += "\nRecent Game History:\n"
                        for game in game_history:
                            score = game.get("score", 0)
                            turns = game.get("turns_played", 0)
                            learning_history += f"- Score: {score}, Turns: {turns}\n"
            except Exception as e:
                logger.warning(f"Error retrieving learning history: {e}")

        # Create the prompt with learning history
        prompt = create_state_analysis_prompt(game_state, agent_id, memory)
        if learning_history:
            prompt += f"\n\nLearning from past games and actions:{learning_history}"

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
                "current_thoughts": current_thoughts,
                "execution_path": execution_path
            }

        return state
    except Exception as e:
        logger.error(f"Error analyzing game state: {e}")
        return state


def generate_thoughts(state: Dict[str, Any], model: Any, agent_id: int, memory: Dict[str, Any] = None, *, store=None, config=None) -> Dict[str, Any]:
    """Generate thoughts about the game state."""
    try:
        # Track execution path
        execution_path = state.get("execution_path", [])
        execution_path.append("generate_thoughts")

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

            # Store the thought in the memory store if available
            if store and config:
                try:
                    # Get the agent instance from the config if available
                    agent = config.get("agent_instance")
                    if agent and hasattr(agent, "store_memory"):
                        # Store the thought with the current turn information
                        thought_entry = {
                            "thought": cleaned_response,
                            "turn": game_state.turn_count,
                            "context": {
                                "score": game_state.score,
                                "clue_tokens": game_state.clue_tokens,
                                "fuse_tokens": game_state.fuse_tokens
                            }
                        }
                        agent.store_memory("thoughts", thought_entry)
                except Exception as e:
                    logger.warning(f"Error storing thought in memory: {e}")

        # Return updated state
        return {
            **state,
            "current_thoughts": current_thoughts,
            "execution_path": execution_path
        }
    except Exception as e:
        logger.error(f"Error generating thoughts: {e}")
        return state


def propose_action(state: Dict[str, Any], model: Any, agent_id: int, memory: Dict[str, Any] = None, *, store=None, config=None) -> Dict[str, Any]:
    """
    Propose an action based on the current game state and thoughts.

    Args:
        state: Current state of the reasoning graph
        model: Language model to use for generating the action
        agent_id: ID of the agent
        memory: Memory store for the agent

    Returns:
        Updated state with the proposed action
    """
    # Log the state keys for debugging
    logger.info(
        f"Agent {agent_id}: State keys before action proposal: {list(state.keys())}")

    # Check if we're in the action phase
    is_action_phase = state.get("is_action_phase", False)
    logger.info(f"Agent {agent_id}: Is action phase: {is_action_phase}")

    # Get the current player
    current_player = state.get("game_state", {}).current_player
    logger.info(f"Agent {agent_id}: Current player: {current_player}")

    # Track the execution path
    execution_path = state.get("execution_path", [])
    execution_path.append("propose_action")
    logger.info(f"Agent {agent_id}: Execution path: {execution_path}")

    # Create a copy of the state to avoid modifying the original
    new_state = state.copy()

    # Set the agent_id and is_action_phase flags
    new_state["agent_id"] = agent_id
    new_state["is_action_phase"] = True
    new_state["execution_path"] = execution_path

    # Add messages to the state for the action phase
    logger.info(f"Agent {agent_id}: Adding messages to state for action phase")

    # If we don't have messages, initialize them
    if "messages" not in new_state:
        new_state["messages"] = []

    # Extract state components for the prompt
    game_state = state.get("game_state", {})
    discussion_history = state.get("discussion_history", [])
    game_history = state.get("game_history", [])
    current_thoughts = state.get("current_thoughts", [])

    # Create a prompt for the action
    from langchain_core.messages import HumanMessage

    # Create a prompt that includes the game state, discussion history, and current thoughts
    prompt = f"""
    You are playing a game of Hanabi. It's your turn to take an action.
    
    Game State:
    {game_state}
    
    Discussion History:
    {discussion_history}
    
    Your Current Thoughts:
    {current_thoughts}
    
    Please take an action by using one of the following tools:
    1. play_card - Play a card from your hand
    2. give_clue - Give a clue to another player
    3. discard - Discard a card from your hand
    
    Choose the most appropriate action based on the current game state and discussion.
    """

    # Add the prompt as a human message
    human_message = HumanMessage(content=prompt)
    new_state["messages"].append(human_message)

    # Use the model to generate the action
    action_message = model.invoke([human_message])

    # Add the action message to the state
    new_state["messages"].append(action_message)

    # Check if the action message has tool calls
    if hasattr(action_message, "tool_calls") and action_message.tool_calls:
        logger.info(
            f"Agent {agent_id}: Generated tool calls: {action_message.tool_calls}")

        # Add the proposed tool calls to the state
        new_state["proposed_tool_calls"] = action_message.tool_calls

        # Log the state keys after adding tool calls
        logger.info(
            f"Agent {agent_id}: State keys after adding tool calls: {list(new_state.keys())}")
        logger.info(
            f"Agent {agent_id}: Tool calls added to state: {action_message.tool_calls}")
    else:
        # If no tool calls, just log the message content
        logger.info(
            f"Agent {agent_id}: No tool calls in action message. Content: {action_message.content}")

    return new_state
