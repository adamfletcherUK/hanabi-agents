from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage
import logging
from ..prompts.state_analysis import create_state_analysis_prompt
from ..prompts.thought_generation import create_thought_generation_prompt
from ..prompts.action_proposal import create_action_proposal_prompt

# Set up logging
logger = logging.getLogger(__name__)


def analyze_game_state(
    state: Dict[str, Any],
    model,
    agent_id: int,
    memory: Dict[str, Any],
    config=None
) -> Dict[str, Any]:
    """
    Analyze the current game state.

    This node analyzes the game state, including the player's hand, other players' hands,
    firework piles, discard pile, and available resources.

    Args:
        state: Current state of the reasoning graph
        model: Language model to use for analysis
        agent_id: ID of the agent
        memory: Agent's memory
        config: Configuration

    Returns:
        Updated state with analysis results
    """
    logger.info(f"Analyzing game state for agent {agent_id}")

    # Create a copy of the state to avoid modifying the original
    new_state = state.copy()

    # Track the execution path
    execution_path = new_state.get("execution_path", [])
    execution_path.append("analyze_state")
    new_state["execution_path"] = execution_path

    # Get the game state
    game_state = new_state.get("game_state")
    if game_state is None:
        logger.error("No game state found in state")
        return new_state

    # Get the card knowledge
    card_knowledge = new_state.get("card_knowledge", [])

    # Get the discussion history
    discussion_history = new_state.get("discussion_history", [])

    # Get the game history
    game_history = new_state.get("game_history", [])

    # Create the prompt
    prompt = create_state_analysis_prompt(
        game_state=game_state,
        agent_id=agent_id,
        card_knowledge=card_knowledge,
        discussion_history=discussion_history,
        game_history=game_history
    )

    # Create the message
    message = HumanMessage(content=prompt)

    # Get the analysis from the model
    response = model.invoke([message])

    # Add the messages to the state
    messages = new_state.get("messages", [])
    messages.append(message)
    messages.append(response)
    new_state["messages"] = messages

    return new_state


def generate_thoughts(
    state: Dict[str, Any],
    model,
    agent_id: int,
    memory: Dict[str, Any],
    config=None
) -> Dict[str, Any]:
    """
    Generate strategic thoughts based on the game state analysis.

    This node generates thoughts about the current game state, including
    what cards the agent might have, what actions would be beneficial,
    and what other players might be trying to communicate.

    Args:
        state: Current state of the reasoning graph
        model: Language model to use for thought generation
        agent_id: ID of the agent
        memory: Agent's memory
        config: Configuration

    Returns:
        Updated state with generated thoughts
    """
    logger.info(f"Generating thoughts for agent {agent_id}")

    # Create a copy of the state to avoid modifying the original
    new_state = state.copy()

    # Track the execution path
    execution_path = new_state.get("execution_path", [])
    execution_path.append("generate_thoughts")
    new_state["execution_path"] = execution_path

    # Get the game state
    game_state = new_state.get("game_state")
    if game_state is None:
        logger.error("No game state found in state")
        return new_state

    # Get the card knowledge
    card_knowledge = new_state.get("card_knowledge", [])

    # Get the discussion history
    discussion_history = new_state.get("discussion_history", [])

    # Get the game history
    game_history = new_state.get("game_history", [])

    # Get the messages from the previous step
    messages = new_state.get("messages", [])

    # Get any errors from previous actions
    errors = new_state.get("errors", [])
    recent_errors = errors[-3:] if errors else []

    # Create the prompt
    prompt = create_thought_generation_prompt(
        game_state=game_state,
        agent_id=agent_id,
        card_knowledge=card_knowledge,
        discussion_history=discussion_history,
        game_history=game_history,
        previous_analysis=messages[-1].content if len(messages) >= 1 else None,
        recent_errors=recent_errors
    )

    # Create the message
    message = HumanMessage(content=prompt)

    # Get the thoughts from the model
    response = model.invoke([message])

    # Add the messages to the state
    messages.append(message)
    messages.append(response)
    new_state["messages"] = messages

    # Extract the thoughts from the response
    thoughts = _extract_thoughts(response.content)
    new_state["current_thoughts"] = thoughts

    # Log the extracted thoughts
    logger.info(f"Extracted thoughts for agent {agent_id}:")
    for i, thought in enumerate(thoughts):
        logger.info(f"  Thought {i+1}: {thought}")

    return new_state


def propose_action(
    state: Dict[str, Any],
    model,
    agent_id: int,
    memory: Dict[str, Any],
    config=None
) -> Dict[str, Any]:
    """
    Propose an action based on the generated thoughts.

    This node proposes a concrete action for the agent to take, such as
    playing a card, giving a clue, or discarding a card.

    Args:
        state: Current state of the reasoning graph
        model: Language model to use for action proposal
        agent_id: ID of the agent
        memory: Agent's memory
        config: Configuration

    Returns:
        Updated state with proposed action
    """
    logger.info(f"Proposing action for agent {agent_id}")

    # Create a copy of the state to avoid modifying the original
    new_state = state.copy()

    # Track the execution path
    execution_path = new_state.get("execution_path", [])
    execution_path.append("propose_action")
    new_state["execution_path"] = execution_path

    # Get the game state
    game_state = new_state.get("game_state")
    if game_state is None:
        logger.error("No game state found in state")
        return new_state

    # Get the card knowledge
    card_knowledge = new_state.get("card_knowledge", [])

    # Get the current thoughts
    current_thoughts = new_state.get("current_thoughts", [])

    # Get the discussion history
    discussion_history = new_state.get("discussion_history", [])

    # Get the game history
    game_history = new_state.get("game_history", [])

    # Get the messages from the previous steps
    messages = new_state.get("messages", [])

    # Get any errors from previous actions
    errors = new_state.get("errors", [])

    # Get action errors from memory if available
    action_errors = []
    if config and "agent_instance" in config:
        agent = config["agent_instance"]
        if hasattr(agent, "get_memory_from_store"):
            action_errors = agent.get_memory_from_store("action_errors", [])
            if action_errors:
                logger.info(
                    f"Retrieved {len(action_errors)} action errors from memory")

    # Combine errors from state and memory
    recent_errors = []

    # Add errors from state
    if errors:
        recent_errors.extend(errors[-3:])

    # Add errors from memory
    if action_errors:
        # Convert memory error format to the format expected by the prompt
        for error in action_errors[-3:]:
            action_type = error.get("action", {}).get("type", "unknown")
            error_message = error.get("error", "Unknown error")

            # Create a structured error record
            error_record = {
                "action_type": action_type,
                "guidance": f"Previous attempt failed: {error_message}. Avoid this exact action."
            }

            # Add specific guidance for clue errors
            if action_type == "give_clue":
                clue = error.get("action", {}).get("clue", {})
                target_id = error.get("action", {}).get("target_id")
                if "no_affected_cards" in error_message or "wouldn't affect any cards" in error_message:
                    error_record["guidance"] = f"Clue {clue.get('type')}={clue.get('value')} to Player {target_id} failed because it doesn't affect any cards in their hand. Check their hand carefully before giving clues."

            recent_errors.append(error_record)

    # Limit to most recent 3 errors to avoid overwhelming the prompt
    recent_errors = recent_errors[-3:]

    if recent_errors:
        logger.info(
            f"Including {len(recent_errors)} recent errors in action proposal prompt")

    # Create the prompt
    prompt = create_action_proposal_prompt(
        game_state=game_state,
        agent_id=agent_id,
        card_knowledge=card_knowledge,
        current_thoughts=current_thoughts,
        discussion_history=discussion_history,
        game_history=game_history,
        recent_errors=recent_errors
    )

    # Create the message
    message = HumanMessage(content=prompt)

    # Get the action proposal from the model
    response = model.invoke([message])

    # Add the messages to the state
    messages.append(message)
    messages.append(response)
    new_state["messages"] = messages

    # Log the tool calls if present
    if hasattr(response, "tool_calls") and response.tool_calls:
        logger.info(f"Model proposed tool calls: {response.tool_calls}")

        # Store the tool calls in the state for reference
        new_state["proposed_tool_calls"] = response.tool_calls

        # Extract the first tool call for backward compatibility
        first_tool_call = response.tool_calls[0]
        tool_name = first_tool_call.get("name")
        tool_args = first_tool_call.get("args", {})

        # Map tool names to action types
        action_type_map = {
            "play_card": "play",
            "give_clue": "clue",
            "discard": "discard"
        }

        # Create a proposed action in the old format for backward compatibility
        proposed_action = {
            "action_type": action_type_map.get(tool_name, "unknown")
        }

        # Add tool-specific arguments
        if tool_name == "play_card" or tool_name == "discard":
            proposed_action["card_index"] = tool_args.get("card_index")
        elif tool_name == "give_clue":
            proposed_action["target_id"] = tool_args.get("target_id")
            proposed_action["clue_type"] = tool_args.get("clue_type")
            proposed_action["clue_value"] = tool_args.get("clue_value")

        # Store the proposed action in the state
        new_state["proposed_action"] = proposed_action
    else:
        logger.warning("No tool calls found in model response")
        new_state["proposed_action"] = None

    return new_state


def _extract_thoughts(content: str) -> List[str]:
    """
    Extract thoughts from the model's response.

    Args:
        content: Content of the model's response

    Returns:
        List of extracted thoughts
    """
    thoughts = []

    # Split the content into lines
    lines = content.strip().split("\n")

    # Look for lines that start with numbers or bullet points
    for line in lines:
        line = line.strip()
        if line and (line[0].isdigit() or line[0] in ["•", "-", "*"]):
            # Remove the number or bullet point
            thought = line
            if line[0].isdigit():
                parts = line.split(".", 1)
                if len(parts) > 1:
                    thought = parts[1].strip()
            else:
                thought = line[1:].strip()

            if thought:
                thoughts.append(thought)

    # If no thoughts were found, use the whole content
    if not thoughts:
        thoughts = [content.strip()]

    return thoughts
