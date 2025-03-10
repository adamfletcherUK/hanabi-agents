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
    store=None,
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
        store: Memory store
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

    # Store the analysis in the memory if store is available
    if store is not None:
        store.put(
            f"agent_{agent_id}_analysis_{len(game_history)}",
            {"prompt": prompt, "response": response.content}
        )

    return new_state


def generate_thoughts(
    state: Dict[str, Any],
    model,
    agent_id: int,
    memory: Dict[str, Any],
    store=None,
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
        store: Memory store
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

    # Store the thoughts in the memory if store is available
    if store is not None:
        store.put(
            f"agent_{agent_id}_thoughts_{len(game_history)}",
            {"prompt": prompt, "response": response.content, "thoughts": thoughts}
        )

    return new_state


def propose_action(
    state: Dict[str, Any],
    model,
    agent_id: int,
    memory: Dict[str, Any],
    store=None,
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
        store: Memory store
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
    recent_errors = errors[-3:] if errors else []

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

    # Extract the proposed action from the response
    proposed_action = _extract_action(response.content)
    new_state["proposed_action"] = proposed_action

    # Store the action proposal in the memory if store is available
    if store is not None:
        store.put(
            f"agent_{agent_id}_action_{len(game_history)}",
            {"prompt": prompt, "response": response.content, "action": proposed_action}
        )

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


def _extract_action(content: str) -> Optional[Dict[str, Any]]:
    """
    Extract the proposed action from the model's response.

    Args:
        content: Content of the model's response

    Returns:
        Dictionary representing the proposed action, or None if no action was found
    """
    # Look for play action
    if "play" in content.lower():
        # Look for card index
        import re
        card_index_match = re.search(
            r"card(?:\sat)?\s+(?:index\s+)?(\d+)", content.lower())
        if card_index_match:
            card_index = int(card_index_match.group(1))
            return {
                "action_type": "play",
                "card_index": card_index
            }

    # Look for clue action
    if "clue" in content.lower() or "hint" in content.lower():
        # Look for target player
        import re
        target_match = re.search(r"(?:player|agent)\s+(\d+)", content.lower())

        # Look for clue type and value
        color_match = re.search(
            r"(red|yellow|green|blue|white)", content.lower())
        number_match = re.search(r"number\s+(\d+)", content.lower())

        if target_match:
            target_id = int(target_match.group(1))

            if color_match:
                return {
                    "action_type": "clue",
                    "target_id": target_id,
                    "clue_type": "color",
                    "clue_value": color_match.group(1)
                }
            elif number_match:
                return {
                    "action_type": "clue",
                    "target_id": target_id,
                    "clue_type": "number",
                    "clue_value": int(number_match.group(1))
                }

    # Look for discard action
    if "discard" in content.lower():
        # Look for card index
        import re
        card_index_match = re.search(
            r"card(?:\sat)?\s+(?:index\s+)?(\d+)", content.lower())
        if card_index_match:
            card_index = int(card_index_match.group(1))
            return {
                "action_type": "discard",
                "card_index": card_index
            }

    # If no action was found, return None
    return None
