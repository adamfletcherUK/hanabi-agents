from typing import Dict, Any, List, Optional, Tuple
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

    # Log the state keys for debugging
    logger.info(
        f"State keys after thought generation: {list(new_state.keys())}")
    logger.info(f"Number of thoughts generated: {len(thoughts)}")

    # Store a hash of the thoughts for tracking
    import hashlib
    thoughts_hash = hashlib.md5(str(thoughts).encode()).hexdigest()
    new_state["thoughts_hash"] = thoughts_hash
    logger.info(f"Thoughts hash: {thoughts_hash}")

    return new_state


def _validate_tool_call_consistency(tool_call: Dict[str, Any], thoughts: List[str]) -> bool:
    """
    Validate that the tool call is consistent with the thoughts.

    Args:
        tool_call: The tool call to validate
        thoughts: The thoughts to validate against

    Returns:
        True if the tool call is consistent with the thoughts, False otherwise
    """
    if not tool_call or not thoughts:
        return False

    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {})

    # Extract key concepts from thoughts
    thought_concepts = []
    for thought in thoughts:
        thought_lower = thought.lower()

        # Check for concepts related to giving clues
        if any(term in thought_lower for term in ["clue", "hint", "information", "tell", "inform"]):
            thought_concepts.append("give_clue")

            # Check for specific clue types
            if "color" in thought_lower:
                thought_concepts.append("color_clue")
            if any(str(num) in thought_lower for num in range(1, 6)):
                thought_concepts.append("number_clue")
                # Extract specific numbers mentioned
                for num in range(1, 6):
                    if str(num) in thought_lower:
                        thought_concepts.append(f"number_{num}")

        # Check for concepts related to playing cards
        if any(term in thought_lower for term in ["play", "playable", "safe to play"]):
            thought_concepts.append("play_card")

        # Check for concepts related to discarding
        if any(term in thought_lower for term in ["discard", "throw", "get rid"]):
            thought_concepts.append("discard")

    # Check if the tool call matches any of the thought concepts
    if tool_name == "play_card_tool" and "play_card" in thought_concepts:
        return True
    elif tool_name == "give_clue_tool":
        if "give_clue" in thought_concepts:
            # Check for more specific clue type consistency
            clue_type = tool_args.get("clue_type", "")
            clue_value = tool_args.get("clue_value", "")

            if clue_type == "color" and "color_clue" in thought_concepts:
                return True
            elif clue_type == "number" and "number_clue" in thought_concepts:
                # Check for specific number consistency
                if f"number_{clue_value}" in thought_concepts:
                    return True
                return "number_clue" in thought_concepts

            # If no specific clue type mentioned in thoughts, any clue is consistent
            return True
    elif tool_name == "discard_tool" and "discard" in thought_concepts:
        return True

    # If no specific action concepts found in thoughts, consider it inconsistent
    return False


def _map_thoughts_to_actions(thoughts: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Map thoughts to potential actions.

    Args:
        thoughts: List of thoughts to map

    Returns:
        Dictionary mapping action types to potential actions
    """
    # Initialize action map with the official tool names
    action_map = {
        "play_card_tool": [],
        "give_clue_tool": [],
        "discard_tool": []
    }

    for thought in thoughts:
        thought_lower = thought.lower()

        # Check for play card actions
        if any(term in thought_lower for term in ["play", "playable", "safe to play"]):
            # Try to extract card indices
            card_indices = []
            for i in range(5):  # Assuming max 5 cards in hand
                if f"card {i}" in thought_lower or f"position {i}" in thought_lower:
                    card_indices.append(i)

            # If no specific card mentioned, consider all cards
            if not card_indices:
                card_indices = list(range(5))

            for card_index in card_indices:
                action_map["play_card_tool"].append({
                    "card_index": card_index,
                    "thought": thought
                })

        # Check for give clue actions
        if any(term in thought_lower for term in ["clue", "hint", "tell", "inform"]):
            # Try to extract target player
            target_ids = []
            for i in range(5):  # Assuming max 5 players
                if f"player {i}" in thought_lower:
                    target_ids.append(i)

            # If no specific player mentioned, consider all players
            if not target_ids:
                target_ids = list(range(5))

            # Try to extract clue type
            clue_types = []
            if any(term in thought_lower for term in ["color", "red", "blue", "green", "yellow", "white"]):
                clue_types.append("color")
            if any(term in thought_lower for term in ["number", "1", "2", "3", "4", "5"]):
                clue_types.append("number")

            # If no specific clue type mentioned, consider both
            if not clue_types:
                clue_types = ["color", "number"]

            # Try to extract clue value
            clue_values = []
            # Check for colors
            for color in ["red", "blue", "green", "yellow", "white"]:
                if color in thought_lower:
                    clue_values.append(color)
            # Check for numbers
            for number in ["1", "2", "3", "4", "5"]:
                if number in thought_lower:
                    clue_values.append(number)

            # If no specific clue value mentioned, consider all
            if not clue_values:
                clue_values = ["red", "1"]  # Default values

            for target_id in target_ids:
                for clue_type in clue_types:
                    for clue_value in clue_values:
                        if (clue_type == "color" and clue_value in ["red", "blue", "green", "yellow", "white"]) or \
                           (clue_type == "number" and clue_value in [str(i) for i in range(1, 6)]):
                            action_map["give_clue_tool"].append({
                                "target_id": target_id,
                                "clue_type": clue_type,
                                "clue_value": clue_value,
                                "thought": thought
                            })

        # Check for discard actions
        if any(term in thought_lower for term in ["discard", "throw", "get rid"]):
            # Try to extract card indices
            card_indices = []
            for i in range(5):  # Assuming max 5 cards in hand
                if f"card {i}" in thought_lower or f"position {i}" in thought_lower:
                    card_indices.append(i)

            # If no specific card mentioned, consider all cards
            if not card_indices:
                card_indices = list(range(5))

            for card_index in card_indices:
                action_map["discard_tool"].append({
                    "card_index": card_index,
                    "thought": thought
                })

    return action_map


def _check_tool_call_against_action_map(tool_call: Dict[str, Any], action_map: Dict[str, List[Dict[str, Any]]]) -> Tuple[bool, str]:
    """
    Check if a tool call is consistent with the action map.

    Args:
        tool_call: The tool call to check
        action_map: The action map to check against

    Returns:
        Tuple of (is_consistent, reason)
    """
    if not tool_call:
        return False, "No tool call provided"

    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {})

    # Check if the tool name is in the action map
    if tool_name not in action_map:
        return False, f"Tool name {tool_name} not in action map"

    # Check if there are any potential actions for this tool
    if not action_map[tool_name]:
        return False, f"No potential actions for {tool_name} in action map"

    # Check if the tool args match any potential action
    for potential_action in action_map[tool_name]:
        is_match = True

        # Check each arg in the tool call
        for arg_name, arg_value in tool_args.items():
            if arg_name in potential_action and str(potential_action[arg_name]) != str(arg_value):
                is_match = False
                break

        if is_match:
            return True, f"Tool call matches potential action derived from thought: {potential_action.get('thought', 'Unknown')}"

    return False, f"Tool call {tool_call} does not match any potential action in the action map"


def _normalize_tool_name(tool_name: str) -> str:
    """
    Normalize tool names to match the official tool names.

    Args:
        tool_name: The tool name to normalize

    Returns:
        Normalized tool name
    """
    # Map of alternative tool names to official tool names
    tool_name_map = {
        "play_card": "play_card_tool",
        "give_clue": "give_clue_tool",
        "discard": "discard_tool"
    }

    # Return the normalized name if it's in the map, otherwise return the original
    return tool_name_map.get(tool_name, tool_name)


def propose_action(
    state: Dict[str, Any],
    model,
    agent_id: int,
    memory: Dict[str, Any],
    config=None
) -> Dict[str, Any]:
    """
    Propose an action based on the current game state and thoughts.

    Args:
        state: Current state of the reasoning graph
        model: Language model to use for generating the action proposal
        agent_id: ID of the agent
        memory: Agent's memory
        config: Optional configuration

    Returns:
        Updated state with the proposed action
    """
    # Get the current thoughts from the state
    current_thoughts = state.get("current_thoughts", [])
    if not current_thoughts:
        logger.warning("No thoughts found in state, cannot propose action")
        return state

    # Create a new state to avoid modifying the input state
    new_state = state.copy()

    # Get the messages from the state or initialize an empty list
    messages = state.get("messages", [])

    # Get the game state from the state
    game_state = state.get("game_state")
    if not game_state:
        logger.warning("No game state found in state, cannot propose action")
        return state

    # Get the discussion history from the state
    discussion_history = state.get("discussion_history", [])

    # Get the game history from the state
    game_history = state.get("game_history", [])

    # Get the card knowledge from the state
    card_knowledge = state.get("card_knowledge", {})

    # Get any recent errors from the state
    recent_errors = state.get("errors", [])

    # Create the action proposal prompt
    prompt = create_action_proposal_prompt(
        game_state=game_state,
        agent_id=agent_id,
        card_knowledge=card_knowledge,
        current_thoughts=current_thoughts,
        discussion_history=discussion_history,
        game_history=game_history,
        recent_errors=recent_errors
    )

    # Map thoughts to potential actions
    action_map = _map_thoughts_to_actions(current_thoughts)

    # Create the message
    message = HumanMessage(content=prompt)

    # Get the action proposal from the model
    # Force the model to make a tool call by using tool_choice="required"
    # First, check if the model already has tool_choice set
    if hasattr(model, "tool_choice") and model.tool_choice == "required":
        # Model already has tool_choice set to "required"
        response = model.invoke([message])
    else:
        # Create a copy of the model with tool_choice="required"
        action_model = model.bind(tool_choice="required")
        response = action_model.invoke([message])

    # Add the messages to the state
    messages.append(message)
    messages.append(response)
    new_state["messages"] = messages

    # Log the tool calls if present
    if hasattr(response, "tool_calls") and response.tool_calls:
        logger.info(f"Model proposed tool calls: {response.tool_calls}")

        # Extract the first tool call for detailed logging
        tool_call = response.tool_calls[0]
        original_tool_name = tool_call.get("name", "")

        # Normalize the tool name to match official names
        normalized_tool_name = _normalize_tool_name(original_tool_name)

        # Update the tool call with the normalized name if it changed
        if normalized_tool_name != original_tool_name:
            logger.info(
                f"Normalized tool name from '{original_tool_name}' to '{normalized_tool_name}'")
            tool_call["name"] = normalized_tool_name
            # Update the tool call in the response
            response.tool_calls[0]["name"] = normalized_tool_name

        tool_args = tool_call.get("args", {})
        logger.info(
            f"Tool call details - Name: {normalized_tool_name}, Args: {tool_args}")

        # Analyze the relationship between thoughts and tool call
        logger.info("Analyzing relationship between thoughts and tool call:")
        for i, thought in enumerate(current_thoughts):
            # Check if the thought relates to the tool call
            relation = []
            if normalized_tool_name == "play_card_tool" and "play" in thought.lower():
                relation.append("play_card")
            if normalized_tool_name == "give_clue_tool" and "clue" in thought.lower():
                relation.append("give_clue")
            if normalized_tool_name == "discard_tool" and "discard" in thought.lower():
                relation.append("discard")

            # Check for value references
            if normalized_tool_name == "give_clue_tool":
                clue_value = tool_args.get("clue_value", "")
                if clue_value and clue_value in thought:
                    relation.append(f"value_{clue_value}")

            if relation:
                logger.info(
                    f"  Thought {i+1} relates to tool call via: {', '.join(relation)}")
            else:
                logger.info(
                    f"  Thought {i+1} has no clear relation to tool call")

        # Check if the tool call is consistent with the action map
        is_consistent, reason = _check_tool_call_against_action_map(
            tool_call, action_map)
        if is_consistent:
            logger.info(f"Tool call is consistent with action map: {reason}")
        else:
            logger.warning(
                f"Tool call is NOT consistent with action map: {reason}")

        # Store the proposed tool calls in the state for later use
        new_state["proposed_tool_calls"] = response.tool_calls
    else:
        logger.warning("No tool calls found in model response")

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
