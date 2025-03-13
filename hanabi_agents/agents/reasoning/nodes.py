from typing import Dict, Any, List, Optional, Tuple
from langchain_core.messages import HumanMessage, AIMessage
import logging
from ..prompts.state_analysis import create_state_analysis_prompt
from ..prompts.thought_generation import create_thought_generation_prompt
from ..prompts.action_proposal import create_action_proposal_prompt
import json
import uuid
import re
import hashlib
from ..state.agent_state import AgentStateDict, ActionError, ActionResult
from ..tools.play_card import _play_card_impl
from ..tools.give_clue import _give_clue_impl
from ..tools.discard import _discard_impl

# Set up logging
logger = logging.getLogger(__name__)

# Add a new DEBUG level check to reduce console noise


def debug_only(message):
    """Log only if debug level is enabled"""
    logger.debug(message)


def analyze_game_state(
    state: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze the current game state.

    Args:
        state: The current state
        config: Optional configuration containing model, agent_id, and agent_instance

    Returns:
        Updated state with analysis
    """
    # Extract parameters from config
    if config is None:
        config = {}

    # Move debug-level logging to debug_only
    debug_only(
        f"analyze_game_state called with state keys: {list(state.keys())}")
    debug_only(
        f"analyze_game_state called with config keys: {list(config.keys())}")

    # Use model from config
    model = config.get("model")
    agent_id = config.get("agent_id")
    agent_instance = config.get("agent_instance")

    # Use debug_only for model and agent debugging
    debug_only(f"model from config: {model}")
    debug_only(f"agent_id from config: {agent_id}")
    debug_only(f"agent_id from state: {state.get('agent_id')}")

    if not model:
        raise ValueError("Model not provided in config")

    # Fix: Properly handle agent_id
    if agent_id is None:
        agent_id = state.get("agent_id")

    if agent_id is None:
        raise ValueError("Agent ID not provided in config or state")

    # Check if we've already logged for this execution path to prevent duplicate logging
    execution_path = state.get("execution_path", [])
    # Only log if this is a new execution of analyze_game_state
    if "analyze_game_state" not in execution_path:
        # Keep this log as it's informative but make it short
        logger.info(f"Agent {agent_id} analyzing game state")

    # Create a copy of the state to avoid modifying the original
    new_state = state.copy()

    # Track the execution path
    execution_path = new_state.get("execution_path", [])
    execution_path.append("analyze_game_state")
    new_state["execution_path"] = execution_path

    # Get the game state, card knowledge, discussion history, and game history from the state
    game_state = state.get("game_state", {})
    card_knowledge = state.get("card_knowledge", {})
    discussion_history = state.get("discussion_history", [])
    game_history = state.get("game_history", [])

    # Create the prompt for state analysis
    prompt = create_state_analysis_prompt(
        game_state=game_state,
        agent_id=agent_id,
        card_knowledge=card_knowledge,
        discussion_history=discussion_history,
        game_history=game_history
    )

    # Log the prompt for debugging
    logger.debug(f"State analysis prompt: {prompt}")

    # Create a human message with the prompt
    message = HumanMessage(content=prompt)

    # Send the message to the model
    response = model.invoke([message])

    # Log the response for debugging
    logger.debug(f"State analysis response: {response}")

    # Add the response to the messages
    if "messages" not in new_state:
        new_state["messages"] = []
    new_state["messages"].append(response)

    return new_state


def generate_thoughts(
    state: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate strategic thoughts based on the game state analysis.

    Args:
        state: The current state
        config: Optional configuration containing thought_model, agent_id, and agent_instance

    Returns:
        Updated state with generated thoughts
    """
    # Extract parameters from config
    if config is None:
        config = {}

    # Move debug logging to debug_only
    debug_only(
        f"generate_thoughts called with state keys: {list(state.keys())}")
    debug_only(
        f"generate_thoughts called with config keys: {list(config.keys())}")

    # Use thought_model if available, otherwise use regular model
    model = config.get("thought_model", config.get("model"))
    agent_id = config.get("agent_id")
    agent_instance = config.get("agent_instance")

    # Move model debug info to debug_only
    debug_only(f"model from config: {model}")
    debug_only(f"agent_id from config: {agent_id}")
    debug_only(f"agent_id from state: {state.get('agent_id')}")

    if not model:
        raise ValueError("Model not provided in config")

    # Fix: Properly handle agent_id
    if agent_id is None:
        agent_id = state.get("agent_id")

    if agent_id is None:
        raise ValueError("Agent ID not provided in config or state")

    # Check if we've already logged for this execution path to prevent duplicate logging
    execution_path = state.get("execution_path", [])
    # Only log if this is a new execution of generate_thoughts
    if "generate_thoughts" not in execution_path:
        # Keep this log as it's informative
        logger.info(f"Agent {agent_id} generating thoughts")

    # Create a copy of the state to avoid modifying the original
    new_state = state.copy()

    # Track the execution path
    execution_path = new_state.get("execution_path", [])
    execution_path.append("generate_thoughts")
    new_state["execution_path"] = execution_path

    # Get the game state, card knowledge, discussion history, and game history from the state
    game_state = state.get("game_state", {})
    card_knowledge = state.get("card_knowledge", {})
    discussion_history = state.get("discussion_history", [])
    game_history = state.get("game_history", [])

    # Get any recent errors
    recent_errors = state.get("errors", [])

    # Create the prompt for thought generation
    prompt = create_thought_generation_prompt(
        game_state=game_state,
        agent_id=agent_id,
        card_knowledge=card_knowledge,
        discussion_history=discussion_history,
        game_history=game_history,
        recent_errors=recent_errors
    )

    # Log the prompt for debugging
    logger.debug(f"Thought generation prompt: {prompt}")

    # Check if the model has tool_choice set to "required"
    has_required_tools = _has_required_tools(model)

    # Temporarily disable tool_choice if it's set to "required"
    if has_required_tools:
        logger.info("Temporarily disabling tool_choice for thought generation")
        # Create a copy of the model without tool_choice
        if hasattr(model, "bind"):
            temp_model = model.bind(tool_choice=None)
        else:
            temp_model = model  # Fallback if bind is not available
    else:
        temp_model = model

    # Create a human message with the prompt
    message = HumanMessage(content=prompt)

    try:
        # Send the message to the model
        response = temp_model.invoke([message])

        # Log the response for debugging
        logger.debug(f"Thought generation response: {response}")

        # Add the response to the messages
        if "messages" not in new_state:
            new_state["messages"] = []
        new_state["messages"].append(response)

        # Extract the thoughts from the response
        thoughts = _extract_thoughts(response.content)

        # If no thoughts were extracted, create a default thought
        if not thoughts:
            logger.warning(
                "No thoughts extracted from response, creating default thought")
            thoughts = [
                "I need to analyze the current game state and determine the best action."]
    except Exception as e:
        logger.error(f"Error generating thoughts: {e}")
        thoughts = [
            "I need to analyze the current game state and determine the best action."]

    # Store the thoughts in the state
    new_state["current_thoughts"] = thoughts

    # Log the thoughts for debugging (only to file, not console)
    # Only log if this is the first time these thoughts are being generated
    thoughts_hash = hashlib.md5(str(thoughts).encode()).hexdigest()
    if "thoughts_hash" not in state or state.get("thoughts_hash") != thoughts_hash:
        # File logging only, no console output
        logger.debug("Generated thoughts:")
        for i, thought in enumerate(thoughts):
            logger.debug(f"  • {thought}")
    logger.debug(f"State keys: {new_state.keys()}")

    # Store a hash of the thoughts for tracking
    new_state["thoughts_hash"] = thoughts_hash

    return new_state


def _validate_tool_call_consistency(tool_call, thoughts):
    """
    Validate that the tool call is consistent with the agent's thoughts.

    Args:
        tool_call: The tool call to validate
        thoughts: The agent's current thoughts as a list of strings

    Returns:
        bool: True if the tool call is consistent with the thoughts, False otherwise
    """
    if not tool_call or not thoughts:
        return True  # Can't validate without both pieces

    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {})

    # Join thoughts into a single string and convert to lowercase for case-insensitive matching
    thoughts_text = " ".join(thoughts) if isinstance(
        thoughts, list) else thoughts
    thoughts_lower = thoughts_text.lower() if thoughts_text else ""

    # Check for basic consistency based on tool type
    if tool_name == "play_card_tool":
        # Check if thoughts mention playing a card
        play_indicators = ["play", "playing",
                           "should play", "can play", "safe to play"]
        card_index = tool_args.get("card_index")

        # Check if any play indicators are in the thoughts
        has_play_intent = any(
            indicator in thoughts_lower for indicator in play_indicators)

        # Check if the specific card index is mentioned
        card_mentioned = f"card {card_index}" in thoughts_lower or f"position {card_index}" in thoughts_lower

        return has_play_intent

    elif tool_name == "discard_tool":
        # Check if thoughts mention discarding
        discard_indicators = ["discard", "discarding",
                              "should discard", "can discard", "safe to discard"]

        # Check if any discard indicators are in the thoughts
        return any(indicator in thoughts_lower for indicator in discard_indicators)

    elif tool_name == "give_clue_tool":
        # Check if thoughts mention giving a clue
        clue_indicators = ["clue", "hint",
                           "give information", "inform", "tell"]
        clue_type = tool_args.get("clue_type", "").lower()
        clue_value = tool_args.get("clue_value", "").lower()

        # Check if any clue indicators are in the thoughts
        has_clue_intent = any(
            indicator in thoughts_lower for indicator in clue_indicators)

        # Check if the specific clue type or value is mentioned
        type_mentioned = clue_type in thoughts_lower
        value_mentioned = clue_value in thoughts_lower

        return has_clue_intent and (type_mentioned or value_mentioned)

    # Default to True for unknown tool types
    return True


def _normalize_tool_name(tool_name: str) -> str:
    """
    Normalize tool names to match the official tool names.

    Args:
        tool_name: The tool name to normalize

    Returns:
        The normalized tool name
    """
    # Map of common variations to official tool names
    tool_name_map = {
        "play_card": "play_card_tool",
        "give_clue": "give_clue_tool",
        "discard": "discard_tool",
        "play": "play_card_tool",
        "clue": "give_clue_tool",
    }

    # If the tool name already ends with "_tool", return it as is
    if tool_name.endswith("_tool"):
        return tool_name

    # Convert to lowercase for case-insensitive matching
    lower_tool_name = tool_name.lower()

    # Check for exact matches first
    if lower_tool_name in tool_name_map:
        return tool_name_map[lower_tool_name]

    # Then check for partial matches
    for key, value in tool_name_map.items():
        if key in lower_tool_name:
            return value

    # If not found in the map, return the original with "_tool" appended
    # This ensures consistency with the expected format
    if not tool_name.endswith("_tool"):
        return f"{tool_name}_tool"

    # If all else fails, return the original
    return tool_name


def _has_required_tools(model) -> bool:
    """
    Check if the model has the required tools defined.

    Args:
        model: The model to check

    Returns:
        True if the model has the required tools, False otherwise
    """
    if not hasattr(model, "tools") or not model.tools:
        return False

    required_tools = ["play_card_tool", "give_clue_tool", "discard_tool"]
    model_tool_names = [tool.get("function", {}).get(
        "name", "") for tool in model.tools]

    # Check if all required tools are defined
    return all(tool in model_tool_names for tool in required_tools)


def propose_action(
    state: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Propose an action based on the game state and thoughts.

    Args:
        state: The current state
        config: Optional configuration containing model, agent_id, and agent_instance

    Returns:
        Updated state with proposed action
    """
    # Extract parameters from config
    if config is None:
        config = {}

    # Move debug logging to debug_only
    debug_only(f"propose_action called with state keys: {list(state.keys())}")
    debug_only(
        f"propose_action called with config keys: {list(config.keys())}")

    # Use model from config
    model = config.get("model")
    agent_id = config.get("agent_id")
    agent_instance = config.get("agent_instance")

    # Move debug logging to debug_only
    debug_only(f"model from config: {model}")
    debug_only(f"agent_id from config: {agent_id}")
    debug_only(f"agent_id from state: {state.get('agent_id')}")

    if not model:
        raise ValueError("Model not provided in config")

    # Fix: Properly handle agent_id
    if agent_id is None:
        agent_id = state.get("agent_id")

    if agent_id is None:
        raise ValueError("Agent ID not provided in config or state")

    # Keep this log as it's informative
    logger.info(f"Agent {agent_id} proposing action")

    # Create a copy of the state to avoid modifying the original
    new_state = state.copy()

    # Track the execution path
    execution_path = new_state.get("execution_path", [])
    execution_path.append("propose_action")
    new_state["execution_path"] = execution_path

    # Get the game state, card knowledge, and thoughts from the state
    game_state = state.get("game_state", {})
    card_knowledge = state.get("card_knowledge", {})
    thoughts = state.get("current_thoughts", [])
    discussion_history = state.get("discussion_history", [])
    game_history = state.get("game_history", [])

    # Create the prompt for action proposal
    prompt = create_action_proposal_prompt(
        game_state=game_state,
        agent_id=agent_id,
        card_knowledge=card_knowledge,
        current_thoughts=thoughts,
        discussion_history=discussion_history,
        game_history=game_history
    )

    # Log the prompt for debugging
    logger.debug(f"Action proposal prompt: {prompt}")

    # Create a human message with the prompt
    message = HumanMessage(content=prompt)

    # Send the message to the model
    response = model.invoke([message])

    # Log the response for debugging
    logger.debug(f"Action proposal response: {response}")

    # Add the response to the messages
    if "messages" not in new_state:
        new_state["messages"] = []
    new_state["messages"].append(response)

    # Extract the tool call from the response
    tool_calls = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_calls = response.tool_calls
    else:
        # Try to extract tool calls from the content
        extracted_tool_call = _extract_tool_call_from_text(response.content)
        if extracted_tool_call:
            tool_calls = [extracted_tool_call]

    # Store the tool calls in the state
    new_state["proposed_tool_calls"] = tool_calls

    # Store the tool calls in the agent's memory if available
    if agent_instance and hasattr(agent_instance, "store_tool_calls"):
        agent_instance.store_tool_calls(tool_calls, new_state)

    # Convert the tool call to the format expected by the game engine
    if tool_calls:
        tool_call = tool_calls[0]
        tool_name = _normalize_tool_name(tool_call.get("name", ""))
        tool_args = tool_call.get("args", {})

        if tool_name == "play_card_tool":
            new_state["action"] = {
                "type": "play_card",
                "card_index": tool_args.get("card_index", 0)
            }
        elif tool_name == "give_clue_tool":
            new_state["action"] = {
                "type": "give_clue",
                "target_id": tool_args.get("target_id", 0),
                "clue": {
                    "type": tool_args.get("clue_type", "color"),
                    "value": tool_args.get("clue_value", "red")
                }
            }
        elif tool_name == "discard_tool":
            # Check if at max clue tokens
            if game_state.get("clue_tokens", 0) >= game_state.get("max_clue_tokens", 8):
                logger.warning(
                    f"Agent {agent_id} attempted to discard when at max clue tokens")
                # Find a valid target for a clue
                target_id = None
                for player_id in game_state.get("hands", {}):
                    if player_id != agent_id and game_state["hands"][player_id]:
                        target_id = player_id
                        break

                if target_id is not None:
                    # Find a valid clue to give
                    target_hand = game_state["hands"][target_id]
                    if target_hand:
                        # Try to give a color clue
                        new_state["action"] = {
                            "type": "give_clue",
                            "target_id": target_id,
                            "clue": {
                                "type": "color",
                                "value": target_hand[0]["color"]
                            }
                        }
                else:
                    # If we couldn't find a valid clue target, try to play a card
                    new_state["action"] = {
                        "type": "play_card",
                        "card_index": 0
                    }
            else:
                new_state["action"] = {
                    "type": "discard",
                    "card_index": tool_args.get("card_index", 0)
                }
        else:
            logger.warning(
                f"Unknown tool name: {tool_name}, defaulting to discard")
            new_state["action"] = {
                "type": "discard",
                "card_index": 0
            }
    else:
        # If no tool calls were found, create a default action
        logger.warning(f"No tool calls found, creating default action")
        new_state["action"] = {
            "type": "discard",
            "card_index": 0
        }

    return new_state


def _extract_thoughts(content: str) -> List[str]:
    """
    Extract thoughts from the model's response.

    Args:
        content: The content of the model's response

    Returns:
        A list of extracted thoughts
    """
    if not content:
        return []

    thoughts = []

    # First, try to find simple numbered thoughts (our preferred format)
    numbered_pattern = r'(?:^|\n)\s*(\d+)[.):]\s*(.*?)(?=\n\s*\d+[.):]\s*|$)'
    numbered_matches = re.findall(numbered_pattern, content, re.DOTALL)
    if numbered_matches:
        thoughts = [match[1].strip()
                    for match in numbered_matches if match[1].strip()]
        debug_only(
            f"Extracted {len(thoughts)} thoughts with simple numbered format")
        return thoughts

    # Try to find thoughts with the explicit THOUGHT prefix
    thought_pattern = r'(?:^|\n)(?:THOUGHT\s*\d*:?\s*)(.*?)(?=\n\s*THOUGHT\s*\d*:?\s*|$)'
    thought_matches = re.findall(
        thought_pattern, content, re.DOTALL | re.IGNORECASE)
    if thought_matches:
        thoughts = [match.strip()
                    for match in thought_matches if match.strip()]
        debug_only(f"Extracted {len(thoughts)} thoughts with THOUGHT prefix")
        return thoughts

    # If no explicit THOUGHT prefixes, try alternative numbered thoughts
    alt_numbered_pattern = r'(?:^|\n)(?:\d+\.?\s*)(.*?)(?=\n\s*\d+\.?\s*|$)'
    alt_numbered_matches = re.findall(alt_numbered_pattern, content, re.DOTALL)
    if alt_numbered_matches:
        thoughts = [match.strip()
                    for match in alt_numbered_matches if match.strip()]
        debug_only(
            f"Extracted {len(thoughts)} thoughts with alternative numbered format")
        return thoughts

    # Try to find sections labeled as thoughts
    labeled_patterns = [
        r'(?:^|\n)(?:Thought|THOUGHT|Reasoning|REASONING|Analysis|ANALYSIS)[^\n]*?:\s*(.*?)(?=\n\s*(?:Thought|THOUGHT|Reasoning|REASONING|Analysis|ANALYSIS)[^\n]*?:|$)',
        r'(?:^|\n)(?:[*\-•]\s*)(.*?)(?=\n\s*[*\-•]\s*|$)'
    ]

    for pattern in labeled_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        if matches:
            thoughts = [match.strip() for match in matches if match.strip()]
            debug_only(
                f"Extracted {len(thoughts)} thoughts with labeled format")
            return thoughts

    # If all else fails, split by paragraphs
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if paragraphs:
        # Filter out very short paragraphs and headers
        thoughts = [p for p in paragraphs if len(
            p) > 15 and not p.startswith('#')]
        debug_only(f"Extracted {len(thoughts)} thoughts from paragraphs")
        if thoughts:
            return thoughts

    # Last resort: use the whole content as one thought
    logger.warning(
        "Could not extract structured thoughts, using entire content")
    return [content.strip()]


def _extract_tool_call_from_text(content: str) -> Optional[Dict[str, Any]]:
    """
    Extract a tool call from a text response.

    Args:
        content: The text content to extract from

    Returns:
        A tool call dictionary if found, None otherwise
    """
    # Try to find a JSON object in the text
    import re
    import json

    # Look for JSON objects in the text
    json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
    matches = re.findall(json_pattern, content)

    for match in matches:
        try:
            data = json.loads(match)
            # Check if this looks like a tool call
            if "name" in data and "args" in data:
                # Add missing fields if needed
                if "id" not in data:
                    data["id"] = f"call_{uuid.uuid4().hex.replace('-', '')}"
                if "type" not in data:
                    data["type"] = "tool_call"
                return data
        except json.JSONDecodeError:
            continue

    return None


def execute_action(state: AgentStateDict, config: Optional[Dict[str, Any]] = None) -> AgentStateDict:
    """
    Execute the proposed action using the appropriate tool implementation.

    Args:
        state: The current state
        config: Optional configuration

    Returns:
        Updated state with action results
    """
    logger.info("Executing proposed action")

    # Move debug logging to debug_only
    debug_only(f"execute_action called with state keys: {list(state.keys())}")
    debug_only(
        f"execute_action called with config keys: {list(config.keys()) if config else []}")

    # Create a copy of the state to avoid modifying the original
    new_state = state.copy()

    try:
        # Get the agent instance and game state from config
        agent_instance = config.get("agent_instance") if config else None
        agent_id = config.get("agent_id") if config else state.get("agent_id")

        # Move debug logging to debug_only
        debug_only(
            f"agent_id from config: {config.get('agent_id') if config else None}")
        debug_only(f"agent_id from state: {state.get('agent_id')}")

        # Fix: Properly handle agent_id
        if agent_id is None:
            agent_id = state.get("agent_id")

        if agent_id is None:
            raise ValueError("Agent ID not provided in config or state")

        game_state = agent_instance.current_game_state if agent_instance else None

        if not game_state:
            raise ValueError("Game state not available for action execution")

        # Get the proposed tool calls from the state
        proposed_tool_calls = state.get("proposed_tool_calls")

        if not proposed_tool_calls:
            logger.warning("No proposed tool calls found in state")
            new_state["action_result"] = ActionResult(
                action="none",
                result="No action was proposed",
                timestamp=state.get("timestamp", "")
            )
            return new_state

        # Get the first tool call
        tool_call = proposed_tool_calls[0]

        # Normalize the tool name
        tool_name = _normalize_tool_name(tool_call.get("name", ""))
        tool_args = tool_call.get("args", {})

        # Execute the appropriate tool based on the name
        result = None
        if tool_name == "play_card_tool":
            card_index = tool_args.get("card_index", 0)
            result = _play_card_impl(agent_id, card_index, game_state)
        elif tool_name == "give_clue_tool":
            target_id = tool_args.get("target_id", 0)
            clue_type = tool_args.get("clue_type", "")
            clue_value = tool_args.get("clue_value", "")
            result = _give_clue_impl(
                agent_id, target_id, clue_type, clue_value, game_state)
        elif tool_name == "discard_tool":
            # Check for max clue tokens before discarding
            if game_state.clue_tokens >= game_state.max_clue_tokens:
                logger.warning(
                    f"Agent {agent_id} attempted to discard when at max clue tokens")
                result = {"error": "Cannot discard when at max clue tokens"}
            else:
                card_index = tool_args.get("card_index", 0)
                result = _discard_impl(agent_id, card_index, game_state)
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        # Check for errors in the result
        if result and "error" in result:
            logger.error(
                f"Error executing tool {tool_name}: {result['error']}")
            new_state["error"] = result["error"]
            new_state["action_error"] = ActionError(
                action={"name": tool_name, "args": tool_args},
                error=result["error"],
                timestamp=state.get("timestamp", ""),
                turn=state.get("turn_count", 0)
            )
        else:
            # Store the successful result
            logger.info(f"Successfully executed tool {tool_name}")
            new_state["action_result"] = ActionResult(
                action=tool_name,
                args=tool_args,
                result=result,
                timestamp=state.get("timestamp", "")
            )

            # Create a properly formatted action for the game engine
            if tool_name == "play_card_tool":
                new_state["action"] = {
                    "type": "play_card",
                    "card_index": tool_args.get("card_index", 0)
                }
            elif tool_name == "give_clue_tool":
                new_state["action"] = {
                    "type": "give_clue",
                    "target_id": tool_args.get("target_id", 0),
                    "clue": {
                        "type": tool_args.get("clue_type", "color"),
                        "value": tool_args.get("clue_value", "red")
                    }
                }
            elif tool_name == "discard_tool":
                new_state["action"] = {
                    "type": "discard",
                    "card_index": tool_args.get("card_index", 0)
                }

            # Store in agent memory if available
            if agent_instance and hasattr(agent_instance, "store_memory"):
                agent_instance.store_memory(
                    "action_result", new_state["action_result"])
                # Also store the formatted action
                if "action" in new_state:
                    agent_instance.store_memory("action", new_state["action"])

        return new_state
    except Exception as e:
        logger.error(f"Error in execute_action: {e}")
        new_state["error"] = str(e)
        return new_state


def handle_error(state: AgentStateDict, config: Optional[Dict[str, Any]] = None) -> AgentStateDict:
    """
    Handle errors that occurred during action execution.

    Args:
        state: The current state
        config: Optional configuration

    Returns:
        Updated state with error handling
    """
    logger.info("Handling error in action execution")

    # Move debug logging to debug_only
    debug_only(f"handle_error called with state keys: {list(state.keys())}")
    debug_only(
        f"handle_error called with config keys: {list(config.keys()) if config else []}")

    # Create a copy of the state to avoid modifying the original
    new_state = state.copy()

    # Get the error from the state
    error = state.get("error", "Unknown error")
    action_error = state.get("action_error")

    # Log the error
    logger.error(f"Error in action execution: {error}")

    # Get the agent instance and agent_id from config
    agent_instance = config.get("agent_instance") if config else None
    agent_id = config.get("agent_id") if config else state.get("agent_id")

    # Move debug logging to debug_only
    debug_only(
        f"agent_id from config: {config.get('agent_id') if config else None}")
    debug_only(f"agent_id from state: {state.get('agent_id')}")

    # Fix: Properly handle agent_id
    if agent_id is None:
        agent_id = state.get("agent_id")

    if agent_id is None:
        logger.warning(
            "Agent ID not provided in config or state for error handling")
        # Continue anyway since this is error handling

    # Store the error in agent memory if available
    if agent_instance and hasattr(agent_instance, "store_memory"):
        # Get existing errors or initialize empty list
        errors = agent_instance.get_memory_from_store("action_errors", [])

        # Add the new error
        if action_error:
            errors.append(action_error)
        else:
            errors.append(ActionError(
                action={"name": "unknown", "args": {}},
                error=error,
                timestamp=state.get("timestamp", ""),
                turn=state.get("turn_count", 0)
            ))

        # Store the updated errors
        agent_instance.store_memory("action_errors", errors)

    # Clear the error from the state to avoid infinite loops
    new_state.pop("error", None)

    # Add the error to the state's errors list for reference
    errors = new_state.get("errors", [])
    errors.append(error)
    new_state["errors"] = errors

    # Return the updated state
    return new_state
