from typing import Dict, Any, List, Optional, Union
import os
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from .base import Agent
from ..game.state import GameState
from .state.state_factory import create_initial_state, create_action_state
from .reasoning.graph import setup_reasoning_graph
from .tools import play_card_tool, give_clue_tool, discard_tool
from .reasoning.nodes import _normalize_tool_name, execute_action
import datetime
from .state.agent_state import AgentMemory, ActionError, ActionResult, AgentStateDict
from .reasoning.schemas import ActionProposal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIAgent(Agent):
    """
    AI agent implementation using LangGraph for reasoning.

    This agent uses a language model and a reasoning graph to analyze the game state,
    generate strategic thoughts, and propose actions.
    """

    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.0,
        agent_id: int = 0,
        **kwargs
    ):
        """
        Initialize the AI agent.

        Args:
            agent_id: The ID of the agent
            model_name: The name of the model to use (defaults to env var)
            temperature: The temperature to use for the model
            **kwargs: Additional arguments including name
        """
        super().__init__(agent_id, kwargs.get("name"))

        # Set up logging to suppress detailed debugging
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("langchain").setLevel(logging.WARNING)
        logging.getLogger("langchain_core").setLevel(logging.WARNING)
        logging.getLogger("langgraph").setLevel(logging.WARNING)

        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in your .env file or environment.")

        # Use default model if none specified
        if model_name is None:
            model_name = os.getenv("MODEL_NAME", "gpt-4")
            logger.info(f"Using model from environment: {model_name}")

        self.model_name = model_name
        self.temperature = temperature

        # Initialize models for different purposes
        self.thought_model = self._initialize_model(
            use_json_format=False)  # For generating thoughts
        self.action_model = self._initialize_model(
            use_json_format=True)    # For proposing actions
        self.model = self.thought_model  # Default model for backward compatibility

        # Initialize agent memory
        self.agent_memory = AgentMemory()

        # Initialize memory store as a simple dictionary
        self.memory_store = {}

        # Initialize checkpointer as None
        self.checkpointer = None

        # Initialize current game state
        self.current_game_state = None

        # Track when thoughts have been displayed (for logging purposes)
        self._thoughts_displayed_for_turn = -1

        # Initialize reasoning graph
        self.reasoning_graph = setup_reasoning_graph(self)

    def _initialize_model(self, use_json_format: bool = False) -> ChatOpenAI:
        """
        Initialize the model with improved structured output support.

        Args:
            use_json_format: Whether to enable JSON response format
        """
        model_kwargs = {}

        if use_json_format:
            # Ensure consistent structured output format
            model_kwargs["response_format"] = {"type": "json_object"}
            # We're using structured output instead of function calling
            model_kwargs["functions"] = None
        else:
            # For the thought model, define consistent tool schemas
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "play_card_tool",
                        "description": "Play a card from your hand. IMPORTANT: Use 1-indexed positions (first card = 1, second card = 2, etc.)",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "card_index": {
                                    "type": "integer",
                                    "description": "The position of the card to play (1-indexed: first card = 1, second card = 2, etc.)"
                                }
                            },
                            "required": ["card_index"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "give_clue_tool",
                        "description": "Give a clue to another player",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "target_id": {
                                    "type": "integer",
                                    "description": "The ID of the player to give the clue to"
                                },
                                "clue_type": {
                                    "type": "string",
                                    "enum": ["color", "number"],
                                    "description": "The type of clue to give (color or number)"
                                },
                                "clue_value": {
                                    "type": "string",
                                    "description": "The value of the clue (a color or a number as a string)"
                                }
                            },
                            "required": ["target_id", "clue_type", "clue_value"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "discard_tool",
                        "description": "Discard a card from your hand. IMPORTANT: Use 1-indexed positions (first card = 1, second card = 2, etc.)",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "card_index": {
                                    "type": "integer",
                                    "description": "The position of the card to discard (1-indexed: first card = 1, second card = 2, etc.)"
                                }
                            },
                            "required": ["card_index"]
                        }
                    }
                }
            ]
            model_kwargs["tools"] = tools

        return ChatOpenAI(
            model=self.model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            verbose=True,  # Enable verbose mode for better debugging
            temperature=self.temperature,
            model_kwargs=model_kwargs
        )

    def verify_tool_call(self, tool_call):
        """
        Verify and standardize a tool call to ensure it's properly formatted.

        Args:
            tool_call: The tool call to verify

        Returns:
            The verified and standardized tool call, or None if invalid
        """
        try:
            import json

            # Basic validation
            if not isinstance(tool_call, dict):
                logger.warning(f"Tool call is not a dictionary: {tool_call}")
                return None

            if "name" not in tool_call:
                logger.warning(f"Tool call missing 'name': {tool_call}")
                return None

            if "args" not in tool_call or not isinstance(tool_call["args"], dict):
                logger.warning(
                    f"Tool call missing or invalid 'args': {tool_call}")
                # Try to convert non-dict args to dict
                if isinstance(tool_call.get("args"), str):
                    try:
                        tool_call["args"] = json.loads(tool_call["args"])
                    except:
                        tool_call["args"] = {}
                else:
                    tool_call["args"] = {}

            # Normalize tool name
            tool_call["name"] = _normalize_tool_name(tool_call["name"])

            return tool_call

        except Exception as e:
            logger.error(f"Error verifying tool call: {e}")
            return None

    def participate_in_discussion(self, game_state: GameState, discussion_history: List[Dict[str, Any]]) -> str:
        """
        Analyze the game state and generate a contribution to the discussion.

        Args:
            game_state: Current state of the game (filtered for this agent)
            discussion_history: History of the discussion so far

        Returns:
            A string containing the agent's contribution to the discussion
        """
        logger.info(f"Agent {self.agent_id} analyzing game state")

        # Store the current game state for tool access
        self.current_game_state = game_state

        # Create the initial state
        state = create_initial_state(
            game_state=game_state,
            agent_id=self.agent_id,
            discussion_history=discussion_history
        )

        # Explicitly set agent_id in the state to ensure it's available
        state["agent_id"] = self.agent_id

        # Create a unique conversation ID for this agent and turn
        conversation_id = f"agent_{self.agent_id}_turn_{game_state.turn_count}"

        # Create a unique thread ID for this conversation
        thread_id = f"thread_{self.agent_id}_{game_state.turn_count}_{datetime.datetime.now().timestamp()}"

        # Run the reasoning graph with explicit model configuration and conversation ID
        result = self.reasoning_graph.invoke(
            state,
            config={
                "agent_id": self.agent_id,
                "agent_instance": self,
                "model": self.model,  # For action proposal
                "configurable": {
                    "conversation_id": conversation_id,
                    "thread_id": thread_id  # Add thread_id for checkpoint system
                }
            }
        )

        # Extract the thoughts and tool calls from the result
        thoughts = result.get("current_thoughts", [])
        messages = result.get("messages", [])

        # Store the thoughts in the agent's memory
        self.agent_memory.thoughts = thoughts
        # Also store in the memory store for immediate access
        self.store_memory("current_thoughts", thoughts)

        # Store the messages for later reference
        self.store_memory("messages", messages)

        # Find the last message with tool calls
        tool_calls = None
        for message in reversed(messages):
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_calls = message.tool_calls
                logger.info(f"Found tool calls in message: {tool_calls}")
                break

        # Store the tool calls for later use with better error handling
        if tool_calls:
            try:
                # Standardize and store tool calls
                updated_state = self.store_tool_calls(tool_calls, {})
                # Extract any standardized actions
                standardized_actions = updated_state.get(
                    "standardized_actions", [])
                if standardized_actions:
                    logger.info(
                        f"Successfully stored standardized actions: {standardized_actions}")
                else:
                    logger.warning(
                        "No standardized actions extracted from tool calls")
            except Exception as e:
                logger.error(f"Error processing tool calls: {e}")
                # Continue with fallback behavior
        else:
            logger.warning(
                "No tool calls found in messages after reasoning graph execution")

        # Format the thoughts and tool calls into a contribution
        contribution = "## Game State Analysis\n\n"
        for i, thought in enumerate(thoughts):
            contribution += f"{i+1}. {thought}\n"

        contribution += "\n## Suggested Tool Call\n\n"
        if tool_calls:
            tool_call = tool_calls[0]  # Get the first tool call
            original_tool_name = tool_call.get("name", "")

            # Normalize the tool name to match official names
            tool_name = _normalize_tool_name(original_tool_name)

            # Log if the tool name was normalized
            if tool_name != original_tool_name:
                logger.info(
                    f"Normalized tool name from '{original_tool_name}' to '{tool_name}'")

            tool_args = tool_call.get("args", {})

            if tool_name == "play_card_tool":
                card_index = tool_args.get("card_index", 0)
                contribution += f"I suggest playing card {card_index}."
            elif tool_name == "give_clue_tool":
                target_id = tool_args.get("target_id", 0)
                clue_type = tool_args.get("clue_type", "unknown")
                clue_value = tool_args.get("clue_value", "unknown")
                contribution += f"I suggest giving a {clue_type} clue about {clue_value} to Player {target_id}."
            elif tool_name == "discard_tool":
                card_index = tool_args.get("card_index", 0)
                contribution += f"I suggest discarding card {card_index}."
            else:
                contribution += "I don't have a specific suggestion at this time."
        else:
            contribution += "I don't have a specific suggestion at this time."

        return contribution

    def decide_action(self, game_state: GameState, discussion_summary: Optional[str] = None) -> Dict[str, Any]:
        """
        Decide on an action to take based on the game state and discussion summary.

        Args:
            game_state: Current state of the game (filtered for this agent)
            discussion_summary: Summary of the discussion (optional)

        Returns:
            A dictionary containing the action to take
        """
        logger.info(f"Agent {self.agent_id} deciding on action")

        # First, check if we have a stored standardized action from the discussion phase
        # These should already be in 0-indexed format from store_tool_calls
        primary_action = self.get_memory_from_store("primary_action")
        if primary_action:
            logger.info(
                f"Using stored primary action from discussion: {primary_action}")
            return primary_action

        # If no stored action, check for tool-specific actions
        # These should already be in 0-indexed format from store_tool_calls
        play_card_action = self.get_memory_from_store("play_card_action")
        if play_card_action:
            logger.info(f"Using stored play_card_action: {play_card_action}")
            return play_card_action

        give_clue_action = self.get_memory_from_store("give_clue_action")
        if give_clue_action:
            logger.info(f"Using stored give_clue_action: {give_clue_action}")
            return give_clue_action

        discard_action = self.get_memory_from_store("discard_action")
        if discard_action:
            logger.info(f"Using stored discard_action: {discard_action}")
            return discard_action

        # If no stored actions found, continue with existing logic to generate a new action
        logger.info(
            "No stored actions found, generating new action through reasoning graph")

        # Store the current game state for tool access
        self.current_game_state = game_state

        # Create the action state
        state = create_action_state(
            game_state=game_state,
            agent_id=self.agent_id,
            discussion_summary=discussion_summary
        )

        # Explicitly set agent_id in the state to ensure it's available
        state["agent_id"] = self.agent_id

        # Create a unique conversation ID for this agent and turn
        conversation_id = f"agent_{self.agent_id}_action_turn_{game_state.turn_count}"

        # Create a unique thread ID for this conversation
        thread_id = f"thread_action_{self.agent_id}_{game_state.turn_count}_{datetime.datetime.now().timestamp()}"

        # Run the reasoning graph with explicit model configuration
        result = self.reasoning_graph.invoke(
            state,
            config={
                "agent_id": self.agent_id,
                "agent_instance": self,
                "model": self.model,
                "configurable": {
                    "conversation_id": conversation_id,
                    "thread_id": thread_id  # Add thread_id for checkpoint system
                }
            }
        )

        # Log the result keys for debugging
        logger.info(f"Result keys from reasoning graph: {list(result.keys())}")

        # Extract the action from the result
        action = result.get("action", {})
        if not action:
            logger.warning(
                f"No action found in result, checking for action_result")
            # Try to extract action from action_result
            action_result = result.get("action_result", {})
            if action_result and hasattr(action_result, "action") and action_result.action:
                logger.info(
                    f"Found action in action_result: {action_result.action}")
                # Try to convert the action_result to a proper action
                if action_result.action == "play_card_tool" and hasattr(action_result, "args"):
                    # Get hand size for validation
                    hand_size = 0
                    if hasattr(game_state, 'hands') and len(game_state.hands) > self.agent_id:
                        hand_size = len(game_state.hands[self.agent_id])

                    # Get the original 1-indexed card position
                    original_index = action_result.args.get(
                        "card_index", 1)  # Default to 1 if missing

                    # Validate and convert to 0-indexed
                    zero_indexed = self.validate_and_convert_index(
                        original_index, hand_size)

                    action = {
                        "type": "play_card",
                        "card_index": zero_indexed
                    }
                elif action_result.action == "give_clue_tool" and hasattr(action_result, "args"):
                    action = {
                        "type": "give_clue",
                        "target_id": action_result.args.get("target_id", 0),
                        "clue": {
                            "type": action_result.args.get("clue_type", "color"),
                            "value": action_result.args.get("clue_value", "red")
                        }
                    }
                elif action_result.action == "discard_tool" and hasattr(action_result, "args"):
                    # Get hand size for validation
                    hand_size = 0
                    if hasattr(game_state, 'hands') and len(game_state.hands) > self.agent_id:
                        hand_size = len(game_state.hands[self.agent_id])

                    # Get the original 1-indexed card position
                    original_index = action_result.args.get(
                        "card_index", 1)  # Default to 1 if missing

                    # Validate and convert to 0-indexed
                    zero_indexed = self.validate_and_convert_index(
                        original_index, hand_size)

                    action = {
                        "type": "discard",
                        "card_index": zero_indexed
                    }

        # If we still don't have an action, use fallback
        if not action:
            logger.warning(f"No action found in result, using fallback action")
            return self._fallback_action(game_state)

        # Get hand size for final validation
        hand_size = 0
        if hasattr(game_state, 'hands') and len(game_state.hands) > self.agent_id:
            hand_size = len(game_state.hands[self.agent_id])

        # One last check - make sure any card_index is correctly 0-indexed
        if action.get("type") in ["play_card", "discard"] and "card_index" in action:
            # Only convert if it seems to be in 1-indexed format (greater than 0)
            if action["card_index"] > 0:
                original_index = action["card_index"]
                action["card_index"] = self.validate_and_convert_index(
                    original_index, hand_size)

        logger.info(f"Extracted action: {action}")

        # Extract the thoughts from the result
        thoughts = result.get("current_thoughts", [])

        # Store the thoughts in the agent's memory
        self.agent_memory.thoughts = thoughts
        # Also store in the memory store for immediate access
        self.store_memory("current_thoughts", thoughts)

        # Store the action in the memory store for immediate access
        self.store_memory("proposed_action", action)

        return action

    def notify_action_result(self, action: Dict[str, Any], result: Union[Dict[str, Any], bool]) -> None:
        """
        Notify the agent of the result of an action.

        Args:
            action: The action that was executed
            result: The result of the action (can be a dictionary or a boolean)
        """
        logger.info(
            f"Agent {self.agent_id} notified of action result: {result}")

        # Convert boolean result to dictionary if needed
        result_dict = result if isinstance(result, dict) else {
            "success": result}

        # Store the action result in structured memory
        self.agent_memory.add_action_result(action, result_dict)

        # Store the action and result in the memory store for immediate access
        self.store_memory("last_action", action)
        self.store_memory("last_result", result_dict)

        # Store the action in a standardized format for history tracking
        standardized_action = {
            "type": action.get("type", "unknown"),
            "timestamp": datetime.datetime.now().isoformat(),
            "turn": self.current_game_state.turn_count if self.current_game_state else 0,
        }

        # Add specific details based on action type
        if action.get("type") == "play_card":
            standardized_action["card_index"] = action.get("card_index")
        elif action.get("type") == "give_clue":
            standardized_action["target_id"] = action.get("target_id")
            standardized_action["clue"] = action.get("clue", {})
        elif action.get("type") == "discard":
            standardized_action["card_index"] = action.get("card_index")

        # Store the standardized action
        self.agent_memory.store_memory(
            "standardized_actions", standardized_action)

        # Log any errors for learning
        if isinstance(result, dict) and not result.get("success", True):
            error_message = result.get("error", "Unknown error")
            logger.warning(f"Action failed: {error_message}")

            # Determine specific error reason based on action type and error message
            error_reason = "unknown_error"
            action_type = action.get("type", "unknown")

            if action_type == "give_clue":
                if "no clue tokens" in error_message.lower():
                    error_reason = "no_clue_tokens"
                elif "would not affect any cards" in error_message.lower() or "doesn't apply to any cards" in error_message.lower():
                    error_reason = "no_affected_cards"
                elif "cannot give clue to yourself" in error_message.lower():
                    error_reason = "self_clue"
                elif "invalid target" in error_message.lower():
                    error_reason = "invalid_target"
            elif action_type == "play_card":
                if "invalid card index" in error_message.lower():
                    error_reason = "invalid_card_index"
            elif action_type == "discard":
                if "maximum clue tokens" in error_message.lower():
                    error_reason = "max_clue_tokens"
                elif "invalid card index" in error_message.lower():
                    error_reason = "invalid_card_index"

            # Store error in structured memory
            self.agent_memory.add_action_error(
                action=action,
                error=error_message,
                error_reason=error_reason,
                turn=self.current_game_state.turn_count if self.current_game_state else 0
            )

        elif isinstance(result, bool) and not result:
            logger.warning(f"Action failed: Unknown error")

            # Store error in structured memory
            self.agent_memory.add_action_error(
                action=action,
                error="Unknown error",
                error_reason="unknown_error",
                turn=self.current_game_state.turn_count if self.current_game_state else 0
            )

        # Save the current state to memory store
        try:
            # Use store_memory instead of direct assignment
            self.store_memory(f"agent_{self.agent_id}",
                              self.agent_memory.dict())
        except Exception as e:
            logger.error(f"Error storing in memory store: {e}")
            # Continue execution even if memory storage fails

    def get_memory_from_store(self, key, default=None):
        """
        Get memory from the memory store using LangGraph's recommended approach.

        Args:
            key: The key to retrieve
            default: Default value if key not found

        Returns:
            The stored value or default
        """
        # First try to get from agent memory
        if hasattr(self, 'agent_memory'):
            value = self.agent_memory.get_memory(key, None)
            if value is not None:
                return value

        # Create a unique key for this agent
        memory_key = f"agent_{self.agent_id}_{key}"

        # Use LangGraph's get method to retrieve from memory store
        try:
            # In LangGraph 0.3.5+, InMemoryStore may not have a get method
            # Just log and return default
            logger.debug(f"Would retrieve memory with key: {memory_key}")
            # value = self.memory_store.get(memory_key)  # This may not work in LangGraph 0.3.5+
            # return value if value is not None else default
            return default
        except Exception as e:
            logger.error(f"Error retrieving from memory store: {e}")
            return default

    def store_memory(self, key, value):
        """
        Store a value in the memory store using LangGraph's recommended approach.

        Args:
            key: Key to store the value under
            value: Value to store
        """
        # Create a unique key for this agent
        memory_key = f"agent_{self.agent_id}_{key}"

        # Use LangGraph's set method to store in memory store
        try:
            # For LangGraph 0.3.5+, InMemoryStore doesn't have a set method
            # Just log the error and continue with agent memory
            logger.debug(f"Storing memory with key: {memory_key}")
            # self.memory_store.set(memory_key, value)  # This doesn't work in LangGraph 0.3.5+
        except Exception as e:
            logger.error(f"Error storing in memory store: {e}")

        # Store in agent memory's custom_memory for persistence
        if hasattr(self, 'agent_memory'):
            self.agent_memory.store_memory(key, value)

        # Create a checkpoint with the memory saver
        try:
            # Create a config dict for the checkpoint
            config_dict = {
                "agent_id": self.agent_id,
                "key": key,
                "timestamp": datetime.datetime.now().isoformat()
            }

            # In LangGraph 0.3.5+, MemorySaver doesn't have a save method
            # Just log and continue
            logger.debug(f"Would create checkpoint with config: {config_dict}")
            # self.checkpointer.save(config_dict)  # This doesn't work in LangGraph 0.3.5+
        except Exception as e:
            logger.error(f"Error creating checkpoint: {e}")

    def load_memory_from_checkpoint(self, config_filter=None):
        """
        Load memory from a checkpoint using LangGraph's recommended approach.

        Args:
            config_filter: Optional filter for checkpoint config

        Returns:
            True if memory was loaded, False otherwise
        """
        try:
            if config_filter is None:
                config_filter = {"agent_id": self.agent_id}

            # Get the latest checkpoint for this agent
            checkpoint = self.checkpointer.get_latest(config_filter)

            if checkpoint:
                # Extract the state and config
                state_dict, config = checkpoint

                # Load the state into agent memory
                self.agent_memory = AgentMemory.from_dict(state_dict)

                logger.info(
                    f"Agent {self.agent_id} loaded memory from checkpoint (timestamp: {config.get('timestamp')})")
                return True

            # If no checkpoint found, try to get from memory store
            memory_key = f"agent_{self.agent_id}"
            try:
                memory_data = self.memory_store.get(memory_key)
                if memory_data:
                    self.agent_memory = AgentMemory.from_dict(memory_data)
                    logger.info(
                        f"Agent {self.agent_id} loaded memory from memory store")
                    return True
            except Exception as e:
                logger.error(f"Error loading from memory store: {e}")

            logger.warning(
                f"No checkpoint or memory store data found for agent {self.agent_id}")
            return False
        except Exception as e:
            logger.error(
                f"Error loading memory for agent {self.agent_id}: {e}")
            return False

    def store_tool_calls(self, tool_calls, state):
        """
        Enhanced method to store tool calls with reliable persistence.
        Converts from 1-indexed (LLM) to 0-indexed (Python) for card positions.

        Args:
            tool_calls: The tool calls to store
            state: The current state

        Returns:
            The updated state with standardized tool calls
        """
        if not tool_calls:
            return state

        # Make a copy of the state to avoid modifying the original
        new_state = state.copy() if state else {}

        # Store the raw tool calls for debugging purposes
        self.store_memory("raw_tool_calls", tool_calls)
        new_state["proposed_tool_calls"] = tool_calls

        # Get hand size for validation if available
        hand_size = 0
        if self.current_game_state and hasattr(self.current_game_state, 'hands') and len(self.current_game_state.hands) > self.agent_id:
            hand_size = len(self.current_game_state.hands[self.agent_id])

        # Create standardized actions from tool calls
        standardized_actions = []

        for tool_call in tool_calls:
            # Verify and validate the tool call
            validated_tool = self.verify_tool_call(tool_call)
            if not validated_tool:
                logger.warning(f"Skipping invalid tool call: {tool_call}")
                continue

            # Extract the verified tool information
            tool_name = validated_tool["name"]
            tool_args = validated_tool["args"]

            # Convert to standard format based on tool type
            if tool_name == "play_card_tool":
                # Get the original 1-indexed card position
                original_index = tool_args.get(
                    "card_index", 1)  # Default to 1 if missing

                # Validate and convert to 0-indexed
                zero_indexed = self.validate_and_convert_index(
                    original_index, hand_size)

                std_action = {
                    "type": "play_card",
                    "card_index": zero_indexed
                }
                # Store specialized version for direct access
                self.store_memory("play_card_action", std_action)

            elif tool_name == "give_clue_tool":
                std_action = {
                    "type": "give_clue",
                    "target_id": tool_args.get("target_id", 0),
                    "clue": {
                        "type": tool_args.get("clue_type", "color"),
                        "value": tool_args.get("clue_value", "red")
                    }
                }
                self.store_memory("give_clue_action", std_action)

            elif tool_name == "discard_tool":
                # Get the original 1-indexed card position
                original_index = tool_args.get(
                    "card_index", 1)  # Default to 1 if missing

                # Validate and convert to 0-indexed
                zero_indexed = self.validate_and_convert_index(
                    original_index, hand_size)

                std_action = {
                    "type": "discard",
                    "card_index": zero_indexed
                }
                self.store_memory("discard_action", std_action)

            else:
                logger.warning(f"Unknown tool name: {tool_name}")
                continue

            standardized_actions.append(std_action)

        # Store all standardized actions
        if standardized_actions:
            self.store_memory("standardized_actions", standardized_actions)
            new_state["standardized_actions"] = standardized_actions
            # Store the first action as the primary one
            self.store_memory("primary_action", standardized_actions[0])
            new_state["primary_action"] = standardized_actions[0]

        logger.info(f"Standardized actions: {standardized_actions}")
        return new_state

    def execute_tool_directly(self, tool_call, game_state):
        """
        Execute a tool call directly and return the result without going through the reasoning graph.
        Useful for direct tool execution from the thought phase.

        Args:
            tool_call: The tool call to execute
            game_state: The current game state

        Returns:
            A dictionary containing the action to take
        """
        validated_tool = self.verify_tool_call(tool_call)
        if not validated_tool:
            logger.warning("Invalid tool call, using fallback action")
            return self._fallback_action(game_state)

        tool_name = validated_tool["name"]
        tool_args = validated_tool["args"]

        logger.info(
            f"Executing tool directly: {tool_name} with args {tool_args}")

        # Get hand size for validation
        hand_size = 0
        if hasattr(game_state, 'hands') and len(game_state.hands) > self.agent_id:
            hand_size = len(game_state.hands[self.agent_id])

        if tool_name == "play_card_tool":
            # Get the original 1-indexed card position
            original_index = tool_args.get(
                "card_index", 1)  # Default to 1 if missing

            # Validate and convert to 0-indexed
            zero_indexed = self.validate_and_convert_index(
                original_index, hand_size)

            return {
                "type": "play_card",
                "card_index": zero_indexed
            }
        elif tool_name == "give_clue_tool":
            return {
                "type": "give_clue",
                "target_id": tool_args.get("target_id", 0),
                "clue": {
                    "type": tool_args.get("clue_type", "color"),
                    "value": tool_args.get("clue_value", "red")
                }
            }
        elif tool_name == "discard_tool":
            # Get the original 1-indexed card position
            original_index = tool_args.get(
                "card_index", 1)  # Default to 1 if missing

            # Validate and convert to 0-indexed
            zero_indexed = self.validate_and_convert_index(
                original_index, hand_size)

            return {
                "type": "discard",
                "card_index": zero_indexed
            }
        else:
            logger.warning(
                f"Unknown tool name: {tool_name}, using fallback action")
            return self._fallback_action(game_state)

    def take_turn(self, game_state, use_unified_approach=False):
        """
        Take a turn in the game, using either a two-phase or unified approach.

        Args:
            game_state: The current state of the game
            use_unified_approach: If True, use the unified think_and_act method

        Returns:
            The action to take

        Note:
            This method handles the conversion between 1-indexed card positions (used by the LLM)
            and 0-indexed positions (used by the Python implementation). The conversion happens
            at the final execution stage, not during the thought or proposal phase.
        """
        logger.info(
            f"Agent {self.agent_id} taking turn with unified_approach={use_unified_approach}")

        # Store the current game state for use in subsequent methods
        self.current_game_state = game_state

        # Log the agent's hand for debugging card indexing issues
        if hasattr(game_state, 'hands') and len(game_state.hands) > self.agent_id:
            hand = game_state.hands[self.agent_id]
            hand_str = ", ".join(
                [f"{i}: '{card}'" for i, card in enumerate(hand)])
            logger.info(
                f"Agent {self.agent_id}'s hand (0-indexed): [{hand_str}]")

            # Also log knowledge if available
            if hasattr(game_state, 'card_knowledge') and len(game_state.card_knowledge) > self.agent_id:
                knowledge = game_state.card_knowledge[self.agent_id]
                knowledge_str = ", ".join(
                    [f"{i}: '{k}'" for i, k in enumerate(knowledge)])
                logger.info(
                    f"Agent {self.agent_id}'s knowledge (0-indexed): [{knowledge_str}]")

        if use_unified_approach:
            # Use the unified approach (single phase)
            return self.think_and_act(game_state)
        else:
            # Use the original two-phase approach
            discussion = self.participate_in_discussion(game_state, [])
            return self.decide_action(game_state, discussion)

    def _fallback_action(self, game_state):
        """
        Provide a fallback action when the reasoning graph fails.

        Args:
            game_state: The current state of the game

        Returns:
            A fallback action
        """
        # Check if we can give a clue
        if game_state.clue_tokens > 0:
            # Get the next player (wrap around if needed)
            next_player = (self.agent_id + 1) % len(game_state.hands)
            return {
                "type": "give_clue",
                "target_id": next_player,
                "clue": {
                    "type": "number",
                    "value": "1"
                }
            }
        # Otherwise, discard the first card
        return {
            "type": "discard",
            "card_index": 0
        }

    def propose_action(
        self,
        game_state: Dict[str, Any],
        card_knowledge: Dict[str, Any],
        current_thoughts: List[str],
        discussion_history: List[Dict[str, Any]],
        game_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Propose an action using structured output."""
        from .prompts.action_proposal import create_action_proposal_prompt

        # Create the prompt
        prompt = create_action_proposal_prompt(
            game_state=game_state,
            agent_id=self.agent_id,
            card_knowledge=card_knowledge,
            current_thoughts=current_thoughts,
            discussion_history=discussion_history,
            game_history=game_history
        )

        # Add instruction about 1-indexed card positions
        prompt += """

CRITICAL INDEXING INSTRUCTION:
When referring to card positions, you MUST use 1-indexed positions where:
- The first card (leftmost) is position 1
- The second card is position 2
- The third card is position 3
- And so on...

For example, if your hand has 5 cards and you want to play the second card from the left, 
use card_index: 2, NOT card_index: 1.

The system will convert your 1-indexed positions to 0-indexed positions when executing your action.
DO NOT make this conversion yourself.
"""

        # Create a human message with the prompt
        message = HumanMessage(content=prompt)

        try:
            # Use the action model for structured output
            response = self.action_model.invoke([message])

            # Parse the response into an ActionProposal
            action_proposal = ActionProposal.parse_raw(response.content)
            action = action_proposal.action.dict()

            # Log the original 1-indexed action for clarity
            if action.get("type") in ["play_card", "discard"] and "card_index" in action:
                logger.info(
                    f"LLM proposed {action.get('type')} with 1-indexed card_index: {action['card_index']}")

            # DO NOT convert here - we'll convert at the final execution stage
            # Keep the 1-indexed values as they are

            # Return the action and explanation
            return {
                "action": action,
                "explanation": action_proposal.explanation
            }

        except Exception as e:
            logger.error(f"Error in propose_action: {e}")
            # Fallback to default action
            return {
                "action": {
                    "type": "discard",
                    "card_index": 0  # 0-indexed since this is our internal fallback
                },
                "explanation": "Error occurred, falling back to default action"
            }

    def think_and_act(self, game_state: GameState) -> Dict[str, Any]:
        """
        Generate thoughts about the game state and immediately execute an action.

        Args:
            game_state: Current state of the game (filtered for this agent)

        Returns:
            A dictionary containing the action to take
        """
        logger.info(
            f"Agent {self.agent_id} thinking and acting in a unified flow")

        # Store the current game state for tool access
        self.current_game_state = game_state

        # Create initial state for thinking
        state = create_initial_state(
            game_state=game_state,
            agent_id=self.agent_id,
            discussion_history=[]  # Empty or retrieve from memory
        )

        # Explicitly set agent_id in the state
        state["agent_id"] = self.agent_id

        # Create unique identifiers for this conversation
        conversation_id = f"agent_{self.agent_id}_think_act_{game_state.turn_count}"
        thread_id = f"thread_{self.agent_id}_{game_state.turn_count}_{datetime.datetime.now().timestamp()}"

        # Run reasoning graph for thought generation
        thought_result = self.reasoning_graph.invoke(
            state,
            config={
                "agent_id": self.agent_id,
                "agent_instance": self,
                "model": self.thought_model,
                "configurable": {
                    "conversation_id": conversation_id,
                    "thread_id": thread_id
                }
            }
        )

        # Extract and store thoughts
        thoughts = thought_result.get("current_thoughts", [])
        self.agent_memory.thoughts = thoughts
        self.store_memory("current_thoughts", thoughts)

        # Log the current thoughts for debugging
        logger.info(f"Current thoughts: {thoughts}")

        # Check if we have tool calls from the thought phase
        tool_calls = None
        messages = thought_result.get("messages", [])
        for message in reversed(messages):
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_calls = message.tool_calls
                logger.info(f"Found tool calls in thought phase: {tool_calls}")
                break

        # If we have tool calls, execute the first one directly
        if tool_calls and len(tool_calls) > 0:
            logger.info("Executing tool call directly from thought phase")
            action = self.execute_tool_directly(tool_calls[0], game_state)
            # Store the action in agent memory using store_memory
            self.store_memory("proposed_action", action)
            return action

        # If no tool calls, use propose_action to get a structured action
        logger.info(
            "No tool calls from thought phase, using structured action proposal")
        action_proposal = self.propose_action(
            game_state=game_state.dict() if hasattr(game_state, "dict") else game_state,
            card_knowledge=getattr(game_state, "card_knowledge", {}),
            current_thoughts=thoughts,
            discussion_history=[],
            game_history=[]
        )

        # Extract the action from the proposal (will be in 1-indexed format)
        action = action_proposal.get("action", {})

        # Log the proposed action
        logger.info(f"Successfully proposed action: {action}")

        # Get hand size for validation
        hand_size = 0
        if hasattr(game_state, 'hands') and len(game_state.hands) > self.agent_id:
            hand_size = len(game_state.hands[self.agent_id])

        # Validate and convert from 1-indexed to 0-indexed for card positions
        if action and action.get("type") in ["play_card", "discard"] and "card_index" in action:
            original_index = action["card_index"]
            action["card_index"] = self.validate_and_convert_index(
                original_index, hand_size)

        # Store the converted action in memory
        self.store_memory("proposed_action", action)

        # Return the action, or fallback if none found
        if action:
            logger.info(f"Returning action from proposal: {action}")
            return action
        else:
            logger.warning("No action from proposal, using fallback")
            fallback_action = self._fallback_action(game_state)
            self.store_memory("proposed_action", fallback_action)
            return fallback_action

    def to_zero_indexed(self, one_indexed_value):
        """
        Convert a 1-indexed value to a 0-indexed value.

        Args:
            one_indexed_value: A value using 1-based indexing

        Returns:
            The equivalent 0-indexed value
        """
        try:
            # Check if the value is a string or number that can be converted to int
            if isinstance(one_indexed_value, (int, str)) and str(one_indexed_value).isdigit():
                zero_indexed = int(one_indexed_value) - 1
                # Ensure we don't return negative values
                return max(0, zero_indexed)
            return one_indexed_value  # Return unchanged if not convertible
        except Exception as e:
            logger.error(f"Error converting to zero-indexed: {e}")
            return one_indexed_value  # Return unchanged on error

    def to_one_indexed(self, zero_indexed_value):
        """
        Convert a 0-indexed value to a 1-indexed value.

        Args:
            zero_indexed_value: A value using 0-based indexing

        Returns:
            The equivalent 1-indexed value
        """
        try:
            # Check if the value is a string or number that can be converted to int
            if isinstance(zero_indexed_value, (int, str)) and str(zero_indexed_value).isdigit():
                return int(zero_indexed_value) + 1
            return zero_indexed_value  # Return unchanged if not convertible
        except Exception as e:
            logger.error(f"Error converting to one-indexed: {e}")
            return zero_indexed_value  # Return unchanged on error

    def validate_and_convert_index(self, original_index, hand_size=None):
        """
        Validate a 1-indexed card position and convert it to 0-indexed.

        Args:
            original_index: The 1-indexed card position from the LLM
            hand_size: Optional hand size for validation

        Returns:
            The validated and converted 0-indexed card position
        """
        # Default to 1 if missing or invalid type
        if not isinstance(original_index, int) or original_index < 1:
            logger.warning(
                f"Invalid 1-indexed card_index: {original_index}. Must be at least 1. Setting to 1.")
            original_index = 1

        # Validate against hand size if provided
        if hand_size is not None and hand_size > 0 and original_index > hand_size:
            logger.warning(
                f"Invalid 1-indexed card_index: {original_index}. Exceeds hand size of {hand_size}. Setting to {hand_size}.")
            original_index = hand_size

        # Convert to 0-indexed
        zero_indexed = self.to_zero_indexed(original_index)
        logger.info(
            f"Converting from 1-indexed {original_index} to 0-indexed {zero_indexed}")

        return zero_indexed
