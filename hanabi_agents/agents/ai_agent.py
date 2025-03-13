from typing import Dict, Any, List, Optional, Union
import os
import logging
from langchain_openai import ChatOpenAI
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIAgent(Agent):
    """
    AI agent implementation using LangGraph for reasoning.

    This agent uses a language model and a reasoning graph to analyze the game state,
    generate strategic thoughts, and propose actions.
    """

    def __init__(self, agent_id: int, name: str = None, model_name: str = None, temperature: float = 0.0):
        """
        Initialize the AI agent.

        Args:
            agent_id: The ID of the agent
            name: The name of the agent
            model_name: The name of the model to use
            temperature: The temperature to use for the model
        """
        super().__init__(agent_id, name)

        # Set up logging to suppress detailed debugging
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("langchain").setLevel(logging.WARNING)
        logging.getLogger("langchain_core").setLevel(logging.WARNING)
        logging.getLogger("langgraph").setLevel(logging.WARNING)

        # Initialize the model
        self.model = self._initialize_model(model_name, temperature)

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

    def _initialize_model(self, model_name: str = None, temperature: float = 0.0) -> ChatOpenAI:
        """
        Initialize the language model.

        Args:
            model_name: Name of the model to use
            temperature: The temperature to use for the model

        Returns:
            Initialized language model
        """
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in your .env file or environment.")

        # Use default model if none specified
        if model_name is None:
            model_name = os.getenv("MODEL_NAME", "o3-mini")
            logger.info(f"Using model from environment: {model_name}")

        # Define the tools
        play_card_tool = {
            "type": "function",
            "function": {
                "name": "play_card_tool",
                "description": "Play a card from your hand",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "card_index": {
                            "type": "integer",
                            "description": "Index of the card to play (0-indexed)"
                        }
                    },
                    "required": ["card_index"],
                    "additionalProperties": False
                }
            }
        }

        give_clue_tool = {
            "type": "function",
            "function": {
                "name": "give_clue_tool",
                "description": "Give a clue to another player",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_id": {
                            "type": "integer",
                            "description": "ID of the player to give the clue to"
                        },
                        "clue_type": {
                            "type": "string",
                            "enum": ["color", "number"],
                            "description": "Type of clue to give"
                        },
                        "clue_value": {
                            "type": "string",
                            "description": "Value of the clue (e.g., 'red', '1')"
                        }
                    },
                    "required": ["target_id", "clue_type", "clue_value"],
                    "additionalProperties": False
                }
            }
        }

        discard_tool = {
            "type": "function",
            "function": {
                "name": "discard_tool",
                "description": "Discard a card from your hand",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "card_index": {
                            "type": "integer",
                            "description": "Index of the card to discard (0-indexed)"
                        }
                    },
                    "required": ["card_index"],
                    "additionalProperties": False
                }
            }
        }

        # Initialize the model with the tools
        model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            verbose=True,  # Enable verbose mode for better debugging
            temperature=temperature,
            tools=[play_card_tool, give_clue_tool, discard_tool]
        )

        # Define tool schemas - simplified approach without strict property
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "play_card_tool",
                    "description": "Play a card from the agent's hand",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "card_index": {
                                "type": "integer",
                                "description": "Index of the card to play (0-indexed, must be between 0 and 4)"
                            }
                        },
                        "required": ["card_index"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "give_clue_tool",
                    "description": "Give a clue to another player about their cards",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_id": {
                                "type": "integer",
                                "description": "ID of the player to give the clue to"
                            },
                            "clue_type": {
                                "type": "string",
                                "enum": ["color", "number"],
                                "description": "Type of clue to give"
                            },
                            "clue_value": {
                                "type": "string",
                                "description": "Value of the clue (e.g., 'red', '1')"
                            }
                        },
                        "required": ["target_id", "clue_type", "clue_value"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "discard_tool",
                    "description": "Discard a card from the agent's hand",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "card_index": {
                                "type": "integer",
                                "description": "Index of the card to discard (0-indexed, must be between 0 and 4)"
                            }
                        },
                        "required": ["card_index"],
                        "additionalProperties": False
                    }
                }
            }
        ]

        # Create two model versions - one for thought generation (without forced tools) and one for action selection (with forced tools)
        # No forced tool choice for thought generation
        self.thought_model = model.bind(tools=tools)

        # Always use tool_choice='required' to force the model to call a tool for action selection
        return model.bind(tools=tools, tool_choice="required")

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
                "thought_model": self.thought_model,  # For thought generation
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
                break

        # Store the tool calls for later use in decide_action
        if tool_calls:
            logger.info(f"Storing tool calls in memory: {tool_calls}")
            self.store_memory("tool_calls", tool_calls)
            # Also store in the agent's memory for persistence
            self.agent_memory.store_memory("proposed_tool_calls", tool_calls)
        else:
            logger.warning(
                f"No tool calls found in messages after reasoning graph execution")

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
                "thought_model": self.thought_model,
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
                    action = {
                        "type": "play_card",
                        "card_index": action_result.args.get("card_index", 0)
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
                    action = {
                        "type": "discard",
                        "card_index": action_result.args.get("card_index", 0)
                    }

        if not action:
            logger.warning(f"No action found in result, using fallback action")
            return self._fallback_action(game_state)

        logger.info(f"Extracted action: {action}")

        # Extract the thoughts from the result
        thoughts = result.get("current_thoughts", [])

        # Store the thoughts in the agent's memory
        self.agent_memory.thoughts = thoughts
        # Also store in the memory store for immediate access
        self.store_memory("current_thoughts", thoughts)

        # Store the action in the agent's memory
        self.agent_memory.proposed_action = action
        # Also store in the memory store for immediate access
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
        Store tool calls in the agent's memory for use between phases.

        Args:
            tool_calls: The tool calls to store
            state: The current state

        Returns:
            The updated state with stored tool calls
        """
        if not tool_calls:
            return state

        # Make a copy of the state to avoid modifying the original
        new_state = state.copy()

        # Store the tool calls
        new_state["proposed_tool_calls"] = tool_calls

        # Log the stored tool calls
        logger.info(f"Stored tool calls in state: {tool_calls}")

        return new_state

    def take_turn(self, game_state):
        """
        Take a turn in the game.

        Args:
            game_state: The current state of the game

        Returns:
            The action to take
        """
        logger.info(f"Agent {self.player_id} taking turn")

        # Store the current game state for use in the reasoning graph
        self.current_game_state = game_state

        # Initialize the reasoning graph
        graph = setup_reasoning_graph(self)

        # Initialize the state
        state = {
            "game_state": game_state,
            "player_id": self.player_id,
            "memory": self.memory,
            "model": self.model
        }

        # Execute the reasoning graph
        try:
            logger.info("Executing reasoning graph")
            final_state = graph.invoke(state)

            # Check if we have an action in the final state
            if "action" in final_state:
                action = final_state["action"]
                logger.info(f"Action from reasoning graph: {action}")
                return action

            # If no action in final state, check for proposed tool calls
            elif "proposed_tool_calls" in final_state and final_state["proposed_tool_calls"]:
                # Use execute_action to convert tool calls to action
                action_state = execute_action(final_state)
                if "action" in action_state:
                    logger.info(
                        f"Action from execute_action: {action_state['action']}")
                    return action_state["action"]

            # If we still don't have an action, use a fallback
            logger.warning(
                "No action or tool calls found in final state, using fallback")
            return self._fallback_action(game_state)

        except Exception as e:
            logger.error(f"Error executing reasoning graph: {e}")
            return self._fallback_action(game_state)

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
