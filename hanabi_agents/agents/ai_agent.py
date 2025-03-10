from typing import Dict, Any, List, Optional, Union
import os
import logging
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore
from .base import Agent
from ..game.state import GameState
from .state.state_factory import create_initial_state, create_action_state
from .reasoning.graph import setup_reasoning_graph
from .tools import play_card_tool, give_clue_tool, discard_tool
from .reasoning.nodes import _normalize_tool_name
import datetime
from .state.agent_state import AgentMemory, ActionError, ActionResult

# Set up logging
logger = logging.getLogger(__name__)


class AIAgent(Agent):
    """
    AI agent implementation using LangGraph for reasoning.

    This agent uses a language model and a reasoning graph to analyze the game state,
    generate strategic thoughts, and propose actions.
    """

    def __init__(self, agent_id: int, name: str = None, model_name: str = None):
        """
        Initialize a new AI agent.

        Args:
            agent_id: Unique identifier for this agent
            name: Optional name for this agent
            model_name: Name of the language model to use
        """
        super().__init__(agent_id, name)

        # Initialize the model
        self.model = self._initialize_model(model_name)

        # Initialize memory storage as a simple dictionary
        self.memory_store = {}

        # Initialize the agent memory
        self.agent_memory = AgentMemory()

        # Initialize the reasoning graph
        self.reasoning_graph = setup_reasoning_graph(self)

        # Store current game state for tool access
        self.current_game_state = None

    def _initialize_model(self, model_name: str = None) -> ChatOpenAI:
        """
        Initialize the language model.

        Args:
            model_name: Name of the model to use

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
            model_name = os.getenv("OPENAI_MODEL_NAME", "o3-mini")

        # Initialize the model and bind tools
        model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            verbose=False  # Disable verbose mode to prevent stdout logging
        )

        # Bind tools to the model with tool_choice="required" to force tool calls
        return model.bind_tools([
            play_card_tool,
            give_clue_tool,
            discard_tool
        ], tool_choice="required")

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

        # Run the reasoning graph
        result = self.reasoning_graph.invoke(
            state,
            config={"agent_id": self.agent_id, "agent_instance": self}
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
        Execute the tool call (action) suggested during the analysis phase.

        Args:
            game_state: Current state of the game (filtered for this agent)
            discussion_summary: Not used in this implementation

        Returns:
            A dictionary representing the chosen action
        """
        logger.info(f"Agent {self.agent_id} executing suggested action")

        # Store the current game state for tool access
        self.current_game_state = game_state

        # Get the tool calls from memory (set during participate_in_discussion)
        tool_calls = self.get_memory_from_store("tool_calls")

        # Also try to get from agent memory if not found in memory store
        if not tool_calls:
            tool_calls = self.agent_memory.get_memory("proposed_tool_calls")
            if tool_calls:
                logger.info(
                    f"Retrieved tool calls from agent memory: {tool_calls}")
                # Store back in memory store for consistency
                self.store_memory("tool_calls", tool_calls)

        # If no tool calls were stored during discussion, run the reasoning graph again
        if not tool_calls:
            logger.warning(
                f"No tool calls found in memory, running reasoning graph again")

            # Create the action state with agent memory to include proposed tool calls
            state = create_action_state(
                game_state=game_state,
                agent_id=self.agent_id,
                discussion_summary="",
                agent_memory=self.agent_memory
            )

            # Create a model with explicit tool_choice="required" for this run
            forced_model = self.model.bind(tool_choice="required")

            # Run the reasoning graph with the forced model
            result = self.reasoning_graph.invoke(
                state,
                config={
                    "agent_id": self.agent_id,
                    "agent_instance": self,
                    "model": forced_model  # Pass the forced model to the graph
                }
            )

            # Extract thoughts and store them in the agent's memory
            thoughts = result.get("current_thoughts", [])
            if thoughts:
                self.agent_memory.thoughts = thoughts
                self.store_memory("current_thoughts", thoughts)

            # Extract tool calls from the result
            messages = result.get("messages", [])
            self.store_memory("messages", messages)

            for message in reversed(messages):
                if hasattr(message, "tool_calls") and message.tool_calls:
                    tool_calls = message.tool_calls
                    # Store the tool calls in memory
                    logger.info(
                        f"Storing tool calls from second reasoning run: {tool_calls}")
                    self.store_memory("tool_calls", tool_calls)
                    # Also store in the agent's memory for persistence
                    self.agent_memory.store_memory(
                        "proposed_tool_calls", tool_calls)
                    break

        # If still no tool calls, create a smart fallback action
        if not tool_calls:
            logger.warning(
                f"Agent {self.agent_id} did not generate any tool calls, creating smart fallback action")

            # Check if we're at max clue tokens
            if game_state.clue_tokens >= game_state.max_clue_tokens:
                # If at max clue tokens, we can't discard - give a clue instead
                logger.info(
                    f"At max clue tokens ({game_state.clue_tokens}), defaulting to give clue")

                # Find a valid target for a clue
                target_id = None
                for player_id in game_state.hands:
                    if player_id != self.agent_id and game_state.hands[player_id]:
                        target_id = player_id
                        break

                if target_id is not None:
                    # Find a valid clue to give
                    target_hand = game_state.hands[target_id]
                    if target_hand:
                        # Try to give a color clue
                        return {
                            "type": "give_clue",
                            "target_id": target_id,
                            "clue": {
                                "type": "color",
                                "value": target_hand[0].color.value
                            }
                        }

                # If we couldn't find a valid clue target, try to play a card
                return {
                    "type": "play_card",
                    "card_index": 0
                }
            else:
                # If not at max clue tokens, we can discard
                return {
                    "type": "discard",
                    "card_index": 0
                }

        # Get the first tool call
        tool_call = tool_calls[0]
        original_tool_name = tool_call.get("name", "")

        # Normalize the tool name to match official names
        tool_name = _normalize_tool_name(original_tool_name)

        # Log if the tool name was normalized
        if tool_name != original_tool_name:
            logger.info(
                f"Normalized tool name from '{original_tool_name}' to '{tool_name}'")

        tool_args = tool_call.get("args", {})

        # Check if the tool call is a discard when at max clue tokens
        if tool_name == "discard_tool" and game_state.clue_tokens >= game_state.max_clue_tokens:
            logger.warning(
                f"Agent {self.agent_id} attempted to discard when at max clue tokens, switching to give clue")

            # Find a valid target for a clue
            target_id = None
            for player_id in game_state.hands:
                if player_id != self.agent_id and game_state.hands[player_id]:
                    target_id = player_id
                    break

            if target_id is not None:
                # Find a valid clue to give
                target_hand = game_state.hands[target_id]
                if target_hand:
                    # Try to give a color clue
                    return {
                        "type": "give_clue",
                        "target_id": target_id,
                        "clue": {
                            "type": "color",
                            "value": target_hand[0].color.value
                        }
                    }

            # If we couldn't find a valid clue target, try to play a card
            return {
                "type": "play_card",
                "card_index": 0
            }

        # Convert the tool call to the format expected by the game engine
        if tool_name == "play_card_tool":
            return {
                "type": "play_card",
                "card_index": tool_args.get("card_index", 0)
            }
        elif tool_name == "give_clue_tool":
            # Ensure we have all the required fields with proper names
            action = {
                "type": "give_clue",
                "target_id": tool_args.get("target_id", 0),
                "clue": {
                    "type": tool_args.get("clue_type", "color"),
                    "value": tool_args.get("clue_value", "red")
                }
            }

            # Log the translation for debugging
            logger.info(f"Translated give_clue_tool to action: {action}")

            return action
        elif tool_name == "discard_tool":
            return {
                "type": "discard",
                "card_index": tool_args.get("card_index", 0)
            }
        else:
            logger.error(f"Unknown tool name: {tool_name}")
            return {
                "type": "discard",
                "card_index": 0
            }

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
        self.memory_store[f"agent_{self.agent_id}"] = self.agent_memory.dict()

    def get_memory_from_store(self, key, default=None):
        """
        Get a value from the memory store.

        Args:
            key: Key to retrieve
            default: Default value to return if key not found

        Returns:
            Value from memory store or default
        """
        # First check direct keys in memory_store
        if key in self.memory_store:
            return self.memory_store[key]

        # Then check in agent memory's custom_memory
        agent_memory_dict = self.memory_store.get(f"agent_{self.agent_id}")
        if agent_memory_dict and isinstance(agent_memory_dict, dict):
            custom_memory = agent_memory_dict.get("custom_memory", {})
            if key in custom_memory:
                return custom_memory[key]

        # If not found in either place, return the default
        return default

    def store_memory(self, key, value):
        """
        Store a value in the memory store.

        Args:
            key: Key to store under
            value: Value to store
        """
        self.memory_store[key] = value

        # Save the current state to memory store after each memory update
        self.memory_store[f"agent_{self.agent_id}"] = self.agent_memory.dict()

    def load_memory_from_checkpoint(self, config_filter=None):
        """
        Load memory from a checkpoint.

        Args:
            config_filter: Optional filter for checkpoint config (not used in this implementation)

        Returns:
            True if memory was loaded, False otherwise
        """
        try:
            # Get the memory for this agent
            memory_data = self.memory_store.get(f"agent_{self.agent_id}")

            if memory_data:
                # Restore agent memory from stored data
                self.agent_memory = AgentMemory.from_dict(memory_data)
                return True
            return False

        except Exception as e:
            logger.error(
                f"Error loading memory for agent {self.agent_id}: {str(e)}")
            return False
