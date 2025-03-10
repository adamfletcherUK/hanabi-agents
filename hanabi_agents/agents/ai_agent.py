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

        # Initialize the memory store
        self.memory_store = InMemoryStore()

        # Initialize the agent memory
        self.agent_memory = AgentMemory()

        # Initialize the checkpointer with the memory store
        self.checkpointer = MemorySaver(self.memory_store)

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
            api_key=api_key
        )

        # Bind tools to the model
        return model.bind_tools([
            play_card_tool,
            give_clue_tool,
            discard_tool
        ])

    def participate_in_discussion(self, game_state: GameState, discussion_history: List[Dict[str, Any]]) -> str:
        """
        Analyze the game state and suggest a tool call (action).

        For the active player, this method analyzes the current game state,
        generates strategic thoughts, and suggests a tool call (action).

        Args:
            game_state: Current state of the game (filtered for this agent)
            discussion_history: History of discussion contributions (empty for our implementation)

        Returns:
            The agent's analysis and suggested tool call
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

        # Find the last message with tool calls
        tool_calls = None
        for message in reversed(messages):
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_calls = message.tool_calls
                break

        # Store the tool calls for later use in decide_action
        if tool_calls:
            self.store_memory("tool_calls", tool_calls)

        # Format the thoughts and tool calls into a contribution
        contribution = "## Game State Analysis\n\n"
        for i, thought in enumerate(thoughts):
            contribution += f"{i+1}. {thought}\n"

        contribution += "\n## Suggested Tool Call\n\n"
        if tool_calls:
            tool_call = tool_calls[0]  # Get the first tool call
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})

            if tool_name == "play_card":
                card_index = tool_args.get("card_index", 0)
                contribution += f"I suggest playing card {card_index}."
            elif tool_name == "give_clue":
                target_id = tool_args.get("target_id", 0)
                clue_type = tool_args.get("clue_type", "unknown")
                clue_value = tool_args.get("clue_value", "unknown")
                contribution += f"I suggest giving a {clue_type} clue about {clue_value} to Player {target_id}."
            elif tool_name == "discard":
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

        # If no tool calls were stored during discussion, run the reasoning graph again
        if not tool_calls:
            logger.warning(
                f"No tool calls found in memory, running reasoning graph again")

            # Create the action state
            state = create_action_state(
                game_state=game_state,
                agent_id=self.agent_id,
                discussion_summary=""
            )

            # Run the reasoning graph
            result = self.reasoning_graph.invoke(
                state,
                config={"agent_id": self.agent_id, "agent_instance": self}
            )

            # Extract tool calls from the result
            messages = result.get("messages", [])
            for message in reversed(messages):
                if hasattr(message, "tool_calls") and message.tool_calls:
                    tool_calls = message.tool_calls
                    break

        # If still no tool calls, default to discarding the first card
        if not tool_calls:
            logger.warning(
                f"Agent {self.agent_id} did not generate any tool calls, defaulting to discard")
            return {
                "type": "discard",
                "card_index": 0
            }

        # Get the first tool call
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})

        # Convert the tool call to the format expected by the game engine
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
                    "value": tool_args.get("clue_value", "red")
                }
            }
        elif tool_name == "discard":
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

        # Save the current state to the checkpoint
        config_dict = {
            "agent_id": self.agent_id,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.checkpointer.save(self.agent_memory.dict(), config_dict)

    def get_memory_from_store(self, key, default=None):
        """
        Get memory from the memory store.

        Args:
            key: The key to retrieve
            default: Default value if key not found

        Returns:
            The stored value or default
        """
        return self.agent_memory.get_memory(key, default)

    def store_memory(self, key, value):
        """
        Store a value in memory.

        Args:
            key: The key to store under
            value: The value to store
        """
        self.agent_memory.store_memory(key, value)

        # Save the current state to the checkpoint after each memory update
        config_dict = {
            "agent_id": self.agent_id,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.checkpointer.save(self.agent_memory.dict(), config_dict)

    def load_memory_from_checkpoint(self, config_filter=None):
        """
        Load memory from a checkpoint.

        Args:
            config_filter: Optional filter for checkpoint config

        Returns:
            True if memory was loaded, False otherwise
        """
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
        else:
            logger.warning(f"No checkpoint found for agent {self.agent_id}")
            return False
