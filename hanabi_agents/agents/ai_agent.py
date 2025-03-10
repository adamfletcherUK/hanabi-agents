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
import datetime

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

        # Initialize the checkpointer
        self.checkpointer = MemorySaver()

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

        # Initialize the model without temperature
        return ChatOpenAI(
            model=model_name,
            api_key=api_key
        )

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
            config={"agent_id": self.agent_id}
        )

        # Extract the thoughts and proposed action from the result
        thoughts = result.get("current_thoughts", [])
        proposed_action = result.get("proposed_action", {})

        # Store the proposed action for later use in decide_action
        self.store_memory("proposed_action", proposed_action)

        # Format the thoughts and proposed action into a contribution
        contribution = "## Game State Analysis\n\n"
        for i, thought in enumerate(thoughts):
            contribution += f"{i+1}. {thought}\n"

        contribution += "\n## Suggested Tool Call\n\n"
        if proposed_action:
            action_type = proposed_action.get("action_type", "unknown")
            if action_type == "play":
                contribution += f"I suggest playing card at index {proposed_action.get('card_index', '?')}.\n"
            elif action_type == "clue":
                contribution += f"I suggest giving a {proposed_action.get('clue_type', '?')} clue to Player {proposed_action.get('target_id', '?')} about {proposed_action.get('clue_value', '?')}.\n"
            elif action_type == "discard":
                contribution += f"I suggest discarding card at index {proposed_action.get('card_index', '?')}.\n"
            else:
                contribution += "I'm not sure what action to take.\n"
        else:
            contribution += "I don't have a specific action to suggest at this time.\n"

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

        # Get the proposed action from memory (set during participate_in_discussion)
        proposed_action = self.get_memory_from_store("proposed_action")

        # If no action was proposed during discussion, run the reasoning graph again
        if proposed_action is None:
            logger.warning(
                f"No proposed action found in memory, running reasoning graph again")

            # Create the action state
            state = create_action_state(
                game_state=game_state,
                agent_id=self.agent_id,
                discussion_summary=""
            )

            # Run the reasoning graph
            result = self.reasoning_graph.invoke(
                state,
                config={"agent_id": self.agent_id}
            )

            # Extract the proposed action from the result
            proposed_action = result.get("proposed_action")

        # If still no action was proposed, default to discarding the first card
        if proposed_action is None:
            logger.warning(
                f"Agent {self.agent_id} did not propose an action, defaulting to discard")
            proposed_action = {
                "action_type": "discard",
                "card_index": 0
            }

        # Convert the action to the format expected by the game engine
        action = self._convert_action_format(proposed_action)

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

        # Store the action and result in memory for future reference
        if hasattr(self, 'memory_store'):
            # Convert boolean result to dictionary if needed
            result_dict = result if isinstance(result, dict) else {
                "success": result}

            action_record = {
                "action": action,
                "result": result_dict,
                "timestamp": datetime.datetime.now().isoformat()
            }

            # Store in memory with a unique key
            key = f"action_result_{self.agent_id}_{len(self.get_memory_from_store('action_results', []))}"
            self.store_memory("action_results", action_record)

            # Log any errors for learning
            if isinstance(result, dict) and not result.get("success", True):
                logger.warning(
                    f"Action failed: {result.get('error', 'Unknown error')}")

                # Store error specifically for learning
                error_record = {
                    "action": action,
                    "error": result.get("error", "Unknown error"),
                    "timestamp": datetime.datetime.now().isoformat()
                }
                self.store_memory("action_errors", error_record)
            elif isinstance(result, bool) and not result:
                logger.warning(f"Action failed: Unknown error")

                # Store error specifically for learning
                error_record = {
                    "action": action,
                    "error": "Unknown error",
                    "timestamp": datetime.datetime.now().isoformat()
                }
                self.store_memory("action_errors", error_record)

    def get_memory_from_store(self, key, default=None):
        """
        Get memory from the memory store.

        Args:
            key: The key to retrieve
            default: Default value if key not found

        Returns:
            The stored value or default
        """
        # Simple implementation for now
        if not hasattr(self, '_memory'):
            self._memory = {}
        return self._memory.get(key, default)

    def store_memory(self, key, value):
        """
        Store a value in memory.

        Args:
            key: The key to store under
            value: The value to store
        """
        # Simple implementation for now
        if not hasattr(self, '_memory'):
            self._memory = {}

        # If storing a list item, append to existing list
        if key in self._memory and isinstance(self._memory[key], list):
            self._memory[key].append(value)
        elif key in self._memory and isinstance(value, list):
            self._memory[key] = value
        else:
            # For non-list values, just store directly
            self._memory[key] = value

    def _convert_action_format(self, proposed_action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert the proposed action to the format expected by the game engine.

        Args:
            proposed_action: Action proposed by the reasoning graph

        Returns:
            Action in the format expected by the game engine
        """
        action_type = proposed_action.get("action_type")

        if action_type == "play":
            return {
                "type": "play_card",
                "card_index": proposed_action.get("card_index")
            }
        elif action_type == "clue":
            return {
                "type": "give_clue",
                "target_id": proposed_action.get("target_id"),
                "clue": {
                    "type": proposed_action.get("clue_type"),
                    "value": proposed_action.get("clue_value")
                }
            }
        elif action_type == "discard":
            return {
                "type": "discard",
                "card_index": proposed_action.get("card_index")
            }
        else:
            logger.error(f"Unknown action type: {action_type}")
            return {
                "type": "discard",
                "card_index": 0
            }
