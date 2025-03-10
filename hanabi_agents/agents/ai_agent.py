from typing import Dict, Any, List, Optional
import os
import logging
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from .base import Agent
from ..game.state import GameState
from .state.state_factory import create_initial_state, create_action_state
from .reasoning.graph import setup_reasoning_graph

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
            model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo")

        # Initialize the model
        return ChatOpenAI(
            model=model_name,
            temperature=0.2,
            api_key=api_key
        )

    def participate_in_discussion(self, game_state: GameState, discussion_history: List[Dict[str, Any]]) -> str:
        """
        Participate in the pre-action discussion phase.

        Args:
            game_state: Current state of the game (filtered for this agent)
            discussion_history: History of discussion contributions

        Returns:
            The agent's contribution to the discussion
        """
        logger.info(f"Agent {self.agent_id} participating in discussion")

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
            store=self.memory_store,
            config={"agent_id": self.agent_id}
        )

        # Extract the thoughts from the result
        thoughts = result.get("current_thoughts", [])

        # Format the thoughts into a contribution
        contribution = "Here are my thoughts about the current game state:\n\n"
        for i, thought in enumerate(thoughts):
            contribution += f"{i+1}. {thought}\n"

        return contribution

    def decide_action(self, game_state: GameState, discussion_summary: Optional[str] = None) -> Dict[str, Any]:
        """
        Decide on an action based on the game state and discussion summary.

        Args:
            game_state: Current state of the game (filtered for this agent)
            discussion_summary: Optional summary of the pre-action discussion

        Returns:
            A dictionary representing the chosen action
        """
        logger.info(f"Agent {self.agent_id} deciding action")

        # Store the current game state for tool access
        self.current_game_state = game_state

        # Create the action state
        state = create_action_state(
            game_state=game_state,
            agent_id=self.agent_id,
            discussion_summary=discussion_summary
        )

        # Run the reasoning graph
        result = self.reasoning_graph.invoke(
            state,
            store=self.memory_store,
            config={"agent_id": self.agent_id}
        )

        # Extract the proposed action from the result
        proposed_action = result.get("proposed_action")

        # If no action was proposed, default to discarding the first card
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
                "type": "play",
                "card_index": proposed_action.get("card_index")
            }
        elif action_type == "clue":
            return {
                "type": "clue",
                "target_id": proposed_action.get("target_id"),
                "clue_type": proposed_action.get("clue_type"),
                "clue_value": proposed_action.get("clue_value")
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
