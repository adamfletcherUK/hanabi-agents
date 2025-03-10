from typing import Dict, Any, List
import os
from langgraph.graph import Graph, StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
import logging
from .base import Agent
from ..game.state import GameState, Card, Color
from ..utils.env import load_environment_variables
from typing_extensions import TypedDict
import json
import re
from langchain_core.tools import Tool
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableLambda
from .state.state_factory import create_initial_state, create_action_state
from .utils.discussion import process_discussion_history
from .reasoning.graph import setup_reasoning_graph
from .discussion.contribution import generate_contribution, generate_active_player_contribution, generate_feedback_contribution
from .actions.extractor import extract_action_from_state
import uuid

# Set up logging
logger = logging.getLogger(__name__)


# Define the state schema using TypedDict for Langgraph compatibility
class AgentStateDict(TypedDict, total=False):
    """State schema for the agent reasoning graph."""
    game_state: Any
    discussion_history: List[str]
    game_history: List[str]
    current_thoughts: List[str]
    proposed_action: Dict[str, Any]


# Define the agent state class
class AgentState:
    """State class for the agent reasoning graph."""

    def __init__(self, game_state: GameState, discussion_history: List[str], game_history: List[str] = None):
        self.game_state = game_state
        self.discussion_history = discussion_history
        self.game_history = game_history or []
        self.current_thoughts = []
        self.proposed_action = None

    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Langgraph compatibility."""
        return {
            "game_state": self.game_state,
            "discussion_history": self.discussion_history,
            "game_history": self.game_history,
            "current_thoughts": self.current_thoughts,
            "proposed_action": self.proposed_action
        }

    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> "AgentState":
        """Create AgentState from dictionary."""
        agent_state = cls(
            game_state=state_dict["game_state"],
            discussion_history=state_dict["discussion_history"],
            game_history=state_dict.get("game_history", [])
        )
        agent_state.current_thoughts = state_dict.get("current_thoughts", [])
        agent_state.proposed_action = state_dict.get("proposed_action", None)
        return agent_state


class AIAgent(Agent):
    def __init__(self, agent_id: int, model_name: str = None):
        super().__init__(agent_id)

        # Initialize environment and API key
        self._initialize_environment()

        # Initialize memory store
        self.memory_store = self._initialize_memory_store()

        # Initialize the model
        self.model = self._initialize_model(model_name)

        # Initialize the reasoning graph
        self.reasoning_graph = setup_reasoning_graph(self)

        # Store current game state for tool access
        self.current_game_state = None

    def _initialize_environment(self):
        """Load environment variables and API key."""
        # Ensure environment variables are loaded
        load_environment_variables()

        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in your .env file or environment.")

    def _initialize_memory_store(self):
        """Initialize the memory store for the agent."""
        from langgraph.store.memory import InMemoryStore

        try:
            # Create an in-memory store for the agent
            memory_store = InMemoryStore()
            logger.info(f"Memory store initialized for Agent {self.agent_id}")
            return memory_store
        except Exception as e:
            logger.error(f"Error initializing memory store: {e}")
            return None

    def _initialize_model(self, model_name):
        """Create and configure the LLM model."""
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")

        # Get model name from environment if not provided
        if model_name is None:
            model_name = os.getenv("MODEL_NAME", "o3-mini")
            logger.info(f"Using model from environment: {model_name}")

        # Initialize the model with the API key
        try:
            # Create the base model
            base_model = ChatOpenAI(model=model_name, api_key=api_key)

            # Define the tools
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "play_card",
                        "description": "Play a card from your hand",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "card_index": {
                                    "type": "integer",
                                    "description": "Index of the card to play (0-4)"
                                }
                            },
                            "required": ["card_index"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "give_clue",
                        "description": "Give a clue to another player",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "target_id": {
                                    "type": "integer",
                                    "description": "ID of the player to give a clue to"
                                },
                                "clue_type": {
                                    "type": "string",
                                    "enum": ["color", "number"],
                                    "description": "Type of clue to give"
                                },
                                "clue_value": {
                                    "type": "string",
                                    "description": "Value of the clue (color name or number 1-5)"
                                }
                            },
                            "required": ["target_id", "clue_type", "clue_value"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "discard",
                        "description": "Discard a card from your hand",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "card_index": {
                                    "type": "integer",
                                    "description": "Index of the card to discard (0-4)"
                                }
                            },
                            "required": ["card_index"]
                        }
                    }
                }
            ]

            # Bind the tools to the model
            return base_model.bind(tools=tools)

        except Exception as e:
            logger.error(f"Error initializing AI Agent {self.agent_id}: {e}")
            raise

    def participate_in_discussion(self, game_state: GameState, discussion_history: list, is_active_player: bool = False, active_player_proposal: str = None) -> str:
        """
        Participate in the pre-action discussion phase.

        Args:
            game_state: Current game state
            discussion_history: History of the discussion
            is_active_player: Whether this agent is the active player
            active_player_proposal: The proposal from the active player (only used for feedback)

        Returns:
            Contribution string
        """
        # Store the current game state for tool access
        self.current_game_state = game_state

        # Create initial state for the reasoning graph
        initial_state = create_initial_state(game_state, discussion_history)

        # Make sure we're not in action phase for discussion
        initial_state["is_action_phase"] = False

        # Create a thread-specific config for checkpointing
        config = {
            "configurable": {
                "thread_id": f"agent_{self.agent_id}_discussion",
                "agent_id": self.agent_id
            },
            "agent_instance": self  # Pass the agent instance for memory access
        }

        try:
            # Run the reasoning graph with checkpointing
            final_state = self.reasoning_graph.invoke(initial_state, config)

            # Generate a contribution based on the final state and agent role
            if is_active_player:
                # Active player proposes an action with clear reasoning
                contribution = generate_active_player_contribution(
                    final_state, self.model, self.agent_id)
            else:
                # Feedback agent provides yes/no feedback on the active player's proposal
                if active_player_proposal:
                    contribution = generate_feedback_contribution(
                        final_state, self.model, self.agent_id, active_player_proposal)
                else:
                    # Fallback to regular contribution if no active player proposal is provided
                    contribution = generate_contribution(
                        final_state, self.model, self.agent_id)

            # Store the proposed tool calls for later use
            if "proposed_tool_calls" in final_state:
                # Convert tool calls to a serializable format if needed
                tool_calls = final_state["proposed_tool_calls"]
                if tool_calls and isinstance(tool_calls, list):
                    serializable_tool_calls = []
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            # Make a copy to avoid modifying the original
                            serializable_tool_call = tool_call.copy()
                            # Convert any non-serializable values
                            if "args" in serializable_tool_call and isinstance(serializable_tool_call["args"], dict):
                                serializable_tool_call["args"] = dict(
                                    serializable_tool_call["args"])
                            serializable_tool_calls.append(
                                serializable_tool_call)
                        else:
                            # If it's not a dict, try to convert it to one
                            try:
                                serializable_tool_call = {
                                    "name": getattr(tool_call, "name", "unknown"),
                                    "args": dict(getattr(tool_call, "args", {})),
                                    "id": getattr(tool_call, "id", "unknown")
                                }
                                serializable_tool_calls.append(
                                    serializable_tool_call)
                            except Exception as e:
                                logger.warning(
                                    f"Could not convert tool call to serializable format: {e}")

                    self.update_memory("proposed_tool_calls",
                                       serializable_tool_calls)
                    logger.info(
                        f"Stored serialized tool calls in memory: {serializable_tool_calls}")

            return contribution
        except Exception as e:
            logger.error(
                f"Error in discussion phase for Agent {self.agent_id}: {e}")
            # Generate a simple contribution without using the graph
            if is_active_player:
                return "I propose to give a number clue to help identify playable 1s because this is the safest way to make progress at this stage."
            else:
                return "I agree with the proposal because it helps us identify playable cards safely."

    def decide_action(self, game_state: GameState, discussion_summary: str) -> Dict[str, Any]:
        """Decide on an action based on the game state and discussion summary."""
        # Store the current game state for tool access
        self.current_game_state = game_state

        # Check for recent tool errors and incorporate them into the decision process
        tool_errors = self.get_tool_error_history()
        recent_errors = []
        if tool_errors:
            # Get the most recent errors (up to 3)
            recent_errors = tool_errors[-3:]
            logger.info(
                f"Agent {self.agent_id}: Considering {len(recent_errors)} recent tool errors in decision process")

            # Add error information to the discussion summary if there are recent errors
            if recent_errors:
                error_explanations = [self._generate_error_explanation(
                    error.get('error_reason', 'unknown'),
                    error.get('action', {}).get('type', 'unknown'),
                    error
                ) for error in recent_errors]

                error_summary = "\n\nPrevious action errors to avoid:\n" + "\n".join(
                    f"- {explanation}" for explanation in error_explanations
                )

                discussion_summary += error_summary
                logger.info(
                    f"Agent {self.agent_id}: Added error information to discussion summary")

        # First, check if we have proposed tool calls from the discussion phase
        proposed_tool_calls = self.get_memory_from_store("proposed_tool_calls")
        if proposed_tool_calls:
            logger.info(
                f"Agent {self.agent_id}: Using proposed tool calls from discussion phase: {proposed_tool_calls}")

            # Extract the first tool call
            tool_call = proposed_tool_calls[0]
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})

            # Check if this tool call is similar to a recent error
            should_skip = False
            for error in recent_errors:
                error_action = error.get('action', {})
                error_type = error_action.get('type', '')

                if (tool_name == 'play_card' and error_type == 'play_card' and
                        tool_args.get('card_index') == error_action.get('card_index')):
                    logger.warning(
                        f"Agent {self.agent_id}: Skipping proposed play_card action that recently failed")
                    should_skip = True
                    break

                elif (tool_name == 'give_clue' and error_type == 'give_clue' and
                      tool_args.get('target_id') == error_action.get('target_id') and
                      tool_args.get('clue_type') == error_action.get('clue', {}).get('type') and
                      tool_args.get('clue_value') == error_action.get('clue', {}).get('value')):
                    logger.warning(
                        f"Agent {self.agent_id}: Skipping proposed give_clue action that recently failed")
                    should_skip = True
                    break

                elif (tool_name == 'discard' and error_type == 'discard' and
                      tool_args.get('card_index') == error_action.get('card_index')):
                    logger.warning(
                        f"Agent {self.agent_id}: Skipping proposed discard action that recently failed")
                    should_skip = True
                    break

            # If the proposed action is not similar to a recent error, proceed with it
            if not should_skip:
                # Map tool names to action types
                if tool_name == "play_card":
                    action = {
                        "type": "play_card",
                        "card_index": tool_args.get("card_index", 0)
                    }
                    logger.info(
                        f"Agent {self.agent_id}: Extracted play_card action from discussion: {action}")
                    return action
                elif tool_name == "give_clue":
                    action = {
                        "type": "give_clue",
                        "target_id": tool_args.get("target_id", 0),
                        "clue": {
                            "type": tool_args.get("clue_type", "color"),
                            "value": tool_args.get("clue_value", "")
                        }
                    }
                    logger.info(
                        f"Agent {self.agent_id}: Extracted give_clue action from discussion: {action}")
                    return action
                elif tool_name == "discard":
                    action = {
                        "type": "discard",
                        "card_index": tool_args.get("card_index", 0)
                    }
                    logger.info(
                        f"Agent {self.agent_id}: Extracted discard action from discussion: {action}")
                    return action

        # If no proposed tool calls or extraction failed, fall back to the reasoning graph
        logger.info(
            f"Agent {self.agent_id}: No valid tool calls from discussion, using reasoning graph")

        # Create initial state for the reasoning graph
        initial_state = create_action_state(
            game_state, discussion_summary, self.memory)

        # Explicitly set the action phase flag
        initial_state["is_action_phase"] = True

        # Add tool error information to the initial state
        if recent_errors:
            initial_state["recent_tool_errors"] = recent_errors
            initial_state["tool_error_explanations"] = [
                self._generate_error_explanation(
                    error.get('error_reason', 'unknown'),
                    error.get('action', {}).get('type', 'unknown'),
                    error
                ) for error in recent_errors
            ]

        # Create a thread-specific config for checkpointing
        config = {
            "configurable": {
                "thread_id": f"agent_{self.agent_id}_action",
                "agent_id": self.agent_id
            },
            "agent_instance": self  # Pass the agent instance for memory access
        }

        try:
            # Log the initial state for debugging
            logger.info(
                f"Agent {self.agent_id}: Initial state keys: {list(initial_state.keys())}")

            # Run the reasoning graph with checkpointing
            final_state = self.reasoning_graph.invoke(initial_state, config)

            # Log the final state for debugging
            logger.info(
                f"Agent {self.agent_id}: Final state keys: {list(final_state.keys())}")

            # Extract the action from the final state
            action = extract_action_from_state(final_state, self.agent_id)

            # Check if this action is similar to a recent error
            should_retry = False
            for error in recent_errors:
                error_action = error.get('action', {})
                if self._actions_are_similar(action, error_action):
                    logger.warning(
                        f"Agent {self.agent_id}: Extracted action is similar to a recent error, will retry")
                    should_retry = True
                    break

            # If the action is similar to a recent error, modify it to avoid repeating the error
            if should_retry:
                action = self._modify_action_to_avoid_error(
                    action, recent_errors, game_state)
                logger.info(
                    f"Agent {self.agent_id}: Modified action to avoid repeating error: {action}")

            # Store the action in memory
            self.update_memory("last_action", action)

            return action
        except Exception as e:
            logger.error(f"Error deciding action: {e}")
            # Create a fallback action
            fallback_action = self._create_fallback_action(game_state)
            logger.warning(
                f"Using fallback action due to error: {fallback_action}")
            return fallback_action

    def _actions_are_similar(self, action1: Dict[str, Any], action2: Dict[str, Any]) -> bool:
        """
        Check if two actions are similar enough to be considered the same.

        Args:
            action1: The first action
            action2: The second action

        Returns:
            True if the actions are similar, False otherwise
        """
        # Check if the action types are the same
        if action1.get('type') != action2.get('type'):
            return False

        action_type = action1.get('type')

        # For play_card and discard, check if the card indices are the same
        if action_type in ['play_card', 'discard']:
            return action1.get('card_index') == action2.get('card_index')

        # For give_clue, check if the target and clue are the same
        elif action_type == 'give_clue':
            clue1 = action1.get('clue', {})
            clue2 = action2.get('clue', {})

            return (action1.get('target_id') == action2.get('target_id') and
                    clue1.get('type') == clue2.get('type') and
                    clue1.get('value') == clue2.get('value'))

        # For unknown action types, assume they're not similar
        return False

    def _modify_action_to_avoid_error(self, action: Dict[str, Any], recent_errors: List[Dict[str, Any]], game_state: GameState) -> Dict[str, Any]:
        """
        Modify an action to avoid repeating a recent error.

        Args:
            action: The action to modify
            recent_errors: List of recent errors
            game_state: The current game state

        Returns:
            A modified action that avoids the recent errors
        """
        action_type = action.get('type')

        # For play_card, try a different card
        if action_type == 'play_card':
            card_index = action.get('card_index')
            hand_size = len(game_state.hands[self.agent_id])

            # Try each card index in order
            for i in range(hand_size):
                if i != card_index:
                    # Check if this card index was in a recent error
                    if not any(error.get('action', {}).get('type') == 'play_card' and
                               error.get('action', {}).get('card_index') == i
                               for error in recent_errors):
                        return {'type': 'play_card', 'card_index': i}

            # If all play actions would repeat errors, try discarding instead
            if game_state.clue_tokens < game_state.max_clue_tokens:
                return {'type': 'discard', 'card_index': 0}

        # For give_clue, try a different target or clue type
        elif action_type == 'give_clue':
            # If we have no clue tokens, switch to discard
            if game_state.clue_tokens <= 0:
                return {'type': 'discard', 'card_index': 0}

            target_id = action.get('target_id')
            clue = action.get('clue', {})
            clue_type = clue.get('type')
            clue_value = clue.get('value')

            # Try different targets
            for i in range(len(game_state.hands)):
                if i != self.agent_id and i != target_id:
                    # Try color clues
                    if clue_type != 'color' or clue_value != 'red':
                        # Check if any cards match this clue
                        for card_idx, card in enumerate(game_state.hands[i]):
                            if card.color.value == 'red':
                                return {
                                    'type': 'give_clue',
                                    'target_id': i,
                                    'clue': {'type': 'color', 'value': 'red'}
                                }

                    # Try number clues
                    for num in range(1, 6):
                        if clue_type != 'number' or clue_value != num:
                            # Check if any cards match this clue
                            for card_idx, card in enumerate(game_state.hands[i]):
                                if card.number == num:
                                    return {
                                        'type': 'give_clue',
                                        'target_id': i,
                                        'clue': {'type': 'number', 'value': num}
                                    }

            # If no valid clue found, try discarding
            if game_state.clue_tokens < game_state.max_clue_tokens:
                return {'type': 'discard', 'card_index': 0}

        # For discard, try a different card
        elif action_type == 'discard':
            # If clue tokens are at max, try playing a card instead
            if game_state.clue_tokens >= game_state.max_clue_tokens:
                return {'type': 'play_card', 'card_index': 0}

            card_index = action.get('card_index')
            hand_size = len(game_state.hands[self.agent_id])

            # Try each card index in order
            for i in range(hand_size):
                if i != card_index:
                    # Check if this card index was in a recent error
                    if not any(error.get('action', {}).get('type') == 'discard' and
                               error.get('action', {}).get('card_index') == i
                               for error in recent_errors):
                        return {'type': 'discard', 'card_index': i}

            # If all discard actions would repeat errors, try giving a clue instead
            if game_state.clue_tokens > 0:
                # Find a target that's not ourselves
                for i in range(len(game_state.hands)):
                    if i != self.agent_id:
                        # Try a color clue
                        for color in ['red', 'green', 'blue', 'white', 'yellow']:
                            # Check if any cards match this clue
                            for card_idx, card in enumerate(game_state.hands[i]):
                                if card.color.value == color:
                                    return {
                                        'type': 'give_clue',
                                        'target_id': i,
                                        'clue': {'type': 'color', 'value': color}
                                    }

        # If we couldn't find a good alternative, return the original action
        # (the game engine will handle the error)
        return action

    def get_reasoning_history(self, phase="action", limit=5):
        """Retrieve the agent's reasoning history from checkpoints."""
        thread_id = f"agent_{self.agent_id}_{phase}"
        config = {"configurable": {"thread_id": thread_id}}

        try:
            # Get the state history from the checkpointer
            history = list(self.reasoning_graph.get_state_history(config))

            # Limit the number of entries if needed
            if limit and len(history) > limit:
                history = history[:limit]

            return history
        except Exception as e:
            logger.error(
                f"Error retrieving reasoning history for Agent {self.agent_id}: {e}")
            return []

    def get_memory_from_store(self, key, query=None):
        """Retrieve information from the memory store."""
        try:
            # Check if we have a memory store
            if not hasattr(self, 'memory_store'):
                logger.warning(
                    f"No memory store available for Agent {self.agent_id}")
                return []

            # Use the memory store directly
            store = self.memory_store

            # Define the namespace for this agent
            namespace = (f"agent_{self.agent_id}", key)

            if query:
                # Perform a semantic search if a query is provided
                try:
                    results = store.search(namespace, query=query, limit=5)
                    return [item.value for item in results]
                except AttributeError:
                    # If search is not available, fall back to get method
                    logger.warning(
                        "Search not available in store, falling back to direct memory access")
                    return self.memory.get(key, [])
            else:
                # Try to list all items in the namespace
                try:
                    items = store.list(namespace)
                    return [item.value for item in items]
                except AttributeError:
                    # If list is not available, fall back to direct memory access
                    logger.warning(
                        "List not available in store, falling back to direct memory access")
                    return self.memory.get(key, [])
        except Exception as e:
            logger.error(
                f"Error retrieving from memory store for Agent {self.agent_id}: {e}")
            return []

    def store_memory(self, key, value, index=True):
        """Store information in the memory store."""
        try:
            # Check if we have a memory store
            if not hasattr(self, 'memory_store'):
                logger.warning(
                    f"No memory store available for Agent {self.agent_id}")
                return

            # Use the memory store directly
            store = self.memory_store

            # Define the namespace for this agent
            namespace = (f"agent_{self.agent_id}", key)

            # Generate a unique ID for this memory
            memory_id = str(uuid.uuid4())

            # Store the memory
            try:
                store.put(namespace, memory_id, value, index=index)
            except TypeError:
                # If index parameter is not supported
                store.put(namespace, memory_id, value)

            return True
        except Exception as e:
            logger.error(
                f"Error storing in memory store for Agent {self.agent_id}: {e}")
            return False

    # Tool functions for Hanabi actions
    def _play_card_tool(self, card_index: int) -> Dict[str, Any]:
        """Play a card from your hand."""
        # Store the current game state for reference
        game_state = self.current_game_state

        # Validate the action
        if not isinstance(card_index, int):
            raise ValueError(
                f"Card index must be an integer, got {type(card_index)}")

        if not (0 <= card_index < len(game_state.hands[self.agent_id])):
            raise ValueError(
                f"Invalid card index: {card_index}. Must be between 0 and {len(game_state.hands[self.agent_id])-1}")

        # Format the action for the game engine
        return {
            "type": "play_card",
            "card_index": card_index
        }

    def _give_clue_tool(self, target_id: int, clue_type: str, clue_value: Any) -> Dict[str, Any]:
        """Give a clue to another player."""
        # Store the current game state for reference
        game_state = self.current_game_state

        # Validate the action
        if game_state.clue_tokens <= 0:
            raise ValueError("No clue tokens available")

        if target_id == self.agent_id:
            raise ValueError("Cannot give clue to yourself")

        if target_id not in game_state.hands:
            raise ValueError(f"Invalid target player: {target_id}")

        if clue_type not in ["color", "number"]:
            raise ValueError(f"Invalid clue type: {clue_type}")

        # Validate clue value based on type
        if clue_type == "color":
            if not isinstance(clue_value, str):
                raise ValueError(
                    f"Color value must be a string, got {type(clue_value)}")

            if clue_value not in [c.value for c in Color]:
                raise ValueError(
                    f"Invalid color value: {clue_value}. Must be one of {[c.value for c in Color]}")

        elif clue_type == "number":
            # Convert to int if it's a string
            if isinstance(clue_value, str):
                try:
                    clue_value = int(clue_value)
                except ValueError:
                    raise ValueError(
                        f"Invalid number value: {clue_value}. Must be convertible to an integer.")

            if not isinstance(clue_value, int):
                raise ValueError(
                    f"Number value must be an integer, got {type(clue_value)}")

            if not (1 <= clue_value <= 5):
                raise ValueError(
                    f"Invalid number value: {clue_value}. Must be between 1 and 5.")

        # Check if the clue matches any cards in the target's hand
        target_hand = game_state.hands[target_id]
        matches = False
        for card in target_hand:
            if (clue_type == "color" and card.color.value == clue_value) or \
               (clue_type == "number" and card.number == clue_value):
                matches = True
                break

        if not matches:
            raise ValueError(
                f"No {clue_type} {clue_value} cards in player {target_id}'s hand")

        # Format the action for the game engine
        return {
            "type": "give_clue",
            "target_id": target_id,
            "clue": {
                "type": clue_type,
                "value": clue_value
            }
        }

    def _discard_tool(self, card_index: int) -> Dict[str, Any]:
        """Discard a card from your hand."""
        # Store the current game state for reference
        game_state = self.current_game_state

        # Validate the action
        if game_state.clue_tokens >= 8:
            raise ValueError("Clue tokens already at maximum (8)")

        if not isinstance(card_index, int):
            raise ValueError(
                f"Card index must be an integer, got {type(card_index)}")

        if not (0 <= card_index < len(game_state.hands[self.agent_id])):
            raise ValueError(
                f"Invalid card index: {card_index}. Must be between 0 and {len(game_state.hands[self.agent_id])-1}")

        # Format the action for the game engine
        return {
            "type": "discard",
            "card_index": card_index
        }

    def _handle_tool_error(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle errors that occur during tool execution.

        This method is called when a tool execution fails. It logs the error,
        creates appropriate error messages, and ensures the graph can continue
        execution rather than terminating.

        Args:
            state: The current state of the agent

        Returns:
            Updated state with error information
        """
        error = state.get("error")

        # Log the error
        logger.error(f"Agent {self.agent_id} tool execution error: {error}")

        # Check if we're in the action phase (messages will be present)
        if "messages" not in state or not state["messages"]:
            logger.warning(
                "Tool error occurred outside action phase or with no messages")
            return state

        # Get the last message which should contain tool calls
        last_message = state["messages"][-1]

        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            logger.warning(
                "Tool error occurred but no tool calls found in the last message")
            return state

        # Create error messages for each tool call
        tool_calls = last_message.tool_calls
        error_messages = []

        for tc in tool_calls:
            tool_call_id = tc.get("id") if isinstance(
                tc, dict) and "id" in tc else "unknown"
            tool_name = tc.get("name") if isinstance(
                tc, dict) and "name" in tc else "unknown tool"

            error_message = ToolMessage(
                content=f"Error executing {tool_name}: {repr(error)}. Please try a different approach.",
                tool_call_id=tool_call_id
            )
            error_messages.append(error_message)

        # Return updated state with error messages
        return {
            **state,
            "messages": state["messages"] + error_messages,
            "tool_error": str(error)  # Store the error for reference
        }

    def notify_incorrect_tool_usage(self, error_record: Dict[str, Any]) -> None:
        """
        Enhanced notification of incorrect tool usage for AI agents.

        This method overrides the base Agent implementation to provide more
        sophisticated handling of tool errors for AI agents.

        Args:
            error_record: A dictionary containing information about the error
        """
        # Call the parent implementation to store the error in memory
        super().notify_incorrect_tool_usage(error_record)

        # Log the error
        logger.warning(
            f"AI Agent {self.agent_id} notified of incorrect tool usage: {error_record['error_message']}")

        # Create a human-readable explanation of the error
        error_reason = error_record.get('error_reason', 'unknown')
        action_type = error_record.get('action', {}).get('type', 'unknown')

        explanation = self._generate_error_explanation(
            error_reason, action_type, error_record)

        # Store the explanation in memory for future reference
        self.memory["last_tool_error_explanation"] = explanation

        # If we have a reasoning graph, add the error to the current thoughts
        if hasattr(self, 'current_thoughts'):
            self.current_thoughts.append(f"ERROR: {explanation}")

    def _generate_error_explanation(self, error_reason: str, action_type: str, error_record: Dict[str, Any]) -> str:
        """
        Generate a human-readable explanation of a tool error.

        Args:
            error_reason: The machine-readable error reason code
            action_type: The type of action that caused the error
            error_record: The complete error record

        Returns:
            A human-readable explanation of the error
        """
        # Common error explanations
        explanations = {
            # Clue errors
            "no_clue_tokens": "I cannot give a clue because there are no clue tokens available. I should consider discarding a card to gain a clue token.",
            "self_clue": "I cannot give a clue to myself. I must select another player as the target.",
            "invalid_clue_format": "The clue format was invalid. Clues must be either color (red, green, blue, white, yellow) or number (1-5).",
            "no_affected_cards": "The clue I tried to give wouldn't affect any cards in the target player's hand. Clues must match at least one card.",

            # Play errors
            "invalid_card_index": "The card index I specified is invalid. I must choose a valid index within my hand.",

            # Discard errors
            "max_clue_tokens": "I cannot discard when clue tokens are at maximum. I should use a clue token first.",

            # General errors
            "unknown_action_type": "I attempted an unknown action type. Valid actions are 'play_card', 'give_clue', and 'discard'.",
            "exception": "An unexpected error occurred while executing the action."
        }

        # Get the explanation for the specific error reason, or a generic one if not found
        explanation = explanations.get(
            error_reason, f"An error occurred with the {action_type} action: {error_record['error_message']}")

        # Add action-specific details
        if action_type == "give_clue":
            clue = error_record.get('action', {}).get('clue', {})
            target_id = error_record.get('action', {}).get('target_id')
            explanation += f" (Attempted to give a {clue.get('type')} clue with value {clue.get('value')} to player {target_id})"
        elif action_type == "play_card" or action_type == "discard":
            card_index = error_record.get('action', {}).get('card_index')
            explanation += f" (Attempted to {action_type.replace('_', ' ')} card at index {card_index})"

        return explanation

    def get_tool_error_history(self) -> List[Dict[str, Any]]:
        """
        Get the agent's history of tool errors.

        Returns:
            A list of error records, or an empty list if none exist
        """
        return self.memory.get("tool_errors", [])

    def _setup_reasoning_graph(self):
        """Set up the reasoning graph for the agent."""
        # Create the reasoning graph
        builder = StateGraph(AgentStateDict)

        # Define the tools
        tools = [
            Tool.from_function(
                func=self._play_card_tool,
                name="play_card",
                description="Play a card from your hand",
                args_schema={
                    "type": "object",
                    "properties": {
                        "card_index": {
                            "type": "integer",
                            "description": "Index of the card to play (0-4)"
                        }
                    },
                    "required": ["card_index"]
                }
            ),
            Tool.from_function(
                func=self._give_clue_tool,
                name="give_clue",
                description="Give a clue to another player",
                args_schema={
                    "type": "object",
                    "properties": {
                        "target_id": {
                            "type": "integer",
                            "description": "ID of the player to give a clue to"
                        },
                        "clue_type": {
                            "type": "string",
                            "enum": ["color", "number"],
                            "description": "Type of clue to give"
                        },
                        "clue_value": {
                            "type": "string",
                            "description": "Value of the clue (color name or number 1-5)"
                        }
                    },
                    "required": ["target_id", "clue_type", "clue_value"]
                }
            ),
            Tool.from_function(
                func=self._discard_tool,
                name="discard",
                description="Discard a card from your hand",
                args_schema={
                    "type": "object",
                    "properties": {
                        "card_index": {
                            "type": "integer",
                            "description": "Index of the card to discard (0-4)"
                        }
                    },
                    "required": ["card_index"]
                }
            )
        ]

        # Create a tool node with error handling
        tool_node = ToolNode(tools).with_fallbacks(
            [RunnableLambda(self._handle_tool_error)],
            exception_key="error"
        )

        # Add nodes for each reasoning step
        builder.add_node("analyze_state", self._analyze_game_state)
        builder.add_node("generate_thoughts", self._generate_thoughts)
        builder.add_node("propose_action", self._propose_action)
        builder.add_node("execute_tools", tool_node)

        # Define a router function to determine if tools should be executed
        def should_execute_tools(state: Dict[str, Any]):
            """
            Determine if tools should be executed based on the state.

            Returns:
                str: "execute_tools" if tools should be executed, "end" otherwise
            """
            # Check if we're in the action phase (messages will be present)
            is_action_phase = "messages" in state

            # If we're not in the action phase, we don't execute tools
            if not is_action_phase:
                return "end"

            # Check if the last message has tool calls
            if not state.get("messages"):
                return "end"

            last_message = state["messages"][-1]
            has_tool_calls = hasattr(
                last_message, "tool_calls") and last_message.tool_calls

            # Only execute tools if we have tool calls
            if has_tool_calls:
                return "execute_tools"

            # Otherwise, we're done
            return "end"

        # Connect the nodes with conditional routing
        builder.add_edge("analyze_state", "generate_thoughts")
        builder.add_edge("generate_thoughts", "propose_action")

        # Add conditional edge from propose_action
        builder.add_conditional_edges(
            "propose_action",
            should_execute_tools,
            {
                "execute_tools": "execute_tools",
                "end": END  # Use the special END constant to end the graph
            }
        )

        # Connect execute_tools back to propose_action to handle any follow-up actions
        builder.add_edge("execute_tools", "propose_action")

        # Set the entry point
        builder.set_entry_point("analyze_state")

        # Compile the graph
        self.reasoning_graph = builder.compile()

    def _analyze_game_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the game state to understand the current situation."""
        try:
            # Extract state components
            game_state = state["game_state"]

            # Create the prompt
            prompt = self._create_state_analysis_prompt(game_state)

            # Generate analysis using the LLM
            response = self.model.invoke([HumanMessage(content=prompt)])

            # Process the response
            if response:
                # Add the analysis to current_thoughts
                current_thoughts = state.get("current_thoughts", [])
                current_thoughts.append(response.content)

                # Return updated state
                return {
                    **state,
                    "current_thoughts": current_thoughts
                }

            return state
        except Exception as e:
            logger.error(f"Error analyzing game state: {e}")
            return state

    def _generate_thoughts(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate thoughts about the game state."""
        try:
            # Extract state components
            game_state = state["game_state"]
            discussion_history = state["discussion_history"]
            current_thoughts = state.get("current_thoughts", [])

            # Check for recent tool errors
            recent_tool_errors = state.get("recent_tool_errors", [])
            tool_error_explanations = state.get("tool_error_explanations", [])

            # Create the prompt
            prompt = self._create_thought_generation_prompt(
                game_state, discussion_history, current_thoughts, tool_error_explanations)

            # Generate thoughts using the LLM
            response = self.model.invoke([HumanMessage(content=prompt)])

            # Process the response
            if response:
                # For thought generation, we want natural language, not JSON
                # Clean up the response
                cleaned_response = response.content.strip()

                # Remove any JSON formatting that might have been included
                cleaned_response = cleaned_response.replace(
                    "```json", "").replace("```", "")

                # Add the new thought to the list
                current_thoughts.append(cleaned_response)

            # Return updated state
            return {
                **state,
                "current_thoughts": current_thoughts
            }
        except Exception as e:
            logger.error(f"Error generating thoughts: {e}")
            return state

    def _create_thought_generation_prompt(self, game_state, discussion_history, current_thoughts, tool_error_explanations=None):
        """Create a prompt for generating thoughts about the game state."""
        # Format the game state for the prompt
        game_state_str = self._format_game_state_for_prompt(game_state)

        # Format the discussion history
        discussion_str = "\n".join(
            discussion_history) if discussion_history else "No discussion yet."

        # Format current thoughts
        thoughts_str = "\n".join(
            current_thoughts) if current_thoughts else "No thoughts yet."

        # Format tool error explanations if available
        error_str = ""
        if tool_error_explanations and len(tool_error_explanations) > 0:
            error_str = "\n\nRECENT ACTION ERRORS TO AVOID:\n" + "\n".join(
                f"- {explanation}" for explanation in tool_error_explanations
            )

        # Create the prompt
        prompt = f"""
You are Player {self.agent_id} in a game of Hanabi. Think through the current game state and what action you should take.

GAME STATE:
{game_state_str}

DISCUSSION HISTORY:
{discussion_str}

YOUR PREVIOUS THOUGHTS:
{thoughts_str}{error_str}

Think step by step about the current game state. Consider:
1. What information do you have about your own hand from clues?
2. What information do you have about other players' hands?
3. What cards are in the discard pile and what does that tell you?
4. What is the current state of the firework piles?
5. How many clue tokens and fuse tokens are left?
6. What action would be most beneficial right now?

YOUR NEW THOUGHTS:
"""

        return prompt

    def _propose_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propose an action based on the game state and thoughts.

        This method handles both the discussion phase and the action phase:
        - In discussion phase: Generates thoughts about potential actions
        - In action phase: Generates actual tool calls for execution

        Args:
            state: The current state of the agent

        Returns:
            Updated state with proposed action or thoughts
        """
        try:
            # Extract state components
            game_state = state["game_state"]
            discussion_history = state["discussion_history"]
            current_thoughts = state.get("current_thoughts", [])

            # Store the game state for tool access
            self.current_game_state = game_state

            # Determine if we're in the action phase (messages will be present)
            is_action_phase = "messages" in state

            # Create the prompt based on the phase
            prompt = self._create_action_proposal_prompt(
                game_state, discussion_history, current_thoughts)

            # Generate response using the LLM with tools
            response = self.model.invoke([HumanMessage(content=prompt)])

            # Log if tool calls were generated
            if hasattr(response, "tool_calls") and response.tool_calls:
                logger.info(
                    f"Agent {self.agent_id} generated tool calls: {response.tool_calls}")

                # If in discussion phase but tool calls were generated, extract the content
                if not is_action_phase:
                    content = response.content if response.content else "I'm considering my options."
                    logger.info(
                        f"Tool calls generated during discussion phase, using content: {content}")
                    current_thoughts.append(f"I'm considering: {content}")

                    return {
                        **state,
                        "current_thoughts": current_thoughts
                    }

            # Process the response based on the phase
            if is_action_phase:
                # In action phase, add the response to messages for potential tool execution
                messages = state.get("messages", [])
                messages.append(response)

                return {
                    **state,
                    "messages": messages
                }
            else:
                # In discussion phase, add the content to thoughts
                content = response.content.strip()
                current_thoughts.append(content)

                return {
                    **state,
                    "current_thoughts": current_thoughts
                }

        except Exception as e:
            logger.error(f"Error proposing action: {e}")
            # Add error to thoughts instead of re-raising to avoid terminating the graph
            if "messages" in state:
                # In action phase, add error message
                messages = state.get("messages", [])
                messages.append(
                    AIMessage(content=f"Error proposing action: {e}"))
                return {
                    **state,
                    "messages": messages
                }
            else:
                # In discussion phase, add error to thoughts
                current_thoughts = state.get("current_thoughts", [])
                current_thoughts.append(f"Error proposing action: {e}")
                return {
                    **state,
                    "current_thoughts": current_thoughts
                }

    def _create_fallback_action(self, game_state: GameState) -> Dict[str, Any]:
        """
        Create a fallback action when the normal action selection fails.

        Args:
            game_state: Current game state

        Returns:
            A fallback action
        """
        logger.info(f"Agent {self.agent_id}: Creating fallback action")

        # Check if we have clue tokens available
        if game_state.clue_tokens > 0:
            # Give a clue to the next player about 1s (usually safe)
            next_player_id = (self.agent_id + 1) % len(game_state.hands)
            return {
                "type": "give_clue",
                "target_id": next_player_id,
                "clue": {
                    "type": "number",
                    "value": "1"  # Clue about 1s is usually safe
                }
            }
        else:
            # If no clue tokens, discard the first card
            return {
                "type": "discard",
                "card_index": 0
            }

    def _format_game_state_for_prompt(self, game_state):
        """Format the game state for inclusion in a prompt."""
        # Get the player's view of the game state
        player_view = game_state.get_view_for(self.agent_id)

        # Format firework piles
        firework_str = []
        for color, pile in player_view.firework_piles.items():
            if pile:
                top_card = pile[-1].number
                firework_str.append(f"{color.value}: {top_card}")
            else:
                firework_str.append(f"{color.value}: empty")

        firework_summary = ", ".join(firework_str)

        # Format player hands
        hand_str = []
        for player_id, hand in player_view.hands.items():
            if player_id == self.agent_id:
                # For the player's own hand, show clue information
                cards = []
                for i, card in enumerate(hand):
                    # For the player's own view, we hide the actual card values
                    card_info = f"Card {i}: [HIDDEN]"

                    # Add clue information if available
                    clues = []
                    if hasattr(card, 'color_clued') and card.color_clued:
                        clues.append(f"color: {card.color.value}")
                    if hasattr(card, 'number_clued') and card.number_clued:
                        clues.append(f"number: {card.number}")

                    if clues:
                        card_info += f" ({', '.join(clues)})"

                    cards.append(card_info)

                hand_str.append(
                    f"Your hand (Player {player_id}): {', '.join(cards)}")
            else:
                # For other players' hands, show the actual cards
                cards = [
                    f"Card {i}: {card.color.value} {card.number}" for i, card in enumerate(hand)]
                hand_str.append(
                    f"Player {player_id}'s hand: {', '.join(cards)}")

        hand_summary = "\n".join(hand_str)

        # Format discard pile
        discard_summary = {}
        for card in player_view.discard_pile:
            key = f"{card.color.value} {card.number}"
            discard_summary[key] = discard_summary.get(key, 0) + 1

        discard_str = [f"{card}: {count}" for card,
                       count in discard_summary.items()]
        discard_summary = ", ".join(discard_str) if discard_str else "empty"

        # Format game status
        game_status = f"""
Current player: Player {player_view.current_player}
Turn count: {player_view.turn_count}
Clue tokens: {player_view.clue_tokens}/{player_view.max_clue_tokens}
Fuse tokens: {player_view.fuse_tokens}/{player_view.max_fuse_tokens}
Deck size: {len(player_view.deck)}
Score: {player_view.score}

Firework piles: {firework_summary}

{hand_summary}

Discard pile: {discard_summary}
"""

        return game_status
