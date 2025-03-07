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
from .discussion.contribution import generate_contribution
from .actions.extractor import extract_action_from_state

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

    def participate_in_discussion(self, game_state: GameState, discussion_history: list) -> str:
        """Participate in the discussion phase by generating a contribution."""
        try:
            # Store the game state for tool access
            self.current_game_state = game_state

            # Process discussion history
            discussion_strings, game_history_strings = process_discussion_history(
                discussion_history, self.memory)

            # Create initial state
            initial_state = create_initial_state(
                game_state, discussion_strings, game_history_strings)

            # Run the reasoning graph
            final_state = self.reasoning_graph.invoke(initial_state)

            # Check if there are proposed tool calls in the final state and store them in memory
            if "proposed_tool_calls" in final_state:
                self.memory["proposed_tool_calls"] = final_state["proposed_tool_calls"]
                logger.info(
                    f"Agent {self.agent_id} stored tool calls in memory from discussion phase: {final_state['proposed_tool_calls']}")

            # Generate contribution based on thoughts
            return generate_contribution(final_state, self.model, self.agent_id)

        except Exception as e:
            logger.error(f"Error in Agent {self.agent_id} discussion: {e}")
            return "I'm having trouble analyzing the game state."

    def decide_action(self, game_state: GameState, discussion_history: list) -> Dict[str, Any]:
        """Decide on an action based on the game state and discussion summary."""
        # Store the game state for tool access
        self.current_game_state = game_state

        try:
            # Process discussion history
            discussion_strings, game_history_strings = process_discussion_history(
                discussion_history, self.memory)

            # Create initial state with messages for action phase
            initial_state = create_action_state(
                game_state, discussion_strings, game_history_strings, self.memory)

            # Run the reasoning graph
            final_state = self.reasoning_graph.invoke(initial_state)

            # Extract action from final state
            return extract_action_from_state(final_state, self.agent_id)

        except Exception as e:
            logger.critical(f"Agent {self.agent_id} failed with error: {e}")
            raise

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

            # Create the prompt
            prompt = self._create_thought_generation_prompt(
                game_state, discussion_history, current_thoughts)

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
