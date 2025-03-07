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

        # Ensure environment variables are loaded
        load_environment_variables()

        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in your .env file or environment.")

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
            self.model = base_model.bind(tools=tools)

            # Set up the reasoning graph
            self._setup_reasoning_graph()
            logger.info(
                f"Initialized AI Agent {agent_id} with model {model_name}")
        except Exception as e:
            logger.error(f"Error initializing AI Agent {agent_id}: {e}")
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

    def participate_in_discussion(self, game_state: GameState, discussion_history: list) -> str:
        """Participate in the discussion phase by generating a contribution."""
        try:
            # Store the game state for tool access
            self.current_game_state = game_state

            # Check if discussion_history is a list of DiscussionEntry objects or strings
            has_discussion_objects = False
            has_game_history = False

            if discussion_history and hasattr(discussion_history[0], 'content'):
                has_discussion_objects = True

            # Check if we have access to game history (entries with turn_number attribute)
            if discussion_history and hasattr(discussion_history[0], 'turn_number'):
                has_game_history = True

            # Convert discussion history to strings if needed
            discussion_strings = []
            game_history_strings = []

            if has_game_history:
                # Separate current discussion from previous turns
                current_turn = max(
                    entry.turn_number for entry in discussion_history)
                current_discussion = [
                    entry for entry in discussion_history if entry.turn_number == current_turn]
                previous_discussions = [
                    entry for entry in discussion_history if entry.turn_number < current_turn]

                # Convert to strings
                discussion_strings = [
                    entry.content for entry in current_discussion]
                game_history_strings = [f"Turn {entry.turn_number+1}, Agent {entry.agent_id}: {entry.content}"
                                        for entry in previous_discussions]

                # Update memory with latest game history
                self.update_memory("game_history", game_history_strings)
            elif has_discussion_objects:
                discussion_strings = [
                    entry.content for entry in discussion_history]
            else:
                discussion_strings = discussion_history

            # Retrieve previous game history from memory if available
            if not has_game_history and "game_history" in self.memory:
                game_history_strings = self.memory["game_history"]

            # Create initial state
            initial_state = {
                "game_state": game_state,
                "discussion_history": discussion_strings,
                "game_history": game_history_strings,
                "current_thoughts": []
            }

            # Run the reasoning graph
            final_state = self.reasoning_graph.invoke(initial_state)

            # Extract the generated thoughts
            current_thoughts = final_state.get("current_thoughts", [])

            # Generate a contribution based on the thoughts
            if current_thoughts:
                # Create a prompt for generating a contribution
                prompt = f"""You are Agent {self.agent_id} in a game of Hanabi, participating in a discussion.

Based on your analysis of the game state, generate a concise contribution to the discussion.

CRITICAL INFORMATION RULES:
1. You MUST distinguish between KNOWN information and INFERENCES.
2. KNOWN information is ONLY what you've been explicitly told through clues.
3. INFERENCES are educated guesses based on game state, but are NOT certainties.
4. You MUST use language like "I believe", "I infer", "likely", "probably", "might be" for inferences.
5. You MUST use language like "I know" ONLY for information directly given through clues.
6. For example, if you received a "green" clue on a card, you can say "I know this card is green" but NOT "I know this is a green 1".
7. You CANNOT claim to know both color AND number of a card unless you've received BOTH clues for that card.
8. You CANNOT claim to know the exact identity of a card based solely on a single clue.

Your thoughts:
{self._format_thoughts(current_thoughts)}

Generate a concise, strategic contribution that follows the information rules above. Focus on what action you're considering and why, without revealing specific card information you shouldn't know.
"""

                # Generate the contribution
                response = self.model.invoke([HumanMessage(content=prompt)])

                contribution = response.content.strip()

                # Log at debug level instead of info to avoid double logging
                logger.debug(
                    f"Agent {self.agent_id} generated contribution: {contribution}")
                return contribution
            else:
                logger.warning(
                    f"Agent {self.agent_id} failed to generate thoughts")
                return "I'm analyzing the game state and considering our options."

        except Exception as e:
            logger.error(f"Error in Agent {self.agent_id} discussion: {e}")
            return "I'm having trouble analyzing the game state."

    def decide_action(self, game_state: GameState, discussion_history: list) -> Dict[str, Any]:
        """Decide on an action based on the game state and discussion summary."""
        # Store the game state for tool access
        self.current_game_state = game_state

        # Check if discussion_history is a list of DiscussionEntry objects or strings
        has_discussion_objects = False
        has_game_history = False

        if discussion_history and hasattr(discussion_history[0], 'content'):
            has_discussion_objects = True

        # Convert discussion history to strings if needed
        discussion_strings = []
        game_history_strings = []

        if has_discussion_objects:
            discussion_strings = [
                entry.content for entry in discussion_history]
        else:
            discussion_strings = discussion_history

        # Retrieve previous game history from memory if available
        if "game_history" in self.memory:
            game_history_strings = self.memory["game_history"]

        # Create initial state
        initial_state = {
            "game_state": game_state,
            "discussion_history": discussion_strings,
            "game_history": game_history_strings,
            "current_thoughts": [],
            "messages": [HumanMessage(content="It's your turn to take an action in the Hanabi game.")]
        }

        # Run the reasoning graph
        try:
            final_state = self.reasoning_graph.invoke(initial_state)

            # Check if we have a tool result
            if "tool_result" in final_state and final_state["tool_result"]:
                action = final_state["tool_result"]
                logger.info(f"Agent {self.agent_id} decided action: {action}")
                return action

            # If we don't have a tool result but have messages, check the last message for tool calls
            elif "messages" in final_state and final_state["messages"]:
                last_message = final_state["messages"][-1]
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    # Extract the action from the tool call
                    tool_call = last_message.tool_calls[0]
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})

                    # Map tool names to action types
                    if tool_name == "play_card":
                        action = {
                            "type": "play_card",
                            "card_index": tool_args.get("card_index", 0)
                        }
                    elif tool_name == "give_clue":
                        action = {
                            "type": "give_clue",
                            "target_id": tool_args.get("target_id", 0),
                            "clue": {
                                "type": tool_args.get("clue_type", "color"),
                                "value": tool_args.get("clue_value", "")
                            }
                        }
                    elif tool_name == "discard":
                        action = {
                            "type": "discard",
                            "card_index": tool_args.get("card_index", 0)
                        }
                    else:
                        raise ValueError(f"Unknown tool name: {tool_name}")

                    logger.info(
                        f"Agent {self.agent_id} decided action from tool call: {action}")
                    return action

            # If we still don't have an action, raise an error
            raise ValueError(
                f"Agent {self.agent_id} failed to propose any action")
        except Exception as e:
            logger.critical(f"Agent {self.agent_id} failed with error: {e}")
            raise

    def _create_state_analysis_prompt(self, game_state: GameState) -> str:
        """Create prompt for analyzing the game state."""
        # Get game context from memory if available
        game_context = ""
        if "game_summary" in self.memory:
            game_context = f"\nGame context: {self.memory['game_summary']}\n"

        return f"""You are Agent {self.agent_id} in a game of Hanabi.
Current game state:
- Firework piles: {self._format_firework_piles(game_state)}
- Discard pile: {self._format_discard_pile(game_state)}
- Clue tokens: {game_state.clue_tokens}
- Fuse tokens: {game_state.fuse_tokens}
- Your hand: {self._format_hand(game_state.hands[self.agent_id])}{game_context}s

EXTREMELY IMPORTANT INSTRUCTIONS:
- Your ENTIRE response must be under 100 words
- Use simple, direct language with no fluff
- Focus only on the most important observations
- Do not use any special formatting, bullet points, or section headers
- Write in plain text only"""

    def _create_thought_generation_prompt(self, game_state: GameState,
                                          discussion_history: List[str],
                                          current_thoughts: List[str]) -> str:
        """Create a prompt for generating thoughts about the game state."""
        # Determine if this agent is the active player or providing feedback
        is_active_player = self.agent_id == game_state.current_player
        active_player_id = game_state.current_player

        # Get the active player's proposal if this agent is providing feedback
        active_player_proposal = ""
        if not is_active_player and discussion_history:
            active_player_proposal = discussion_history[0]

        # Format the game state information
        firework_piles = self._format_firework_piles(game_state)
        discard_pile = self._format_discard_pile(game_state)

        # Format information about other players' hands
        other_players_info = []
        for player_id, hand in game_state.hands.items():
            if player_id != self.agent_id:
                hand_str = self._format_hand(hand)

                # Add information about what clues would be valid for this player
                valid_color_clues = set()
                valid_number_clues = set()
                for card in hand:
                    valid_color_clues.add(card.color.value)
                    valid_number_clues.add(str(card.number))

                clue_info = f"Valid clues for Player {player_id}: "
                clue_info += f"Colors: {', '.join(sorted(valid_color_clues))}, "
                clue_info += f"Numbers: {', '.join(sorted(valid_number_clues))}"

                other_players_info.append(
                    f"Player {player_id}'s hand: {hand_str}\n{clue_info}")

        other_players = "\n".join(other_players_info)

        # Format information about the agent's own hand (which they can't see)
        my_hand = "Your hand (you cannot see the actual cards):"
        if self.agent_id in game_state.hands:
            for i, card in enumerate(game_state.hands[self.agent_id]):
                card_info = f"\nCard {i}: [HIDDEN]"

                # Add clue information if available
                if card.is_visible:
                    clues = []
                    if hasattr(card, "color_clued") and card.color_clued:
                        clues.append(f"color: {card.color.value}")
                    if hasattr(card, "number_clued") and card.number_clued:
                        clues.append(f"number: {card.number}")

                    if not clues and "clue_history" in self.memory:
                        for clue in self.memory["clue_history"]:
                            if clue["receiver_id"] == self.agent_id and i in clue["affected_indices"]:
                                if clue["clue_type"] == "color":
                                    clues.append(
                                        f"color: {clue['clue_value']}")
                                else:  # number clue
                                    clues.append(
                                        f"number: {clue['clue_value']}")

                    if clues:
                        card_info += f" ({', '.join(clues)})"
                        # Add inference information
                        inferences = []
                        # We could add logic here to generate inferences
                        # For now, we'll leave it empty
                    else:
                        card_info += " (has received clues)"

                my_hand += card_info

        # Format the discussion history
        discussion = self._format_discussion(discussion_history)

        # Format the current thoughts
        thoughts = self._format_thoughts(current_thoughts)

        # Create the prompt
        prompt = f"""You are Agent {self.agent_id} in a game of Hanabi.

Current Game State:
- Score: {game_state.score}
- Clue tokens: {game_state.clue_tokens}
- Fuse tokens: {game_state.fuse_tokens}
- Current player: {game_state.current_player}
- Firework piles: {firework_piles}
- Discard pile: {discard_pile}

{other_players}

{my_hand}

CRITICAL INFORMATION RULES:
1. You MUST distinguish between KNOWN information and INFERENCES.
2. KNOWN information is ONLY what you've been explicitly told through clues.
3. INFERENCES are educated guesses based on game state, but are NOT certainties.
4. You MUST use language like "I believe", "I infer", "likely", "probably", "might be" for inferences.
5. You MUST use language like "I know" ONLY for information directly given through clues.
6. For example, if you received a "green" clue on a card, you can say "I know this card is green" but NOT "I know this is a green 1".
7. You CANNOT claim to know both color AND number of a card unless you've received BOTH clues for that card.
8. You CANNOT claim to know the exact identity of a card based solely on a single clue.

HANABI COMMUNICATION RULES:
1. You CANNOT directly tell other players what cards they have in their hands.
2. You CANNOT indirectly hint at specific card values outside of the official clue mechanism.
3. You CANNOT discuss specific card values you see in other players' hands.
4. You CANNOT say things like "Player 2 has a red 3" or "Player 3's second card is a 1".
5. You CANNOT say "I see a red 1 in your hand" or "Your third card is a 5".
6. You CANNOT say "You should play your red card" or "You should play your 1".
7. You CANNOT say "Give a red clue to Player 3" or "Give a 1 clue to Player 4".
8. You CAN say "I'll give information about Player 3's 2nd card".
9. You CAN say "I'll give a color clue to Player 3" (without specifying which color).
10. You CAN say "I'll give a number clue to Player 3" (without specifying which number).
11. You CAN discuss general strategy like "We should focus on building the red firework next".
12. You CAN say "Consider giving information about Player 3's first card".

"""

        # Add role-specific instructions
        if is_active_player:
            prompt += f"""You are the active player (Player {self.agent_id}) and need to decide on an action.
Think through the current game state, what you know about your hand, and what would be the most strategic move.
Consider the balance between giving clues, playing cards, and discarding.
Remember to clearly distinguish between what you KNOW from clues and what you INFER from the game state.
"""
        else:
            prompt += f"""You are providing feedback to Player {active_player_id}, who is the active player.
Their proposal is: {active_player_proposal}

Think about whether their proposed action is strategic and how it fits with the team's goals.
Consider if there might be better alternatives they should consider.
Remember to clearly distinguish between what you KNOW from clues and what you INFER from the game state.
"""

        # Add discussion history if available
        if discussion:
            prompt += f"\nDiscussion so far:\n{discussion}\n"

        # Add current thoughts if available
        if thoughts:
            prompt += f"\nYour current thoughts:\n{thoughts}\n"

        # Add final instruction
        prompt += """
Generate your next thoughts about the game state and potential actions.
Be concise and focus on the most important strategic considerations.
IMPORTANT: Clearly distinguish between what you KNOW from clues and what you INFER from the game state.
"""

        return prompt

    def _create_action_proposal_prompt(self, game_state: GameState,
                                       discussion_history: List[str],
                                       current_thoughts: List[str]) -> str:
        """Create a prompt for proposing an action based on the game state and discussion."""
        # Format the game state information
        firework_piles = self._format_firework_piles(game_state)
        discard_pile = self._format_discard_pile(game_state)

        # Format information about other players' hands
        other_players_info = []
        for player_id, hand in game_state.hands.items():
            if player_id != self.agent_id:
                hand_str = self._format_hand(hand)

                # Add information about what clues would be valid for this player
                valid_color_clues = set()
                valid_number_clues = set()
                for card in hand:
                    valid_color_clues.add(card.color.value)
                    # Convert to string for consistency
                    valid_number_clues.add(str(card.number))

                clue_info = f"Valid clues for Player {player_id}: "
                clue_info += f"Colors: {', '.join(sorted(valid_color_clues))}, "
                clue_info += f"Numbers: {', '.join(sorted(valid_number_clues))}"

                other_players_info.append(
                    f"Player {player_id}'s hand: {hand_str}\n{clue_info}")

        other_players = "\n".join(other_players_info)

        # Format information about the agent's own hand (which they can't see)
        my_hand = "Your hand (you cannot see the actual cards):"
        if self.agent_id in game_state.hands:
            for i, card in enumerate(game_state.hands[self.agent_id]):
                card_info = f"\nCard {i}: [HIDDEN]"

                # Add clue information if available
                if card.is_visible:
                    # In a real game, we would track what clues have been given
                    # For now, we'll use the card's actual properties to simulate clue information
                    clues = []

                    # Check if this card has received color clues
                    if hasattr(card, "color_clued") and card.color_clued:
                        clues.append(f"color: {card.color.value}")

                    # Check if this card has received number clues
                    if hasattr(card, "number_clued") and card.number_clued:
                        clues.append(f"number: {card.number}")

                    # If we don't have specific clue tracking, just indicate it's been clued
                    if not clues:
                        # Try to get clue information from memory
                        if "clue_history" in self.memory:
                            for clue in self.memory["clue_history"]:
                                if clue["receiver_id"] == self.agent_id and i in clue["affected_indices"]:
                                    if clue["clue_type"] == "color":
                                        clues.append(
                                            f"color: {clue['clue_value']}")
                                    else:  # number clue
                                        clues.append(
                                            f"number: {clue['clue_value']}")

                    if clues:
                        card_info += f" ({', '.join(clues)})"
                    else:
                        card_info += " (has received clues)"

                my_hand += card_info

        # Format the discussion history
        discussion = self._format_discussion(discussion_history)

        # Format the current thoughts
        thoughts = self._format_thoughts(current_thoughts)

        # Generate information about valid actions
        valid_actions_info = self._generate_valid_actions_info(game_state)

        # Create the prompt
        prompt = f"""You are Agent {self.agent_id} in a game of Hanabi and it's your turn to decide on an action.

Current Game State:
- Score: {game_state.score}
- Clue tokens: {game_state.clue_tokens}
- Fuse tokens: {game_state.fuse_tokens}
- Current player: {game_state.current_player}
- Firework piles: {firework_piles}
- Discard pile: {discard_pile}

{other_players}

{my_hand}

HANABI DECK COMPOSITION:
- Each color (red, yellow, green, blue, white) has:
  - Three 1s
  - Two 2s, 3s, and 4s
  - Only one 5
- This means discarding a 5 makes it impossible to complete that color's firework

CRITICAL INFORMATION RULES:
1. You MUST distinguish between KNOWN information and INFERENCES.
2. KNOWN information is ONLY what you've been explicitly told through clues.
3. INFERENCES are educated guesses based on game state, but are NOT certainties.
4. You MUST use language like "I believe", "I infer", "likely", "probably", "might be" for inferences.
5. You MUST use language like "I know" ONLY for information directly given through clues.
6. For example, if you received a "green" clue on a card, you can say "I know this card is green" but NOT "I know this is a green 1".
7. You CANNOT claim to know both color AND number of a card unless you've received BOTH clues for that card.
8. You CANNOT claim to know the exact identity of a card based solely on a single clue.

HANABI COMMUNICATION RULES:
1. You CANNOT directly tell other players what cards they have in their hands.
2. You CANNOT indirectly hint at specific card values outside of the official clue mechanism.
3. You CANNOT discuss specific card values you see in other players' hands.
4. You CANNOT say things like "Player 2 has a red 3" or "Player 3's second card is a 1".
5. You CANNOT say "I see a red 1 in your hand" or "Your third card is a 5".
6. You CANNOT say "You should play your red card" or "You should play your 1".
7. You CANNOT say "Give a red clue to Player 3" or "Give a 1 clue to Player 4".
8. You CAN say "I'll give information about Player 3's 2nd card".
9. You CAN say "I'll give a color clue to Player 3" (without specifying which color).
10. You CAN say "I'll give a number clue to Player 3" (without specifying which number).
11. You CAN discuss general strategy like "We should focus on building the red firework next".
12. You CAN say "Consider giving information about Player 3's first card".

EXAMPLES OF FORBIDDEN STATEMENTS:
- "I know my card 0 is a green 1" (when you've only received a green clue)
- "I know my card 1 is a blue 4" (when you've only received a blue clue)
- "I know my card 2 is a red 3" (when you've only received a red clue)
- "I know my card 3 is a white 2" (when you've only received a 2 clue)
- "Player 2 has a red 3 in position 1"
- "Your second card is a blue 4"
- "I see you have a 1 in your hand"
- "Give a red clue to Player 3"
- "Player 4 has two 1s"
- "Your hand has a playable red card"
- "I'll give you a 1 clue"
- "I'll give Player 3 a red clue"
- "You should play your red card"
- "You should play your 1"

EXAMPLES OF ALLOWED STATEMENTS:
- "I know my card 0 is green" (when you've received a green clue)
- "I know my card 1 is a 4" (when you've received a 4 clue)
- "I believe my card 0 might be a green 1 based on the current game state" (inference)
- "I infer that my green card is likely a 1 since no green cards have been played yet" (inference)
- "I'll give information about Player 3's cards"
- "I'll give a color clue to Player 3"
- "I'll give a number clue to Player 3"
- "Consider giving information about your teammate's first card"
- "I think we should focus on building the red firework next"
- "I'll play my first card"
- "I'll discard my third card"

HANABI STRATEGY PRINCIPLES:
- Balance information gathering with progress - sometimes it's worth taking calculated risks
- Clues are a limited resource - use them efficiently to convey maximum information
- Cards that have been clued are usually important - either playable now or valuable for later
- Consider what information other players already have when deciding your action
- Sometimes discarding is necessary to regain clue tokens, even if it means losing potential points
- The team's overall strategy is more important than individual perfect plays
- Pay attention to the discard pile to track which cards remain in the deck

{valid_actions_info}

"""

        # Add discussion history if available
        if discussion:
            prompt += f"\nDiscussion:\n{discussion}\n"

        # Add current thoughts if available
        if thoughts:
            prompt += f"\nYour thoughts:\n{thoughts}\n"

        # Add final instruction for using tools
        prompt += """
Based on the game state, discussion, and your thoughts, decide on your final action.
Remember to clearly distinguish between KNOWN information (from clues) and INFERENCES.

You have the following tools available:

1. play_card: Play a card from your hand
   - card_index: Index of the card to play (0-4)

2. give_clue: Give a clue to another player
   - target_id: ID of the player to give a clue to
   - clue_type: Type of clue ("color" or "number")
   - clue_value: Value of the clue (color name or number 1-5)

3. discard: Discard a card from your hand
   - card_index: Index of the card to discard (0-4)

Before using a tool, provide a brief explanation of your reasoning that follows the information rules.
"""

        return prompt

    def _generate_valid_actions_info(self, game_state: GameState) -> str:
        """Generate information about valid actions based on the current game state."""
        valid_actions = []

        # Check if giving clues is valid
        if game_state.clue_tokens > 0:
            valid_actions.append(
                "- You can give clues (clue tokens available: {})".format(game_state.clue_tokens))

            # List valid targets for clues
            valid_targets = [i for i in range(
                len(game_state.hands)) if i != self.agent_id]
            valid_actions.append("  Valid targets for clues: {}".format(
                ", ".join(map(str, valid_targets))))
        else:
            valid_actions.append(
                "- You CANNOT give clues (no clue tokens available)")

        # Check if discarding is valid
        if game_state.clue_tokens < 8:
            valid_actions.append(
                "- You can discard cards (clue tokens: {}/8)".format(game_state.clue_tokens))

            # List valid indices for discarding
            valid_indices = list(range(len(game_state.hands[self.agent_id])))
            valid_actions.append("  Valid indices for discarding: {}".format(
                ", ".join(map(str, valid_indices))))
        else:
            valid_actions.append(
                "- You CANNOT discard cards (clue tokens already at maximum: 8/8)")

        # Information about playing cards
        valid_actions.append(
            "- You can play a card if it's the next in sequence for its color")

        # Current state of firework piles
        firework_info = []
        for color in Color:
            pile_height = len(game_state.firework_piles.get(color, []))
            next_card = pile_height + 1
            if next_card <= 5:
                firework_info.append(
                    f"  {color.value}: Next playable card is {next_card}")
            else:
                firework_info.append(f"  {color.value}: Firework complete")

        valid_actions.extend(firework_info)

        return "\n".join(valid_actions)

    def _parse_action_response(self, response: str) -> Dict[str, Any]:
        """Parse the action response from the LLM."""
        try:
            # Store the raw response for debugging
            self._last_raw_response = response

            # Log the raw response for debugging
            logger.debug(
                f"Agent {self.agent_id}: Raw action response: {response}")

            # Extract JSON from the response
            json_match = re.search(
                r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)

            if not json_match:
                # Try to find any JSON-like structure without code blocks
                json_match = re.search(r'({[\s\S]*?})', response)

            if json_match:
                json_str = json_match.group(1)
                logger.debug(
                    f"Agent {self.agent_id}: Extracted JSON: {json_str}")

                try:
                    # Try to parse the JSON
                    action = json.loads(json_str)

                    # Ensure the action has the required fields
                    if "type" not in action:
                        logger.warning(
                            f"Agent {self.agent_id}: Missing 'type' in action: {action}")
                        return {}

                    # Fix common issues with action format
                    action_type = action["type"].lower().strip()

                    # Normalize action type
                    if "play" in action_type:
                        action["type"] = "play_card"
                    elif "clue" in action_type or "hint" in action_type:
                        action["type"] = "give_clue"
                    elif "discard" in action_type:
                        action["type"] = "discard"

                    # Fix play_card actions
                    if action["type"] == "play_card":
                        # Ensure card_index is present and an integer
                        if "card_index" not in action:
                            if "index" in action:
                                action["card_index"] = action["index"]
                            elif "position" in action:
                                action["card_index"] = action["position"]
                            elif "card" in action and isinstance(action["card"], int):
                                action["card_index"] = action["card"]
                            else:
                                # Default to first card
                                action["card_index"] = 0

                        # Convert card_index to int if it's a string
                        if isinstance(action["card_index"], str):
                            try:
                                action["card_index"] = int(
                                    action["card_index"])
                            except ValueError:
                                action["card_index"] = 0

                    # Fix give_clue actions
                    elif action["type"] == "give_clue":
                        # Ensure target_id is present
                        if "target_id" not in action:
                            if "target" in action:
                                action["target_id"] = action["target"]
                            elif "player" in action:
                                action["target_id"] = action["player"]
                            else:
                                # Default to next player
                                action["target_id"] = (self.agent_id + 1) % 5

                        # Convert target_id to int if it's a string
                        if isinstance(action["target_id"], str):
                            try:
                                action["target_id"] = int(action["target_id"])
                            except ValueError:
                                action["target_id"] = (self.agent_id + 1) % 5

                        # Ensure clue is present and properly formatted
                        if "clue" not in action or not isinstance(action["clue"], dict):
                            clue_type = None
                            clue_value = None

                            # Try to extract clue type and value from action
                            if "color" in action:
                                clue_type = "color"
                                clue_value = action["color"]
                            elif "number" in action:
                                clue_type = "number"
                                clue_value = action["number"]

                            # Create clue dict
                            if clue_type and clue_value:
                                action["clue"] = {
                                    "type": clue_type,
                                    "value": clue_value
                                }
                            else:
                                # Default clue
                                action["clue"] = {
                                    "type": "color",
                                    "value": "red"
                                }

                        # Ensure clue has type and value
                        if "type" not in action["clue"]:
                            action["clue"]["type"] = "color"
                        if "value" not in action["clue"]:
                            action["clue"]["value"] = "red"

                        # Normalize clue type
                        clue_type = action["clue"]["type"].lower().strip()
                        if "color" in clue_type:
                            action["clue"]["type"] = "color"
                        elif "number" in clue_type or "value" in clue_type:
                            action["clue"]["type"] = "number"

                        # Normalize color values
                        if action["clue"]["type"] == "color":
                            color_value = str(
                                action["clue"]["value"]).lower().strip()
                            valid_colors = [c.value for c in Color]

                            # Find closest match
                            for valid_color in valid_colors:
                                if valid_color in color_value:
                                    action["clue"]["value"] = valid_color
                                    break

                        # Normalize number values
                        if action["clue"]["type"] == "number":
                            try:
                                num_value = int(action["clue"]["value"])
                                if 1 <= num_value <= 5:
                                    action["clue"]["value"] = num_value
                                else:
                                    action["clue"]["value"] = max(
                                        1, min(5, num_value))
                            except (ValueError, TypeError):
                                action["clue"]["value"] = 1

                    # Fix discard actions
                    elif action["type"] == "discard":
                        # Ensure card_index is present and an integer
                        if "card_index" not in action:
                            if "index" in action:
                                action["card_index"] = action["index"]
                            elif "position" in action:
                                action["card_index"] = action["position"]
                            elif "card" in action and isinstance(action["card"], int):
                                action["card_index"] = action["card"]
                            else:
                                # Default to first card
                                action["card_index"] = 0

                        # Convert card_index to int if it's a string
                        if isinstance(action["card_index"], str):
                            try:
                                action["card_index"] = int(
                                    action["card_index"])
                            except ValueError:
                                action["card_index"] = 0

                    logger.debug(
                        f"Agent {self.agent_id}: Normalized action: {action}")
                    return action

                except json.JSONDecodeError as e:
                    logger.error(
                        f"Agent {self.agent_id}: JSON decode error: {e}")
                    logger.debug(
                        f"Agent {self.agent_id}: Problematic JSON: {json_str}")

            # If we couldn't extract or parse JSON, try to infer the action from the text
            return self._infer_action_from_text(response)

        except Exception as e:
            logger.error(
                f"Agent {self.agent_id}: Error parsing action response: {e}")
            return {}

    def _infer_action_from_text(self, text: str) -> Dict[str, Any]:
        """Infer an action from plain text when JSON parsing fails."""
        text = text.lower()

        # Check for play card action
        play_match = re.search(
            r'play(?:\s+card)?(?:\s+at)?\s+(?:index|position|card)?\s*(\d+)', text)
        if play_match:
            try:
                card_index = int(play_match.group(1))
                return {
                    "type": "play_card",
                    "card_index": card_index
                }
            except (ValueError, IndexError):
                pass

        # Check for discard action
        discard_match = re.search(
            r'discard(?:\s+card)?(?:\s+at)?\s+(?:index|position|card)?\s*(\d+)', text)
        if discard_match:
            try:
                card_index = int(discard_match.group(1))
                return {
                    "type": "discard",
                    "card_index": card_index
                }
            except (ValueError, IndexError):
                pass

        # Check for give clue action
        clue_match = re.search(
            r'(?:give|hint)(?:\s+a)?\s+(?:clue|hint)(?:\s+to)?\s+(?:player|agent)?\s*(\d+)', text)
        color_match = re.search(r'(red|blue|green|yellow|white)', text)
        number_match = re.search(r'number\s*(\d+)', text)

        if clue_match:
            try:
                target_id = int(clue_match.group(1))

                # Determine clue type and value
                if color_match:
                    return {
                        "type": "give_clue",
                        "target_id": target_id,
                        "clue": {
                            "type": "color",
                            "value": color_match.group(1)
                        }
                    }
                elif number_match:
                    return {
                        "type": "give_clue",
                        "target_id": target_id,
                        "clue": {
                            "type": "number",
                            "value": int(number_match.group(1))
                        }
                    }
                else:
                    # Default to red if no color/number found
                    return {
                        "type": "give_clue",
                        "target_id": target_id,
                        "clue": {
                            "type": "color",
                            "value": "red"
                        }
                    }
            except (ValueError, IndexError):
                pass

        # If we couldn't infer anything, return empty dict
        logger.warning(
            f"Agent {self.agent_id}: Could not infer action from text: {text[:100]}...")
        return {}

    def _validate_action_format(self, action: Dict[str, Any]) -> bool:
        """Validate the format of an action."""
        if not action or not isinstance(action, dict):
            return False

        # Check if action has a type
        if "type" not in action:
            return False

        action_type = action["type"]

        # Validate play_card action
        if action_type == "play_card":
            return "card_index" in action

        # Validate give_clue action
        elif action_type == "give_clue":
            if "target_id" not in action:
                return False

            if "clue" not in action or not isinstance(action["clue"], dict):
                return False

            clue = action["clue"]
            if "type" not in clue or "value" not in clue:
                return False

            if clue["type"] not in ["color", "number"]:
                return False

            return True

        # Validate discard action
        elif action_type == "discard":
            return "card_index" in action

        # Unknown action type
        return False

    def _format_firework_piles(self, game_state: GameState) -> str:
        """Format firework piles for display."""
        if not game_state.firework_piles:
            return "No fireworks started"

        formatted_piles = []
        for color, pile in game_state.firework_piles.items():
            if pile:
                top_card = pile[-1].number if pile else 0
                formatted_piles.append(f"{color}: {top_card}")
            else:
                formatted_piles.append(f"{color}: 0")

        return ", ".join(formatted_piles)

    def _format_discard_pile(self, game_state: GameState) -> str:
        """Format discard pile for display."""
        if not game_state.discard_pile:
            return "Empty"

        # Group cards by color and number for better readability
        card_counts = {}
        for card in game_state.discard_pile:
            key = f"{card.color} {card.number}"
            card_counts[key] = card_counts.get(key, 0) + 1

        formatted_cards = [f"{key} (x{count})" if count > 1 else key
                           for key, count in card_counts.items()]
        return ", ".join(formatted_cards)

    def _format_hand(self, hand: List[Card]) -> str:
        """Format hand for display."""
        if not hand:
            return "Empty"

        formatted_cards = []
        for i, card in enumerate(hand):
            if card.is_visible:
                # For visible cards, show the color and number in concise format
                # First letter of color
                color_abbr = card.color.value[0].upper()
                formatted_cards.append(f"{i}: {color_abbr}{card.number}")
            else:
                # For hidden cards, show [?]
                formatted_cards.append(f"{i}: [?]")

        return ", ".join(formatted_cards)

    def _format_discussion(self, discussion_history: List[str]) -> str:
        """Format discussion history for display."""
        if not discussion_history:
            return "No discussion yet"

        # Limit the length of each message to avoid extremely long prompts
        # Set to None to disable truncation
        max_message_length = 500
        formatted_messages = []

        for i, msg in enumerate(discussion_history):
            if max_message_length and len(msg) > max_message_length:
                short_msg = msg[:max_message_length] + "... (truncated)"
            else:
                short_msg = msg

            formatted_messages.append(f"Message {i+1}: {short_msg}")

        return "\n".join(formatted_messages)

    def _format_thoughts(self, thoughts: List[str]) -> str:
        """Format current thoughts for display."""
        if not thoughts:
            return "No thoughts yet"

        # Limit the length of each thought to avoid extremely long prompts
        # Set to None to disable truncation
        max_thought_length = 500
        formatted_thoughts = []

        for i, thought in enumerate(thoughts):
            if max_thought_length and len(thought) > max_thought_length:
                short_thought = thought[:max_thought_length] + \
                    "... (truncated)"
            else:
                short_thought = thought

            formatted_thoughts.append(f"Thought {i+1}: {short_thought}")

        return "\n".join(formatted_thoughts)

    def _validate_action_before_submission(self, game_state: GameState, action: Dict[str, Any]) -> bool:
        """Validate an action before submitting it to the game engine."""
        if not action:
            logger.warning(f"Agent {self.agent_id}: Empty action")
            return False

        try:
            # Validate action format
            if not self._validate_action_format(action):
                logger.error(
                    f"Agent {self.agent_id}: Invalid action format: {action}")
                return False

            action_type = action.get("type")

            # Validate play_card action
            if action_type == "play_card":
                card_index = action.get("card_index")
                if not isinstance(card_index, int) or card_index < 0 or card_index >= len(game_state.hands[self.agent_id]):
                    logger.error(
                        f"Agent {self.agent_id}: Invalid card index: {card_index}")
                    return False
                return True

            # Validate give_clue action
            elif action_type == "give_clue":
                # Check if we have clue tokens
                if game_state.clue_tokens <= 0:
                    logger.error(
                        f"Agent {self.agent_id}: No clue tokens available")
                    return False

                # Check target player
                target_id = action.get("target_id")
                if target_id == self.agent_id:
                    logger.error(
                        f"Agent {self.agent_id}: Cannot give clue to yourself")
                    return False
                if not isinstance(target_id, int) or target_id < 0 or target_id >= len(game_state.hands):
                    logger.error(
                        f"Agent {self.agent_id}: Invalid target ID: {target_id}")
                    return False

                # Check clue format
                clue = action.get("clue")
                if not isinstance(clue, dict):
                    logger.error(
                        f"Agent {self.agent_id}: Invalid clue format: {clue}")
                    return False

                # Check clue type and value
                clue_type = clue.get("type")
                clue_value = clue.get("value")
                if not clue_type or not clue_value:
                    logger.error(
                        f"Agent {self.agent_id}: Missing clue type or value: {clue}")
                    return False

                # Validate clue type
                if clue_type not in ["color", "number"]:
                    logger.error(
                        f"Agent {self.agent_id}: Invalid clue type: {clue_type}")
                    return False

                # Validate clue value based on type
                if clue_type == "color":
                    if clue_value not in [c.value for c in Color]:
                        logger.error(
                            f"Agent {self.agent_id}: Invalid color value: {clue_value}")
                        return False
                elif clue_type == "number":
                    try:
                        num_value = int(clue_value)
                        if num_value < 1 or num_value > 5:
                            logger.error(
                                f"Agent {self.agent_id}: Invalid number value (out of range): {clue_value}")
                            return False
                    except ValueError:
                        logger.error(
                            f"Agent {self.agent_id}: Invalid number value (not convertible to int): {clue_value}")
                        return False

                # Check if the clue matches any cards in the target player's hand
                matches = False
                for card in game_state.hands[target_id]:
                    if clue_type == "color" and card.color.value == clue_value:
                        matches = True
                        break
                    elif clue_type == "number" and card.number == int(clue_value):
                        matches = True
                        break

                if not matches:
                    logger.error(
                        f"Agent {self.agent_id}: No {clue_type} {clue_value} cards in player {target_id}'s hand")
                    return False

                return True

            # Validate discard action
            elif action_type == "discard":
                # Check if we need more clue tokens
                if game_state.clue_tokens >= 8:
                    logger.error(
                        f"Agent {self.agent_id}: Cannot discard when clue tokens are at maximum (8)")
                    return False

                # Check card index
                card_index = action.get("card_index")
                if not isinstance(card_index, int) or card_index < 0 or card_index >= len(game_state.hands[self.agent_id]):
                    logger.error(
                        f"Agent {self.agent_id}: Invalid card index: {card_index}")
                    return False

                return True

            # Unknown action type
            else:
                logger.error(
                    f"Agent {self.agent_id}: Unknown action type: {action_type}")
                return False

        except Exception as e:
            logger.error(
                f"Agent {self.agent_id}: Error validating action: {e}")
            return False
