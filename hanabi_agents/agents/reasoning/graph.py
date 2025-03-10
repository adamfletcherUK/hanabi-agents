from typing import Dict, Any
from langgraph.graph import StateGraph, END, START
from langchain_core.tools import Tool
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableLambda
from ..state.agent_state import AgentStateDict
from ..tools.play_card import play_card_tool
from ..tools.give_clue import give_clue_tool
from ..tools.discard import discard_tool
from ..tools.error_handler import handle_tool_error
from .nodes import analyze_game_state, generate_thoughts, propose_action
from .router import should_execute_tools
import logging

# Set up logging
logger = logging.getLogger(__name__)


def setup_reasoning_graph(agent):
    """
    Set up the reasoning graph for the agent.

    Args:
        agent: The agent instance

    Returns:
        Compiled reasoning graph
    """
    # Create the reasoning graph
    builder = StateGraph(AgentStateDict)

    # Define the tools
    tools = [
        Tool.from_function(
            func=lambda card_index: play_card_tool(
                agent.agent_id, card_index, agent.current_game_state),
            name="play_card",
            description="Play a card from your hand",
            args_schema={
                "type": "object",
                "properties": {
                    "card_index": {
                        "type": "integer",
                        "description": "Index of the card to play (0-indexed)"
                    }
                },
                "required": ["card_index"]
            }
        ),
        Tool.from_function(
            func=lambda target_id, clue_type, clue_value: give_clue_tool(
                agent.agent_id, target_id, clue_type, clue_value, agent.current_game_state),
            name="give_clue",
            description="Give a clue to another player",
            args_schema={
                "type": "object",
                "properties": {
                    "target_id": {
                        "type": "integer",
                        "description": "ID of the player to give the clue to"
                    },
                    "clue_type": {
                        "type": "string",
                        "enum": ["color", "number"],
                        "description": "Type of clue to give (color or number)"
                    },
                    "clue_value": {
                        "type": "string",
                        "description": "Value of the clue (e.g., 'red', '1')"
                    }
                },
                "required": ["target_id", "clue_type", "clue_value"]
            }
        ),
        Tool.from_function(
            func=lambda card_index: discard_tool(
                agent.agent_id, card_index, agent.current_game_state),
            name="discard",
            description="Discard a card from your hand",
            args_schema={
                "type": "object",
                "properties": {
                    "card_index": {
                        "type": "integer",
                        "description": "Index of the card to discard (0-indexed)"
                    }
                },
                "required": ["card_index"]
            }
        )
    ]

    # Create a tool node with error handling
    tool_node = ToolNode(tools)
    tool_node_with_error_handling = RunnableLambda(
        lambda state, config=None: tool_node.invoke(state, config=config)
    ).with_fallbacks([
        RunnableLambda(
            lambda state, error, config=None: handle_tool_error(
                state, agent_id=agent.agent_id)
        )
    ])

    # Wrap the tool node to ensure it has messages
    def execute_tools_wrapper(state, store=None, config=None):
        """
        Wrapper function for executing tools with error handling.

        Args:
            state: Current state of the reasoning graph
            store: Memory store
            config: Configuration

        Returns:
            Updated state with tool execution results
        """
        try:
            # Log the state for debugging
            logger.info(
                f"Executing tools with state keys: {list(state.keys())}")

            # Make sure we have the agent_id in the state
            agent_id = state.get("agent_id")
            if agent_id is None:
                logger.warning("No agent_id in state, using default agent_id")
                agent_id = agent.agent_id

            # Track the execution path
            execution_path = state.get("execution_path", [])
            execution_path.append("execute_tools")

            # Create a copy of the state to avoid modifying the original
            new_state = state.copy()
            new_state["execution_path"] = execution_path
            new_state["agent_id"] = agent_id

            # Check if we have proposed tool calls or messages with tool calls
            has_tool_calls = False

            # First check for proposed_tool_calls
            if "proposed_tool_calls" in new_state and new_state["proposed_tool_calls"]:
                logger.info(
                    f"Found proposed_tool_calls in state: {new_state['proposed_tool_calls']}")
                has_tool_calls = True

            # Then check for messages with tool calls
            elif "messages" in new_state and new_state["messages"]:
                last_message = new_state["messages"][-1]
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    logger.info(
                        f"Found tool_calls in last message: {last_message.tool_calls}")
                    has_tool_calls = True

            # If we don't have tool calls, return the state as is
            if not has_tool_calls:
                logger.warning(
                    "No tool calls found in state, returning state as is")
                return new_state

            # Execute the tools
            logger.info(f"Executing tools for agent {agent_id}")

            # Create a config with the agent instance
            tool_config = {"agent_id": agent_id, "agent_instance": agent}

            # Execute the tools with the config
            result = tool_node_with_error_handling.invoke(
                new_state, config=tool_config)

            # Return the result
            return result
        except Exception as e:
            # Log the error
            logger.error(f"Error executing tools: {e}")

            # Handle the error
            return handle_tool_error(state, agent_id=agent.agent_id)

    # Create a custom node to handle the routing and execution
    def router_and_execute(state, store=None, config=None):
        """
        Router function to determine whether to execute tools or end the reasoning process.

        Args:
            state: Current state of the reasoning graph
            store: Memory store
            config: Configuration

        Returns:
            Updated state with tool execution results or the original state
        """
        # Log the state for debugging
        logger.info(
            f"Router and execute node with state keys: {list(state.keys())}")

        # Make sure we have the agent_id in the state
        agent_id = state.get("agent_id")
        if agent_id is None:
            agent_id = agent.agent_id
            logger.info(f"Using agent_id from agent: {agent_id}")

        # Create a copy of the state to avoid modifying the original
        new_state = state.copy()

        # Set the agent_id in the state
        new_state["agent_id"] = agent_id

        # Check if we should execute tools
        should_execute = should_execute_tools(new_state)

        if should_execute == "execute_tools":
            logger.info("Router decided to execute tools")
            return execute_tools_wrapper(new_state, store, config)
        else:
            logger.info("Router decided not to execute tools, returning state")
            return new_state

    # Add nodes for each reasoning step with store and config access
    builder.add_node("analyze_state", lambda state, store=None, config=None: analyze_game_state(
        state, agent.model, agent.agent_id, agent.memory, store=store, config=config))
    builder.add_node("generate_thoughts", lambda state, store=None, config=None: generate_thoughts(
        state, agent.model, agent.agent_id, agent.memory, store=store, config=config))
    builder.add_node("propose_action", lambda state, store=None, config=None: propose_action(
        state, agent.model, agent.agent_id, agent.memory, store=store, config=config))
    builder.add_node("router_and_execute", router_and_execute)

    # Add edges between nodes
    builder.add_edge(START, "analyze_state")
    builder.add_edge("analyze_state", "generate_thoughts")
    builder.add_edge("generate_thoughts", "propose_action")
    builder.add_edge("propose_action", "router_and_execute")
    builder.add_edge("router_and_execute", END)

    # Compile the graph
    return builder.compile()
