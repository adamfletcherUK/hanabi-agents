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
    def execute_tools_wrapper(state, config=None):
        """
        Wrapper function to execute tools.

        Args:
            state: Current state of the reasoning graph
            config: Configuration

        Returns:
            Updated state with tool execution results
        """
        # Log the state for debugging
        logger.info(
            f"Executing tools with state keys: {list(state.keys())}")

        # Check if we have tool calls in the state
        tool_calls = state.get("proposed_tool_calls", [])
        if not tool_calls:
            logger.warning(
                "No tool calls found in state, returning state as is")
            return state

        # Create a copy of the state to avoid modifying the original
        new_state = state.copy()

        # Execute each tool call
        for tool_call in tool_calls:
            # Extract the tool name and arguments
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})

            # Find the matching tool
            matching_tool = next(
                (tool for tool in tools if tool.name == tool_name), None)

            if matching_tool:
                try:
                    # Execute the tool
                    result = matching_tool.invoke(tool_args)

                    # Store the result in the state
                    new_state["action_result"] = result
                except Exception as e:
                    # Handle tool execution errors
                    logger.error(f"Error executing tool {tool_name}: {e}")
                    new_state["errors"] = new_state.get("errors", []) + \
                        [f"Error executing tool {tool_name}: {e}"]
            else:
                # Handle unknown tool
                logger.error(f"Unknown tool: {tool_name}")
                new_state["errors"] = new_state.get("errors", []) + \
                    [f"Unknown tool: {tool_name}"]

        return new_state

    # Create a custom node to handle the routing and execution
    def router_and_execute(state, config=None):
        """
        Router function to determine whether to execute tools or end the reasoning process.

        Args:
            state: Current state of the reasoning graph
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
            return execute_tools_wrapper(new_state, config)
        else:
            logger.info("Router decided not to execute tools, returning state")
            return new_state

    # Add nodes for each reasoning step with store and config access
    builder.add_node("analyze_state", lambda state, config=None: analyze_game_state(
        state, agent.model, agent.agent_id, agent.memory, config=config))
    builder.add_node("generate_thoughts", lambda state, config=None: generate_thoughts(
        state, agent.model, agent.agent_id, agent.memory, config=config))
    builder.add_node("propose_action", lambda state, config=None: propose_action(
        state, agent.model, agent.agent_id, agent.memory, config=config))
    builder.add_node("router_and_execute", router_and_execute)

    # Add edges between nodes
    builder.add_edge(START, "analyze_state")
    builder.add_edge("analyze_state", "generate_thoughts")
    builder.add_edge("generate_thoughts", "propose_action")
    builder.add_edge("propose_action", "router_and_execute")
    builder.add_edge("router_and_execute", END)

    # Compile the graph
    return builder.compile()
