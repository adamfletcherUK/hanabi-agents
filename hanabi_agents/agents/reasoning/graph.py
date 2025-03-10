from typing import Dict, Any
from langgraph.graph import StateGraph, END, START
from langchain_core.tools import Tool
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import ToolMessage
from ..state.agent_state import AgentStateDict
from ..tools.play_card import play_card_tool, _play_card_impl
from ..tools.give_clue import give_clue_tool, _give_clue_impl
from ..tools.discard import discard_tool, _discard_impl
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

    # Create tool implementations that have access to agent_id and game_state
    def play_card_with_context(args):
        return _play_card_impl(agent.agent_id, args["card_index"], agent.current_game_state)

    def give_clue_with_context(args):
        return _give_clue_impl(
            agent.agent_id,
            args["target_id"],
            args["clue_type"],
            args["clue_value"],
            agent.current_game_state
        )

    def discard_with_context(args):
        return _discard_impl(agent.agent_id, args["card_index"], agent.current_game_state)

    # Define the tools with proper context
    tools = [
        Tool.from_function(
            func=play_card_with_context,
            name="play_card",
            description="Play a card from your hand",
            args_schema=play_card_tool.args_schema
        ),
        Tool.from_function(
            func=give_clue_with_context,
            name="give_clue",
            description="Give a clue to another player",
            args_schema=give_clue_tool.args_schema
        ),
        Tool.from_function(
            func=discard_with_context,
            name="discard",
            description="Discard a card from your hand",
            args_schema=discard_tool.args_schema
        )
    ]

    # Create a ToolNode with error handling
    tool_node = ToolNode(tools)
    tool_node_with_error_handling = tool_node.with_fallbacks([
        RunnableLambda(
            lambda state, error, config=None: handle_tool_error(
                state, agent_id=agent.agent_id)
        )
    ])

    # Add nodes for each reasoning step with store and config access
    builder.add_node("analyze_state", lambda state, config=None: analyze_game_state(
        state, agent.model, agent.agent_id, agent.memory, config=config))
    builder.add_node("generate_thoughts", lambda state, config=None: generate_thoughts(
        state, agent.model, agent.agent_id, agent.memory, config=config))
    builder.add_node("propose_action", lambda state, config=None: propose_action(
        state, agent.model, agent.agent_id, agent.memory, config=config))
    builder.add_node("execute_tools", tool_node_with_error_handling)

    # Add edges between nodes
    builder.add_edge(START, "analyze_state")
    builder.add_edge("analyze_state", "generate_thoughts")
    builder.add_edge("generate_thoughts", "propose_action")

    # Add conditional edge from propose_action to either execute_tools or END
    builder.add_conditional_edges(
        "propose_action",
        should_execute_tools,
        {
            "execute_tools": "execute_tools",
            "end": END
        }
    )

    # Add edge from execute_tools back to END
    builder.add_edge("execute_tools", END)

    # Compile the graph
    return builder.compile()
