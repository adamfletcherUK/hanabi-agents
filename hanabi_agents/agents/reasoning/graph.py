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
        # Add explicit check for max clue tokens before calling the implementation
        if agent.current_game_state.clue_tokens >= agent.current_game_state.max_clue_tokens:
            logger.warning(
                f"Agent {agent.agent_id} attempted to discard when at max clue tokens")
            return {
                "success": False,
                "error": f"Cannot discard when clue tokens are at maximum ({agent.current_game_state.max_clue_tokens})",
                "action_type": "discard",
                "guidance": "When at max clue tokens, you must either play a card or give a clue instead of discarding."
            }
        return _discard_impl(agent.agent_id, args["card_index"], agent.current_game_state)

    # Define the tools with proper context
    tools = [
        Tool.from_function(
            func=play_card_with_context,
            name="play_card_tool",
            description="Play a card from your hand",
            args_schema=play_card_tool.args_schema
        ),
        Tool.from_function(
            func=give_clue_with_context,
            name="give_clue_tool",
            description="Give a clue to another player",
            args_schema=give_clue_tool.args_schema
        ),
        Tool.from_function(
            func=discard_with_context,
            name="discard_tool",
            description="Discard a card from your hand (only when not at max clue tokens)",
            args_schema=discard_tool.args_schema
        )
    ]

    # Create a ToolNode with error handling
    tool_node = ToolNode(tools)

    # Enhanced error handling function
    def enhanced_error_handler(state, error, config=None):
        """Enhanced error handler that provides better guidance for common errors"""
        logger.error(f"Tool error: {error}")

        # Create a copy of the state to avoid modifying the original
        new_state = state.copy()

        # Add the error to the state
        errors = new_state.get("errors", [])

        # Check if the error is about max clue tokens
        if "max" in str(error).lower() and "clue" in str(error).lower() and "token" in str(error).lower():
            error_info = {
                "action_type": "discard",
                "error": str(error),
                "guidance": "When at max clue tokens, you must either play a card or give a clue instead of discarding."
            }
        else:
            # Generic error handling
            error_info = handle_tool_error(state, agent_id=agent.agent_id)

        errors.append(error_info)
        new_state["errors"] = errors

        # Add a message about the error
        messages = new_state.get("messages", [])
        messages.append(ToolMessage(
            content=f"Error: {error_info.get('error')}. {error_info.get('guidance', '')}",
            tool_call_id="error"
        ))
        new_state["messages"] = messages

        return new_state

    tool_node_with_error_handling = tool_node.with_fallbacks([
        RunnableLambda(enhanced_error_handler)
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
