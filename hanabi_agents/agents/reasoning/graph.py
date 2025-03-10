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
from .nodes import analyze_game_state, generate_thoughts, propose_action, _normalize_tool_name
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
            return {"error": "Cannot discard when at max clue tokens"}
        return _discard_impl(agent.agent_id, args["card_index"], agent.current_game_state)

    # Create the tools with the context-aware implementations
    tools = [
        Tool(
            name="play_card_tool",
            description="Play a card from your hand",
            func=play_card_with_context,
            args_schema=play_card_tool.args_schema
        ),
        Tool(
            name="give_clue_tool",
            description="Give a clue to another player",
            func=give_clue_with_context,
            args_schema=give_clue_tool.args_schema
        ),
        Tool(
            name="discard_tool",
            description="Discard a card from your hand",
            func=discard_with_context,
            args_schema=discard_tool.args_schema
        )
    ]

    # Create a tool node with the tools
    tool_node = ToolNode(tools)

    # Create an enhanced error handler that stores errors in the state
    def enhanced_error_handler(state, error, config=None):
        logger.error(f"Error in tool execution: {error}")

        # Create a copy of the state to avoid modifying the original
        new_state = state.copy()

        # Get the errors from the state or initialize an empty list
        errors = new_state.get("errors", [])

        # Add the error to the list
        errors.append(str(error))

        # Store the errors in the state
        new_state["errors"] = errors

        # Store the error in the agent's memory for persistence
        if config and "agent_instance" in config:
            agent = config["agent_instance"]
            if hasattr(agent, "store_memory"):
                agent.store_memory("action_errors", errors)

        # Return the updated state
        return new_state

    # Create a wrapper for the tool node that handles errors
    def tool_node_with_error_handling(state, config=None):
        try:
            # Get the proposed tool calls from the state
            proposed_tool_calls = state.get("proposed_tool_calls")

            # Create a copy of the state to avoid modifying the original
            new_state = state.copy()

            # If we have proposed tool calls, use them
            if proposed_tool_calls:
                logger.info(
                    f"Using proposed tool calls from state: {proposed_tool_calls}")

                # Get the first tool call
                tool_call = proposed_tool_calls[0]

                # Check if the tool call is already in the correct format
                if not isinstance(tool_call, dict) or "name" not in tool_call or "args" not in tool_call:
                    logger.warning(
                        f"Tool call is not in the correct format: {tool_call}")
                    return new_state

                # Normalize the tool name to match official names
                original_tool_name = tool_call.get("name", "")
                normalized_tool_name = _normalize_tool_name(original_tool_name)

                # Update the tool call with the normalized name if it changed
                if normalized_tool_name != original_tool_name:
                    logger.info(
                        f"Normalized tool name from '{original_tool_name}' to '{normalized_tool_name}'")
                    tool_call["name"] = normalized_tool_name
                    # Update the tool call in the state
                    proposed_tool_calls[0]["name"] = normalized_tool_name
                    new_state["proposed_tool_calls"] = proposed_tool_calls

                # Add the tool call to the messages if not already there
                messages = new_state.get("messages", [])
                has_tool_call_message = False

                for message in messages:
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        has_tool_call_message = True
                        break

                if not has_tool_call_message:
                    # Create a message with the tool call
                    from langchain_core.messages import AIMessage
                    new_state["messages"].append(AIMessage(
                        content="",
                        tool_calls=[tool_call]
                    ))

            # Execute the tool node with the updated state
            return tool_node.with_fallbacks([
                RunnableLambda(enhanced_error_handler)
            ]).invoke(new_state, config=config)
        except Exception as e:
            logger.error(f"Error in tool_node_with_error_handling: {e}")
            return enhanced_error_handler(state, e, config)

    # Add nodes for each reasoning step with store and config access
    builder.add_node("analyze_state", lambda state, config=None: analyze_game_state(
        state,
        config.get("model", agent.model) if config else agent.model,
        agent.agent_id,
        agent.memory,
        config=config
    ))
    builder.add_node("generate_thoughts", lambda state, config=None: generate_thoughts(
        state,
        config.get("model", agent.model) if config else agent.model,
        agent.agent_id,
        agent.memory,
        config=config
    ))
    builder.add_node("propose_action", lambda state, config=None: propose_action(
        state,
        config.get("model", agent.model) if config else agent.model,
        agent.agent_id,
        agent.memory,
        config=config
    ))
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
