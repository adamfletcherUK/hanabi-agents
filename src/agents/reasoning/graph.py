from typing import Dict, Any
from langgraph.graph import StateGraph, END
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


def setup_reasoning_graph(agent):
    """Set up the reasoning graph for the agent."""
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
                        "description": "Index of the card to play (0-4)"
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
            func=lambda card_index: discard_tool(
                agent.agent_id, card_index, agent.current_game_state),
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
        [RunnableLambda(lambda state: handle_tool_error(
            state, agent.agent_id))],
        exception_key="error"
    )

    # Add nodes for each reasoning step
    builder.add_node("analyze_state", lambda state: analyze_game_state(
        state, agent.model, agent.agent_id, agent.memory))
    builder.add_node("generate_thoughts", lambda state: generate_thoughts(
        state, agent.model, agent.agent_id, agent.memory))
    builder.add_node("propose_action", lambda state: propose_action(
        state, agent.model, agent.agent_id, agent.memory))
    builder.add_node("execute_tools", tool_node)

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
    return builder.compile()
