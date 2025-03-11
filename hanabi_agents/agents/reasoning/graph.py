from typing import Dict, Any, List, Optional, Callable
import logging
from langgraph.graph import StateGraph, END, START
from ..state.agent_state import AgentStateDict
from .nodes import (
    analyze_game_state,
    generate_thoughts,
    propose_action,
    execute_action,
    handle_error,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_reasoning_graph(agent):
    """
    Set up the reasoning graph for the agent.

    Args:
        agent: The agent instance

    Returns:
        The reasoning graph
    """
    # Define the graph
    graph = StateGraph(AgentStateDict)

    # Create node wrappers that pass the agent's model and agent_id
    def analyze_with_model(state, config=None):
        if config is None:
            config = {}
        # Ensure model is in config
        if "model" not in config:
            config["model"] = agent.model
        if "agent_id" not in config:
            config["agent_id"] = agent.agent_id
        if "agent_instance" not in config:
            config["agent_instance"] = agent

        # Debug logging
        logger.info(
            f"analyze_with_model called with state keys: {list(state.keys())}")
        logger.info(
            f"analyze_with_model called with config keys: {list(config.keys())}")
        logger.info(f"agent_id in config: {config.get('agent_id')}")
        logger.info(f"agent_id in state: {state.get('agent_id')}")

        return analyze_game_state(state, config)

    def generate_thoughts_with_model(state, config=None):
        if config is None:
            config = {}
        # Ensure thought_model is in config
        if "thought_model" not in config:
            config["thought_model"] = agent.thought_model
        if "model" not in config:
            config["model"] = agent.model
        if "agent_id" not in config:
            config["agent_id"] = agent.agent_id
        if "agent_instance" not in config:
            config["agent_instance"] = agent
        return generate_thoughts(state, config)

    def propose_action_with_model(state, config=None):
        if config is None:
            config = {}
        # Ensure model is in config
        if "model" not in config:
            config["model"] = agent.model
        if "agent_id" not in config:
            config["agent_id"] = agent.agent_id
        if "agent_instance" not in config:
            config["agent_instance"] = agent
        return propose_action(state, config)

    def execute_action_with_agent(state, config=None):
        if config is None:
            config = {}
        # Ensure agent_instance is in config
        if "agent_instance" not in config:
            config["agent_instance"] = agent
        if "agent_id" not in config:
            config["agent_id"] = agent.agent_id
        return execute_action(state, config)

    def handle_error_with_agent(state, config=None):
        if config is None:
            config = {}
        # Ensure agent_instance is in config
        if "agent_instance" not in config:
            config["agent_instance"] = agent
        if "agent_id" not in config:
            config["agent_id"] = agent.agent_id
        return handle_error(state, config)

    # Add nodes to the graph with the wrapped functions
    graph.add_node("analyze_game_state", analyze_with_model)
    graph.add_node("generate_thoughts", generate_thoughts_with_model)
    graph.add_node("propose_action", propose_action_with_model)
    graph.add_node("execute_action", execute_action_with_agent)
    graph.add_node("handle_error", handle_error_with_agent)

    # Define the edges - add START edge to provide an entrypoint
    graph.add_edge(START, "analyze_game_state")
    graph.add_edge("analyze_game_state", "generate_thoughts")
    graph.add_edge("generate_thoughts", "propose_action")
    graph.add_edge("propose_action", "execute_action")
    graph.add_edge("execute_action", END)

    # Add conditional edges for error handling
    def has_error(state: AgentStateDict) -> str:
        """Check if there is an error in the state."""
        return "handle_error" if state.get("error") else END

    graph.add_conditional_edges(
        "execute_action",
        has_error,
        {
            "handle_error": "handle_error",
            END: END
        }
    )

    # Add edge from error handler back to propose action
    graph.add_edge("handle_error", "propose_action")

    # Compile the graph with memory and state persistence
    # In LangGraph 0.3.5+, checkpointer may not be supported
    try:
        if agent.checkpointer:
            return graph.compile(checkpointer=agent.checkpointer)
        else:
            return graph.compile()
    except Exception as e:
        logger.warning(f"Error compiling graph with checkpointer: {e}")
        # Fall back to compiling without checkpointer
        return graph.compile()
