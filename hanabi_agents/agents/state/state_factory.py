from typing import Dict, Any, List
from ...game.state import GameState
from .agent_state import AgentState


def create_initial_state(
    game_state: GameState,
    agent_id: int,
    discussion_history: List[Dict[str, Any]] = None,
    game_history: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create the initial state for an agent's reasoning graph.

    Args:
        game_state: Current state of the game
        agent_id: ID of the agent
        discussion_history: History of discussion contributions
        game_history: History of game actions

    Returns:
        Dictionary representation of the initial state
    """
    # Create a filtered view of the game state for this agent
    filtered_game_state = game_state.get_view_for(agent_id)

    # Create the agent state
    agent_state = AgentState(
        game_state=filtered_game_state,
        agent_id=agent_id,
        discussion_history=discussion_history or [],
        game_history=game_history or []
    )

    # Initialize card knowledge based on the game state
    agent_state.card_knowledge = _initialize_card_knowledge(
        filtered_game_state, agent_id)

    return agent_state.dict()


def create_action_state(
    game_state: GameState,
    agent_id: int,
    discussion_summary: str = None,
    agent_memory=None
) -> Dict[str, Any]:
    """
    Create a state for action decision-making.

    Args:
        game_state: Current state of the game
        agent_id: ID of the agent
        discussion_summary: Summary of the pre-action discussion
        agent_memory: Optional agent memory object to include proposed tool calls

    Returns:
        Dictionary representation of the action state
    """
    # Create a filtered view of the game state for this agent
    filtered_game_state = game_state.get_view_for(agent_id)

    # Create the agent state
    agent_state = AgentState(
        game_state=filtered_game_state,
        agent_id=agent_id
    )

    # Initialize card knowledge based on the game state
    agent_state.card_knowledge = _initialize_card_knowledge(
        filtered_game_state, agent_id)

    # Add the discussion summary if provided
    if discussion_summary:
        agent_state.discussion_history = [{"summary": discussion_summary}]

    # Add a flag to indicate this is the action phase
    agent_state.is_action_phase = True

    # Include proposed tool calls from agent memory if available
    if agent_memory:
        if hasattr(agent_memory, "get_memory"):
            # If it's an AgentMemory object
            proposed_tool_calls = agent_memory.get_memory(
                "proposed_tool_calls")
            if proposed_tool_calls:
                agent_state.proposed_tool_calls = proposed_tool_calls
        elif isinstance(agent_memory, dict) and "proposed_tool_calls" in agent_memory:
            # If it's a dictionary
            agent_state.proposed_tool_calls = agent_memory["proposed_tool_calls"]

    return agent_state.dict()


def _initialize_card_knowledge(game_state: GameState, agent_id: int) -> List[Dict[str, Any]]:
    """
    Initialize the agent's knowledge about cards in their hand.

    Args:
        game_state: Filtered game state for this agent
        agent_id: ID of the agent

    Returns:
        List of knowledge about each card in the agent's hand
    """
    # Get the agent's hand
    hand = game_state.hands.get(agent_id, [])

    # Initialize knowledge for each card
    card_knowledge = []
    for i, card in enumerate(hand):
        knowledge = {
            "index": i,
            "color_clued": card.color_clued,
            "number_clued": card.number_clued,
            "possible_colors": [c.value for c in game_state.get_possible_colors(agent_id, i)],
            "possible_numbers": game_state.get_possible_numbers(agent_id, i)
        }
        card_knowledge.append(knowledge)

    return card_knowledge
