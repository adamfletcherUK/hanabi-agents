from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
from ...game.state import GameState


class AgentStateDict(TypedDict, total=False):
    """
    State schema for the agent reasoning graph.

    This TypedDict defines the structure of the state that flows through the LangGraph.
    It includes all the information needed for the agent to reason and make decisions.
    """
    # Core game information
    game_state: GameState
    agent_id: int

    # Discussion and history
    discussion_history: List[Dict[str, Any]]
    game_history: List[Dict[str, Any]]

    # Reasoning components
    current_thoughts: List[str]
    card_knowledge: List[Dict[str, Any]]

    # Action components
    proposed_action: Optional[Dict[str, Any]]
    action_result: Optional[Dict[str, Any]]

    # Tool execution
    messages: List[Any]
    proposed_tool_calls: List[Dict[str, Any]]

    # Error handling
    errors: List[Dict[str, Any]]

    # Execution tracking
    execution_path: List[str]


class AgentState:
    """
    State class for the agent reasoning graph.

    This class provides a more structured interface for working with the agent state,
    with methods for converting to and from dictionaries for LangGraph compatibility.
    """

    def __init__(
        self,
        game_state: GameState,
        agent_id: int,
        discussion_history: List[Dict[str, Any]] = None,
        game_history: List[Dict[str, Any]] = None
    ):
        """
        Initialize a new agent state.

        Args:
            game_state: Current state of the game
            agent_id: ID of the agent
            discussion_history: History of discussion contributions
            game_history: History of game actions
        """
        self.game_state = game_state
        self.agent_id = agent_id
        self.discussion_history = discussion_history or []
        self.game_history = game_history or []
        self.current_thoughts = []
        self.card_knowledge = []
        self.proposed_action = None
        self.action_result = None
        self.messages = []
        self.proposed_tool_calls = []
        self.errors = []
        self.execution_path = []

    def dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for LangGraph compatibility.

        Returns:
            Dictionary representation of the state
        """
        return {
            "game_state": self.game_state,
            "agent_id": self.agent_id,
            "discussion_history": self.discussion_history,
            "game_history": self.game_history,
            "current_thoughts": self.current_thoughts,
            "card_knowledge": self.card_knowledge,
            "proposed_action": self.proposed_action,
            "action_result": self.action_result,
            "messages": self.messages,
            "proposed_tool_calls": self.proposed_tool_calls,
            "errors": self.errors,
            "execution_path": self.execution_path
        }

    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> "AgentState":
        """
        Create AgentState from dictionary.

        Args:
            state_dict: Dictionary representation of the state

        Returns:
            New AgentState instance
        """
        agent_state = cls(
            game_state=state_dict["game_state"],
            agent_id=state_dict["agent_id"],
            discussion_history=state_dict.get("discussion_history", []),
            game_history=state_dict.get("game_history", [])
        )
        agent_state.current_thoughts = state_dict.get("current_thoughts", [])
        agent_state.card_knowledge = state_dict.get("card_knowledge", [])
        agent_state.proposed_action = state_dict.get("proposed_action")
        agent_state.action_result = state_dict.get("action_result")
        agent_state.messages = state_dict.get("messages", [])
        agent_state.proposed_tool_calls = state_dict.get(
            "proposed_tool_calls", [])
        agent_state.errors = state_dict.get("errors", [])
        agent_state.execution_path = state_dict.get("execution_path", [])
        return agent_state
