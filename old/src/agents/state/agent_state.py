from typing import Dict, Any, List
from typing_extensions import TypedDict
from ...game.state import GameState


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
