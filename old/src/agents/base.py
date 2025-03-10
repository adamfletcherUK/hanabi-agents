from abc import ABC, abstractmethod
from typing import Dict, Any
from ..game.state import GameState


class Agent(ABC):
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.memory: Dict[str, Any] = {}  # For storing agent's internal state

    @abstractmethod
    def participate_in_discussion(self, game_state: GameState, discussion_history: list) -> str:
        """Participate in the pre-action discussion phase."""
        pass

    @abstractmethod
    def decide_action(self, game_state: GameState, discussion_summary: str) -> Dict[str, Any]:
        """Decide on an action based on the game state and discussion summary."""
        pass

    def update_memory(self, key: str, value: Any) -> None:
        """Update the agent's internal memory."""
        self.memory[key] = value

    def get_memory(self, key: str) -> Any:
        """Retrieve information from the agent's internal memory."""
        return self.memory.get(key)

    def notify_incorrect_tool_usage(self, error_record: Dict[str, Any]) -> None:
        """
        Notify the agent of incorrect tool usage.

        This method is called by the game engine when the agent attempts to use a tool incorrectly.
        The default implementation stores the error in the agent's memory.
        Subclasses can override this method to provide more sophisticated handling.

        Args:
            error_record: A dictionary containing information about the error
        """
        # Store the error in the agent's memory
        tool_errors = self.memory.get("tool_errors", [])
        tool_errors.append(error_record)
        self.memory["tool_errors"] = tool_errors

        # Store the most recent error separately for easy access
        self.memory["last_tool_error"] = error_record
