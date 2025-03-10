from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from ..game.state import GameState


class Agent(ABC):
    """
    Base class for all Hanabi agents.

    This abstract class defines the interface that all agent implementations must follow.
    It includes methods for participating in discussions, making decisions, and managing memory.
    """

    def __init__(self, agent_id: int, name: str = None):
        """
        Initialize a new agent.

        Args:
            agent_id: Unique identifier for this agent
            name: Optional name for this agent
        """
        self.agent_id = agent_id
        self.name = name or f"Agent {agent_id}"
        self.memory: Dict[str, Any] = {}  # For storing agent's internal state

    @abstractmethod
    def participate_in_discussion(self, game_state: GameState, discussion_history: List[Dict[str, Any]]) -> str:
        """
        Participate in the pre-action discussion phase.

        Args:
            game_state: Current state of the game (filtered for this agent)
            discussion_history: History of discussion contributions

        Returns:
            The agent's contribution to the discussion
        """
        pass

    @abstractmethod
    def decide_action(self, game_state: GameState, discussion_summary: Optional[str] = None) -> Dict[str, Any]:
        """
        Decide on an action based on the game state and discussion summary.

        Args:
            game_state: Current state of the game (filtered for this agent)
            discussion_summary: Optional summary of the pre-action discussion

        Returns:
            A dictionary representing the chosen action
        """
        pass

    def update_memory(self, key: str, value: Any) -> None:
        """
        Update the agent's internal memory.

        Args:
            key: The key to store the value under
            value: The value to store
        """
        self.memory[key] = value

    def get_memory(self, key: str, default: Any = None) -> Any:
        """
        Retrieve information from the agent's internal memory.

        Args:
            key: The key to retrieve
            default: Default value to return if key is not found

        Returns:
            The stored value or the default
        """
        return self.memory.get(key, default)

    def notify_action_result(self, action: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Notify the agent of the result of its action.

        Args:
            action: The action that was taken
            result: The result of the action
        """
        # Store the action and result in memory
        action_history = self.memory.get("action_history", [])
        action_history.append({"action": action, "result": result})
        self.memory["action_history"] = action_history

        # Store the most recent action and result separately for easy access
        self.memory["last_action"] = action
        self.memory["last_result"] = result

    def notify_incorrect_action(self, action: Dict[str, Any], error: str) -> None:
        """
        Notify the agent of an incorrect action attempt.

        Args:
            action: The action that was attempted
            error: Description of why the action was incorrect
        """
        # Store the error in the agent's memory
        action_errors = self.memory.get("action_errors", [])
        error_record = {"action": action, "error": error,
                        "turn": self.memory.get("turn_count", 0)}
        action_errors.append(error_record)
        self.memory["action_errors"] = action_errors

        # Store the most recent error separately for easy access
        self.memory["last_action_error"] = error_record
