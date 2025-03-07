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
