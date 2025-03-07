from typing import List, Dict, Any
from ..game.state import GameState
from ..agents.base import Agent
from datetime import datetime
from pydantic import BaseModel
import logging

# Set up logging
logger = logging.getLogger(__name__)


class DiscussionEntry(BaseModel):
    """Represents a single entry in the discussion."""
    agent_id: int
    timestamp: datetime
    content: str
    round_number: int
    turn_number: int  # Add turn number to track which turn this entry belongs to


class DiscussionManager:
    def __init__(self, max_rounds: int = 3):
        self.max_rounds = max_rounds
        # Current turn's discussion
        self.discussion_history: List[DiscussionEntry] = []
        # Persistent game-level history
        self.game_history: List[DiscussionEntry] = []
        self.current_round = 0
        self.current_turn = 0

    def start_new_discussion(self) -> None:
        """Start a new discussion round for a new turn."""
        # Save current discussion to game history before resetting
        if self.discussion_history:
            self.game_history.extend(self.discussion_history)

        # Reset for new turn
        self.discussion_history = []
        self.current_round = 0
        self.current_turn += 1

    def add_contribution(self, agent_id: int, content: str) -> None:
        """Add a new contribution to the discussion."""
        entry = DiscussionEntry(
            agent_id=agent_id,
            timestamp=datetime.now(),
            content=content,
            round_number=self.current_round,
            turn_number=self.current_turn
        )
        self.discussion_history.append(entry)

    def get_discussion_summary(self) -> str:
        """Get a formatted summary of the current turn's discussion."""
        if not self.discussion_history:
            return "No discussion yet."

        summary = []
        current_round = -1

        for entry in self.discussion_history:
            if entry.round_number != current_round:
                current_round = entry.round_number
                summary.append(f"\nRound {current_round + 1}:")
            summary.append(f"Agent {entry.agent_id}: {entry.content}")

        return "\n".join(summary)

    def get_game_discussion_summary(self, last_n_turns: int = None) -> str:
        """Get a formatted summary of the entire game's discussion, optionally limited to last N turns."""
        if not self.game_history and not self.discussion_history:
            return "No discussion yet."

        # Combine game history with current discussion
        all_history = self.game_history + self.discussion_history

        # Filter by last N turns if specified
        if last_n_turns is not None:
            current_turn = self.current_turn
            min_turn = max(0, current_turn - last_n_turns)
            all_history = [
                entry for entry in all_history if entry.turn_number >= min_turn]

        summary = []
        current_turn = -1
        current_round = -1

        for entry in all_history:
            if entry.turn_number != current_turn:
                current_turn = entry.turn_number
                current_round = -1
                summary.append(f"\nTurn {current_turn + 1}:")

            if entry.round_number != current_round:
                current_round = entry.round_number
                summary.append(f"  Round {current_round + 1}:")

            summary.append(f"    Agent {entry.agent_id}: {entry.content}")

        return "\n".join(summary)

    def has_reached_consensus(self) -> bool:
        """Check if the discussion has reached a consensus."""
        if not self.discussion_history:
            return False

        # If we've reached max rounds, consider it a consensus
        if self.current_round >= self.max_rounds - 1:
            return True

        # Check if all agents have contributed in the current round
        current_round_contributions = [
            entry for entry in self.discussion_history
            if entry.round_number == self.current_round
        ]
        unique_agents = len(
            set(entry.agent_id for entry in current_round_contributions))

        # If all agents have contributed and there's agreement on the next action
        return unique_agents >= 4  # We have 5 agents total

    def advance_round(self) -> None:
        """Advance to the next discussion round."""
        self.current_round += 1

    def get_current_round(self) -> int:
        """Get the current discussion round number."""
        return self.current_round

    def get_current_turn(self) -> int:
        """Get the current turn number."""
        return self.current_turn

    def get_discussion_history(self) -> List[DiscussionEntry]:
        """Get the current turn's discussion history."""
        return self.discussion_history

    def get_game_history(self, last_n_turns: int = None) -> List[DiscussionEntry]:
        """Get the entire game's discussion history, optionally limited to last N turns."""
        all_history = self.game_history + self.discussion_history

        if last_n_turns is not None:
            current_turn = self.current_turn
            min_turn = max(0, current_turn - last_n_turns)
            all_history = [
                entry for entry in all_history if entry.turn_number >= min_turn]

        return all_history

    def conduct_discussion(
        self,
        agents: List[Agent],
        game_state: GameState,
        active_agent_id: int
    ) -> str:
        """Conduct a discussion among all agents about the current game state."""
        logger = logging.getLogger(__name__)
        logger.info(f"Starting discussion for Agent {active_agent_id}'s turn")

        # Start a new discussion
        self.start_new_discussion()

        # Continue discussion until consensus is reached or max rounds are hit
        while not self.has_reached_consensus() and self.current_round < self.max_rounds:
            logger.info(f"Discussion round {self.current_round + 1}")

            # Get contributions from all agents except the active one
            for agent in agents:
                if agent.agent_id != active_agent_id:
                    # Get agent's view of the game state
                    agent_view = game_state.get_view_for(agent.agent_id)

                    # Get agent's contribution
                    try:
                        contribution = agent.participate_in_discussion(
                            agent_view,
                            self.get_discussion_history()
                        )
                        self.add_contribution(agent.agent_id, contribution)
                        logger.debug(
                            f"Agent {agent.agent_id} contributed to discussion")
                    except Exception as e:
                        logger.error(
                            f"Error getting contribution from Agent {agent.agent_id}: {e}")
                        self.add_contribution(
                            agent.agent_id,
                            "I'm having trouble analyzing the current situation."
                        )

            # Check if consensus has been reached
            if self.has_reached_consensus():
                logger.info("Consensus reached in discussion")
                break

            # Move to next round
            self.advance_round()

        # Get summary of discussion
        summary = self.get_discussion_summary()
        logger.info(
            f"Discussion completed after {self.current_round + 1} rounds")

        return summary

    def _check_consensus(self, contributions: List[str]) -> bool:
        """
        Check if the agents have reached a consensus.
        This is a simplified implementation that can be enhanced.
        """
        # For now, we'll consider it a consensus if all agents made contributions
        return len(contributions) > 0 and all(c.strip() for c in contributions)

    def _summarize_discussion(self, discussion_history: List[List[str]]) -> str:
        """Create a summary of the discussion."""
        summary = []
        for round_num, round_contributions in enumerate(discussion_history, 1):
            summary.append(f"Round {round_num}:")
            for agent_id, contribution in enumerate(round_contributions):
                summary.append(f"Agent {agent_id}: {contribution}")
        return "\n".join(summary)
