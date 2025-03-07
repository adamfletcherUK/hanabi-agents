from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging

# Set up logging
logger = logging.getLogger(__name__)


class Color(str, Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    BLUE = "blue"
    WHITE = "white"


class Card(BaseModel):
    color: Color
    number: int = Field(ge=1, le=5)
    is_visible: bool = False


class ScoreHistory(BaseModel):
    """Tracks the history of score changes."""
    timestamp: datetime
    score_change: int
    action_type: str
    details: str


class GameState(BaseModel):
    deck: List[Card] = Field(default_factory=list)
    hands: Dict[int, List[Card]] = Field(
        default_factory=dict)  # agent_id -> cards
    firework_piles: Dict[Color, List[Card]] = Field(default_factory=dict)
    discard_pile: List[Card] = Field(default_factory=list)
    clue_tokens: int = 8
    fuse_tokens: int = 3
    current_player: int = 0
    turn_count: int = 0
    game_over: bool = False
    score: int = 0
    score_history: List[ScoreHistory] = Field(default_factory=list)
    completed_fireworks: Dict[Color, bool] = Field(
        default_factory=lambda: {color: False for color in Color})

    def get_view_for(self, agent_id: int) -> "GameState":
        """Get a filtered view of the game state for the given agent."""
        logger.debug(f"Creating view for agent {agent_id}")

        # Create a copy of the game state
        view = GameState(
            deck=self.deck.copy(),
            hands={},
            firework_piles={color: pile.copy()
                            for color, pile in self.firework_piles.items()},
            discard_pile=self.discard_pile.copy(),
            clue_tokens=self.clue_tokens,
            fuse_tokens=self.fuse_tokens,
            current_player=self.current_player,
            turn_count=self.turn_count,
            game_over=self.game_over,
            score=self.score,
            score_history=self.score_history.copy(),
            completed_fireworks={color: completed for color,
                                 completed in self.completed_fireworks.items()}
        )

        # Copy all hands except the agent's own hand
        for player_id, hand in self.hands.items():
            if player_id == agent_id:
                # For the agent's own hand, create cards with visibility based on clues
                view.hands[player_id] = hand.copy()
            else:
                # For other players' hands, show all cards
                view.hands[player_id] = hand.copy()

        logger.debug(f"View created for agent {agent_id}")
        return view

    def is_valid_move(self, agent_id: int, action_type: str, **kwargs) -> bool:
        """Validates if a move is legal according to Hanabi rules."""
        # Basic game state checks
        if self.game_over:
            return False

        if agent_id != self.current_player:
            return False

        # Action-specific validation
        if action_type == "give_clue":
            return self._validate_clue_action(agent_id, **kwargs)
        elif action_type == "play_card":
            return self._validate_play_action(agent_id, **kwargs)
        elif action_type == "discard":
            return self._validate_discard_action(agent_id, **kwargs)
        else:
            return False

    def _validate_clue_action(self, agent_id: int, **kwargs) -> bool:
        """Validates if a clue action is legal."""
        # Check if we have clue tokens
        if self.clue_tokens <= 0:
            return False

        # Extract clue parameters
        target_id = kwargs.get("target_id")
        clue = kwargs.get("clue", {})
        clue_type = clue.get("type")
        value = clue.get("value")

        # Check if target is valid
        if target_id is None or target_id == agent_id or target_id not in self.hands:
            return False

        # Check if clue type is valid
        if clue_type not in ["color", "number"]:
            return False

        # Check if clue value is valid
        if clue_type == "color" and value not in [c.value for c in Color]:
            return False

        if clue_type == "number":
            # Convert value to int if it's a string
            try:
                if isinstance(value, str):
                    value = int(value)
                if not (1 <= value <= 5):
                    return False
            except (ValueError, TypeError):
                return False

        # Check if the clue applies to at least one card
        target_hand = self.hands[target_id]
        applies_to_any_card = False

        for card in target_hand:
            if clue_type == "color" and card.color.value == value:
                applies_to_any_card = True
                break
            elif clue_type == "number":
                # Convert value to int if it's a string for comparison
                card_value = card.number
                clue_value = int(value) if isinstance(value, str) else value
                if card_value == clue_value:
                    applies_to_any_card = True
                    break

        return applies_to_any_card

    def _validate_play_action(self, agent_id: int, **kwargs) -> bool:
        """Validate a card-playing action."""
        # Check card index
        card_index = kwargs.get("card_index")
        if card_index is None or not isinstance(card_index, int):
            return False

        # Check if card exists in hand
        if not (0 <= card_index < len(self.hands[agent_id])):
            return False

        # Check if firework is already complete
        card = self.hands[agent_id][card_index]
        if self.completed_fireworks[card.color]:
            return False

        # Check if the play would be valid
        firework_pile = self.firework_piles[card.color]
        next_number = len(firework_pile) + 1
        if card.number != next_number:
            return False

        return True

    def _validate_discard_action(self, agent_id: int, **kwargs) -> bool:
        """Validate a card-discarding action."""
        # Check clue tokens
        if self.clue_tokens >= 8:
            return False

        # Check card index
        card_index = kwargs.get("card_index")
        if card_index is None or not isinstance(card_index, int):
            return False

        # Check if card exists in hand
        if not (0 <= card_index < len(self.hands[agent_id])):
            return False

        return True

    def _card_matches_clue(self, card: Card, clue: Dict[str, Any]) -> bool:
        """Check if a card matches the given clue."""
        logger.debug(
            f"Checking if card {card.color} {card.number} matches clue {clue}")
        if clue["type"] == "color":
            matches = card.color.value == clue["value"]
            logger.debug(f"Color match: {matches}")
            return matches
        else:  # number clue
            matches = card.number == clue["value"]
            logger.debug(f"Number match: {matches}")
            return matches

    def update_score(self, points: int, action_type: str, details: str) -> None:
        """Update the game score and record the change in history."""
        logger.info(
            f"Updating score: {points} points for {action_type} - {details}")
        self.score += points
        self.score_history.append(ScoreHistory(
            timestamp=datetime.now(),
            score_change=points,
            action_type=action_type,
            details=details
        ))
        logger.debug(
            f"New score: {self.score}, History length: {len(self.score_history)}")

    def check_firework_completion(self, color: Color) -> bool:
        """Check if a firework is complete and update state if it is."""
        logger.debug(f"Checking firework completion for {color}")
        if self.completed_fireworks[color]:
            logger.debug(f"Firework {color} already completed")
            return True

        firework_pile = self.firework_piles[color]
        if len(firework_pile) == 5:
            logger.info(f"Firework {color} completed!")
            self.completed_fireworks[color] = True
            return True
        logger.debug(
            f"Firework {color} not complete yet ({len(firework_pile)}/5)")
        return False

    def get_completed_fireworks_count(self) -> int:
        """Get the number of completed fireworks."""
        return sum(1 for completed in self.completed_fireworks.values() if completed)

    def get_max_possible_score(self) -> int:
        """Calculate the maximum possible score based on completed fireworks."""
        return len(Color) * 5  # 5 points per color

    def get_score_percentage(self) -> float:
        """Calculate the score as a percentage of the maximum possible score."""
        max_score = self.get_max_possible_score()
        return (self.score / max_score) * 100 if max_score > 0 else 0.0

    def get_score_summary(self) -> Dict[str, Any]:
        """Get a summary of the current game scoring state."""
        return {
            "current_score": self.score,
            "completed_fireworks": self.get_completed_fireworks_count(),
            "max_possible_score": self.get_max_possible_score(),
            "score_percentage": self.get_score_percentage(),
            "score_history": [
                {
                    "timestamp": entry.timestamp.isoformat(),
                    "change": entry.score_change,
                    "action": entry.action_type,
                    "details": entry.details
                }
                for entry in self.score_history
            ]
        }
