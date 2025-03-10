from enum import Enum
from typing import List, Dict, Optional, Any, Set
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
import logging

# Set up logging
logger = logging.getLogger(__name__)


class Color(str, Enum):
    """Represents the colors of cards in Hanabi."""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    BLUE = "blue"
    WHITE = "white"


class Card(BaseModel):
    """Represents a card in the game of Hanabi."""
    color: Color
    number: int = Field(ge=1, le=5)
    # Visibility flags for the player who owns this card
    is_visible: bool = False
    color_clued: bool = False
    number_clued: bool = False

    def __str__(self) -> str:
        """String representation of the card."""
        return f"{self.color.value} {self.number}"


class ClueAction(BaseModel):
    """Represents a clue given by a player."""
    type: str = Field(..., pattern="^(color|number)$")
    value: Any

    @field_validator('value')
    def validate_value(cls, v, info):
        """Validate that the clue value is appropriate for the clue type."""
        field_name = info.field_name
        if field_name == 'value':
            clue_type = info.data.get('type')
            if clue_type == 'color':
                if v not in [c.value for c in Color]:
                    raise ValueError(f"Invalid color value: {v}")
            elif clue_type == 'number':
                try:
                    num_value = int(v) if isinstance(v, str) else v
                    if not (1 <= num_value <= 5):
                        raise ValueError(
                            f"Number must be between 1 and 5, got {v}")
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid number value: {v}")
        return v


class ScoreHistory(BaseModel):
    """Tracks the history of score changes."""
    timestamp: datetime
    score_change: int
    action_type: str
    details: str


class ClueHistory(BaseModel):
    """Tracks the history of clues given."""
    timestamp: datetime
    giver_id: int
    receiver_id: int
    clue_type: str
    clue_value: Any
    affected_indices: List[int]


class GameState(BaseModel):
    """Represents the complete state of a Hanabi game."""
    # Game components
    deck: List[Card] = Field(default_factory=list)
    hands: Dict[int, List[Card]] = Field(
        default_factory=dict)  # agent_id -> cards
    firework_piles: Dict[Color, List[Card]] = Field(default_factory=dict)
    discard_pile: List[Card] = Field(default_factory=list)

    # Game resources
    clue_tokens: int = 8
    max_clue_tokens: int = 8
    fuse_tokens: int = 3

    # Game state tracking
    current_player: int = 0
    turn_count: int = 0
    game_over: bool = False
    score: int = 0

    # History tracking
    score_history: List[ScoreHistory] = Field(default_factory=list)
    clue_history: List[ClueHistory] = Field(default_factory=list)

    # Game completion tracking
    completed_fireworks: Dict[Color, bool] = Field(
        default_factory=lambda: {color: False for color in Color})

    # Last action tracking
    last_action: Optional[Dict[str, Any]] = None
    last_action_result: Optional[Dict[str, Any]] = None

    def get_view_for(self, agent_id: int) -> "GameState":
        """
        Get a filtered view of the game state for the given agent.

        This creates a copy of the game state where the agent's own cards
        have visibility based on clues received, while other players' cards
        are fully visible.

        Args:
            agent_id: The ID of the agent to create the view for

        Returns:
            A filtered GameState object
        """
        logger.debug(f"Creating view for agent {agent_id}")

        # Create a copy of the game state
        view = GameState(
            deck=[],  # Hide the deck contents
            hands={},
            firework_piles={color: pile.copy()
                            for color, pile in self.firework_piles.items()},
            discard_pile=self.discard_pile.copy(),
            clue_tokens=self.clue_tokens,
            max_clue_tokens=self.max_clue_tokens,
            fuse_tokens=self.fuse_tokens,
            current_player=self.current_player,
            turn_count=self.turn_count,
            game_over=self.game_over,
            score=self.score,
            score_history=self.score_history.copy(),
            clue_history=self.clue_history.copy(),
            completed_fireworks={color: completed for color,
                                 completed in self.completed_fireworks.items()},
            last_action=self.last_action.copy() if self.last_action else None,
            last_action_result=self.last_action_result.copy() if self.last_action_result else None
        )

        # Copy all hands except the agent's own hand
        for player_id, hand in self.hands.items():
            if player_id == agent_id:
                # For the agent's own hand, create cards with visibility based on clues
                view.hands[player_id] = [
                    Card(
                        color=card.color,
                        number=card.number,
                        is_visible=False,  # The agent can't see their own cards
                        color_clued=card.color_clued,
                        number_clued=card.number_clued
                    ) for card in hand
                ]
            else:
                # For other players' hands, show all cards
                view.hands[player_id] = hand.copy()

        logger.debug(f"View created for agent {agent_id}")
        return view

    def is_valid_move(self, agent_id: int, action_type: str, **kwargs) -> bool:
        """
        Validates if a move is legal according to Hanabi rules.

        Args:
            agent_id: The ID of the agent making the move
            action_type: The type of action ("play_card", "give_clue", "discard")
            **kwargs: Additional parameters specific to the action type

        Returns:
            True if the move is valid, False otherwise
        """
        # Basic game state checks
        if self.game_over:
            logger.warning(f"Game is over, no moves are valid")
            return False

        if agent_id != self.current_player:
            logger.warning(
                f"Not agent {agent_id}'s turn, current player is {self.current_player}")
            return False

        # Action-specific validation
        if action_type == "give_clue":
            return self._validate_clue_action(agent_id, **kwargs)
        elif action_type == "play_card":
            return self._validate_play_action(agent_id, **kwargs)
        elif action_type == "discard":
            return self._validate_discard_action(agent_id, **kwargs)
        else:
            logger.warning(f"Unknown action type: {action_type}")
            return False

    def _validate_clue_action(self, agent_id: int, **kwargs) -> bool:
        """
        Validates if a clue action is legal.

        Args:
            agent_id: The ID of the agent giving the clue
            **kwargs: Must contain "target_id" and "clue" parameters

        Returns:
            True if the clue action is valid, False otherwise
        """
        # Check if we have clue tokens
        if self.clue_tokens <= 0:
            logger.warning("No clue tokens available")
            return False

        # Extract clue parameters
        target_id = kwargs.get("target_id")
        clue = kwargs.get("clue", {})

        # Validate clue format
        try:
            clue_obj = ClueAction(**clue)
            clue_type = clue_obj.type
            value = clue_obj.value
        except Exception as e:
            logger.warning(f"Invalid clue format: {e}")
            return False

        # Check if target is valid
        if target_id is None or target_id == agent_id or target_id not in self.hands:
            logger.warning(f"Invalid target ID: {target_id}")
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

        if not applies_to_any_card:
            logger.warning(
                f"Clue {clue_type}={value} doesn't apply to any cards in target's hand")
            return False

        return True

    def _validate_play_action(self, agent_id: int, **kwargs) -> bool:
        """
        Validate a card-playing action.

        Args:
            agent_id: The ID of the agent playing the card
            **kwargs: Must contain "card_index" parameter

        Returns:
            True if the play action is valid, False otherwise
        """
        # Check card index
        card_index = kwargs.get("card_index")
        if card_index is None or not isinstance(card_index, int):
            logger.warning(f"Invalid card index: {card_index}")
            return False

        # Check if card exists in hand
        if not (0 <= card_index < len(self.hands[agent_id])):
            logger.warning(
                f"Card index {card_index} out of range for hand size {len(self.hands[agent_id])}")
            return False

        # In the new implementation, we allow any card to be played
        # The consequences (success or failure) are determined during execution
        return True

    def _validate_discard_action(self, agent_id: int, **kwargs) -> bool:
        """
        Validate a card-discarding action.

        Args:
            agent_id: The ID of the agent discarding the card
            **kwargs: Must contain "card_index" parameter

        Returns:
            True if the discard action is valid, False otherwise
        """
        # Check if we're at max clue tokens (can't discard if we are)
        if self.clue_tokens >= self.max_clue_tokens:
            logger.warning(
                f"Cannot discard when at maximum clue tokens ({self.clue_tokens})")
            return False

        # Check card index
        card_index = kwargs.get("card_index")
        if card_index is None or not isinstance(card_index, int):
            logger.warning(f"Invalid card index: {card_index}")
            return False

        # Check if card exists in hand
        if not (0 <= card_index < len(self.hands[agent_id])):
            logger.warning(
                f"Card index {card_index} out of range for hand size {len(self.hands[agent_id])}")
            return False

        return True

    def add_clue_event(self, giver_id: int, receiver_id: int, clue_type: str, clue_value: Any, affected_indices: List[int]) -> None:
        """
        Add a clue event to the history.

        Args:
            giver_id: The ID of the agent giving the clue
            receiver_id: The ID of the agent receiving the clue
            clue_type: The type of clue ("color" or "number")
            clue_value: The value of the clue
            affected_indices: The indices of cards affected by the clue
        """
        self.clue_history.append(ClueHistory(
            timestamp=datetime.now(),
            giver_id=giver_id,
            receiver_id=receiver_id,
            clue_type=clue_type,
            clue_value=clue_value,
            affected_indices=affected_indices
        ))
        logger.debug(
            f"Added clue event: {giver_id} -> {receiver_id}, {clue_type}={clue_value}, affected: {affected_indices}")

    def add_score_event(self, action_type: str, details: str, score_change: int = 1) -> None:
        """
        Add a score event to the history.

        Args:
            action_type: The type of action that led to the score change
            details: Details about the score change
            score_change: The amount the score changed (default: 1)
        """
        self.score_history.append(ScoreHistory(
            timestamp=datetime.now(),
            score_change=score_change,
            action_type=action_type,
            details=details
        ))
        logger.debug(
            f"Added score event: {action_type}, {details}, change: {score_change}")

    def check_firework_completion(self, color: Color) -> bool:
        """
        Check if a firework is complete and update state if it is.

        Args:
            color: The color of the firework to check

        Returns:
            True if the firework is complete, False otherwise
        """
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
        """
        Get the number of completed fireworks.

        Returns:
            The number of completed fireworks
        """
        return sum(1 for completed in self.completed_fireworks.values() if completed)

    def get_max_possible_score(self) -> int:
        """
        Calculate the maximum possible score based on the number of colors.

        Returns:
            The maximum possible score (5 points per color)
        """
        return len(Color) * 5  # 5 points per color

    def get_score_percentage(self) -> float:
        """
        Calculate the score as a percentage of the maximum possible score.

        Returns:
            The score as a percentage
        """
        max_score = self.get_max_possible_score()
        return (self.score / max_score) * 100 if max_score > 0 else 0

    def get_score_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current score.

        Returns:
            A dictionary containing score information
        """
        return {
            "score": self.score,
            "max_score": self.get_max_possible_score(),
            "percentage": self.get_score_percentage(),
            "completed_fireworks": self.get_completed_fireworks_count(),
            "firework_status": {color.value: len(self.firework_piles[color]) for color in Color}
        }

    def get_remaining_cards(self) -> Dict[str, int]:
        """
        Get a count of remaining cards in the deck.

        Returns:
            A dictionary mapping card descriptions to counts
        """
        card_counts = {}
        for card in self.deck:
            key = f"{card.color.value} {card.number}"
            card_counts[key] = card_counts.get(key, 0) + 1
        return card_counts

    def get_discarded_cards(self) -> Dict[str, int]:
        """
        Get a count of discarded cards.

        Returns:
            A dictionary mapping card descriptions to counts
        """
        card_counts = {}
        for card in self.discard_pile:
            key = f"{card.color.value} {card.number}"
            card_counts[key] = card_counts.get(key, 0) + 1
        return card_counts

    def get_played_cards(self) -> Dict[str, int]:
        """
        Get a count of played cards.

        Returns:
            A dictionary mapping card descriptions to counts
        """
        card_counts = {}
        for color in Color:
            for card in self.firework_piles[color]:
                key = f"{card.color.value} {card.number}"
                card_counts[key] = card_counts.get(key, 0) + 1
        return card_counts

    def get_card_knowledge(self, agent_id: int) -> List[Dict[str, Any]]:
        """
        Get the knowledge an agent has about their own cards.

        Args:
            agent_id: The ID of the agent

        Returns:
            A list of dictionaries containing card knowledge
        """
        knowledge = []
        for i, card in enumerate(self.hands[agent_id]):
            card_info = {
                "index": i,
                "color_clued": card.color_clued,
                "number_clued": card.number_clued,
                "known_color": card.color.value if card.color_clued else None,
                "known_number": card.number if card.number_clued else None
            }
            knowledge.append(card_info)
        return knowledge
