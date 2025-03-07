import random
import logging
from typing import List, Dict, Any
from .state import GameState, Card, Color
from ..agents.base import Agent
from ..communication.discussion import DiscussionManager

# Set up logging
logger = logging.getLogger(__name__)


class GameEngine:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.logger = logger  # Store reference to the logger
        self.state = self._initialize_game()
        self.discussion_manager = DiscussionManager()
        self._setup_logging()
        logger.info("Game engine initialized successfully")

    def _setup_logging(self):
        """Set up logging for the game engine."""
        # Already using the module-level logger, just add a handler if needed
        pass

    def _initialize_game(self) -> GameState:
        """Initialize a new game of Hanabi."""
        logger.debug("Creating and shuffling deck")
        # Create and shuffle deck
        deck = []
        for color in Color:
            # Add cards with numbers 1-5
            # 1s appear three times
            for _ in range(3):
                deck.append(Card(color=color, number=1))
            # 2s, 3s, 4s appear twice
            for number in range(2, 5):
                for _ in range(2):
                    deck.append(Card(color=color, number=number))
            # 5s appear once
            deck.append(Card(color=color, number=5))

        # Shuffle the deck
        random.shuffle(deck)
        self.logger.debug(f"Deck created with {len(deck)} cards and shuffled")

        # Initialize game state
        state = GameState()
        state.deck = deck

        # Initialize firework piles
        for color in Color:
            state.firework_piles[color] = []
            state.completed_fireworks[color] = False

        # Deal cards to players
        for i in range(len(self.agents)):
            state.hands[i] = []
            for _ in range(5):  # Each player gets 5 cards
                if deck:
                    card = deck.pop()
                    state.hands[i].append(card)

        self.logger.debug(f"Cards dealt. Remaining deck size: {len(deck)}")
        return state

    def play_game(self) -> int:
        """Play a complete game of Hanabi."""
        self.logger.info("Starting new game of Hanabi")

        # Main game loop
        while not self.state.game_over:
            self._play_turn()

            # Check for game over conditions
            if self.state.fuse_tokens <= 0:
                self.logger.info("Game over: All fuse tokens used")
                self.state.game_over = True

            if len(self.state.deck) == 0:
                self.logger.info("Game over: Deck is empty")
                self.state.game_over = True

        # Calculate final score
        final_score = self.state.score
        self.logger.info(f"Game finished with score: {final_score}")
        return final_score

    def _play_turn(self):
        """Play a single turn of the game."""
        active_agent_id = self.state.current_player
        active_agent = self.agents[active_agent_id]

        self.logger.info(
            f"Turn {self.state.turn_count + 1}: Agent {active_agent_id}'s turn")

        # Conduct discussion
        discussion_summary = self.discussion_manager.conduct_discussion(
            self.agents, self.state, active_agent_id)

        # Get action from active agent
        agent_view = self.state.get_view_for(active_agent_id)
        try:
            action = active_agent.decide_action(agent_view, discussion_summary)
            self.logger.info(f"Agent {active_agent_id} chose action: {action}")
        except Exception as e:
            self.logger.critical(
                f"CRITICAL ERROR: Agent {active_agent_id} failed to propose a valid action: {e}")
            self.logger.critical("Game terminated due to critical error")
            self.state.game_over = True
            raise RuntimeError(f"Game terminated due to critical error: {e}")

        # Validate and execute action
        if self.execute_action(active_agent.agent_id, action):
            self._update_game_state()
        else:
            self.logger.error(
                f"Invalid action attempted by Agent {active_agent.agent_id}")
            self.logger.critical("Game terminated due to invalid action")
            self.state.game_over = True
            raise RuntimeError(
                f"Game terminated due to invalid action: {action}")

    def execute_action(self, agent_id: int, action: Dict[str, Any]) -> bool:
        """Execute an action for the given agent."""
        logger.info(f"Executing action for agent {agent_id}: {action}")
        logger.debug(
            f"Before action - Current player: {self.state.current_player}, Turn count: {self.state.turn_count}")

        # Validate the action
        try:
            if not self.state.is_valid_move(agent_id, action["type"], **action):
                # Enhanced error logging to explain why the action is invalid
                action_type = action["type"]
                error_message = f"Invalid {action_type} action by Agent {agent_id}"

                if action_type == "give_clue":
                    target_id = action.get("target_id")
                    clue = action.get("clue", {})
                    clue_type = clue.get("type")
                    clue_value = clue.get("value")

                    if self.state.clue_tokens <= 0:
                        error_message = f"Invalid clue action: No clue tokens available"
                    elif target_id == agent_id:
                        error_message = f"Invalid clue action: Cannot give clue to yourself"
                    elif target_id not in self.state.hands:
                        error_message = f"Invalid clue action: Target player {target_id} does not exist"
                    elif not clue or not isinstance(clue, dict):
                        error_message = f"Invalid clue action: Malformed clue format"
                    elif not clue_type or not clue_value:
                        error_message = f"Invalid clue action: Missing clue type or value"
                    elif clue_type not in ["color", "number"]:
                        error_message = f"Invalid clue action: Invalid clue type '{clue_type}'"
                    elif clue_type == "color" and clue_value not in [c.value for c in Color]:
                        error_message = f"Invalid clue action: Invalid color value '{clue_value}'"
                    elif clue_type == "number":
                        try:
                            # Convert value to int if it's a string for comparison
                            clue_value_int = int(clue_value) if isinstance(
                                clue_value, str) else clue_value
                            if not (1 <= clue_value_int <= 5):
                                error_message = f"Invalid clue action: Invalid number value '{clue_value}'"
                        except (ValueError, TypeError):
                            error_message = f"Invalid clue action: Invalid number value '{clue_value}'"
                    else:
                        # Check if the clue matches any cards in target's hand
                        target_hand = self.state.hands[target_id]
                        matches = [i for i, card in enumerate(target_hand)
                                   if self._card_matches_clue(card, clue)]
                        if not matches:
                            error_message = f"Invalid clue action: No {clue_type} {clue_value} cards in player {target_id}'s hand"

                elif action_type == "play_card":
                    card_index = action.get("card_index")
                    if card_index is None or not isinstance(card_index, int):
                        error_message = f"Invalid play action: Invalid card index format"
                    elif not (0 <= card_index < len(self.state.hands[agent_id])):
                        error_message = f"Invalid play action: Card index {card_index} out of range"
                    else:
                        card = self.state.hands[agent_id][card_index]
                        if self.state.completed_fireworks[card.color]:
                            error_message = f"Invalid play action: Firework for {card.color} is already complete"
                        else:
                            firework_pile = self.state.firework_piles[card.color]
                            next_number = len(firework_pile) + 1
                            error_message = f"Invalid play action: Card {card.color} {card.number} cannot be played. Next needed card is {card.color} {next_number}"

                elif action_type == "discard":
                    card_index = action.get("card_index")
                    if self.state.clue_tokens >= 8:
                        error_message = f"Invalid discard action: Clue tokens already at maximum (8)"
                    elif card_index is None or not isinstance(card_index, int):
                        error_message = f"Invalid discard action: Invalid card index format"
                    elif not (0 <= card_index < len(self.state.hands[agent_id])):
                        error_message = f"Invalid discard action: Card index {card_index} out of range"
                else:
                    error_message = f"Invalid action: Unknown action type '{action_type}'"

            logger.error(error_message)
            # Instead of returning False, raise an exception
            raise ValueError(error_message)

            # Execute the action
            action_type = action["type"]
            logger.debug(f"Processing {action_type} action")

            if action_type == "play_card":
                success = self._execute_play_card(
                    agent_id, action["card_index"])
            elif action_type == "give_clue":
                success = self._execute_give_clue(
                    agent_id, action["target_id"], action["clue"])
            elif action_type == "discard":
                success = self._execute_discard(agent_id, action["card_index"])
            else:
                logger.error(f"Unknown action type: {action_type}")
                raise ValueError(f"Unknown action type: {action_type}")

            if success:
                logger.info(f"Action executed successfully: {action}")
                # Update current player and turn count
                old_player = self.state.current_player
                old_turn = self.state.turn_count
                self.state.current_player = (
                    self.state.current_player + 1) % len(self.agents)
                self.state.turn_count += 1
                logger.debug(
                    f"Player advanced: {old_player} -> {self.state.current_player}, Turn count: {old_turn} -> {self.state.turn_count}")
            else:
                logger.warning(f"Action execution failed: {action}")
            # Instead of returning False, raise an exception
            raise ValueError(f"Action execution failed: {action}")

            return success
        except Exception as e:
            logger.critical(f"CRITICAL ERROR: Action execution failed: {e}")
            logger.critical("Game terminated due to critical error")
            self.state.game_over = True
            raise RuntimeError(f"Game terminated due to critical error: {e}")

    def _execute_play_card(self, agent_id: int, card_index: int) -> bool:
        """Execute playing a card."""
        logger.debug(
            f"Playing card at index {card_index} for agent {agent_id}")
        card = self.state.hands[agent_id][card_index]
        color = card.color
        number = card.number

        # Check if the play is valid (next card in sequence)
        firework_pile = self.state.firework_piles[color]
        next_number = len(firework_pile) + 1

        if number == next_number:
            # Successful play
            self.state.firework_piles[color].append(card)

            # Update score
            self.state.update_score(
                1, "play_card", f"Successfully played {color} {number}")

            # Check if firework is complete
            if self.state.check_firework_completion(color):
                self.logger.info(f"Firework {color} completed!")
                self.state.update_score(
                    1, "complete_firework", f"Completed {color} firework")
                self.logger.info(
                    f"Current score: {self.state.score} ({self.state.get_score_percentage():.1f}%)")

            # Remove card from hand and draw new card if available
            self.state.hands[agent_id].pop(card_index)
            if self.state.deck:
                new_card = self.state.deck.pop()
                self.state.hands[agent_id].append(new_card)
                self.logger.debug(
                    f"Drew new card: {new_card.color} {new_card.number}")
            return True
        else:
            # Failed play
            self.state.discard_pile.append(card)
            self.state.fuse_tokens -= 1

            self.logger.warning(
                f"Failed play: {color} {number}. Fuse tokens: {self.state.fuse_tokens}")

            # Remove card from hand and draw new card if available
            self.state.hands[agent_id].pop(card_index)
            if self.state.deck:
                new_card = self.state.deck.pop()
                self.state.hands[agent_id].append(new_card)
                self.logger.debug(
                    f"Drew new card: {new_card.color} {new_card.number}")
            return False

    def _execute_give_clue(self, agent_id: int, target_id: int, clue: Dict[str, Any]) -> bool:
        """Execute giving a clue."""
        logger.debug(f"Giving clue to agent {target_id}: {clue}")
        self.state.clue_tokens -= 1
        logger.debug(f"Remaining clue tokens: {self.state.clue_tokens}")

        # Validate clue format
        if not self._validate_clue(clue):
            self.logger.error(f"Invalid clue format from Agent {agent_id}")
            return False

        # Update target player's card visibility based on clue
        for card in self.state.hands[target_id]:
            if self._card_matches_clue(card, clue):
                card.is_visible = True

        self.logger.info(
            f"Agent {agent_id} gave clue to Agent {target_id}: {clue}")
        return True

    def _execute_discard(self, agent_id: int, card_index: int) -> bool:
        """Execute discarding a card."""
        logger.debug(
            f"Discarding card at index {card_index} for agent {agent_id}")
        card = self.state.hands[agent_id][card_index]

        # Add card to discard pile
        self.state.discard_pile.append(card)

        # Remove card from hand
        self.state.hands[agent_id].pop(card_index)

        # Recover clue token
        self.state.clue_tokens = min(self.state.clue_tokens + 1, 8)

        # Draw new card if available
        if self.state.deck:
            new_card = self.state.deck.pop()
            self.state.hands[agent_id].append(new_card)
            self.logger.debug(
                f"Drew new card: {new_card.color} {new_card.number}")

        self.logger.info(
            f"Agent {agent_id} discarded {card.color} {card.number}")
        return True

    def _validate_clue(self, clue: Dict[str, Any]) -> bool:
        """Validate the format and content of a clue."""
        if not isinstance(clue, dict):
            return False

        if not all(field in clue for field in ["type", "value"]):
            return False

        if clue["type"] not in ["color", "number"]:
            return False

        if clue["type"] == "color" and clue["value"] not in [c.value for c in Color]:
            return False

        if clue["type"] == "number":
            try:
                # Convert value to int if it's a string
                value = int(clue["value"]) if isinstance(
                    clue["value"], str) else clue["value"]
                if not (1 <= value <= 5):
                    return False
            except (ValueError, TypeError):
                return False

        return True

    def _card_matches_clue(self, card: Card, clue: Dict[str, Any]) -> bool:
        """Check if a card matches the given clue."""
        if clue["type"] == "color":
            return card.color.value == clue["value"]
        else:  # number clue
            try:
                # Convert value to int if it's a string
                clue_value = int(clue["value"]) if isinstance(
                    clue["value"], str) else clue["value"]
                return card.number == clue_value
            except (ValueError, TypeError):
                return False

    def _update_game_state(self):
        """Update the game state after an action."""
        # Check for game over conditions
        if self.state.fuse_tokens <= 0:
            self.state.game_over = True
            self.logger.info("Game over: All fuse tokens used")
        elif not self.state.deck:
            self.state.game_over = True
            self.logger.info("Game over: Deck exhausted")

    def get_turn_info(self) -> Dict[str, Any]:
        """Get information about the current turn in a consistent format."""
        return {
            "turn_count": self.state.turn_count,
            # 1-indexed for human readability
            "human_readable_turn": self.state.turn_count + 1,
            "current_player": self.state.current_player,
            "players_count": len(self.agents)
        }
