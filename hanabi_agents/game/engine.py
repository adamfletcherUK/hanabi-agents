import random
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from .state import GameState, Card, Color, ClueAction

# Set up logging
logger = logging.getLogger(__name__)


class GameEngine:
    """
    The main engine for running a Hanabi game.

    This class is responsible for:
    - Initializing the game state
    - Managing the game loop
    - Executing player actions
    - Enforcing game rules
    - Tracking game completion and scoring
    """

    def __init__(self, num_players: int = 2, seed: Optional[int] = None):
        """
        Initialize a new game engine.

        Args:
            num_players: The number of players in the game (2-5)
            seed: Optional random seed for reproducibility
        """
        if not 2 <= num_players <= 5:
            raise ValueError(
                f"Number of players must be between 2 and 5, got {num_players}")

        self.num_players = num_players
        self.seed = seed

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)

        # Initialize game state
        self.state = self._initialize_game()

        # Track incorrect actions for debugging
        self.action_errors = {i: [] for i in range(num_players)}

        logger.info(f"Game engine initialized with {num_players} players")
        if seed is not None:
            logger.info(f"Random seed set to {seed}")

    def _initialize_game(self) -> GameState:
        """
        Initialize a new game of Hanabi.

        This includes:
        - Creating and shuffling the deck
        - Setting up firework piles
        - Dealing initial hands to players

        Returns:
            A new GameState object
        """
        logger.debug("Creating and shuffling deck")

        # Create deck
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
        logger.debug(f"Deck created with {len(deck)} cards and shuffled")

        # Initialize game state
        state = GameState()
        state.deck = deck

        # Initialize firework piles
        for color in Color:
            state.firework_piles[color] = []

        # Deal cards to players
        cards_per_player = 5 if self.num_players <= 3 else 4
        for i in range(self.num_players):
            state.hands[i] = []
            for _ in range(cards_per_player):
                if deck:
                    card = deck.pop(0)
                    state.hands[i].append(card)

        logger.debug(f"Cards dealt. Remaining deck size: {len(deck)}")
        return state

    def play_game(self, agents: List[Any]) -> int:
        """
        Play a complete game of Hanabi.

        Args:
            agents: A list of agent objects that will make decisions

        Returns:
            The final score
        """
        if len(agents) != self.num_players:
            raise ValueError(
                f"Expected {self.num_players} agents, got {len(agents)}")

        logger.info("Starting new game of Hanabi")

        # Main game loop
        final_player_turns = None

        while not self.state.game_over:
            # Play a turn
            self._play_turn(agents)

            # Check for game over conditions
            if self.state.fuse_tokens <= 0:
                logger.info("Game over: All fuse tokens used")
                self.state.game_over = True

            # If deck is empty, start counting final rounds
            if len(self.state.deck) == 0 and final_player_turns is None:
                logger.info("Deck is empty, each player gets one more turn")
                final_player_turns = self.num_players

            # Count down final turns if deck is empty
            if final_player_turns is not None:
                final_player_turns -= 1
                if final_player_turns <= 0:
                    logger.info("Game over: Final round completed")
                    self.state.game_over = True

        # Calculate final score according to official Hanabi rules
        # The score is the sum of the highest card in each firework pile
        final_score = 0
        for color, pile in self.state.firework_piles.items():
            if pile:  # If the pile is not empty
                final_score += len(pile)  # Each card adds 1 to the score

        # Update the state score to match the final calculation
        self.state.score = final_score

        logger.info(f"Game finished with score: {final_score}")
        return final_score

    def _play_turn(self, agents: List[Any]) -> bool:
        """
        Play a single turn of the game.

        Args:
            agents: A list of agent objects that will make decisions

        Returns:
            True if the turn was completed successfully, False otherwise
        """
        active_agent_id = self.state.current_player
        active_agent = agents[active_agent_id]

        logger.info(
            f"Turn {self.state.turn_count + 1}: Player {active_agent_id}'s turn")

        # Get action from active agent
        agent_view = self.state.get_view_for(active_agent_id)
        try:
            action = active_agent.decide_action(agent_view)
            logger.info(f"Player {active_agent_id} chose action: {action}")

            # Store the action for reference
            self.state.last_action = action
        except Exception as e:
            logger.error(
                f"Player {active_agent_id} failed to propose a valid action: {e}")

            # Create a fallback action
            action = self._create_fallback_action(active_agent_id)
            logger.warning(f"Using fallback action: {action}")
            self.state.last_action = action

        # Validate and execute action
        if self.execute_action(active_agent_id, action):
            # Update game state
            self._update_game_state()
            return True
        else:
            logger.error(
                f"Invalid action attempted by Player {active_agent_id}: {action}")

            # Track the invalid action
            self._track_action_error(
                active_agent_id, action, "Invalid action format")

            # Try with fallback action
            fallback_action = self._create_fallback_action(active_agent_id)
            if self.execute_action(active_agent_id, fallback_action):
                self._update_game_state()
                return True

            return False

    def _create_fallback_action(self, agent_id: int) -> Dict[str, Any]:
        """
        Create a fallback action when an agent fails to provide a valid one.

        Args:
            agent_id: The ID of the agent

        Returns:
            A valid action dictionary
        """
        # Try to create a simple fallback action
        if self.state.clue_tokens > 0:
            # Find a player to give a clue to
            target_id = (agent_id + 1) % self.num_players

            # Find a valid clue to give (color or number)
            target_hand = self.state.hands[target_id]
            if target_hand:
                # Try to give a color clue first
                action = {
                    "type": "give_clue",
                    "target_id": target_id,
                    "clue": {
                        "type": "color",
                        "value": target_hand[0].color.value
                    }
                }
            else:
                # If target has no cards, try another player
                for i in range(self.num_players):
                    if i != agent_id and self.state.hands[i]:
                        action = {
                            "type": "give_clue",
                            "target_id": i,
                            "clue": {
                                "type": "color",
                                "value": self.state.hands[i][0].color.value
                            }
                        }
                        break
                else:
                    # If no one has cards, discard
                    action = {
                        "type": "discard",
                        "card_index": 0
                    }
        else:
            # If no clue tokens, discard the first card
            action = {
                "type": "discard",
                "card_index": 0
            }

        return action

    def execute_action(self, agent_id: int, action: Dict[str, Any]) -> bool:
        """
        Execute an action for the given agent.

        Args:
            agent_id: The ID of the agent performing the action
            action: The action to execute

        Returns:
            True if the action was executed successfully, False otherwise
        """
        logger.info(f"Executing action for player {agent_id}: {action}")

        # Store the action for reference
        self.state.last_action = action

        # Validate the action
        try:
            # Check if the action is valid
            if "type" not in action:
                logger.error(f"Action missing 'type' field: {action}")
                self._track_action_error(
                    agent_id, action, "Action missing 'type' field", "missing_type")
                return False

            action_type = action["type"]

            if not self.state.is_valid_move(agent_id, action_type, **action):
                # Determine specific error reason based on action type
                error_reason = "invalid_action"
                error_message = f"Invalid {action_type} action by Player {agent_id}"

                if action_type == "give_clue":
                    target_id = action.get("target_id")
                    clue = action.get("clue", {})
                    clue_type = clue.get("type")
                    clue_value = clue.get("value")

                    if self.state.clue_tokens <= 0:
                        error_reason = "no_clue_tokens"
                        error_message += " (No clue tokens available)"
                    elif target_id == agent_id:
                        error_reason = "self_clue"
                        error_message += " (Cannot give clue to yourself)"
                    elif target_id is None or target_id not in self.state.hands:
                        error_reason = "invalid_target"
                        error_message += f" (Invalid target ID: {target_id})"
                    elif clue_type not in ["color", "number"]:
                        error_reason = "invalid_clue_type"
                        error_message += f" (Invalid clue type: {clue_type})"
                    else:
                        # Check if the clue would affect any cards
                        affected_cards = []
                        for card in self.state.hands[target_id]:
                            if (clue_type == "color" and card.color.value == clue_value) or \
                               (clue_type == "number" and card.number == clue_value):
                                affected_cards.append(card)

                        if not affected_cards:
                            error_reason = "no_affected_cards"
                            error_message += " (Clue would not affect any cards)"

                elif action_type == "play_card":
                    card_index = action.get("card_index")
                    if card_index is None or not isinstance(card_index, int):
                        error_reason = "invalid_card_index_type"
                        error_message += f" (Card index must be an integer, got {card_index})"
                    elif not (0 <= card_index < len(self.state.hands[agent_id])):
                        error_reason = "invalid_card_index_range"
                        error_message += f" (Card index {card_index} out of range for hand size {len(self.state.hands[agent_id])})"

                elif action_type == "discard":
                    card_index = action.get("card_index")
                    if self.state.clue_tokens >= self.state.max_clue_tokens:
                        error_reason = "max_clue_tokens"
                        error_message += " (Cannot discard when at maximum clue tokens)"
                    elif card_index is None or not isinstance(card_index, int):
                        error_reason = "invalid_card_index_type"
                        error_message += f" (Card index must be an integer, got {card_index})"
                    elif not (0 <= card_index < len(self.state.hands[agent_id])):
                        error_reason = "invalid_card_index_range"
                        error_message += f" (Card index {card_index} out of range for hand size {len(self.state.hands[agent_id])})"

                else:
                    error_reason = "unknown_action_type"
                    error_message += f" (Unknown action type: {action_type})"

                logger.error(error_message)
                self._track_action_error(
                    agent_id, action, error_message, error_reason)
                return False

            # Execute the action based on its type
            result = None

            if action_type == "play_card":
                card_index = action["card_index"]
                result = self._execute_play_card(agent_id, card_index)
            elif action_type == "give_clue":
                target_id = action["target_id"]
                clue = action["clue"]
                result = self._execute_give_clue(agent_id, target_id, clue)
            elif action_type == "discard":
                card_index = action["card_index"]
                result = self._execute_discard(agent_id, card_index)
            else:
                logger.error(f"Unknown action type: {action_type}")
                self._track_action_error(
                    agent_id, action, f"Unknown action type: {action_type}", "unknown_action_type")
                return False

            # Store the action result
            self.state.last_action_result = result

            # Advance to the next player
            self.state.current_player = (
                self.state.current_player + 1) % self.num_players
            self.state.turn_count += 1

            logger.debug(
                f"Player advanced: {agent_id} -> {self.state.current_player}, Turn count: {self.state.turn_count}")

            return True
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            self._track_action_error(
                agent_id, action, f"Error executing action: {e}", "exception")
            return False

    def _execute_play_card(self, agent_id: int, card_index: int) -> Dict[str, Any]:
        """
        Execute a play card action.

        Args:
            agent_id: The ID of the agent playing the card
            card_index: The index of the card in the agent's hand

        Returns:
            A dictionary containing the result of the action
        """
        logger.debug(f"Processing play_card action")

        # Get the card from the player's hand
        hand = self.state.hands[agent_id]
        card = hand[card_index]

        # Check if the card can be played on the firework pile
        color = card.color
        number = card.number
        pile = self.state.firework_piles[color]

        # Determine if the play is successful
        success = False
        if not pile and number == 1:
            # Starting a new pile with a 1
            success = True
        elif pile and pile[-1].number == number - 1:
            # Adding to an existing pile with the next number
            success = True

        # Process the play
        if success:
            logger.info(
                f"Player {agent_id} successfully played {color.value} {number}")

            # Add the card to the firework pile
            self.state.firework_piles[color].append(card)

            # Update the score
            self.state.score += 1
            self.state.add_score_event(
                "play_card", f"Player {agent_id} played {color.value} {number}")

            # Check if we completed a firework (reached 5)
            if number == 5:
                logger.info(f"Completed the {color.value} firework!")
                self.state.check_firework_completion(color)

                # Award a clue token if not already at max
                if self.state.clue_tokens < self.state.max_clue_tokens:
                    self.state.clue_tokens += 1
                    logger.info(
                        f"Awarded a clue token for completing a firework. Now at {self.state.clue_tokens}")
        else:
            logger.info(
                f"Player {agent_id} failed to play {color.value} {number}")

            # Failed play - add to discard pile and lose a fuse token
            self.state.discard_pile.append(card)
            self.state.fuse_tokens -= 1
            logger.warning(
                f"Lost a fuse token. Now at {self.state.fuse_tokens}")

            # Check if we're out of fuse tokens
            if self.state.fuse_tokens <= 0:
                logger.critical("Out of fuse tokens! Game over.")
                self.state.game_over = True

        # Remove the card from the player's hand
        hand.pop(card_index)

        # Draw a new card if there are cards left in the deck
        if self.state.deck:
            new_card = self.state.deck.pop(0)
            hand.append(new_card)
            logger.debug(
                f"Player {agent_id} drew a new card: {new_card.color.value} {new_card.number}")
        else:
            logger.info(
                f"No cards left in the deck for Player {agent_id} to draw")

        # Return the result
        return {
            "success": success,
            "card": str(card),
            "score": self.state.score,
            "fuse_tokens": self.state.fuse_tokens
        }

    def _execute_give_clue(self, agent_id: int, target_id: int, clue: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a give clue action.

        Args:
            agent_id: The ID of the agent giving the clue
            target_id: The ID of the agent receiving the clue
            clue: The clue to give

        Returns:
            A dictionary containing the result of the action
        """
        logger.debug(f"Processing give_clue action")

        # Validate clue format
        try:
            clue_obj = ClueAction(**clue)
            clue_type = clue_obj.type
            value = clue_obj.value
        except Exception as e:
            logger.error(f"Invalid clue format: {e}")
            return {"success": False, "error": str(e)}

        # Spend a clue token
        self.state.clue_tokens -= 1
        logger.debug(f"Remaining clue tokens: {self.state.clue_tokens}")

        # Apply the clue to the target player's hand
        affected_indices = []
        for i, card in enumerate(self.state.hands[target_id]):
            if self._card_matches_clue(card, clue):
                affected_indices.append(i)

                # Mark the specific clue type
                if clue_type == "color":
                    card.color_clued = True
                elif clue_type == "number":
                    card.number_clued = True

        # Log the clue
        logger.info(
            f"Player {agent_id} gave clue to Player {target_id}: {clue_type}={value}, affecting positions {affected_indices}")

        # Track the clue in game history
        self.state.add_clue_event(
            agent_id, target_id, clue_type, value, affected_indices)

        # Return the result
        return {
            "success": True,
            "affected_indices": affected_indices,
            "clue_tokens": self.state.clue_tokens
        }

    def _execute_discard(self, agent_id: int, card_index: int) -> Dict[str, Any]:
        """
        Execute a discard action.

        Args:
            agent_id: The ID of the agent discarding the card
            card_index: The index of the card in the agent's hand

        Returns:
            A dictionary containing the result of the action
        """
        logger.debug(f"Processing discard action")

        # Get the card from the player's hand
        hand = self.state.hands[agent_id]
        card = hand[card_index]

        # Add the card to the discard pile
        self.state.discard_pile.append(card)
        logger.info(
            f"Player {agent_id} discarded {card.color.value} {card.number}")

        # Gain a clue token
        self.state.clue_tokens += 1
        if self.state.clue_tokens > self.state.max_clue_tokens:
            self.state.clue_tokens = self.state.max_clue_tokens
        logger.debug(f"Gained a clue token. Now at {self.state.clue_tokens}")

        # Remove the card from the player's hand
        hand.pop(card_index)

        # Draw a new card if there are cards left in the deck
        if self.state.deck:
            new_card = self.state.deck.pop(0)
            hand.append(new_card)
            logger.debug(
                f"Player {agent_id} drew a new card: {new_card.color.value} {new_card.number}")
        else:
            logger.info(
                f"No cards left in the deck for Player {agent_id} to draw")

        # Return the result
        return {
            "success": True,
            "card": str(card),
            "clue_tokens": self.state.clue_tokens
        }

    def _card_matches_clue(self, card: Card, clue: Dict[str, Any]) -> bool:
        """
        Check if a card matches the given clue.

        Args:
            card: The card to check
            clue: The clue to match against

        Returns:
            True if the card matches the clue, False otherwise
        """
        clue_type = clue["type"]
        value = clue["value"]

        if clue_type == "color":
            return card.color.value == value
        else:  # number clue
            try:
                # Convert value to int if it's a string
                clue_value = int(value) if isinstance(value, str) else value
                return card.number == clue_value
            except (ValueError, TypeError):
                return False

    def _update_game_state(self) -> None:
        """
        Update the game state after an action.

        This checks for game over conditions and updates any state that needs
        to be recalculated after an action.
        """
        # Check for game over conditions
        if self.state.fuse_tokens <= 0:
            self.state.game_over = True
            logger.info("Game over: All fuse tokens used")

        # Check for completed fireworks
        for color in Color:
            self.state.check_firework_completion(color)

    def _track_action_error(self, agent_id: int, action: Dict[str, Any], error_message: str, error_reason: str = "unknown"):
        """
        Track an invalid action for debugging and agent learning.

        Args:
            agent_id: The ID of the agent who attempted the invalid action
            action: The invalid action
            error_message: A human-readable error message
            error_reason: A specific error reason code
        """
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "error_message": error_message,
            "error_reason": error_reason,
            "turn": self.state.turn_count
        }
        self.action_errors[agent_id].append(error_entry)
        logger.warning(
            f"Action error by Player {agent_id}: {error_message} (Reason: {error_reason})")

    def get_game_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current game state.

        Returns:
            A dictionary containing game summary information
        """
        return {
            "turn_count": self.state.turn_count,
            "score": self.state.score,
            "max_score": self.state.get_max_possible_score(),
            "score_percentage": self.state.get_score_percentage(),
            "completed_fireworks": self.state.get_completed_fireworks_count(),
            "clue_tokens": self.state.clue_tokens,
            "fuse_tokens": self.state.fuse_tokens,
            "deck_size": len(self.state.deck),
            "discard_pile_size": len(self.state.discard_pile),
            "game_over": self.state.game_over
        }

    def get_action_errors(self, agent_id: Optional[int] = None) -> Dict[int, List[Dict[str, Any]]]:
        """
        Get the history of action errors.

        Args:
            agent_id: Optional agent ID to filter by

        Returns:
            Dictionary mapping agent IDs to lists of error records,
            or a single list if agent_id is specified
        """
        if agent_id is not None:
            return {agent_id: self.action_errors.get(agent_id, [])}
        return self.action_errors
