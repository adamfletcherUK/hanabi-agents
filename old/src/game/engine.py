import random
import logging
import os
from typing import List, Dict, Any
from .state import GameState, Card, Color
from ..agents.base import Agent
from ..communication.discussion import DiscussionManager
import datetime

# Set up logging
logger = logging.getLogger(__name__)


class GameEngine:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.logger = logger  # Store reference to the logger
        self.state = self._initialize_game()
        self.discussion_manager = DiscussionManager()
        self._setup_logging()
        # Track the last action and its result
        self.last_action = None
        self.last_action_result = None
        # Track incorrect tool usage
        self.incorrect_tool_usage = {}
        for i in range(len(agents)):
            self.incorrect_tool_usage[i] = []
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

        # Check if we have a pre-action discussion summary
        if hasattr(self, 'pre_action_discussion_summary'):
            # Use the pre-action discussion summary
            discussion_summary = self.pre_action_discussion_summary
            self.logger.info("Using pre-action discussion summary")
            # Clear the pre-action discussion summary for the next turn
            delattr(self, 'pre_action_discussion_summary')
        else:
            # Conduct discussion
            discussion_summary = self.discussion_manager.conduct_discussion(
                self.agents, self.state, active_agent_id)

        # Get action from active agent
        agent_view = self.state.get_view_for(active_agent_id)
        try:
            action = active_agent.decide_action(agent_view, discussion_summary)
            self.logger.info(f"Agent {active_agent_id} chose action: {action}")
            # Store the action for reference
            self.last_action = action
        except Exception as e:
            self.logger.error(
                f"Agent {active_agent_id} failed to propose a valid action: {e}")

            # Create a fallback action instead of terminating the game
            try:
                # Try to create a simple fallback action
                if self.state.clue_tokens > 0:
                    # Find a player to give a clue to
                    target_id = (active_agent_id + 1) % len(self.agents)
                    action = {
                        "type": "give_clue",
                        "target_id": target_id,
                        "clue": {
                            "type": "color",
                            "value": "red"  # Default to red as a common color
                        }
                    }
                else:
                    # If no clue tokens, discard the first card
                    action = {
                        "type": "discard",
                        "card_index": 0
                    }

                self.logger.warning(f"Using fallback action: {action}")
                self.last_action = action
            except Exception as fallback_error:
                self.logger.critical(
                    f"CRITICAL ERROR: Failed to create fallback action: {fallback_error}")
                self.logger.critical("Game terminated due to critical error")
                self.state.game_over = True
                raise RuntimeError(
                    f"Game terminated due to critical error: {e}, fallback error: {fallback_error}")

        # Validate and execute action
        if self.execute_action(active_agent.agent_id, action):
            self._update_game_state()
            return True
        else:
            self.logger.error(
                f"Invalid action attempted by Agent {active_agent.agent_id}: {action}")

            # Track the invalid action in the agent's error history
            error_message = f"Invalid action: {action}"
            error_reason = "invalid_action_format"
            self._track_incorrect_tool_usage(
                active_agent.agent_id, action, error_message, error_reason)

            # Instead of terminating, return False to indicate failure
            return False

    def execute_action(self, agent_id: int, action: Dict[str, Any]) -> bool:
        """Execute an action for the given agent."""
        logger.info(f"Executing action for agent {agent_id}: {action}")
        logger.debug(
            f"Before action - Current player: {self.state.current_player}, Turn count: {self.state.turn_count}")

        # Store the action for reference
        self.last_action = action

        # Validate the action
        try:
            # Initialize error_message outside the if block
            error_message = None

            if not self.state.is_valid_move(agent_id, action["type"], **action):
                # Enhanced error logging to explain why the action is invalid
                action_type = action["type"]
                error_message = f"Invalid {action_type} action by Agent {agent_id}"

                if action_type == "give_clue":
                    target_id = action.get("target_id")
                    clue = action.get("clue", {})
                    clue_type = clue.get("type")
                    clue_value = clue.get("value")
                    error_message += f" - Clue: {clue_type}={clue_value} to Player {target_id}"

                    # Check specific clue errors
                    if self.state.clue_tokens <= 0:
                        error_message += " (No clue tokens available)"
                        error_reason = "no_clue_tokens"
                    elif target_id == agent_id:
                        error_message += " (Cannot give clue to yourself)"
                        error_reason = "self_clue"
                    elif not self._validate_clue(clue):
                        error_message += " (Invalid clue format)"
                        error_reason = "invalid_clue_format"
                    else:
                        # Check if the clue would affect any cards
                        affected_cards = []
                        for i, card in enumerate(self.state.hands[target_id]):
                            if self._card_matches_clue(card, clue):
                                affected_cards.append(i)

                        if not affected_cards:
                            error_message += " (Clue would not affect any cards)"
                            error_reason = "no_affected_cards"
                        else:
                            error_reason = "unknown_clue_error"

                elif action_type == "play_card":
                    card_index = action.get("card_index")
                    error_message += f" - Card index: {card_index}"

                    # Check specific play errors
                    if card_index is None or card_index < 0 or card_index >= len(self.state.hands[agent_id]):
                        error_message += f" (Invalid card index, hand size: {len(self.state.hands[agent_id])})"
                        error_reason = "invalid_card_index"
                    else:
                        error_reason = "unknown_play_error"

                elif action_type == "discard":
                    card_index = action.get("card_index")
                    error_message += f" - Card index: {card_index}"

                    # Check specific discard errors
                    if self.state.clue_tokens >= self.state.max_clue_tokens:
                        error_message += " (Cannot discard when clue tokens are at maximum)"
                        error_reason = "max_clue_tokens"
                    elif card_index is None or card_index < 0 or card_index >= len(self.state.hands[agent_id]):
                        error_message += f" (Invalid card index, hand size: {len(self.state.hands[agent_id])})"
                        error_reason = "invalid_card_index"
                    else:
                        error_reason = "unknown_discard_error"
                else:
                    error_reason = "unknown_action_type"

                logger.error(error_message)

                # Track the incorrect tool usage
                self._track_incorrect_tool_usage(
                    agent_id, action, error_message, error_reason)

                return False

            # Execute the action based on its type
            action_type = action["type"]
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
                self._track_incorrect_tool_usage(
                    agent_id, action, f"Unknown action type: {action_type}", "unknown_action_type")
                return False

            # Store the action result
            self.last_action_result = result

            # Advance to the next player
            self.state.current_player = (
                self.state.current_player + 1) % len(self.agents)
            self.state.turn_count += 1
            logger.debug(
                f"Player advanced: {agent_id} -> {self.state.current_player}, Turn count: {self.state.turn_count - 1} -> {self.state.turn_count}")

            return True
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            self._track_incorrect_tool_usage(
                agent_id, action, f"Error executing action: {e}", "exception")
            return False

    def _execute_play_card(self, agent_id: int, card_index: int) -> bool:
        """Execute a play card action."""
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
                f"Agent {agent_id} successfully played {color.value} {number}")

            # Add the card to the firework pile
            self.state.firework_piles[color].append(card)

            # Check if we completed a firework (reached 5)
            if number == 5:
                logger.info(f"Completed the {color.value} firework!")
                # Award a clue token if not already at max
                if self.state.clue_tokens < self.state.max_clue_tokens:
                    self.state.clue_tokens += 1
                    logger.info(
                        f"Awarded a clue token for completing a firework. Now at {self.state.clue_tokens}")

            # Update the score
            self.state.score = sum(len(pile)
                                   for pile in self.state.firework_piles.values())

            # Add to score history
            self.state.add_score_event(
                "play_card", f"Agent {agent_id} played {color.value} {number}")
        else:
            logger.info(
                f"Agent {agent_id} failed to play {color.value} {number}")

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
                f"Agent {agent_id} drew a new card: {new_card.color.value} {new_card.number}")
        else:
            logger.info(
                f"No cards left in the deck for Agent {agent_id} to draw")

        # Return the result
        return {
            "success": success,
            "card": card,
            "score": self.state.score,
            "fuse_tokens": self.state.fuse_tokens
        }

    def _execute_give_clue(self, agent_id: int, target_id: int, clue: Dict[str, Any]) -> bool:
        """Execute a give clue action."""
        logger.debug(f"Processing give_clue action")
        logger.debug(f"Giving clue to agent {target_id}: {clue}")

        # Spend a clue token
        self.state.clue_tokens -= 1
        logger.debug(f"Remaining clue tokens: {self.state.clue_tokens}")

        # Apply the clue to the target player's hand
        affected_cards = []
        for i, card in enumerate(self.state.hands[target_id]):
            if self._card_matches_clue(card, clue):
                affected_cards.append(i)
                card.is_visible = True

                # Mark the specific clue type
                if clue["type"] == "color":
                    card.color_clued = True
                elif clue["type"] == "number":
                    card.number_clued = True

        # Log the clue
        clue_type = clue["type"]
        clue_value = clue["value"]
        logger.info(
            f"Agent {agent_id} gave clue to Agent {target_id}: {clue_type}={clue_value}, affecting positions {affected_cards}")

        # Track the clue in game history
        try:
            self.state.add_clue_event(
                agent_id, target_id, clue_type, clue_value, affected_cards)
        except Exception as e:
            logger.error(f"Error tracking clue event: {e}")
            # Continue execution even if tracking fails

        # Return the result
        return {
            "success": True,
            "affected_cards": affected_cards,
            "clue_tokens": self.state.clue_tokens
        }

    def _execute_discard(self, agent_id: int, card_index: int) -> bool:
        """Execute a discard action."""
        logger.debug(f"Processing discard action")

        # Get the card from the player's hand
        hand = self.state.hands[agent_id]
        card = hand[card_index]

        # Add the card to the discard pile
        self.state.discard_pile.append(card)
        logger.info(
            f"Agent {agent_id} discarded {card.color.value} {card.number}")

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
                f"Agent {agent_id} drew a new card: {new_card.color.value} {new_card.number}")
        else:
            logger.info(
                f"No cards left in the deck for Agent {agent_id} to draw")

        # Return the result
        return {
            "success": True,
            "card": card,
            "clue_tokens": self.state.clue_tokens
        }

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

    def _track_incorrect_tool_usage(self, agent_id: int, action: Dict[str, Any], error_message: str, error_reason: str) -> None:
        """
        Track incorrect tool usage by an agent.

        Args:
            agent_id: The ID of the agent that made the incorrect action
            action: The action that was attempted
            error_message: A human-readable error message
            error_reason: A machine-readable error reason code
        """
        # Create an error record
        error_record = {
            "turn": self.state.turn_count,
            "action": action,
            "error_message": error_message,
            "error_reason": error_reason,
            "timestamp": datetime.datetime.now().isoformat()
        }

        # Add to the agent's error history
        self.incorrect_tool_usage[agent_id].append(error_record)

        # Log the error
        logger.warning(
            f"Incorrect tool usage by Agent {agent_id}: {error_message}")

        # Notify the agent of the error if possible
        try:
            agent = self.agents[agent_id]
            if hasattr(agent, 'notify_incorrect_tool_usage') and callable(getattr(agent, 'notify_incorrect_tool_usage')):
                agent.notify_incorrect_tool_usage(error_record)
        except Exception as e:
            logger.error(
                f"Failed to notify agent {agent_id} of incorrect tool usage: {e}")

    def get_incorrect_tool_usage(self, agent_id: int = None) -> Dict[int, List[Dict[str, Any]]]:
        """
        Get the history of incorrect tool usage.

        Args:
            agent_id: Optional agent ID to filter by

        Returns:
            Dictionary mapping agent IDs to lists of error records,
            or a single list if agent_id is specified
        """
        if agent_id is not None:
            return self.incorrect_tool_usage.get(agent_id, [])
        return self.incorrect_tool_usage
