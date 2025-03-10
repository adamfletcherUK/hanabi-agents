import logging
from typing import Dict, Optional

# Set up logger for this module
logger = logging.getLogger(__name__)

# Try to get the console logger, if it exists
try:
    console_logger = logging.getLogger('console')
    # Check if the console logger has any handlers
    if not console_logger.handlers:
        # If no handlers, use the module logger to avoid duplicate output
        console_logger = logger
except:
    # Fallback to regular logger if console logger doesn't exist
    console_logger = logger

# Color emoji mapping for consistent display
COLOR_EMOJI = {
    "red": "🔴",
    "yellow": "🟡",
    "green": "🟢",
    "blue": "🔵",
    "white": "⚪"
}


def log_game_state(engine, print_to_console=False):
    """
    Log detailed information about the current game state, including:
    - Each player's actual hand (visible to others)
    - Each player's knowledge about their own hand (what they know from clues)

    This function can be used by any script running Hanabi games.

    Args:
        engine: The GameEngine instance
        print_to_console: Whether to also print the information to the console
    """
    # Always log to file
    logger.info("----- DETAILED GAME STATE -----")

    # Only log to console if requested and if console_logger is different from logger
    if print_to_console and console_logger is not logger:
        console_logger.info("\n----- DETAILED GAME STATE -----")

    # Log firework piles
    firework_piles = []
    for color, pile in engine.state.firework_piles.items():
        emoji = COLOR_EMOJI.get(color.value, "")
        if pile:
            top_card = pile[-1].number
            firework_piles.append(f"{emoji}: {top_card}")
        else:
            firework_piles.append(f"{emoji}: empty")

    firework_str = f"🎆 Firework piles: {', '.join(firework_piles)}"
    # Always log to file
    logger.info(firework_str)
    # Only log to console if requested and if console_logger is different from logger
    if print_to_console and console_logger is not logger:
        console_logger.info(firework_str)

    # Log player hands and their knowledge
    for player_id, hand in engine.state.hands.items():
        # Log the actual hand (visible to others)
        actual_hand = []
        for i, card in enumerate(hand):
            emoji = COLOR_EMOJI.get(card.color.value, "")
            actual_hand.append(f"{i}: '{emoji}{card.number}'")

        actual_hand_str = f"👁️ Player {player_id}'s actual hand: [{', '.join(actual_hand)}]"
        # Always log to file
        logger.info(actual_hand_str)
        # Only log to console if requested and if console_logger is different from logger
        if print_to_console and console_logger is not logger:
            console_logger.info(actual_hand_str)

        # Log the player's knowledge about their own hand
        card_knowledge = engine.state.get_card_knowledge(player_id)
        player_view = []
        for i, knowledge in enumerate(card_knowledge):
            if knowledge["color_clued"]:
                emoji = COLOR_EMOJI.get(knowledge["known_color"], "")
                color_info = f"C:{emoji}"
            else:
                color_info = "C:❓"

            if knowledge["number_clued"]:
                number_info = f"#:{knowledge['known_number']}"
            else:
                number_info = "#:❓"

            player_view.append(f"{i}: '{color_info} {number_info}'")

        player_view_str = f"🧠 Player {player_id}'s knowledge: [{', '.join(player_view)}]"
        # Always log to file
        logger.info(player_view_str)
        # Only log to console if requested and if console_logger is different from logger
        if print_to_console and console_logger is not logger:
            console_logger.info(player_view_str)

    # Log discard pile summary
    discard_summary = engine.state.get_discarded_cards()
    discard_info = []
    for card, count in discard_summary.items():
        # Try to extract color and number from the card string
        parts = card.split()
        if len(parts) >= 2:
            color = parts[0]
            number = parts[1]
            emoji = COLOR_EMOJI.get(color, "")
            discard_info.append(f"{emoji}{number}: {count}")
        else:
            discard_info.append(f"{card}: {count}")

    discard_str = f"🗑️ Discard pile: {', '.join(discard_info) if discard_info else 'empty'}"
    # Always log to file
    logger.info(discard_str)
    # Only log to console if requested and if console_logger is different from logger
    if print_to_console and console_logger is not logger:
        console_logger.info(discard_str)

    # Log remaining deck size
    deck_str = f"🎴 Remaining deck size: {len(engine.state.deck)}"
    # Always log to file
    logger.info(deck_str)
    # Only log to console if requested and if console_logger is different from logger
    if print_to_console and console_logger is not logger:
        console_logger.info(deck_str)

    # Always log to file
    logger.info("----- END DETAILED STATE -----")
    # Only log to console if requested and if console_logger is different from logger
    if print_to_console and console_logger is not logger:
        console_logger.info("----- END DETAILED STATE -----\n")


def log_turn_info(turn_count, current_player_name, current_player_id, clue_tokens, max_clue_tokens, fuse_tokens, score, print_to_console=False):
    """
    Log information about the current turn.

    Args:
        turn_count: The current turn count
        current_player_name: The name of the current player
        current_player_id: The ID of the current player
        clue_tokens: The number of clue tokens available
        max_clue_tokens: The maximum number of clue tokens
        fuse_tokens: The number of fuse tokens remaining
        score: The current score
        print_to_console: Whether to also print the information to the console
    """
    logger.info(f"=== Turn {turn_count} ===")
    logger.info(
        f"Current player: {current_player_name} (Player {current_player_id})")
    logger.info(f"Clue tokens: {clue_tokens}/{max_clue_tokens}")
    logger.info(f"Fuse tokens: {fuse_tokens}/3")
    logger.info(f"Score: {score}/25")

    if print_to_console and console_logger is not logger:
        console_logger.info(f"\n=== 🎲 Turn {turn_count} ===")
        console_logger.info(
            f"Current player: 👤 {current_player_name} (Player {current_player_id})")
        console_logger.info(f"🔍 Clue tokens: {clue_tokens}/{max_clue_tokens}")
        console_logger.info(f"💣 Fuse tokens: {fuse_tokens}/3")
        console_logger.info(f"🏆 Score: {score}/25")


def log_action_result(action, result, player_name, print_to_console=False):
    """
    Log the result of an action.

    Args:
        action: The action that was taken
        result: The result of the action
        player_name: The name of the player who took the action
        print_to_console: Whether to also print the information to the console
    """
    action_type = action.get("type", "unknown")

    if action_type == "play_card":
        card_index = action.get("card_index", "?")
        logger.info(
            f"Player {player_name} played card at position {card_index}")

        if isinstance(result, dict) and "card" in result:
            card = result["card"]
            success = result.get("success", False)
            if hasattr(card, "color") and hasattr(card, "number"):
                color_name = card.color.value if hasattr(
                    card.color, "value") else str(card.color)
                color_emoji = COLOR_EMOJI.get(color_name, "")
                logger.info(
                    f"Card was {color_emoji}{card.number}, play was {'successful' if success else 'unsuccessful'}")
                if print_to_console and console_logger is not logger:
                    console_logger.info(
                        f"Card was {color_emoji}{card.number}, play was {'✅ successful' if success else '❌ unsuccessful'}")
            else:
                logger.info(
                    f"Card was {card}, play was {'successful' if success else 'unsuccessful'}")
                if print_to_console and console_logger is not logger:
                    console_logger.info(
                        f"Card was {card}, play was {'✅ successful' if success else '❌ unsuccessful'}")

    elif action_type == "give_clue":
        target_id = action.get("target_id", "?")
        clue = action.get("clue", {})
        clue_type = clue.get("type", "?")
        clue_value = clue.get("value", "?")

        if clue_type == "color":
            color_emoji = COLOR_EMOJI.get(clue_value, "")
            logger.info(
                f"Player {player_name} gave {color_emoji} color clue to Player {target_id}")
            if print_to_console and console_logger is not logger:
                console_logger.info(
                    f"Player {player_name} gave {color_emoji} color clue to Player {target_id}")
        else:
            logger.info(
                f"Player {player_name} gave number {clue_value} clue to Player {target_id}")
            if print_to_console and console_logger is not logger:
                console_logger.info(
                    f"Player {player_name} gave number {clue_value} clue to Player {target_id}")

        if isinstance(result, dict) and "affected_cards" in result:
            affected_cards = result["affected_cards"]
            logger.info(
                f"Clue affected {len(affected_cards)} cards: {affected_cards}")
            if print_to_console and console_logger is not logger:
                console_logger.info(
                    f"Clue affected {len(affected_cards)} cards: {affected_cards}")

    elif action_type == "discard":
        card_index = action.get("card_index", "?")
        logger.info(
            f"Player {player_name} discarded card at position {card_index}")

        if isinstance(result, dict) and "card" in result:
            card = result["card"]
            if hasattr(card, "color") and hasattr(card, "number"):
                color_name = card.color.value if hasattr(
                    card.color, "value") else str(card.color)
                color_emoji = COLOR_EMOJI.get(color_name, "")
                logger.info(f"Discarded card was {color_emoji}{card.number}")
                if print_to_console and console_logger is not logger:
                    console_logger.info(
                        f"Discarded card was {color_emoji}{card.number}")
            else:
                logger.info(f"Discarded card was {card}")
                if print_to_console and console_logger is not logger:
                    console_logger.info(f"Discarded card was {card}")

    # Log the raw result for debugging
    logger.info(f"Action completed: {result}")
    if print_to_console and console_logger is not logger:
        console_logger.info(f"Action completed: {result}")


def format_action_for_display(action, player_name):
    """
    Format an action for display.

    Args:
        action: The action to format
        player_name: The name of the player taking the action

    Returns:
        A formatted string describing the action
    """
    action_type = action.get("type", "unknown")

    if action_type == "play_card":
        card_index = action.get("card_index", 0)
        return f"👤 {player_name} decides to: play card at position {card_index}"

    elif action_type == "give_clue":
        target_id = action.get("target_id", 0)
        clue = action.get("clue", {})
        clue_type = clue.get("type", "unknown")
        clue_value = clue.get("value", "unknown")

        # Format the clue value with emoji if it's a color
        if clue_type == "color":
            color_emoji = COLOR_EMOJI.get(clue_value, "")
            return f"👤 {player_name} decides to: give a {color_emoji} color clue to Player {target_id}"
        else:
            return f"👤 {player_name} decides to: give a number {clue_value} clue to Player {target_id}"

    elif action_type == "discard":
        card_index = action.get("card_index", 0)
        return f"👤 {player_name} decides to: discard card at position {card_index}"

    else:
        return f"👤 {player_name} decides to: {action}"


def log_game_over(score, reason, print_to_console=False):
    """
    Log information about the game ending.

    Args:
        score: The final score
        reason: The reason the game ended
        print_to_console: Whether to also print the information to the console
    """
    logger.info(f"Game over! Final score: {score}/25")
    logger.info(f"Reason: {reason}")

    if print_to_console:
        console_logger.info(f"\n🎮 GAME OVER! 🎮")
        console_logger.info(f"Final score: {score}/25")
        console_logger.info(f"Reason: {reason}")
