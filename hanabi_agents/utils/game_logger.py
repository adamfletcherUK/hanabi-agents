import logging
from typing import Dict, Optional

# Set up logger for this module
logger = logging.getLogger(__name__)

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
    logger.info("----- DETAILED GAME STATE -----")

    if print_to_console:
        print("\n----- DETAILED GAME STATE -----")

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
    logger.info(firework_str)
    if print_to_console:
        print(firework_str)

    # Log player hands and their knowledge
    for player_id, hand in engine.state.hands.items():
        # Log the actual hand (visible to others)
        actual_hand = []
        for i, card in enumerate(hand):
            emoji = COLOR_EMOJI.get(card.color.value, "")
            actual_hand.append(f"{i}: '{emoji}{card.number}'")

        actual_hand_str = f"👁️ Player {player_id}'s actual hand: [{', '.join(actual_hand)}]"
        logger.info(actual_hand_str)
        if print_to_console:
            print(actual_hand_str)

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
        logger.info(player_view_str)
        if print_to_console:
            print(player_view_str)

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
    logger.info(discard_str)
    if print_to_console:
        print(discard_str)

    # Log remaining deck size
    deck_str = f"🎴 Remaining deck size: {len(engine.state.deck)}"
    logger.info(deck_str)
    if print_to_console:
        print(deck_str)

    logger.info("----- END DETAILED STATE -----")
    if print_to_console:
        print("----- END DETAILED STATE -----\n")


def log_turn_info(turn_count, current_player_name, current_player_id, clue_tokens, max_clue_tokens, fuse_tokens, score, print_to_console=False):
    """
    Log information about the current turn.

    Args:
        turn_count: The current turn number
        current_player_name: The name of the current player
        current_player_id: The ID of the current player
        clue_tokens: The number of clue tokens available
        max_clue_tokens: The maximum number of clue tokens
        fuse_tokens: The number of fuse tokens remaining
        score: The current score
        print_to_console: Whether to also print the information to the console
    """
    turn_info = f"=== Turn {turn_count} ==="
    player_info = f"Current player: {current_player_name} (Player {current_player_id})"
    clue_info = f"Clue tokens: {clue_tokens}/{max_clue_tokens}"
    fuse_info = f"Fuse tokens: {fuse_tokens}/3"
    score_info = f"Score: {score}/25"

    logger.info(turn_info)
    logger.info(player_info)
    logger.info(clue_info)
    logger.info(fuse_info)
    logger.info(score_info)

    if print_to_console:
        print(f"\n=== 🎲 Turn {turn_count} ===")
        print(
            f"Current player: 👤 {current_player_name} (Player {current_player_id})")
        print(f"🔍 Clue tokens: {clue_tokens}/{max_clue_tokens}")
        print(f"💣 Fuse tokens: {fuse_tokens}/3")
        print(f"🏆 Score: {score}/25")


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

    # Handle case where result is a boolean or not a dictionary
    if not isinstance(result, dict):
        action_str = f"Action completed: {result}"
        logger.info(action_str)
        if print_to_console:
            print(action_str)
        return

    if action_type == "play_card" and "card" in result:
        card = result["card"]
        success = result.get("success", False)
        emoji = "✅" if success else "❌"

        if hasattr(card, "color") and hasattr(card, "number"):
            color_emoji = COLOR_EMOJI.get(card.color.value, "")
            action_str = f"Card played: {color_emoji}{card.number} {emoji}"
        else:
            action_str = f"Card played: {card} {emoji}"

        logger.info(action_str)
        if print_to_console:
            print(action_str)

    elif action_type == "give_clue":
        # Extract clue details
        clue = action.get("clue", {})
        clue_type = clue.get("type", "unknown")
        clue_value = clue.get("value", "unknown")
        target_id = action.get("target_id", "?")

        # Format the clue value with emoji if it's a color
        if clue_type == "color":
            color_emoji = COLOR_EMOJI.get(clue_value, "")
            clue_desc = f"{color_emoji} color"
        else:
            clue_desc = f"number {clue_value}"

        affected_count = len(result.get('affected_cards', []))
        action_str = f"Clue given: {clue_desc} to Player {target_id}, affected {affected_count} cards"

        logger.info(action_str)
        if print_to_console:
            print(action_str)

    elif action_type == "discard" and "card" in result:
        card = result["card"]

        if hasattr(card, "color") and hasattr(card, "number"):
            color_emoji = COLOR_EMOJI.get(card.color.value, "")
            action_str = f"Card discarded: {color_emoji}{card.number}"
        else:
            action_str = f"Card discarded: {card}"

        logger.info(action_str)
        if print_to_console:
            print(action_str)
    else:
        # Generic fallback for other result formats
        action_str = f"Action result: {result}"
        logger.info(action_str)
        if print_to_console:
            print(action_str)


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
    Log game over information.

    Args:
        score: The final score
        reason: The reason the game ended
        print_to_console: Whether to also print the information to the console
    """
    game_over_str = "=== Game Over ==="
    score_str = f"Final score: {score}/25"
    reason_str = f"Reason: {reason}"

    logger.info(game_over_str)
    logger.info(score_str)
    logger.info(reason_str)

    if print_to_console:
        print("\n=== 🏁 Game Over ===")
        print(f"🏆 Final score: {score}/25")
        print(f"📋 Reason: {reason}")
