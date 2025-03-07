from typing import List
from ...game.state import GameState, Card, Color


def format_firework_piles(game_state: GameState) -> str:
    """Format firework piles for display."""
    if not game_state.firework_piles:
        return "No fireworks started"

    formatted_piles = []
    for color, pile in game_state.firework_piles.items():
        if pile:
            top_card = pile[-1].number if pile else 0
            formatted_piles.append(f"{color}: {top_card}")
        else:
            formatted_piles.append(f"{color}: 0")

    return ", ".join(formatted_piles)


def format_discard_pile(game_state: GameState) -> str:
    """Format discard pile for display."""
    if not game_state.discard_pile:
        return "Empty"

    # Group cards by color and number for better readability
    card_counts = {}
    for card in game_state.discard_pile:
        key = f"{card.color} {card.number}"
        card_counts[key] = card_counts.get(key, 0) + 1

    formatted_cards = [f"{key} (x{count})" if count > 1 else key
                       for key, count in card_counts.items()]
    return ", ".join(formatted_cards)


def format_hand(hand: List[Card]) -> str:
    """Format hand for display."""
    if not hand:
        return "Empty"

    formatted_cards = []
    for i, card in enumerate(hand):
        if card.is_visible:
            # For visible cards, show the color and number in concise format
            # First letter of color
            color_abbr = card.color.value[0].upper()
            formatted_cards.append(f"{i}: {color_abbr}{card.number}")
        else:
            # For hidden cards, show [?]
            formatted_cards.append(f"{i}: [?]")

    return ", ".join(formatted_cards)
