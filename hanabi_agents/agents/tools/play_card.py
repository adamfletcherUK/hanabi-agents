from typing import Dict, Any
from ...game.state import GameState


def play_card_tool(agent_id: int, card_index: int, game_state: GameState) -> Dict[str, Any]:
    """
    Tool for playing a card from the agent's hand.

    Args:
        agent_id: ID of the agent playing the card
        card_index: Index of the card to play (0-indexed)
        game_state: Current state of the game

    Returns:
        Dictionary with the result of the action
    """
    # Validate the action
    if not game_state.is_valid_move(agent_id, "play", card_index=card_index):
        return {
            "success": False,
            "error": f"Invalid play action: card_index={card_index}",
            "action_type": "play"
        }

    # Get the card being played (for information purposes)
    hand = game_state.hands.get(agent_id, [])
    if 0 <= card_index < len(hand):
        card = hand[card_index]
        card_info = f"{card.color.value} {card.number}"
    else:
        card_info = "unknown card"

    return {
        "success": True,
        "action_type": "play",
        "card_index": card_index,
        "card_info": card_info,
        "message": f"Playing card at index {card_index} ({card_info})"
    }
