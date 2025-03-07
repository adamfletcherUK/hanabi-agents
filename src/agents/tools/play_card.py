from typing import Dict, Any


def play_card_tool(agent_id: int, card_index: int, game_state: Any) -> Dict[str, Any]:
    """Play a card from your hand."""
    # Validate the action
    if not isinstance(card_index, int):
        raise ValueError(
            f"Card index must be an integer, got {type(card_index)}")

    if not (0 <= card_index < len(game_state.hands[agent_id])):
        raise ValueError(
            f"Invalid card index: {card_index}. Must be between 0 and {len(game_state.hands[agent_id])-1}")

    # Format the action for the game engine
    return {
        "type": "play_card",
        "card_index": card_index
    }
