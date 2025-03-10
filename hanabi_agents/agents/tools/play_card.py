from typing import Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from ...game.state import GameState


class PlayCardInput(BaseModel):
    card_index: int = Field(
        description="Index of the card to play (0-indexed)")


@tool(args_schema=PlayCardInput)
def play_card_tool(card_index: int) -> Dict[str, Any]:
    """
    Play a card from the agent's hand.

    Args:
        card_index: Index of the card to play (0-indexed)

    Returns:
        Dictionary with the result of the action
    """
    # This function will be called with the agent_id and game_state from the graph
    # We'll implement a wrapper in the graph to provide these values

    # The actual implementation will be in _play_card_impl
    # This is just a placeholder that will be properly bound in the graph
    return {"success": False, "error": "Tool not properly bound to agent and game state"}


def _play_card_impl(agent_id: int, card_index: int, game_state: GameState) -> Dict[str, Any]:
    """
    Implementation of the play card tool with access to agent_id and game_state.

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
