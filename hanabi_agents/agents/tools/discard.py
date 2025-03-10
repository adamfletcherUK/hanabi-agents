from typing import Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from ...game.state import GameState


class DiscardInput(BaseModel):
    card_index: int = Field(
        description="Index of the card to discard (0-indexed)")


@tool(args_schema=DiscardInput)
def discard_tool(card_index: int) -> Dict[str, Any]:
    """
    Discard a card from the agent's hand.

    Args:
        card_index: Index of the card to discard (0-indexed)

    Returns:
        Dictionary with the result of the action
    """
    # This function will be called with the agent_id and game_state from the graph
    # We'll implement a wrapper in the graph to provide these values

    # The actual implementation will be in _discard_impl
    # This is just a placeholder that will be properly bound in the graph
    return {"success": False, "error": "Tool not properly bound to agent and game state"}


def _discard_impl(agent_id: int, card_index: int, game_state: GameState) -> Dict[str, Any]:
    """
    Implementation of the discard tool with access to agent_id and game_state.

    Args:
        agent_id: ID of the agent discarding the card
        card_index: Index of the card to discard (0-indexed)
        game_state: Current state of the game

    Returns:
        Dictionary with the result of the action
    """
    # Explicitly check for max clue tokens first
    if game_state.clue_tokens >= game_state.max_clue_tokens:
        return {
            "success": False,
            "error": f"Cannot discard when clue tokens are at maximum ({game_state.max_clue_tokens})",
            "action_type": "discard",
            "guidance": "When at max clue tokens, you must either play a card or give a clue instead of discarding."
        }

    # Validate the action
    if not game_state.is_valid_move(agent_id, "discard", card_index=card_index):
        return {
            "success": False,
            "error": f"Invalid discard action: card_index={card_index}",
            "action_type": "discard"
        }

    # Get the card being discarded (for information purposes)
    hand = game_state.hands.get(agent_id, [])
    if 0 <= card_index < len(hand):
        card = hand[card_index]
        card_info = f"{card.color.value} {card.number}"
    else:
        card_info = "unknown card"

    return {
        "success": True,
        "action_type": "discard",
        "card_index": card_index,
        "card_info": card_info,
        "message": f"Discarding card at index {card_index} ({card_info})"
    }
