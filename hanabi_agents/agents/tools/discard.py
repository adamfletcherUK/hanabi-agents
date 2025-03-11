from typing import Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from ...game.state import GameState


class DiscardInput(BaseModel):
    """Schema for discarding a card from the agent's hand."""
    card_index: int = Field(
        description="Index of the card to discard (0-indexed)",
        ge=0,  # greater than or equal to 0
        le=4   # less than or equal to 4
    )


class DiscardOutput(BaseModel):
    """Schema for the output of discarding a card."""
    success: bool = Field(description="Whether the action was successful")
    error: str = Field(
        default="", description="Error message if the action failed")
    action_type: str = Field(
        default="discard", description="Type of action performed")
    card_info: str = Field(
        default="", description="Information about the discarded card")


@tool(args_schema=DiscardInput)
def discard_tool(card_index: int) -> Dict[str, Any]:
    """
    Discard a card from the agent's hand.

    Args:
        card_index: Index of the card to discard (0-indexed, must be between 0 and 4)

    Returns:
        Dictionary with the result of the action, including:
        - success: Whether the action was successful
        - error: Error message if the action failed
        - action_type: Type of action performed
        - card_info: Information about the discarded card
    """
    # This function will be called with the agent_id and game_state from the graph
    return DiscardOutput(
        success=False,
        error="Tool not properly bound to agent and game state"
    ).dict()


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
    # Validate the action
    if game_state.clue_tokens >= game_state.max_clue_tokens:
        return DiscardOutput(
            success=False,
            error="Cannot discard when clue tokens are at maximum",
            action_type="discard"
        ).dict()

    # Get the card being discarded (for information purposes)
    hand = game_state.hands.get(agent_id, [])
    if not (0 <= card_index < len(hand)):
        return DiscardOutput(
            success=False,
            error=f"Invalid card index: {card_index}. Must be between 0 and {len(hand)-1}",
            action_type="discard"
        ).dict()

    card = hand[card_index]
    card_info = f"{card.color.value} {card.number}"

    return DiscardOutput(
        success=True,
        action_type="discard",
        card_info=card_info,
        error=""
    ).dict()
