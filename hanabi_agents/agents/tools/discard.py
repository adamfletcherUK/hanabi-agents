from typing import Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from ...game.state import GameState


class DiscardInput(BaseModel):
    """Schema for discarding a card from the agent's hand."""
    card_index: int = Field(
        description="The position of the card to discard (1-indexed: first card = 1, second card = 2, etc.)",
        # 1-indexed input should be at least 1
        ge=1,
        # 1-indexed input should be at most 5 (for a standard hand size of 5 cards)
        le=5
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
    clue_tokens: int = Field(
        default=0, description="Number of clue tokens after discarding")


@tool(args_schema=DiscardInput)
def discard_tool(card_index: int) -> Dict[str, Any]:
    """
    Discard a card from the agent's hand.

    Args:
        card_index: Position of the card to discard (1-indexed: first card = 1, second card = 2, etc.)

    Returns:
        Dictionary with the result of the action, including:
        - success: Whether the action was successful
        - error: Error message if the action failed
        - action_type: Type of action performed
        - card_info: Information about the discarded card
        - clue_tokens: Number of clue tokens after discarding
    """
    # This function will be called with the agent_id and game_state from the graph
    return DiscardOutput(
        success=False,
        error="Tool not properly bound to agent and game state"
    ).model_dump()


def _discard_impl(agent_id: int, card_index: int, game_state: GameState) -> Dict[str, Any]:
    """
    Implementation of the discard tool with access to agent_id and game_state.

    Args:
        agent_id: ID of the agent discarding the card
        card_index: Index of the card to discard (0-indexed, after conversion from 1-indexed input)
        game_state: Current state of the game

    Returns:
        Dictionary with the result of the action
    """
    # Validate the action
    if not game_state.is_valid_move(agent_id, "discard", card_index=card_index):
        return DiscardOutput(
            success=False,
            error=f"Invalid discard action: card_index={card_index}",
            action_type="discard"
        ).model_dump()

    # Get the card being discarded (for information purposes)
    hand = game_state.hands.get(agent_id, [])
    if 0 <= card_index < len(hand):
        card = hand[card_index]
        card_info = f"{card.color.value} {card.number}"
    else:
        card_info = "unknown card"

    # We assume the discard is successful if we've reached this point
    # This is just for testing the tool outside of the actual game engine
    return DiscardOutput(
        success=True,
        action_type="discard",
        card_info=card_info,
        error="",
        clue_tokens=min(game_state.clue_tokens + 1, game_state.max_clue_tokens)
    ).model_dump()
