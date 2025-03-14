from typing import Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from ...game.state import GameState


class PlayCardInput(BaseModel):
    """Schema for playing a card from the agent's hand."""
    card_index: int = Field(
        description="The position of the card to play (1-indexed: first card = 1, second card = 2, etc.)",
        # 1-indexed input should be at least 1
        ge=1,
        # 1-indexed input should be at most 5 (for a standard hand size of 5 cards)
        le=5
    )


class PlayCardOutput(BaseModel):
    """Schema for the output of playing a card."""
    success: bool = Field(description="Whether the action was successful")
    error: str = Field(
        default="", description="Error message if the action failed")
    action_type: str = Field(
        default="play_card", description="Type of action performed")
    card_info: str = Field(
        default="", description="Information about the played card")


@tool(args_schema=PlayCardInput)
def play_card_tool(card_index: int) -> Dict[str, Any]:
    """
    Play a card from the agent's hand.

    Args:
        card_index: Position of the card to play (1-indexed: first card = 1, second card = 2, etc.)

    Returns:
        Dictionary with the result of the action, including:
        - success: Whether the action was successful
        - error: Error message if the action failed
        - action_type: Type of action performed
        - card_info: Information about the played card
    """
    # This function will be called with the agent_id and game_state from the graph
    return PlayCardOutput(
        success=False,
        error="Tool not properly bound to agent and game state"
    ).model_dump()


def _play_card_impl(agent_id: int, card_index: int, game_state: GameState) -> Dict[str, Any]:
    """
    Implementation of the play card tool with access to agent_id and game_state.

    Args:
        agent_id: ID of the agent playing the card
        card_index: Index of the card to play (0-indexed, after conversion from 1-indexed input)
        game_state: Current state of the game

    Returns:
        Dictionary with the result of the action
    """
    # Validate the action
    if not game_state.is_valid_move(agent_id, "play_card", card_index=card_index):
        return PlayCardOutput(
            success=False,
            error=f"Invalid play_card action: card_index={card_index}",
            action_type="play_card"
        ).model_dump()

    # Get the card being played (for information purposes)
    hand = game_state.hands.get(agent_id, [])
    if 0 <= card_index < len(hand):
        card = hand[card_index]
        card_info = f"{card.color.value} {card.number}"
    else:
        card_info = "unknown card"

    return PlayCardOutput(
        success=True,
        action_type="play_card",
        card_info=card_info,
        error=""
    ).model_dump()
