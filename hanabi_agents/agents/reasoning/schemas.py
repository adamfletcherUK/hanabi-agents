from typing import Literal, Union
from pydantic import BaseModel, Field


class PlayCardAction(BaseModel):
    """Schema for playing a card action."""
    type: Literal["play_card"] = Field(description="Type of action")
    card_index: int = Field(
        description="Index of the card to play (1-indexed: first card = 1, second card = 2, etc.)",
        # 1-indexed values should be at least 1
        ge=1,
        # 1-indexed values should be at most 5 (for a standard hand size of 5 cards)
        le=5
    )


class GiveClueAction(BaseModel):
    """Schema for giving a clue action."""
    type: Literal["give_clue"] = Field(description="Type of action")
    target_id: int = Field(
        description="ID of the player to give the clue to",
        ge=0  # greater than or equal to 0
    )
    clue: dict = Field(
        description="Clue information",
        example={
            "type": "color",
            "value": "red"
        }
    )


class DiscardAction(BaseModel):
    """Schema for discarding a card action."""
    type: Literal["discard"] = Field(description="Type of action")
    card_index: int = Field(
        description="Index of the card to discard (1-indexed: first card = 1, second card = 2, etc.)",
        # 1-indexed values should be at least 1
        ge=1,
        # 1-indexed values should be at most 5 (for a standard hand size of 5 cards)
        le=5
    )


class ActionProposal(BaseModel):
    """Schema for the complete action proposal."""
    action: Union[PlayCardAction, GiveClueAction, DiscardAction] = Field(
        description="The proposed action to take"
    )
    explanation: str = Field(
        description="Explanation of why this action was chosen"
    )
