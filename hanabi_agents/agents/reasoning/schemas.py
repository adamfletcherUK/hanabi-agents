from typing import Literal, Union
from pydantic import BaseModel, Field


class PlayCardAction(BaseModel):
    """Schema for playing a card action."""
    type: Literal["play_card"] = Field(description="Type of action")
    card_index: int = Field(
        description="Index of the card to play (1-indexed: first card = 1, second card = 2, etc.)",
        # greater than or equal to 0 (validation will accept 0 though we expect 1-indexed input)
        ge=0,
        # less than or equal to 4 (validation will accept 0-4 though we expect 1-5)
        le=4
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
        # greater than or equal to 0 (validation will accept 0 though we expect 1-indexed input)
        ge=0,
        # less than or equal to 4 (validation will accept 0-4 though we expect 1-5)
        le=4
    )


class ActionProposal(BaseModel):
    """Schema for the complete action proposal."""
    action: Union[PlayCardAction, GiveClueAction, DiscardAction] = Field(
        description="The proposed action to take"
    )
    explanation: str = Field(
        description="Explanation of why this action was chosen"
    )
