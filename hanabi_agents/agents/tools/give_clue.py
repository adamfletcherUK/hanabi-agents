from typing import Dict, Any, Literal
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from ...game.state import GameState, Color


class GiveClueInput(BaseModel):
    """Schema for giving a clue to another player."""
    target_id: int = Field(
        description="ID of the player to give the clue to",
        ge=0  # greater than or equal to 0
    )
    clue_type: Literal["color", "number"] = Field(
        description="Type of clue to give (color or number)"
    )
    clue_value: str = Field(
        description="Value of the clue (e.g., 'red', '1')"
    )


class GiveClueOutput(BaseModel):
    """Schema for the output of giving a clue."""
    success: bool = Field(description="Whether the action was successful")
    error: str = Field(
        default="", description="Error message if the action failed")
    action_type: str = Field(
        default="give_clue", description="Type of action performed")
    clue_info: str = Field(
        default="", description="Information about the given clue")


@tool(args_schema=GiveClueInput)
def give_clue_tool(target_id: int, clue_type: str, clue_value: str) -> Dict[str, Any]:
    """
    Give a clue to another player about their cards.

    Args:
        target_id: ID of the player to give the clue to
        clue_type: Type of clue to give (color or number)
        clue_value: Value of the clue (e.g., 'red', '1')

    Returns:
        Dictionary with the result of the action, including:
        - success: Whether the action was successful
        - error: Error message if the action failed
        - action_type: Type of action performed
        - clue_info: Information about the given clue
    """
    # This function will be called with the agent_id and game_state from the graph
    # We'll implement a wrapper in the graph to provide these values

    # The actual implementation will be in _give_clue_impl
    # This is just a placeholder that will be properly bound in the graph
    return GiveClueOutput(
        success=False,
        error="Tool not properly bound to agent and game state"
    ).dict()


def _give_clue_impl(agent_id: int, target_id: int, clue_type: str, clue_value: str, game_state: GameState) -> Dict[str, Any]:
    """
    Implementation of the give clue tool with access to agent_id and game_state.

    Args:
        agent_id: ID of the agent giving the clue
        target_id: ID of the player to give the clue to
        clue_type: Type of clue to give (color or number)
        clue_value: Value of the clue (e.g., 'red', '1')
        game_state: Current state of the game

    Returns:
        Dictionary with the result of the action
    """
    # Validate the action
    if game_state.clue_tokens <= 0:
        return GiveClueOutput(
            success=False,
            error="No clue tokens available",
            action_type="give_clue"
        ).dict()

    if target_id == agent_id:
        return GiveClueOutput(
            success=False,
            error="Cannot give clue to yourself",
            action_type="give_clue"
        ).dict()

    if target_id not in game_state.hands:
        return GiveClueOutput(
            success=False,
            error=f"Invalid target player: {target_id}",
            action_type="give_clue"
        ).dict()

    target_hand = game_state.hands[target_id]
    if not target_hand:
        return GiveClueOutput(
            success=False,
            error=f"Player {target_id} has no cards",
            action_type="give_clue"
        ).dict()

    # Check if the clue matches any cards
    matches = False
    for card in target_hand:
        if (clue_type == "color" and card.color.value == clue_value) or \
           (clue_type == "number" and str(card.number) == str(clue_value)):
            matches = True
            break

    if not matches:
        return GiveClueOutput(
            success=False,
            error=f"No {clue_type} {clue_value} cards in player {target_id}'s hand",
            action_type="give_clue"
        ).dict()

    return GiveClueOutput(
        success=True,
        action_type="give_clue",
        clue_info=f"Gave {clue_type} clue {clue_value} to player {target_id}",
        error=""
    ).dict()
