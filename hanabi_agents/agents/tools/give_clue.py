from typing import Dict, Any, Literal
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from ...game.state import GameState, Color


class GiveClueInput(BaseModel):
    target_id: int = Field(description="ID of the player to give the clue to")
    clue_type: Literal["color", "number"] = Field(
        description="Type of clue to give (color or number)")
    clue_value: str = Field(description="Value of the clue (e.g., 'red', '1')")


@tool(args_schema=GiveClueInput)
def give_clue_tool(target_id: int, clue_type: str, clue_value: str) -> Dict[str, Any]:
    """
    Give a clue to another player about their cards.

    Args:
        target_id: ID of the player to give the clue to
        clue_type: Type of clue to give (color or number)
        clue_value: Value of the clue (e.g., 'red', '1')

    Returns:
        Dictionary with the result of the action
    """
    # This function will be called with the agent_id and game_state from the graph
    # We'll implement a wrapper in the graph to provide these values

    # The actual implementation will be in _give_clue_impl
    # This is just a placeholder that will be properly bound in the graph
    return {"success": False, "error": "Tool not properly bound to agent and game state"}


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
    if not game_state.is_valid_move(agent_id, "clue", target_id=target_id, clue_type=clue_type, clue_value=clue_value):
        # Check for specific error conditions
        if game_state.clue_tokens <= 0:
            return {
                "success": False,
                "error": "Cannot give clue: no clue tokens available",
                "action_type": "clue"
            }

        if target_id == agent_id:
            return {
                "success": False,
                "error": "Cannot give clue to yourself",
                "action_type": "clue"
            }

        if target_id not in game_state.hands:
            return {
                "success": False,
                "error": f"Invalid target player: {target_id}",
                "action_type": "clue"
            }

        # Check if the clue would affect any cards
        target_hand = game_state.hands.get(target_id, [])
        affected_cards = []

        for i, card in enumerate(target_hand):
            if (clue_type == "color" and card.color.value == clue_value) or \
               (clue_type == "number" and str(card.number) == str(clue_value)):
                affected_cards.append(i)

        if not affected_cards:
            return {
                "success": False,
                "error": f"Clue {clue_type}={clue_value} wouldn't affect any cards in player {target_id}'s hand",
                "action_type": "clue"
            }

        # Generic error for other cases
        return {
            "success": False,
            "error": f"Invalid clue action: target_id={target_id}, clue_type={clue_type}, clue_value={clue_value}",
            "action_type": "clue"
        }

    # Get the affected cards
    target_hand = game_state.hands.get(target_id, [])
    affected_cards = []

    for i, card in enumerate(target_hand):
        if (clue_type == "color" and card.color.value == clue_value) or \
           (clue_type == "number" and str(card.number) == str(clue_value)):
            affected_cards.append(i)

    return {
        "success": True,
        "action_type": "clue",
        "target_id": target_id,
        "clue_type": clue_type,
        "clue_value": clue_value,
        "affected_cards": affected_cards,
        "message": f"Giving {clue_type} clue about {clue_value} to player {target_id}, affecting cards at positions {affected_cards}"
    }
