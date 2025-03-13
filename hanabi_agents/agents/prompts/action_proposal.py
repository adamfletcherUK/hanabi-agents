from typing import Dict, Any, List
from ...game.state import GameState
from ..reasoning.schemas import ActionProposal


def create_action_proposal_prompt(
    game_state: Dict[str, Any],
    agent_id: int,
    card_knowledge: Dict[str, Any],
    current_thoughts: List[str],
    discussion_history: List[Dict[str, Any]],
    game_history: List[Dict[str, Any]]
) -> str:
    """
    Create a prompt for proposing an action.

    Args:
        game_state: Current state of the game
        agent_id: ID of the agent
        card_knowledge: Knowledge about the agent's cards
        current_thoughts: Current thoughts about the game state
        discussion_history: History of discussion contributions
        game_history: History of game actions

    Returns:
        Prompt for proposing an action
    """
    # Get the schema as a JSON string
    schema_json = ActionProposal.schema_json(indent=2)

    return f"""Based on the current game state and your thoughts, propose an action to take.
    
Game State:
{game_state}

Your Thoughts:
{current_thoughts}

Discussion History:
{discussion_history}

Game History:
{game_history}

You must return a JSON object that matches this schema exactly:
{schema_json}

The action must be one of:
- play_card: Play a card from your hand (card_index: 1-5, where 1 is your first card, 2 is your second card, etc.)
- give_clue: Give a clue to another player (target_id: player ID, clue: {{type: "color"|"number", value: string}})
- discard: Discard a card from your hand (card_index: 1-5, where 1 is your first card, 2 is your second card, etc.)

Important rules:
1. The response must be a valid JSON object
2. The action must be valid for the current game state
3. The explanation should clearly justify your choice
4. Do not add any fields not in the schema
5. Do not modify the schema structure
6. When referring to card positions, always use 1-indexed positions (first card = 1, second card = 2, etc.)

Provide your response as a valid JSON object that matches the schema exactly."""
