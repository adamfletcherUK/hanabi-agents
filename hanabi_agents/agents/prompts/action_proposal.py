from typing import Dict, Any, List
from ...game.state import GameState


def create_action_proposal_prompt(
    game_state: GameState,
    agent_id: int,
    card_knowledge: List[Dict[str, Any]],
    current_thoughts: List[str],
    discussion_history: List[Dict[str, Any]],
    game_history: List[Dict[str, Any]],
    recent_errors: List[Dict[str, Any]] = None
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
        recent_errors: Recent errors from failed actions

    Returns:
        Prompt for proposing an action
    """
    prompt = f"""
# Hanabi Action Proposal

You are Player {agent_id} in a game of Hanabi. Based on your analysis and strategic thoughts, propose a specific action to take.

## Current Game State Summary

- Clue tokens: {game_state.clue_tokens}/{game_state.max_clue_tokens}
- Fuse tokens: {game_state.fuse_tokens}/3
- Current score: {game_state.score}/25
- Current player: Player {game_state.current_player} ({"You" if game_state.current_player == agent_id else "Not you"})

## Your Strategic Thoughts
"""

    # Add the current thoughts
    for i, thought in enumerate(current_thoughts):
        prompt += f"{i+1}. {thought}\n"

    prompt += """
## Available Actions

You can take one of the following actions:

1. Play a card: `play card X` (where X is the index of the card in your hand, 0-indexed)
2. Give a clue: `give clue to player Y about color Z` or `give clue to player Y about number Z`
3. Discard a card: `discard card X` (where X is the index of the card in your hand, 0-indexed)

## Action Proposal Task

Propose a single, specific action to take based on your analysis and thoughts. Be explicit about:
- The exact action type (play, clue, or discard)
- The specific card index or clue details
- Why this is the best action to take right now

Your response should clearly state the action in a format that can be parsed, such as:
"I will play card 2" or "I will give a clue to player 1 about red cards" or "I will discard card 0"
"""

    # Add error information if available
    if recent_errors and len(recent_errors) > 0:
        prompt += "\n## Recent Errors to Consider\n"
        for error in recent_errors:
            action_type = error.get("action_type", "unknown")
            guidance = error.get("guidance", "No guidance available.")
            prompt += f"- Error with {action_type} action: {guidance}\n"

    return prompt
