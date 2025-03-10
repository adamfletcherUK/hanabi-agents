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

## Other Players' Hands and Valid Clues
"""

    # Add information about other players' hands and valid clues
    for player_id, hand in game_state.hands.items():
        if player_id != agent_id:
            prompt += f"\nPlayer {player_id}'s hand:\n"

            # Show the cards in the player's hand
            for i, card in enumerate(hand):
                prompt += f"- Card {i}: {card.color.value} {card.number}\n"

            # Calculate valid clues for this player
            valid_colors = set()
            valid_numbers = set()

            for card in hand:
                valid_colors.add(card.color.value)
                valid_numbers.add(card.number)

            # Add valid clue information
            prompt += f"\nValid clues for Player {player_id}:\n"
            prompt += f"- Colors: {', '.join(sorted(valid_colors))}\n"
            prompt += f"- Numbers: {', '.join(str(n) for n in sorted(valid_numbers))}\n"

    prompt += """
## Your Strategic Thoughts
"""

    # Add the current thoughts
    for i, thought in enumerate(current_thoughts):
        prompt += f"{i+1}. {thought}\n"

    prompt += """
## Available Tools

You have access to the following tools:

1. play_card_tool: Play a card from your hand
   - card_index: Index of the card to play (0-indexed)

2. give_clue_tool: Give a clue to another player
   - target_id: ID of the player to give the clue to
   - clue_type: Type of clue to give ("color" or "number")
   - clue_value: Value of the clue (e.g., "red", "1")

3. discard_tool: Discard a card from your hand
   - card_index: Index of the card to discard (0-indexed)
"""

    # Add a clear warning about discarding when at max clue tokens
    if game_state.clue_tokens >= game_state.max_clue_tokens:
        prompt += f"""
## IMPORTANT RESTRICTION
⚠️ You currently have {game_state.clue_tokens}/{game_state.max_clue_tokens} clue tokens, which is the maximum.
⚠️ You CANNOT discard when at maximum clue tokens.
⚠️ You MUST either play a card or give a clue.
"""

    prompt += """
## Action Proposal Task

Based on your analysis and thoughts, you must call ONE of the available tools to take an action in the game.

IMPORTANT: 
- You MUST call a tool - responding with natural language only is not allowed.
- Your tool call MUST directly address your strategic thoughts listed above.
- For EACH thought, explain how your chosen action addresses or relates to that thought.
- Your action should be the logical conclusion of your strategic reasoning.
- When giving clues, make sure the clue will actually affect at least one card in the target player's hand.
- Check the "Valid clues" section above to ensure your clue is valid.
- You MUST use one of the tools above - do not respond with natural language.
- Only call ONE tool.

Before calling the tool, provide a brief explanation of how your chosen action addresses each of your thoughts, numbered to match the thought list above.
"""

    # Add error information if available
    if recent_errors and len(recent_errors) > 0:
        prompt += "\n## Recent Errors to Consider\n"
        for error in recent_errors:
            action_type = error.get("action_type", "unknown")
            guidance = error.get("guidance", "No guidance available.")
            prompt += f"- Error with {action_type} action: {guidance}\n"

    return prompt
