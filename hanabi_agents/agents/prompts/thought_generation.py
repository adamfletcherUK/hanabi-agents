from typing import Dict, Any, List, Optional
from ...game.state import GameState


def create_thought_generation_prompt(
    game_state: GameState,
    agent_id: int,
    card_knowledge: List[Dict[str, Any]],
    discussion_history: List[Dict[str, Any]],
    game_history: List[Dict[str, Any]],
    previous_analysis: Optional[str] = None,
    recent_errors: List[Dict[str, Any]] = None
) -> str:
    """
    Create a prompt for generating strategic thoughts.

    Args:
        game_state: Current state of the game
        agent_id: ID of the agent
        card_knowledge: Knowledge about the agent's cards
        discussion_history: History of discussion contributions
        game_history: History of game actions
        previous_analysis: Analysis from the previous step
        recent_errors: Recent errors from failed actions

    Returns:
        Prompt for generating thoughts
    """
    prompt = f"""
# Hanabi Strategic Thinking

You are Player {agent_id} in a game of Hanabi. Based on the game state analysis, generate strategic thoughts about what to do next.

## Current Game State Summary

### Resources
- Clue tokens: {game_state.clue_tokens}/{game_state.max_clue_tokens}
- Fuse tokens: {game_state.fuse_tokens}/3
- Cards left in deck: {len(game_state.deck)}
- Current score: {game_state.score}/25
- Current player: Player {game_state.current_player}
- Turn count: {game_state.turn_count}

### Firework Piles
"""

    # Add firework piles
    for color, pile in game_state.firework_piles.items():
        top_card = pile[-1] if pile else None
        value = top_card.number if top_card else 0
        prompt += f"- {color.value.capitalize()}: {value}\n"

    prompt += "\n### Players' Hands\n"

    # Add other players' hands
    for player_id, hand in game_state.hands.items():
        if player_id != agent_id:
            prompt += f"\nPlayer {player_id}'s hand:\n"
            for i, card in enumerate(hand):
                prompt += f"- Card {i}: {card.color.value} {card.number}\n"

    # Add your hand (with hidden information)
    prompt += f"\nYour hand (Player {agent_id}):\n"
    for i, knowledge in enumerate(card_knowledge):
        if knowledge.get("color_clued"):
            color_info = f"Known: {knowledge['color_clued']}"
        else:
            color_info = "Unknown"

        if knowledge.get("number_clued"):
            number_info = f"Known: {knowledge['number_clued']}"
        else:
            number_info = "Unknown"

        possible_colors = ", ".join(
            knowledge["possible_colors"]) if knowledge.get("possible_colors") else "All colors possible"
        possible_numbers = ", ".join(map(
            str, knowledge["possible_numbers"])) if knowledge.get("possible_numbers") else "All numbers possible"

        prompt += f"- Card {i}: Color: {color_info} (Possible: {possible_colors}), Number: {number_info} (Possible: {possible_numbers})\n"

    prompt += "\n## Your Previous Analysis\n"

    if previous_analysis:
        prompt += previous_analysis
    else:
        prompt += "No previous analysis available."

    prompt += """
## Thought Generation Task

Generate a list of strategic thoughts about the current game state. Consider:

1. What do you know about your own cards based on clues and game context?
2. What is the most valuable action you can take right now?
3. What information do your teammates need?
4. What risks are worth taking in the current state?
5. How can you best use the available clue tokens?

IMPORTANT: Format your response EXACTLY as a numbered list of thoughts. Each thought must be on a new line and start with a number followed by a period. For example:

1. I know my first card is red based on the clue from Player 2.
2. Playing my first card seems safe since it's likely a red 1.
3. Player 2 might need information about their blue cards.
4. We should prioritize playing cards that advance the fireworks.
5. I should avoid discarding cards that might be needed later.

DO NOT use any other format. DO NOT include any tool calls or function calls in this response.
Be specific and strategic in your thinking.
"""

    # Add error information if available
    if recent_errors and len(recent_errors) > 0:
        prompt += "\n## Recent Errors to Consider\n"
        for error in recent_errors:
            action_type = error.get("action_type", "unknown")
            guidance = error.get("guidance", "No guidance available.")
            prompt += f"- Error with {action_type} action: {guidance}\n"

    return prompt
