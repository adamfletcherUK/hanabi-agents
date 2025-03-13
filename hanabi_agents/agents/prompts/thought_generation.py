from typing import Dict, Any, List, Optional
from ...game.state import GameState
from .strategy_guidelines import get_strategy_guidelines


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
    # Get the strategy guidelines
    strategy_guidelines = get_strategy_guidelines()

    prompt = f"""
# Hanabi Strategic Thinking

You are Player {agent_id} in a game of Hanabi. Based on the game state analysis, generate strategic thoughts about what to do next.

## Strategy Guidelines
{strategy_guidelines}

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
                # Note: Using 1-indexing for display
                prompt += f"- Card {i+1}: {card.color.value} {card.number}\n"

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

        # Note: Using 1-indexing for display
        prompt += f"- Card {i+1}: Color: {color_info} (Possible: {possible_colors}), Number: {number_info} (Possible: {possible_numbers})\n"

    # Add information about recent failed actions if available
    if recent_errors and len(recent_errors) > 0:
        prompt += "\n## Recent Action Errors\n"
        for error in recent_errors:
            action_type = error.get("action", {}).get("type", "unknown")
            error_msg = error.get("error", "unknown error")
            prompt += f"- Failed {action_type}: {error_msg}\n"

    prompt += "\n## Your Previous Analysis\n"

    if previous_analysis:
        prompt += previous_analysis
    else:
        prompt += "No previous analysis available."

    prompt += f"""
## Game History (Last 3 Turns)
"""
    # Add recent game history
    recent_history = game_history[-3:] if game_history and len(
        game_history) > 0 else []
    for i, event in enumerate(recent_history):
        player = event.get("player", "?")
        action_type = event.get("action", {}).get("type", "unknown")
        if action_type == "play_card":
            card_index = event.get("action", {}).get("card_index", "?")
            card_info = event.get("result", {}).get("card", "unknown card")
            success = event.get("result", {}).get("success", False)
            status = "succeeded" if success else "failed"
            # Note: Using 1-indexing for display
            prompt += f"- Turn {event.get('turn', '?')}: Player {player} played card at position {card_index+1} ({card_info}) - {status}\n"
        elif action_type == "give_clue":
            target = event.get("action", {}).get("target_id", "?")
            clue_type = event.get("action", {}).get(
                "clue", {}).get("type", "?")
            clue_value = event.get("action", {}).get(
                "clue", {}).get("value", "?")
            prompt += f"- Turn {event.get('turn', '?')}: Player {player} gave {clue_type} clue '{clue_value}' to Player {target}\n"
        elif action_type == "discard":
            card_index = event.get("action", {}).get("card_index", "?")
            card_info = event.get("result", {}).get("card", "unknown card")
            # Note: Using 1-indexing for display
            prompt += f"- Turn {event.get('turn', '?')}: Player {player} discarded card at position {card_index+1} ({card_info})\n"

    prompt += """
## Thought Generation Task

Generate a list of strategic thoughts about the current game state. Consider:

1. What do you know about your own cards based on clues and game context?
2. What is the most valuable action you can take right now based on the strategy guidelines?
3. What information do your teammates need?
4. Is it safer to play a card, give a clue, or discard in the current situation?
5. If playing a card, which position (1-5) is safest and most beneficial based on your knowledge?
6. If discarding, which position (1-5) is least likely to hurt the team?
7. If giving a clue, what specific information will be most valuable to your teammates?

IMPORTANT: 
- Format your response EXACTLY as a numbered list of thoughts. Each thought must be on a new line and start with a number followed by a period.
- Refer to card positions using 1-indexed positions (first card = 1, second card = 2, etc.)
- Apply the strategy guidelines above to make optimal decisions.
- Be explicit about why you believe certain cards are playable or safe to discard.

Example format:
1. I know my first card (position 1) is a red 2 based on previous clues.
2. The red 1 is already played, so my red 2 is immediately playable.
3. Player 2 has a blue 1 in position 3 that they don't know about.
4. I should play my red 2 (position 1) to advance the red firework pile.
"""

    return prompt
