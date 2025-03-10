from typing import Dict, Any, List
from ...game.state import GameState


def create_state_analysis_prompt(
    game_state: GameState,
    agent_id: int,
    card_knowledge: List[Dict[str, Any]],
    discussion_history: List[Dict[str, Any]],
    game_history: List[Dict[str, Any]]
) -> str:
    """
    Create a prompt for analyzing the current game state.

    Args:
        game_state: Current state of the game
        agent_id: ID of the agent
        card_knowledge: Knowledge about the agent's cards
        discussion_history: History of discussion contributions
        game_history: History of game actions

    Returns:
        Prompt for analyzing the game state
    """
    prompt = f"""
# Hanabi Game State Analysis

You are Player {agent_id} in a game of Hanabi. Analyze the current game state to understand the situation.

## Game Rules Reminder
- Hanabi is a cooperative card game where players work together to build 5 firework stacks (one for each color) in ascending order (1-5).
- Players cannot see their own cards but can see everyone else's cards.
- On your turn, you must take one of three actions:
  1. Play a card from your hand onto the appropriate firework pile.
  2. Give a clue to another player (costs 1 clue token).
  3. Discard a card to gain 1 clue token.
- When playing a card:
  - If it's the next card in sequence for its color, it's added to that firework pile.
  - If it's not playable, a fuse token is lost.
- The game ends when:
  - All 5 firework piles are complete (perfect score of 25).
  - All 3 fuse tokens are lost (the game is lost).
  - The deck is empty and each player has taken one more turn.

## Current Game State

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
        color_info = "Known: " + \
            knowledge["color_clued"] if knowledge["color_clued"] else "Unknown"
        number_info = "Known: " + \
            str(knowledge["number_clued"]
                ) if knowledge["number_clued"] else "Unknown"

        possible_colors = ", ".join(
            knowledge["possible_colors"]) if knowledge["possible_colors"] else "All colors possible"
        possible_numbers = ", ".join(map(
            str, knowledge["possible_numbers"])) if knowledge["possible_numbers"] else "All numbers possible"

        prompt += f"- Card {i}: Color: {color_info} (Possible: {possible_colors}), Number: {number_info} (Possible: {possible_numbers})\n"

    # Add discard pile
    prompt += "\n### Discard Pile\n"
    if game_state.discard_pile:
        for card in game_state.discard_pile:
            prompt += f"- {card.color.value} {card.number}\n"
    else:
        prompt += "- Empty\n"

    # Add recent game history
    prompt += "\n### Recent Game History\n"
    recent_history = game_history[-5:] if game_history else []
    if recent_history:
        for entry in recent_history:
            prompt += f"- {entry['description']}\n"
    else:
        prompt += "- No history yet\n"

    # Add recent discussion
    prompt += "\n### Recent Discussion\n"
    recent_discussion = discussion_history[-3:] if discussion_history else []
    if recent_discussion:
        for entry in recent_discussion:
            if "player_id" in entry and "content" in entry:
                prompt += f"- Player {entry['player_id']}: {entry['content']}\n"
            elif "summary" in entry:
                prompt += f"- Summary: {entry['summary']}\n"
    else:
        prompt += "- No discussion yet\n"

    prompt += """
## Analysis Task
Analyze the current game state and provide insights on:
1. What cards are in your hand based on clues and game context
2. What cards are playable in the current state
3. What cards are safe to discard
4. What information other players need
5. The overall team strategy based on the current state

Be thorough in your analysis, considering all available information.
"""

    return prompt
