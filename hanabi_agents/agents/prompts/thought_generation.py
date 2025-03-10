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

## Your Previous Analysis
"""

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

Format your thoughts as a numbered list. Be specific and strategic in your thinking.
"""

    # Add error information if available
    if recent_errors and len(recent_errors) > 0:
        prompt += "\n## Recent Errors to Consider\n"
        for error in recent_errors:
            action_type = error.get("action_type", "unknown")
            guidance = error.get("guidance", "No guidance available.")
            prompt += f"- Error with {action_type} action: {guidance}\n"

    return prompt
