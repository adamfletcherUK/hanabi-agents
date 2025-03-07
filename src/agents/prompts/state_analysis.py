from ...game.state import GameState
from ..formatters.game_state import format_firework_piles, format_discard_pile, format_hand


def create_state_analysis_prompt(game_state: GameState, agent_id: int, memory: dict = None) -> str:
    """Create prompt for analyzing the game state."""
    # Get game context from memory if available
    game_context = ""
    if memory is not None and "game_summary" in memory:
        game_context = f"\nGame context: {memory['game_summary']}\n"

    return f"""You are Agent {agent_id} in a game of Hanabi.
Current game state:
- Firework piles: {format_firework_piles(game_state)}
- Discard pile: {format_discard_pile(game_state)}
- Clue tokens: {game_state.clue_tokens}
- Fuse tokens: {game_state.fuse_tokens}
- Your hand: {format_hand(game_state.hands[agent_id])}{game_context}s

EXTREMELY IMPORTANT INSTRUCTIONS:
- Your ENTIRE response must be under 100 words
- Use simple, direct language with no fluff
- Focus only on the most important observations
- Do not use any special formatting, bullet points, or section headers
- Write in plain text only"""
