from typing import List
from ...game.state import GameState, Color
from ..formatters.game_state import format_firework_piles, format_discard_pile, format_hand
from ..formatters.discussion import format_discussion
from ..formatters.thoughts import format_thoughts


def create_thought_generation_prompt(game_state: GameState,
                                     discussion_history: List[str],
                                     current_thoughts: List[str],
                                     agent_id: int,
                                     memory: dict = None) -> str:
    """Create a prompt for generating thoughts about the game state."""
    # Determine if this agent is the active player or providing feedback
    is_active_player = agent_id == game_state.current_player
    active_player_id = game_state.current_player

    # Get the active player's proposal if this agent is providing feedback
    active_player_proposal = ""
    if not is_active_player and discussion_history:
        active_player_proposal = discussion_history[0]

    # Format the game state information
    firework_piles = format_firework_piles(game_state)
    discard_pile = format_discard_pile(game_state)

    # Format information about other players' hands
    other_players_info = []
    for player_id, hand in game_state.hands.items():
        if player_id != agent_id:
            hand_str = format_hand(hand)

            # Add information about what clues would be valid for this player
            valid_color_clues = set()
            valid_number_clues = set()
            for card in hand:
                valid_color_clues.add(card.color.value)
                valid_number_clues.add(str(card.number))

            clue_info = f"Valid clues for Player {player_id}: "
            clue_info += f"Colors: {', '.join(sorted(valid_color_clues))}, "
            clue_info += f"Numbers: {', '.join(sorted(valid_number_clues))}"

            other_players_info.append(
                f"Player {player_id}'s hand: {hand_str}\n{clue_info}")

    other_players = "\n".join(other_players_info)

    # Format information about the agent's own hand (which they can't see)
    my_hand = "Your hand (you cannot see the actual cards):"
    if agent_id in game_state.hands:
        for i, card in enumerate(game_state.hands[agent_id]):
            card_info = f"\nCard {i}: [HIDDEN]"

            # Add clue information if available
            if card.is_visible:
                clues = []
                if hasattr(card, "color_clued") and card.color_clued:
                    clues.append(f"color: {card.color.value}")
                if hasattr(card, "number_clued") and card.number_clued:
                    clues.append(f"number: {card.number}")

                if not clues and memory is not None and "clue_history" in memory:
                    for clue in memory["clue_history"]:
                        if clue["receiver_id"] == agent_id and i in clue["affected_indices"]:
                            if clue["clue_type"] == "color":
                                clues.append(
                                    f"color: {clue['clue_value']}")
                            else:  # number clue
                                clues.append(
                                    f"number: {clue['clue_value']}")

                if clues:
                    card_info += f" ({', '.join(clues)})"
                    # Add inference information
                    inferences = []
                    # We could add logic here to generate inferences
                    # For now, we'll leave it empty
                else:
                    card_info += " (has received clues)"

            my_hand += card_info

    # Format the discussion history
    discussion = format_discussion(discussion_history)

    # Format the current thoughts
    thoughts = format_thoughts(current_thoughts)

    # Create the prompt
    prompt = f"""You are Agent {agent_id} in a game of Hanabi.

Current Game State:
- Score: {game_state.score}
- Clue tokens: {game_state.clue_tokens}
- Fuse tokens: {game_state.fuse_tokens}
- Current player: {game_state.current_player}
- Firework piles: {firework_piles}
- Discard pile: {discard_pile}

{other_players}

{my_hand}

CRITICAL INFORMATION RULES:
1. You MUST distinguish between KNOWN information and INFERENCES.
2. KNOWN information is ONLY what you've been explicitly told through clues.
3. INFERENCES are educated guesses based on game state, but are NOT certainties.
4. You MUST use language like "I believe", "I infer", "likely", "probably", "might be" for inferences.
5. You MUST use language like "I know" ONLY for information directly given through clues.
6. For example, if you received a "green" clue on a card, you can say "I know this card is green" but NOT "I know this is a green 1".
7. You CANNOT claim to know both color AND number of a card unless you've received BOTH clues for that card.
8. You CANNOT claim to know the exact identity of a card based solely on a single clue.

HANABI COMMUNICATION RULES:
1. You CANNOT directly tell other players what cards they have in their hands.
2. You CANNOT indirectly hint at specific card values outside of the official clue mechanism.
3. You CANNOT discuss specific card values you see in other players' hands.
4. You CANNOT say things like "Player 2 has a red 3" or "Player 3's second card is a 1".
5. You CANNOT say "I see a red 1 in your hand" or "Your third card is a 5".
6. You CANNOT say "You should play your red card" or "You should play your 1".
7. You CANNOT say "Give a red clue to Player 3" or "Give a 1 clue to Player 4".
8. You CAN say "I'll give information about Player 3's 2nd card".
9. You CAN say "I'll give a color clue to Player 3" (without specifying which color).
10. You CAN say "I'll give a number clue to Player 3" (without specifying which number).
11. You CAN discuss general strategy like "We should focus on building the red firework next".
12. You CAN say "Consider giving information about Player 3's first card".

"""

    # Add role-specific instructions
    if is_active_player:
        prompt += f"""You are the active player (Player {agent_id}) and need to decide on an action.
Think through the current game state, what you know about your hand, and what would be the most strategic move.
Consider the balance between giving clues, playing cards, and discarding.
Remember to clearly distinguish between what you KNOW from clues and what you INFER from the game state.
"""
    else:
        prompt += f"""You are providing feedback to Player {active_player_id}, who is the active player.
Their proposal is: {active_player_proposal}

Think about whether their proposed action is strategic and how it fits with the team's goals.
Consider if there might be better alternatives they should consider.
Remember to clearly distinguish between what you KNOW from clues and what you INFER from the game state.
"""

    # Add discussion history if available
    if discussion:
        prompt += f"\nDiscussion so far:\n{discussion}\n"

    # Add current thoughts if available
    if thoughts:
        prompt += f"\nYour current thoughts:\n{thoughts}\n"

    # Add final instruction
    prompt += """
Generate your next thoughts about the game state and potential actions.
Be concise and focus on the most important strategic considerations.
IMPORTANT: Clearly distinguish between what you KNOW from clues and what you INFER from the game state.
"""

    return prompt
