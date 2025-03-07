from typing import List
from ...game.state import GameState, Color
from ..formatters.game_state import format_firework_piles, format_discard_pile, format_hand
from ..formatters.discussion import format_discussion
from ..formatters.thoughts import format_thoughts


def generate_valid_actions_info(game_state: GameState, agent_id: int) -> str:
    """Generate information about valid actions based on the current game state."""
    valid_actions = []

    # Check if giving clues is valid
    if game_state.clue_tokens > 0:
        valid_actions.append(
            "- You can give clues (clue tokens available: {})".format(game_state.clue_tokens))

        # List valid targets for clues
        valid_targets = [i for i in range(
            len(game_state.hands)) if i != agent_id]
        valid_actions.append("  Valid targets for clues: {}".format(
            ", ".join(map(str, valid_targets))))
    else:
        valid_actions.append(
            "- You CANNOT give clues (no clue tokens available)")

    # Check if discarding is valid
    if game_state.clue_tokens < 8:
        valid_actions.append(
            "- You can discard cards (clue tokens: {}/8)".format(game_state.clue_tokens))

        # List valid indices for discarding
        valid_indices = list(range(len(game_state.hands[agent_id])))
        valid_actions.append("  Valid indices for discarding: {}".format(
            ", ".join(map(str, valid_indices))))
    else:
        valid_actions.append(
            "- You CANNOT discard cards (clue tokens already at maximum: 8/8)")

    # Information about playing cards
    valid_actions.append(
        "- You can play a card if it's the next in sequence for its color")

    # Current state of firework piles
    firework_info = []
    for color in Color:
        pile_height = len(game_state.firework_piles.get(color, []))
        next_card = pile_height + 1
        if next_card <= 5:
            firework_info.append(
                f"  {color.value}: Next playable card is {next_card}")
        else:
            firework_info.append(f"  {color.value}: Firework complete")

    valid_actions.extend(firework_info)

    return "\n".join(valid_actions)


def create_action_proposal_prompt(game_state: GameState,
                                  discussion_history: List[str],
                                  current_thoughts: List[str],
                                  agent_id: int,
                                  memory: dict = None) -> str:
    """Create a prompt for proposing an action based on the game state and discussion."""
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
                # Convert to string for consistency
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
                # In a real game, we would track what clues have been given
                # For now, we'll use the card's actual properties to simulate clue information
                clues = []

                # Check if this card has received color clues
                if hasattr(card, "color_clued") and card.color_clued:
                    clues.append(f"color: {card.color.value}")

                # Check if this card has received number clues
                if hasattr(card, "number_clued") and card.number_clued:
                    clues.append(f"number: {card.number}")

                # If we don't have specific clue tracking, just indicate it's been clued
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
                else:
                    card_info += " (has received clues)"

            my_hand += card_info

    # Format the discussion history
    discussion = format_discussion(discussion_history)

    # Format the current thoughts
    thoughts = format_thoughts(current_thoughts)

    # Generate information about valid actions
    valid_actions_info = generate_valid_actions_info(game_state, agent_id)

    # Create the prompt
    prompt = f"""You are Agent {agent_id} in a game of Hanabi and it's your turn to decide on an action.

Current Game State:
- Score: {game_state.score}
- Clue tokens: {game_state.clue_tokens}
- Fuse tokens: {game_state.fuse_tokens}
- Current player: {game_state.current_player}
- Firework piles: {firework_piles}
- Discard pile: {discard_pile}

{other_players}

{my_hand}

HANABI DECK COMPOSITION:
- Each color (red, yellow, green, blue, white) has:
  - Three 1s
  - Two 2s, 3s, and 4s
  - Only one 5
- This means discarding a 5 makes it impossible to complete that color's firework

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

EXAMPLES OF FORBIDDEN STATEMENTS:
- "I know my card 0 is a green 1" (when you've only received a green clue)
- "I know my card 1 is a blue 4" (when you've only received a blue clue)
- "I know my card 2 is a red 3" (when you've only received a red clue)
- "I know my card 3 is a white 2" (when you've only received a 2 clue)
- "Player 2 has a red 3 in position 1"
- "Your second card is a blue 4"
- "I see you have a 1 in your hand"
- "Give a red clue to Player 3"
- "Player 4 has two 1s"
- "Your hand has a playable red card"
- "I'll give you a 1 clue"
- "I'll give Player 3 a red clue"
- "You should play your red card"
- "You should play your 1"

EXAMPLES OF ALLOWED STATEMENTS:
- "I know my card 0 is green" (when you've received a green clue)
- "I know my card 1 is a 4" (when you've received a 4 clue)
- "I believe my card 0 might be a green 1 based on the current game state" (inference)
- "I infer that my green card is likely a 1 since no green cards have been played yet" (inference)
- "I'll give information about Player 3's cards"
- "I'll give a color clue to Player 3"
- "I'll give a number clue to Player 3"
- "Consider giving information about your teammate's first card"
- "I think we should focus on building the red firework next"
- "I'll play my first card"
- "I'll discard my third card"

HANABI STRATEGY PRINCIPLES:
- Balance information gathering with progress - sometimes it's worth taking calculated risks
- Clues are a limited resource - use them efficiently to convey maximum information
- Cards that have been clued are usually important - either playable now or valuable for later
- Consider what information other players already have when deciding your action
- Sometimes discarding is necessary to regain clue tokens, even if it means losing potential points
- The team's overall strategy is more important than individual perfect plays
- Pay attention to the discard pile to track which cards remain in the deck

{valid_actions_info}

"""

    # Add discussion history if available
    if discussion:
        prompt += f"\nDiscussion:\n{discussion}\n"

    # Add current thoughts if available
    if thoughts:
        prompt += f"\nYour thoughts:\n{thoughts}\n"

    # Add final instruction for using tools
    prompt += """
Based on the game state, discussion, and your thoughts, decide on your final action.
Remember to clearly distinguish between KNOWN information (from clues) and INFERENCES.

You have the following tools available:

1. play_card: Play a card from your hand
   - card_index: Index of the card to play (0-4)

2. give_clue: Give a clue to another player
   - target_id: ID of the player to give a clue to
   - clue_type: Type of clue ("color" or "number")
   - clue_value: Value of the clue (color name or number 1-5)

3. discard: Discard a card from your hand
   - card_index: Index of the card to discard (0-4)

Before using a tool, provide a brief explanation of your reasoning that follows the information rules.
"""

    return prompt
