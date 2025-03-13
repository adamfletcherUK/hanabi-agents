"""
Strategic guidelines for playing Hanabi effectively.
"""

HANABI_STRATEGY_GUIDELINES = """
# Hanabi Strategy Guidelines

## Card Play Principles
1. **Safe Plays First**: Always prioritize playing cards you are 100% certain about over risky plays.
2. **Newest Card Rule**: Newly drawn cards (rightmost in your hand) have no information, making them the safest to discard.
3. **Save 5s**: Fives are the rarest cards (only one of each color). Avoid discarding known or suspected 5s.
4. **Consider Playability Window**: A card is only valuable if it can be played soon. Cards too far ahead in sequence have lower priority.

## Clue Efficiency
1. **Maximum Information**: Give clues that reveal information about multiple cards when possible.
2. **Play Clues vs. Save Clues**: 
   - "Play clues" indicate a card is immediately playable.
   - "Save clues" indicate a card is important but not immediately playable (usually 5s or critical 2s, 3s, 4s).
3. **Implied Information**: If you highlight a card as playable through a clue, it implies all other cards of the same type previously clued are not playable.

## Discard Strategy
1. **Discard Known Safe Cards**: Prefer discarding cards you know are duplicates or no longer needed.
2. **Oldest First**: If unsure, discard the oldest (leftmost) card that has no positive clues about it.
3. **Avoid Discarding Unclued Cards** when another player has signaled they may be important.

## Reading Teammate Actions
1. **Clue Interpretation**: If a teammate gives you a clue about a specific card, they likely want you to play it soon.
2. **Delayed Play**: If a teammate clues a card but it's not immediately playable, they're saving it for later.
3. **Discard Signals**: When a teammate discards a seemingly valuable card, they may be signaling that duplicates exist or that the card isn't needed.

## Game State Awareness
1. **Track the Discard Pile**: Know which cards have been discarded to assess the risk of discarding further cards.
2. **Remaining Deck**: Be aware of how many cards remain in the deck to plan for the endgame.
3. **Token Management**: 
   - If clue tokens are low (0-1), prioritize discarding.
   - If fuse tokens are low (1-2), be extremely cautious about playing cards without certainty.
   - If clue tokens are high (7-8), prioritize playing cards or giving clues before discarding.
"""


def get_strategy_guidelines() -> str:
    """
    Returns the Hanabi strategy guidelines.

    Returns:
        String containing Hanabi strategy guidelines
    """
    return HANABI_STRATEGY_GUIDELINES
