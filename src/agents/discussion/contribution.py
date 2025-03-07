from typing import Dict, Any, List
import logging
from langchain_core.messages import HumanMessage
from ..formatters.thoughts import format_thoughts

# Set up logging
logger = logging.getLogger(__name__)


def generate_contribution(
    final_state: Dict[str, Any],
    model: Any,
    agent_id: int
) -> str:
    """
    Generate a contribution to the discussion based on the agent's thoughts.

    Args:
        final_state: Final state from the reasoning graph
        model: LLM model to use for generation
        agent_id: ID of the agent

    Returns:
        Contribution string
    """
    # Extract the generated thoughts
    current_thoughts = final_state.get("current_thoughts", [])

    if not current_thoughts:
        return "I'm analyzing the game state and considering our options."

    # Create a prompt for generating a contribution
    prompt = f"""You are Agent {agent_id} in a game of Hanabi, participating in a discussion.

Based on your analysis of the game state, generate a concise contribution to the discussion.

CRITICAL INFORMATION RULES:
1. You MUST distinguish between KNOWN information and INFERENCES.
2. KNOWN information is ONLY what you've been explicitly told through clues.
3. INFERENCES are educated guesses based on game state, but are NOT certainties.
4. You MUST use language like "I believe", "I infer", "likely", "probably", "might be" for inferences.
5. You MUST use language like "I know" ONLY for information directly given through clues.
6. For example, if you received a "green" clue on a card, you can say "I know this card is green" but NOT "I know this is a green 1".
7. You CANNOT claim to know both color AND number of a card unless you've received BOTH clues for that card.
8. You CANNOT claim to know the exact identity of a card based solely on a single clue.

Your thoughts:
{format_thoughts(current_thoughts)}

Generate a concise, strategic contribution that follows the information rules above. Focus on what action you're considering and why, without revealing specific card information you shouldn't know.
"""

    # Generate the contribution
    response = model.invoke([HumanMessage(content=prompt)])
    contribution = response.content.strip()

    # Log at debug level instead of info to avoid double logging
    logger.debug(f"Agent {agent_id} generated contribution: {contribution}")
    return contribution
