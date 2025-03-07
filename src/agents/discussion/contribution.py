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


def generate_active_player_contribution(
    final_state: Dict[str, Any],
    model: Any,
    agent_id: int
) -> str:
    """
    Generate a contribution for the active player that clearly states their proposed action and reasoning.

    Args:
        final_state: Final state from the reasoning graph
        model: LLM model to use for generation
        agent_id: ID of the agent

    Returns:
        Contribution string with proposed action and reasoning
    """
    # Extract the generated thoughts
    current_thoughts = final_state.get("current_thoughts", [])

    if not current_thoughts:
        return "I'm analyzing the game state and considering our options."

    # Create a prompt for generating a contribution with clear action proposal
    prompt = f"""You are Agent {agent_id} in a game of Hanabi, and it's your turn to propose an action.

Based on your analysis of the game state, you need to clearly state:
1. The specific action you propose to take (play a card, give a clue, or discard)
2. Your detailed reasoning for this action

PROPOSED ACTION FORMAT:
- Start with "I propose to [action]..." where [action] is one of:
  * "play card [index]" (e.g., "play card 2")
  * "give a [color/number] clue to Player [id]" (e.g., "give a red clue to Player 3")
  * "discard card [index]" (e.g., "discard card 0")

REASONING FORMAT:
- After stating your proposed action, explain your reasoning with "because..."
- Distinguish between KNOWN information (from clues) and INFERENCES (educated guesses)
- Use "I know" only for information directly given through clues
- Use "I believe/infer/think" for inferences

Your thoughts:
{format_thoughts(current_thoughts)}

Generate a clear, strategic proposal that follows the format above. Be specific about your proposed action and provide detailed reasoning.
"""

    # Generate the contribution
    response = model.invoke([HumanMessage(content=prompt)])
    contribution = response.content.strip()

    # Log at debug level instead of info to avoid double logging
    logger.debug(f"Active Agent {agent_id} generated proposal: {contribution}")
    return contribution


def generate_feedback_contribution(
    final_state: Dict[str, Any],
    model: Any,
    agent_id: int,
    active_player_proposal: str
) -> str:
    """
    Generate a yes/no feedback contribution based on the active player's proposal.

    Args:
        final_state: Final state from the reasoning graph
        model: LLM model to use for generation
        agent_id: ID of the agent
        active_player_proposal: The proposal from the active player

    Returns:
        Feedback contribution with yes/no answer and brief reasoning
    """
    # Extract the generated thoughts
    current_thoughts = final_state.get("current_thoughts", [])

    if not current_thoughts:
        return "I need more information to evaluate this proposal."

    # Create a prompt for generating a yes/no feedback contribution
    prompt = f"""You are Agent {agent_id} in a game of Hanabi, providing feedback on another player's proposed action.

The active player has proposed the following action:
"{active_player_proposal}"

Based on your analysis of the game state, you need to:
1. Clearly state whether you AGREE or DISAGREE with this proposal
2. Provide a BRIEF explanation of your reasoning (1-2 sentences)

FEEDBACK FORMAT:
- Start with "I [agree/disagree] with this proposal because..."
- Keep your explanation concise and focused on the most important factors
- Consider the current game state, available clues, and potential risks

Your thoughts:
{format_thoughts(current_thoughts)}

Generate a clear yes/no feedback response that follows the format above.
"""

    # Generate the contribution
    response = model.invoke([HumanMessage(content=prompt)])
    contribution = response.content.strip()

    # Log at debug level instead of info to avoid double logging
    logger.debug(f"Agent {agent_id} generated feedback: {contribution}")
    return contribution
