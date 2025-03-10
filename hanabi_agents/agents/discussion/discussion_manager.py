from typing import Dict, Any, List
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from ...game.state import GameState

# Set up logging
logger = logging.getLogger(__name__)


class DiscussionManager:
    """
    Manager for handling pre-action discussions between agents.

    This class manages the discussion phase, where agents can share their thoughts
    and strategies before the active player takes an action.
    """

    def __init__(self, model_name: str = None):
        """
        Initialize a new discussion manager.

        Args:
            model_name: Name of the language model to use for summarization
        """
        self.model = self._initialize_model(model_name)

    def _initialize_model(self, model_name: str = None) -> ChatOpenAI:
        """
        Initialize the language model for summarization.

        Args:
            model_name: Name of the model to use

        Returns:
            Initialized language model
        """
        # Use default model if none specified
        if model_name is None:
            model_name = "o3-mini"

        # Initialize the model without temperature
        return ChatOpenAI(
            model=model_name,
            verbose=False  # Disable verbose mode to prevent stdout logging
        )

    def conduct_discussion(self, game_state: GameState, agents: List[Any]) -> str:
        """
        Conduct a discussion between agents.

        Args:
            game_state: Current state of the game
            agents: List of agent instances

        Returns:
            Summary of the discussion
        """
        logger.info("Starting discussion phase")

        # Initialize discussion history
        discussion_history = []

        # Get the active player
        active_player_id = game_state.current_player

        # First, let the active player share their thoughts
        active_agent = next(
            (agent for agent in agents if agent.agent_id == active_player_id), None)
        if active_agent:
            logger.info(f"Active player {active_player_id} sharing thoughts")
            contribution = active_agent.participate_in_discussion(
                game_state, discussion_history)
            discussion_history.append({
                "player_id": active_player_id,
                "content": contribution,
                "is_active_player": True
            })

        # Then, let other players share their thoughts
        for agent in agents:
            if agent.agent_id != active_player_id:
                logger.info(f"Player {agent.agent_id} sharing thoughts")
                contribution = agent.participate_in_discussion(
                    game_state, discussion_history)
                discussion_history.append({
                    "player_id": agent.agent_id,
                    "content": contribution,
                    "is_active_player": False
                })

        # Summarize the discussion
        summary = self.summarize_discussion(
            discussion_history, active_player_id)

        return summary

    def summarize_discussion(self, discussion_history: List[Dict[str, Any]], active_player_id: int) -> str:
        """
        Summarize the discussion for the active player.

        Args:
            discussion_history: History of discussion contributions
            active_player_id: ID of the active player

        Returns:
            Summary of the discussion
        """
        logger.info("Summarizing discussion")

        # Create the prompt
        prompt = f"""
# Hanabi Discussion Summary

Summarize the following discussion between Hanabi players. Focus on the most important strategic insights and recommendations for Player {active_player_id}, who is the active player.

## Discussion Contributions
"""

        # Add the discussion contributions
        for entry in discussion_history:
            player_id = entry["player_id"]
            content = entry["content"]
            is_active = entry.get("is_active_player", False)
            active_label = " (Active Player)" if is_active else ""
            prompt += f"\n### Player {player_id}{active_label}\n{content}\n"

        prompt += """
## Summary Task

Provide a concise summary of the discussion, focusing on:
1. Key insights about the game state
2. Recommendations for the active player's next move
3. Important information shared about card knowledge
4. Any strategic disagreements or alternative perspectives

Keep the summary clear and actionable for the active player.
"""

        # Create the message
        message = HumanMessage(content=prompt)

        # Get the summary from the model
        response = self.model.invoke([message])

        return response.content
