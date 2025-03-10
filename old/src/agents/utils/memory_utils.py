from typing import Dict, Any, List
import logging
import uuid
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)


def store_game_outcome(agent, game_state, outcome_type="game_end"):
    """
    Store the outcome of a game in the agent's memory store.

    Args:
        agent: The agent instance
        game_state: The final game state
        outcome_type: Type of outcome (game_end, round_end, etc.)

    Returns:
        Boolean indicating success
    """
    try:
        # Create a memory entry for the game outcome
        memory_entry = {
            "timestamp": datetime.now().isoformat(),
            "outcome_type": outcome_type,
            "score": game_state.score,
            "max_score": game_state.get_max_possible_score() if hasattr(game_state, "get_max_possible_score") else 25,
            "completed_fireworks": game_state.get_completed_fireworks_count() if hasattr(game_state, "get_completed_fireworks_count") else 0,
            "turns_played": game_state.turn_count,
            "remaining_fuse_tokens": game_state.fuse_tokens,
            "remaining_clue_tokens": game_state.clue_tokens
        }

        # Store the memory
        return agent.store_memory("game_outcomes", memory_entry)
    except Exception as e:
        logger.error(
            f"Error storing game outcome for Agent {agent.agent_id}: {e}")
        return False


def store_action_outcome(agent, action, result, game_state):
    """
    Store the outcome of an action in the agent's memory store.

    Args:
        agent: The agent instance
        action: The action that was taken
        result: The result of the action (success, failure, etc.)
        game_state: The game state after the action

    Returns:
        Boolean indicating success
    """
    try:
        # Create a memory entry for the action outcome
        memory_entry = {
            "timestamp": datetime.now().isoformat(),
            "action_type": action.get("type"),
            "action_details": action,
            "result": result,
            "turn": game_state.turn_count,
            "score_before": agent.memory.get("last_game_state", {}).get("score", 0) if "last_game_state" in agent.memory else 0,
            "score_after": game_state.score,
            "clue_tokens_before": agent.memory.get("last_game_state", {}).get("clue_tokens", 8) if "last_game_state" in agent.memory else 8,
            "clue_tokens_after": game_state.clue_tokens,
            "fuse_tokens_before": agent.memory.get("last_game_state", {}).get("fuse_tokens", 3) if "last_game_state" in agent.memory else 3,
            "fuse_tokens_after": game_state.fuse_tokens
        }

        # Store the memory
        return agent.store_memory("action_outcomes", memory_entry)
    except Exception as e:
        logger.error(
            f"Error storing action outcome for Agent {agent.agent_id}: {e}")
        return False


def get_action_history(agent, limit=10, action_type=None):
    """
    Get the agent's action history from the memory store.

    Args:
        agent: The agent instance
        limit: Maximum number of actions to retrieve
        action_type: Optional filter for action type

    Returns:
        List of action outcomes
    """
    try:
        # Get all action outcomes
        action_outcomes = agent.get_memory_from_store("action_outcomes")

        # Filter by action type if specified
        if action_type:
            action_outcomes = [outcome for outcome in action_outcomes if outcome.get(
                "action_type") == action_type]

        # Sort by timestamp (newest first)
        action_outcomes.sort(key=lambda x: x.get(
            "timestamp", ""), reverse=True)

        # Limit the number of results
        return action_outcomes[:limit]
    except Exception as e:
        logger.error(
            f"Error retrieving action history for Agent {agent.agent_id}: {e}")
        return []


def get_game_history(agent, limit=5):
    """
    Get the agent's game history from the memory store.

    Args:
        agent: The agent instance
        limit: Maximum number of games to retrieve

    Returns:
        List of game outcomes
    """
    try:
        # Get all game outcomes
        game_outcomes = agent.get_memory_from_store("game_outcomes")

        # Sort by timestamp (newest first)
        game_outcomes.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # Limit the number of results
        return game_outcomes[:limit]
    except Exception as e:
        logger.error(
            f"Error retrieving game history for Agent {agent.agent_id}: {e}")
        return []


def generate_learning_summary(agent):
    """
    Generate a summary of what the agent has learned from past games.

    Args:
        agent: The agent instance

    Returns:
        String summary of learnings
    """
    try:
        # Get game history
        game_history = get_game_history(agent)

        # Get action history
        action_history = get_action_history(agent, limit=50)

        # Calculate success rates for different action types
        action_stats = {}
        for action in action_history:
            action_type = action.get("action_type")
            result = action.get("result", {}).get("success", False)

            if action_type not in action_stats:
                action_stats[action_type] = {"total": 0, "success": 0}

            action_stats[action_type]["total"] += 1
            if result:
                action_stats[action_type]["success"] += 1

        # Calculate success rates
        for action_type in action_stats:
            total = action_stats[action_type]["total"]
            success = action_stats[action_type]["success"]
            action_stats[action_type]["success_rate"] = (
                success / total) if total > 0 else 0

        # Generate summary
        summary = "Learning Summary:\n\n"

        # Game history summary
        if game_history:
            avg_score = sum(game.get("score", 0)
                            for game in game_history) / len(game_history)
            avg_turns = sum(game.get("turns_played", 0)
                            for game in game_history) / len(game_history)

            summary += f"Game History ({len(game_history)} games):\n"
            summary += f"- Average Score: {avg_score:.1f}\n"
            summary += f"- Average Turns: {avg_turns:.1f}\n\n"

        # Action statistics
        if action_stats:
            summary += "Action Statistics:\n"
            for action_type, stats in action_stats.items():
                summary += f"- {action_type}: {stats['success_rate']*100:.1f}% success rate ({stats['success']}/{stats['total']})\n"

        return summary
    except Exception as e:
        logger.error(
            f"Error generating learning summary for Agent {agent.agent_id}: {e}")
        return "Unable to generate learning summary due to an error."
