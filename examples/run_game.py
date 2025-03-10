import logging
import os
import sys
import datetime
from pathlib import Path
from dotenv import load_dotenv
from src.game.engine import GameEngine
from src.agents.ai_agent import AIAgent
from src.communication.discussion import DiscussionManager
from src.utils.env import load_environment_variables

# Create logs directory at the project root if it doesn't exist
project_root = Path(__file__).parent.parent
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)

# Generate a timestamped log filename
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = logs_dir / f"hanabi_game_{timestamp}.log"

# Set up logging with more detail but filter out noisy HTTP logs
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG level to capture more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

# Set specific loggers to higher levels to reduce noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Our main logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Ensure our game logs are visible
logger.info(f"Logging to file: {log_file}")

# Global variable to track clue history
clue_history = []


def track_clue(turn, giver_id, receiver_id, clue_type, clue_value, affected_indices):
    """Track a clue given during the game."""
    clue_history.append({
        "turn": turn,
        "giver_id": giver_id,
        "receiver_id": receiver_id,
        "clue_type": clue_type,
        "clue_value": clue_value,
        "affected_indices": affected_indices
    })


def log_detailed_game_state(engine):
    """Log detailed information about the current game state."""
    logger.info("----- DETAILED GAME STATE -----")

    # Log firework piles
    firework_piles = []
    for color, pile in engine.state.firework_piles.items():
        if pile:
            top_card = pile[-1].number
            firework_piles.append(f"{color.value}: {top_card}")
        else:
            firework_piles.append(f"{color.value}: empty")
    logger.info(f"Firework piles: {', '.join(firework_piles)}")

    # Log player hands
    for player_id, hand in engine.state.hands.items():
        # First, log how the player sees their own hand
        player_view = []
        for i, card in enumerate(hand):
            # For the player's own view, we hide the actual card values
            card_info = f"Card {i}: [HIDDEN]"

            # Add clue information if available
            clues = []
            if hasattr(card, 'color_clued') and card.color_clued:
                clues.append(f"color: {card.color.value}")
            if hasattr(card, 'number_clued') and card.number_clued:
                clues.append(f"number: {card.number}")

            if clues:
                card_info += f" ({', '.join(clues)})"

            player_view.append(card_info)

        logger.info(
            f"Player {player_id}'s view of their own hand: {player_view}")

        # Now log how this player's hand appears to others (for debugging)
        hand_info = []
        for i, card in enumerate(hand):
            # For logging purposes, we show the actual card values
            card_info = f"Card {i}: {card.color.value} {card.number}"

            # Add visibility information (if card has received clues)
            if hasattr(card, 'is_visible') and card.is_visible:
                card_info += " (clued)"

            hand_info.append(card_info)

        logger.info(
            f"Player {player_id}'s actual hand (visible to others): {hand_info}")

    # Log discard pile summary
    discard_summary = {}
    for card in engine.state.discard_pile:
        key = f"{card.color.value} {card.number}"
        discard_summary[key] = discard_summary.get(key, 0) + 1

    discard_info = [f"{card}: {count}" for card,
                    count in discard_summary.items()]
    logger.info(
        f"Discard pile: {', '.join(discard_info) if discard_info else 'empty'}")

    # Log remaining deck size
    logger.info(f"Remaining deck size: {len(engine.state.deck)}")

    # Log game progress
    total_cards = 5 * 5  # 5 colors, 5 numbers
    completed_cards = sum(len(pile)
                          for pile in engine.state.firework_piles.values())
    logger.info(
        f"Game progress: {completed_cards}/{total_cards} cards played ({(completed_cards/total_cards)*100:.1f}%)")

    logger.info("----- END DETAILED STATE -----")


def log_action_summary(agent_id, action, result=None):
    """Log a summary of an agent's action and its result."""
    action_type = action.get("type", "unknown")

    if action_type == "play_card":
        card_index = action.get("card_index", 0)
        message = f"🎮 Player {agent_id} played card at position {card_index}"
        if result and "card" in result:
            card = result["card"]
            message += f" ({card.color.value} {card.number})"
        if result and "success" in result:
            if result["success"]:
                message += " ✅"
            else:
                message += " ❌"
        logger.info(message)

    elif action_type == "give_clue":
        target_id = action.get("target_id", 0)
        clue = action.get("clue", {})
        clue_type = clue.get("type", "unknown")
        clue_value = clue.get("value", "unknown")
        message = f"💬 Player {agent_id} gave a {clue_type} clue ({clue_value}) to Player {target_id}"
        if result and "affected_cards" in result:
            message += f" affecting {len(result['affected_cards'])} cards"
        logger.info(message)

    elif action_type == "discard":
        card_index = action.get("card_index", 0)
        message = f"🗑️ Player {agent_id} discarded card at position {card_index}"
        if result and "card" in result:
            card = result["card"]
            message += f" ({card.color.value} {card.number})"
        logger.info(message)

    else:
        logger.info(f"Player {agent_id} performed action: {action}")


def log_game_status(engine):
    """Log a concise summary of the current game status."""
    logger.info("=" * 50)
    logger.info(
        f"🏆 SCORE: {engine.state.score} | 🔍 CLUES: {engine.state.clue_tokens} | 💣 FUSES: {engine.state.fuse_tokens}")

    # Log firework piles in a compact format
    fireworks = []
    for color, pile in engine.state.firework_piles.items():
        height = len(pile)
        fireworks.append(f"{color.value[0].upper()}{height}")
    logger.info(f"🎆 FIREWORKS: {' '.join(fireworks)}")

    # Log remaining deck size and turns
    logger.info(
        f"🎴 DECK: {len(engine.state.deck)} cards | 🔄 TURN: {engine.state.turn_count + 1}")
    logger.info("=" * 50)


def log_incorrect_tool_usage(engine):
    """Log information about incorrect tool usage by agents."""
    incorrect_usage = engine.get_incorrect_tool_usage()

    if not any(incorrect_usage.values()):
        return  # No incorrect tool usage to log

    logger.info("=" * 50)
    logger.info("⚠️ INCORRECT TOOL USAGE SUMMARY ⚠️")

    for agent_id, errors in incorrect_usage.items():
        if not errors:
            continue

        logger.info(
            f"Agent {agent_id} has made {len(errors)} incorrect tool uses:")

        # Group errors by type
        error_types = {}
        for error in errors:
            action_type = error.get('action', {}).get('type', 'unknown')
            error_reason = error.get('error_reason', 'unknown')
            key = f"{action_type}:{error_reason}"

            if key not in error_types:
                error_types[key] = 0
            error_types[key] += 1

        # Log summary of error types
        for key, count in error_types.items():
            action_type, error_reason = key.split(':')
            logger.info(f"  - {count}x {action_type} errors ({error_reason})")

        # Log the most recent error in detail
        if errors:
            most_recent = errors[-1]
            action = most_recent.get('action', {})
            error_message = most_recent.get('error_message', 'Unknown error')

            logger.info(
                f"  Most recent error (turn {most_recent.get('turn', '?')}):")
            logger.info(f"  Action: {action}")
            logger.info(f"  Error: {error_message}")

    logger.info("=" * 50)


def run_game():
    """Run a complete game with AI agents."""
    logger.info("=" * 50)
    logger.info("🎮 STARTING NEW GAME OF HANABI 🎮")
    logger.info("=" * 50)

    # Load environment variables
    if not load_environment_variables():
        logger.error("Failed to load required environment variables. Exiting.")
        return

    # Print the API key (first few characters) for debugging
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        logger.info(f"Using OpenAI API key: {api_key[:5]}...")
    else:
        logger.error("OPENAI_API_KEY not found. Exiting.")
        return

    # Initialize game components
    logger.info("Initializing game components...")
    try:
        agents = [AIAgent(agent_id=i) for i in range(5)]
        engine = GameEngine(agents=agents)
        discussion_manager = DiscussionManager(
            max_rounds=1)  # Only one round needed now
        logger.info(f"Game initialized with {len(agents)} agents")
    except Exception as e:
        logger.error(f"Error initializing game components: {e}")
        return

    # Log initial game status
    log_game_status(engine)
    log_detailed_game_state(engine)

    # Main game loop
    try:
        while not engine.state.game_over:
            # Get turn information
            turn_info = engine.get_turn_info()
            current_player = turn_info["current_player"]
            human_readable_turn = turn_info["human_readable_turn"]

            logger.info("=" * 50)
            logger.info(
                f"🎲 TURN {human_readable_turn} - PLAYER {current_player}'S TURN 🎲")
            logger.info("=" * 50)

            # Log detailed game state at the start of each turn
            log_detailed_game_state(engine)

            # Log incorrect tool usage information
            log_incorrect_tool_usage(engine)

            # Start new discussion - now just for the active player to rationalize their action
            logger.info("🧠 ----- REASONING PHASE BEGINS ----- 🧠")
            discussion_manager.start_new_discussion()

            # The active player proposes and rationalizes their action
            current_agent = agents[current_player]
            game_view = engine.state.get_view_for(current_player)

            logger.info(
                f"🎯 Player {current_player} is analyzing the game state")
            try:
                # Get the active player's proposal with clear action and reasoning
                action_proposal = current_agent.participate_in_discussion(
                    game_view,
                    discussion_manager.get_game_history(last_n_turns=3),
                    is_active_player=True  # Indicate this is the active player
                )
                logger.info(
                    f"💡 Player {current_player} proposes: {action_proposal}")
                # Add the proposal to the discussion history
                discussion_manager.add_contribution(
                    current_player, action_proposal)
            except Exception as e:
                logger.error(
                    f"Error getting proposal from Player {current_player}: {e}")
                action_proposal = "I propose to give a number clue to help identify playable 1s because this is the safest way to make progress at this stage."
                discussion_manager.add_contribution(
                    current_player, action_proposal)

            logger.info("🧠 ----- REASONING PHASE ENDS ----- 🧠")

            # Get the reasoning summary
            reasoning_summary = discussion_manager.get_discussion_summary()
            logger.info(f"📝 Reasoning summary: {reasoning_summary}")

            # Action phase - Direct action without feedback
            logger.info("🎬 ----- ACTION PHASE BEGINS ----- 🎬")
            logger.info(f"⚡ Player {current_player} is executing action")

            # The active agent directly executes the action
            game_view = engine.state.get_view_for(current_player)

            try:
                # Pass the reasoning summary to the engine's _play_turn method
                engine.pre_action_discussion_summary = reasoning_summary

                # Store the current state for comparison
                prev_score = engine.state.score
                prev_clue_tokens = engine.state.clue_tokens
                prev_fuse_tokens = engine.state.fuse_tokens

                # This will return False if the agent fails to propose a valid action
                turn_success = engine._play_turn()

                if not turn_success:
                    logger.warning(
                        f"Player {current_player}'s turn failed. Moving to next player.")
                    # Log the incorrect tool usage
                    log_incorrect_tool_usage(engine)
                    # Skip the rest of this iteration and continue with the next player
                    continue

                # Log the action result
                if hasattr(engine, 'last_action') and hasattr(engine, 'last_action_result'):
                    log_action_summary(
                        current_player, engine.last_action, engine.last_action_result)

                # Log changes in game state
                if engine.state.score > prev_score:
                    logger.info(f"🎉 Score increased to {engine.state.score}!")
                if engine.state.clue_tokens != prev_clue_tokens:
                    logger.info(
                        f"🔍 Clue tokens changed: {prev_clue_tokens} → {engine.state.clue_tokens}")
                if engine.state.fuse_tokens != prev_fuse_tokens:
                    logger.info(
                        f"💣 Fuse tokens changed: {prev_fuse_tokens} → {engine.state.fuse_tokens}")

                # If we get here, the action was successful
                logger.info("🎬 ----- ACTION PHASE ENDS ----- 🎬")

                # Log the updated game status
                log_game_status(engine)
            except RuntimeError as e:
                # Handle game termination errors more gracefully
                error_message = str(e)
                logger.critical(f"Error during action phase: {error_message}")

                # Check if this is an invalid action error
                if "invalid action" in error_message.lower():
                    # Extract the action from the error message if possible
                    action_str = error_message.split(
                        "invalid action:")[-1].strip() if "invalid action:" in error_message else "unknown"

                    # Log the error and continue with the next player if possible
                    logger.error(
                        f"Invalid action attempted by Player {current_player}: {action_str}")
                    logger.info(
                        "Attempting to continue the game with the next player...")

                    # Advance to the next player manually
                    engine.state.current_player = (
                        engine.state.current_player + 1) % len(engine.agents)
                    engine.state.turn_count += 1

                    # Log the recovery attempt
                    logger.info(
                        f"Recovered from error. Next player: {engine.state.current_player}")

                    # Continue the game loop
                    continue
                else:
                    # For other runtime errors, log and terminate
                    logger.critical(
                        f"Game terminated due to critical error: {error_message}")
                    logger.critical(f"Stack trace: ", exc_info=True)
                    print(
                        f"\n*** CRITICAL ERROR: Game terminated ***\n{error_message}\n")
                    break
            except Exception as e:
                # Handle other exceptions
                logger.critical(f"Unexpected error during action phase: {e}")
                logger.critical("Stack trace: ", exc_info=True)
                print(
                    f"\n*** CRITICAL ERROR: Game terminated due to unexpected error ***\n{e}\n")
                break

            # Check game over conditions
            if engine.state.fuse_tokens <= 0:
                logger.info("💥 Game Over! Ran out of fuse tokens!")
                break

            if engine.state.get_completed_fireworks_count() == 5:
                logger.info("🎆 Game Over! All fireworks completed!")
                break
    except Exception as e:
        logger.critical(f"Game terminated due to critical error: {e}")
        logger.critical(f"Stack trace: ", exc_info=True)
        engine.state.game_over = True
        # Print a clear message to the console
        print(f"\n\n*** CRITICAL ERROR: Game terminated ***\n{e}\n")
        # Exit with a non-zero status code to indicate an error
        sys.exit(1)

    # Print final game state
    logger.info("=" * 50)
    logger.info("GAME OVER")
    logger.info("=" * 50)
    logger.info(f"Final Score: {engine.state.score}")
    logger.info(
        f"Completed Fireworks: {engine.state.get_completed_fireworks_count()}")
    logger.info(
        f"Score Percentage: {engine.state.get_score_percentage():.1f}%")

    # Log final detailed game state
    log_detailed_game_state(engine)

    # Print score history
    logger.info("\nScore History:")
    for entry in engine.state.score_history:
        logger.info(
            f"{entry.timestamp}: {entry.action_type} - {entry.details}")

    # Store game outcomes in agent memory
    try:
        # Import here to avoid circular imports
        # No need to re-import sys or os as they're already imported at the top of the file

        # Add the project root to the path if needed
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), ".."))
        if project_root not in sys.path:
            sys.path.append(project_root)

        from src.agents.utils.memory_utils import store_game_outcome, generate_learning_summary

        logger.info("\nStoring game outcomes in agent memory...")
        for agent in agents:
            store_game_outcome(agent, engine.state)

        # Generate and log learning summaries for each agent
        logger.info("\nAgent Learning Summaries:")
        for agent in agents:
            learning_summary = generate_learning_summary(agent)
            logger.info(
                f"\nAgent {agent.agent_id} Learning Summary:\n{learning_summary}")
    except Exception as e:
        logger.error(f"Error storing game outcomes: {e}")

    # Log final incorrect tool usage summary
    log_incorrect_tool_usage(engine)

    logger.info("=" * 50)
    logger.info(f"🎮 GAME OVER! FINAL SCORE: {engine.state.score} 🎮")
    logger.info("=" * 50)


if __name__ == "__main__":
    run_game()
