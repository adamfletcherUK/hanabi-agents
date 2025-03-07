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
    """Log detailed information about player hands and clue information."""
    logger.info("----- DETAILED GAME STATE -----")

    # Log firework piles
    firework_info = []
    for color in engine.state.firework_piles:
        pile = engine.state.firework_piles[color]
        if pile:
            top_card = pile[-1].number
            firework_info.append(f"{color.value}: {top_card}")
        else:
            firework_info.append(f"{color.value}: empty")

    logger.info(f"Firework piles: {', '.join(firework_info)}")

    # Log player hands with clue information
    for player_id, hand in engine.state.hands.items():
        # First, log the hand as the player would see it (with hidden information for their own cards)
        player_view = []
        for i, card in enumerate(hand):
            # The player can't see their own cards
            card_info = f"Card {i}: [HIDDEN]"

            # But they might have clue information
            if card.is_visible:
                # Find clues for this card from the clue history
                card_clues = []
                for clue in clue_history:
                    if clue["receiver_id"] == player_id and i in clue["affected_indices"]:
                        if clue["clue_type"] == "color":
                            card_clues.append(f"color: {clue['clue_value']}")
                        else:  # number clue
                            card_clues.append(f"number: {clue['clue_value']}")

                if card_clues:
                    card_info += f" ({', '.join(card_clues)})"
                else:
                    card_info += " (clued but details unknown)"

            player_view.append(card_info)

        logger.info(
            f"Player {player_id}'s view of their own hand: {player_view}")

        # Now log how this player's hand appears to others (for debugging)
        hand_info = []
        for i, card in enumerate(hand):
            # For logging purposes, we show the actual card values
            card_info = f"Card {i}: {card.color.value} {card.number}"

            # Add visibility information (if card has received clues)
            if card.is_visible:
                # Find clues for this card from the clue history
                card_clues = []
                for clue in clue_history:
                    if clue["receiver_id"] == player_id and i in clue["affected_indices"]:
                        if clue["clue_type"] == "color":
                            card_clues.append(f"color: {clue['clue_value']}")
                        else:  # number clue
                            card_clues.append(f"number: {clue['clue_value']}")

                if card_clues:
                    card_info += f" (clued: {', '.join(card_clues)})"
                else:
                    card_info += " (clued)"

            hand_info.append(card_info)

        logger.info(
            f"Player {player_id}'s actual hand (visible to others): {hand_info}")

    # Log clue history for each player
    if clue_history:
        logger.info("----- CLUE HISTORY -----")
        for player_id in engine.state.hands.keys():
            player_clues = [
                clue for clue in clue_history if clue["receiver_id"] == player_id]
            if player_clues:
                logger.info(f"Clues given to Player {player_id}:")
                for clue in player_clues:
                    logger.info(
                        f"  Turn {clue['turn']}: Player {clue['giver_id']} gave {clue['clue_type']} {clue['clue_value']} clue, affecting cards at positions {clue['affected_indices']}")

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


def run_game():
    """Run a complete game with AI agents."""
    logger.info("=" * 50)
    logger.info("STARTING NEW GAME OF HANABI")
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
        discussion_manager = DiscussionManager(max_rounds=3)
        logger.info(f"Game initialized with {len(agents)} agents")
    except Exception as e:
        logger.error(f"Error initializing game components: {e}")
        return

    logger.info("Initial game state:")
    logger.info(f"Deck size: {len(engine.state.deck)}")
    logger.info(f"Clue tokens: {engine.state.clue_tokens}")
    logger.info(f"Fuse tokens: {engine.state.fuse_tokens}")
    logger.info(f"Current player: {engine.state.current_player}")

    # Log detailed initial game state
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
                f"TURN {human_readable_turn} - PLAYER {current_player}'S TURN")
            logger.info(
                f"Internal turn count: {turn_info['turn_count']}, Player index: {current_player}")
            logger.info("=" * 50)

            # Log detailed game state at the start of each turn
            log_detailed_game_state(engine)

            # Start new discussion
            logger.info("----- DISCUSSION PHASE BEGINS -----")
            discussion_manager.start_new_discussion()

            # First, the active player proposes an action
            current_agent = agents[current_player]
            game_view = engine.state.get_view_for(current_player)

            logger.info(
                f"Active Player {current_player} is proposing an action")
            try:
                # Get the active player's initial proposal
                initial_proposal = current_agent.participate_in_discussion(
                    game_view,
                    discussion_manager.get_game_history(last_n_turns=3)
                )
                logger.info(
                    f"Player {current_player} proposes: {initial_proposal}")
                # Add the proposal to the discussion history BEFORE requesting feedback
                discussion_manager.add_contribution(
                    current_player, initial_proposal)
            except Exception as e:
                logger.error(
                    f"Error getting proposal from Player {current_player}: {e}")
                initial_proposal = "I'm having trouble analyzing the game state."
                discussion_manager.add_contribution(
                    current_player, initial_proposal)

            # Select 2-3 other agents to provide feedback
            # We'll use a simple strategy: pick agents that are 1, 2, and sometimes 3 positions ahead
            num_agents = len(agents)
            feedback_agents = [(current_player + i) %
                               num_agents for i in range(1, min(4, num_agents))]

            # For the first player, we'll only get feedback from 2 agents to keep things moving
            if current_player == 0:
                feedback_agents = feedback_agents[:2]

            logger.info(f"Requesting feedback from agents: {feedback_agents}")

            # Get feedback from each selected agent
            for feedback_agent_id in feedback_agents:
                logger.info(f"Getting feedback from Agent {feedback_agent_id}")
                feedback_agent = agents[feedback_agent_id]
                feedback_view = engine.state.get_view_for(feedback_agent_id)

                try:
                    feedback = feedback_agent.participate_in_discussion(
                        feedback_view,
                        discussion_manager.get_discussion_history()
                    )
                    logger.info(f"Agent {feedback_agent_id} says: {feedback}")
                    discussion_manager.add_contribution(
                        feedback_agent_id, feedback)
                except Exception as e:
                    logger.error(
                        f"Error getting feedback from Agent {feedback_agent_id}: {e}")
                    feedback = "I'm having trouble analyzing the current situation."
                    discussion_manager.add_contribution(
                        feedback_agent_id, feedback)

            logger.info("----- DISCUSSION PHASE ENDS -----")

            # Action phase
            logger.info("----- ACTION PHASE BEGINS -----")
            logger.info(f"Agent {current_player} is making final decision")

            # The active agent makes the final decision
            game_view = engine.state.get_view_for(current_player)

            try:
                # This will raise an exception if the agent fails to propose a valid action
                engine._play_turn()
                # If we get here, the action was successful
                logger.info("----- ACTION PHASE ENDS -----")
            except Exception as e:
                # The exception will be caught by the outer try-except block
                logger.critical(f"Error during action phase: {e}")
                raise

            # Check game over conditions
            if engine.state.fuse_tokens <= 0:
                logger.info("Game Over! Ran out of fuse tokens!")
                break

            if engine.state.get_completed_fireworks_count() == 5:
                logger.info("Game Over! All fireworks completed!")
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


if __name__ == "__main__":
    run_game()
