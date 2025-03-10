from hanabi_agents.agents.discussion.discussion_manager import DiscussionManager
from hanabi_agents.agents.ai_agent import AIAgent
from hanabi_agents.game.engine import GameEngine
from hanabi_agents.utils import logging as hanabi_logging
from hanabi_agents.utils import game_logger
from hanabi_agents.utils.game_logger import COLOR_EMOJI
import os
import sys
import logging
from dotenv import load_dotenv
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Set up logging based on environment variables
log_level_str = os.getenv("LOG_LEVEL", "INFO")
log_level = getattr(logging, log_level_str.upper(), logging.INFO)

# Configure logging
# Create a timestamped log file in the /logs directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, f"hanabi_game_{timestamp}.log")
print(f"Setting up logging to file: {log_file_path}")

# Set up logging directly
root_logger = logging.getLogger()
root_logger.setLevel(log_level)

# Remove existing handlers
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Create console handler - only for important messages
console_handler = logging.StreamHandler(sys.stdout)
# Set console handler to INFO level to show detailed output
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter(
    '%(message)s')  # Simplified format for console - just the message
console_handler.setFormatter(console_format)
root_logger.addHandler(console_handler)

# Create file handler - for all detailed logs
try:
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(log_level)  # Full logging to file
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)
    print(f"Successfully set up logging to file: {log_file_path}")
except Exception as e:
    print(f"Error setting up file logging: {str(e)}")

# Configure langchain and OpenAI logging to go to file only
for module in ['langchain', 'langchain_openai', 'openai', 'httpx', 'httpcore']:
    module_logger = logging.getLogger(module)
    # Capture all logs from these modules
    module_logger.setLevel(logging.DEBUG)
    # Remove any existing handlers
    for handler in module_logger.handlers[:]:
        module_logger.removeHandler(handler)
    # Add only the file handler (no console handler)
    module_logger.addHandler(file_handler)
    # Prevent propagation to root logger to avoid duplicate logs
    module_logger.propagate = False

# Disable urllib3 and requests logging to console
logging.getLogger('urllib3').propagate = False
logging.getLogger('requests').propagate = False

# Create a dedicated console logger for game output
console_logger = logging.getLogger('console')
console_logger.setLevel(logging.INFO)
# Remove any existing handlers
for handler in console_logger.handlers[:]:
    console_logger.removeHandler(handler)
# Add the console handler to the console logger
console_logger.addHandler(console_handler)
console_logger.propagate = False  # Don't propagate to root logger

logger = logging.getLogger(__name__)
logger.info(f"Game logging initialized to file: {log_file_path}")

# Get game configuration from environment
max_turns = int(os.getenv("MAX_TURNS", "20"))
num_players = int(os.getenv("NUM_PLAYERS", "3"))
model_name = os.getenv("MODEL_NAME", "o3-mini")


def log_agent_reasoning(agent, turn_count, print_to_console=True):
    """
    Log the reasoning chain of an AI agent and validate memory storage.

    Args:
        agent: The AI agent
        turn_count: The current turn count
        print_to_console: Whether to print to console
    """
    logger.info(
        f"--- Agent {agent.agent_id} Reasoning Chain (Turn {turn_count}) ---")

    # Check if the agent has the necessary attributes
    if hasattr(agent, 'agent_memory'):
        # Get the thoughts from the agent's memory
        thoughts = agent.agent_memory.thoughts

        if thoughts:
            logger.info(
                f"Found {len(thoughts)} thoughts in agent's memory")

            # Log each thought in the reasoning chain to file
            logger.info(
                f"🧠 REASONING CHAIN FOR AGENT {agent.agent_id} (Turn {turn_count}):")
            for i, thought in enumerate(thoughts):
                logger.info(f"Thought {i+1}: {thought}")

            # Print to console if requested - but only a simplified version
            if print_to_console:
                # Use a different format for console to avoid duplication
                console_logger.info(
                    f"\n🧠 REASONING CHAIN FOR AGENT {agent.agent_id} (Turn {turn_count}):")
                for i, thought in enumerate(thoughts):
                    console_logger.info(f"  Thought {i+1}: {thought}")
        else:
            logger.warning(
                f"No thoughts found for agent {agent.agent_id} in turn {turn_count}")
            if print_to_console:
                console_logger.info(
                    f"\n⚠️ No thoughts found for agent {agent.agent_id} in turn {turn_count}")
    else:
        logger.warning(
            "Agent does not have reasoning_graph or checkpointer attributes")
        if print_to_console:
            console_logger.info(
                "\n⚠️ Agent does not have reasoning capabilities")


def log_action_history(agents, turn_count, print_to_console=True):
    """
    Log the action history for all agents.

    Args:
        agents: List of AI agents
        turn_count: The current turn count
        print_to_console: Whether to print to console
    """
    logger.info(f"--- Action History (Turn {turn_count}) ---")

    if print_to_console:
        # Use a different format for console to avoid duplication
        console_logger.info(f"\n📜 ACTION HISTORY (Turn {turn_count}):")
        console_logger.info("=" * 80)

    # Collect all actions from all agents
    all_actions = []

    for agent in agents:
        if hasattr(agent, 'agent_memory'):
            # Get action results from agent memory
            action_results = agent.agent_memory.action_results

            for action_result in action_results:
                all_actions.append({
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "action": action_result.action,
                    "result": action_result.result,
                    # Use the current turn count instead of trying to access action_result.turn
                    "turn": turn_count
                })

    # Sort actions by turn
    all_actions.sort(key=lambda x: x.get("turn", 0))

    # Log each action
    if all_actions:
        for i, action_data in enumerate(all_actions):
            agent_id = action_data.get("agent_id", "?")
            agent_name = action_data.get("agent_name", "?")
            action = action_data.get("action", {})
            action_type = action.get("type", "unknown")

            # Format the action for display
            log_message = f"Action {i+1}: {agent_name} (Player {agent_id}) - "

            if action_type == "play_card":
                card_index = action.get("card_index", "?")
                log_message += f"Play card at position {card_index}"
            elif action_type == "give_clue":
                target_id = action.get("target_id", "?")
                clue = action.get("clue", {})
                clue_type = clue.get("type", "?")
                clue_value = clue.get("value", "?")

                # Add emoji for color clues
                if clue_type == "color":
                    emoji = COLOR_EMOJI.get(clue_value, "")
                    log_message += f"Give {emoji} color clue to Player {target_id}"
                else:
                    log_message += f"Give {clue_value} number clue to Player {target_id}"
            elif action_type == "discard":
                card_index = action.get("card_index", "?")
                log_message += f"Discard card at position {card_index}"
            else:
                log_message += f"Unknown action: {action_type}"

            logger.info(log_message)

            if print_to_console:
                # Use a different format for console to avoid duplication
                console_logger.info(log_message)
    else:
        logger.info("No actions recorded yet")
        if print_to_console:
            console_logger.info("No actions recorded yet")

    if print_to_console:
        console_logger.info("=" * 80)


def main():
    """Run a game of Hanabi with AI agents."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables")
        console_logger.info(
            "Please set your OPENAI_API_KEY in the .env file or environment")
        return

    # Create the game engine
    logger.info(f"Creating game engine with {num_players} players")
    engine = GameEngine(num_players=num_players)

    # Create the AI agents
    logger.info(f"Creating AI agents using model: {model_name}")
    agents = [
        AIAgent(agent_id=i, name=f"Player {i}", model_name=model_name)
        for i in range(num_players)
    ]

    # Initialize the game
    logger.info("Initializing game")
    console_logger.info("\n🎮 STARTING NEW GAME OF HANABI 🎮")
    console_logger.info("=" * 50)

    # Log initial game state to file only
    game_logger.log_game_state(engine, print_to_console=False)

    # Print simplified game state to console
    console_logger.info("\n🎆 INITIAL GAME STATE:")
    console_logger.info(f"Players: {num_players}")
    console_logger.info(
        f"Clue tokens: {engine.state.clue_tokens}/{engine.state.max_clue_tokens}")
    console_logger.info(
        f"Fuse tokens: {engine.state.fuse_tokens}/{engine.state.max_fuse_tokens}")
    console_logger.info(f"Score: {engine.state.score}/25")
    console_logger.info("=" * 50)

    # Play the game
    game_over = False
    turn_count = 0

    while not game_over and turn_count < max_turns:
        # Get the current game state
        game_state = engine.get_game_state()

        # Get the current player
        current_player_id = game_state.current_player
        current_agent = next(
            agent for agent in agents if agent.agent_id == current_player_id)

        # Log turn information to file only
        game_logger.log_turn_info(
            turn_count + 1,
            current_agent.name,
            current_player_id,
            game_state.clue_tokens,
            game_state.max_clue_tokens,
            game_state.fuse_tokens,
            game_state.score,
            print_to_console=False
        )

        # Print simplified turn info to console
        console_logger.info(f"\n=== 🎲 Turn {turn_count + 1} ===")
        console_logger.info(
            f"Current player: 👤 {current_agent.name} (Player {current_player_id})")
        console_logger.info(
            f"🔍 Clue tokens: {game_state.clue_tokens}/{game_state.max_clue_tokens}")
        console_logger.info(
            f"💣 Fuse tokens: {game_state.fuse_tokens}/{game_state.max_fuse_tokens}")
        console_logger.info(f"🏆 Score: {game_state.score}/25")

        # Log detailed game state to console and file
        game_logger.log_game_state(engine, print_to_console=True)

        # Modified: Only the active player analyzes the game state and suggests an action
        logger.info("Starting active player analysis phase")
        console_logger.info("\n--- 🧠 ACTIVE PLAYER ANALYSIS PHASE ---")

        # Get the active player's reasoning
        active_player_id = game_state.current_player
        active_agent = next(
            agent for agent in agents if agent.agent_id == active_player_id)

        console_logger.info(
            f"\n👤 PLAYER {active_player_id} (ACTIVE PLAYER) REASONING:")

        # Empty discussion contributions - only the active player will contribute
        discussion_contributions = []

        # Get the active player's contribution and display raw LLM outputs
        contribution = active_agent.participate_in_discussion(
            game_state, discussion_contributions)

        # Display the raw messages from the agent's memory - but only to the log file
        messages = active_agent.get_memory_from_store("messages", [])
        if messages:
            for i, msg in enumerate(messages):
                # Only log LLM responses (odd indices) to the file
                if hasattr(msg, 'content') and i % 2 == 1:
                    content = msg.content
                    # Log to file but not console
                    logger.debug(f"LLM OUTPUT (STEP {i//2 + 1}):")
                    logger.debug(content)

                    # If this is the thought generation step (step 2)
                    if i == 3:  # Third response (index 3) is thought generation
                        # Extract and log thoughts
                        thoughts = []
                        for line in content.strip().split("\n"):
                            line = line.strip()
                            if line and (line[0].isdigit() or line[0] in ["•", "-", "*"]):
                                # Remove the number or bullet point
                                thought = line
                                if line[0].isdigit():
                                    parts = line.split(".", 1)
                                    if len(parts) > 1:
                                        thought = parts[1].strip()
                                else:
                                    thought = line[1:].strip()

                                if thought:
                                    thoughts.append(thought)

                        # Log the extracted thoughts to file only
                        if thoughts:
                            logger.debug(
                                f"Extracted thoughts for agent {active_agent.agent_id} (Turn {turn_count + 1}):")
                            for j, thought in enumerate(thoughts):
                                logger.debug(f"  Thought {j+1}: {thought}")
                        else:
                            logger.debug(
                                f"No thoughts could be extracted from the LLM output for agent {active_agent.agent_id}")
                            console_logger.info(
                                "\n⚠️ No thoughts could be extracted from the LLM output")
                # Also log input prompts to file only
                elif hasattr(msg, 'content') and i % 2 == 0:
                    content = msg.content
                    # Log to file but not console
                    logger.debug(f"INPUT PROMPT (STEP {i//2 + 1}):")
                    logger.debug(content)

        # Add the contribution to the discussion history
        discussion_contributions.append({
            "player_id": active_player_id,
            "content": contribution,
            "is_active_player": True
        })

        # Modified: Skip other players' contributions and discussion summary
        # Directly decide on an action based on the active player's analysis
        logger.info("Starting action phase")
        console_logger.info("\n--- 🎬 Action Phase ---")

        # No discussion summary needed - pass empty string
        action = current_agent.decide_action(game_state, "")
        logger.info(f"Action decided: {action}")

        # Log the agent's reasoning chain
        log_agent_reasoning(current_agent, turn_count + 1)

        # Log the agent's thoughts to file only
        current_thoughts = current_agent.get_memory_from_store(
            "current_thoughts", [])
        if current_thoughts:
            logger.info(
                f"Agent {current_agent.agent_id}'s thoughts for turn {turn_count + 1}:")
            for i, thought in enumerate(current_thoughts):
                logger.info(f"  Thought {i+1}: {thought}")
        else:
            logger.warning(
                f"No thoughts found for agent {current_agent.agent_id} in turn {turn_count + 1}")

        # Display the formatted action with better highlighting
        action_display = game_logger.format_action_for_display(
            action, current_agent.name)
        console_logger.info(f"\n🎯 DECIDED ACTION: {action_display}")

        # Execute the action
        result = engine.execute_action(current_player_id, action)
        logger.info(f"Action result: {result}")

        # Log the action result
        game_logger.log_action_result(
            action, result, current_agent.name, print_to_console=False)

        # Print simplified action result to console
        if isinstance(result, bool):
            console_logger.info(f"Action completed: {result}")
        elif isinstance(result, dict) and "card" in result:
            card = result["card"]
            if hasattr(card, "color") and hasattr(card, "number"):
                color_name = card.color.value if hasattr(
                    card.color, "value") else str(card.color)
                color_emoji = COLOR_EMOJI.get(color_name, "")
                console_logger.info(f"Card: {color_emoji}{card.number}")

        # Notify the agent of the result
        current_agent.notify_action_result(action, result)

        # Create a detailed action history entry for logging
        action_history_entry = {
            "turn": turn_count + 1,
            "player_id": current_player_id,
            "player_name": current_agent.name,
            "action_type": action.get("type", "unknown"),
            "action_details": action,
            "result": result
        }

        # Store the action history in each agent's memory
        for agent in agents:
            agent.agent_memory.store_memory(
                "action_history", action_history_entry)

            # Update game summary for each agent
            game_summary = f"Turn {turn_count + 1}: Player {current_player_id} ({current_agent.name}) - "

            if action.get("type") == "play_card":
                card_index = action.get("card_index", "?")
                game_summary += f"Played card at position {card_index}"
                if isinstance(result, dict) and "card" in result:
                    card = result["card"]
                    if hasattr(card, "color") and hasattr(card, "number"):
                        color_name = card.color.value if hasattr(
                            card.color, "value") else str(card.color)
                        color_emoji = COLOR_EMOJI.get(color_name, "")
                        game_summary += f" ({color_emoji}{card.number})"
            elif action.get("type") == "give_clue":
                target_id = action.get("target_id", "?")
                clue = action.get("clue", {})
                clue_type = clue.get("type", "?")
                clue_value = clue.get("value", "?")
                game_summary += f"Gave {clue_type} clue ({clue_value}) to Player {target_id}"
            elif action.get("type") == "discard":
                card_index = action.get("card_index", "?")
                game_summary += f"Discarded card at position {card_index}"
                if isinstance(result, dict) and "card" in result:
                    card = result["card"]
                    if hasattr(card, "color") and hasattr(card, "number"):
                        color_name = card.color.value if hasattr(
                            card.color, "value") else str(card.color)
                        color_emoji = COLOR_EMOJI.get(color_name, "")
                        game_summary += f" ({color_emoji}{card.number})"

            # Add game state information
            game_summary += f". Score: {engine.state.score}, Clues: {engine.state.clue_tokens}, Fuses: {engine.state.fuse_tokens}"

            # Store the game summary
            agent.agent_memory.store_memory("game_summary", game_summary)

        # Log the action history for all agents
        log_action_history(agents, turn_count + 1)

        # Check if the game is over
        game_over = engine.is_game_over()

        # Increment the turn count
        turn_count += 1

    # Log game over information
    if game_over:
        reason = "Deck empty" if engine.state.deck_empty else "All fuses used"
        logger.info(
            f"Game over! Final score: {engine.state.score}/25. Reason: {reason}")
        console_logger.info(f"\n🎮 GAME OVER! 🎮")
        console_logger.info(f"Final score: {engine.state.score}/25")
        console_logger.info(f"Reason: {reason}")
    else:
        logger.info(
            f"Maximum turns ({max_turns}) reached. Final score: {engine.state.score}/25")
        console_logger.info(f"\n🎮 MAXIMUM TURNS REACHED 🎮")
        console_logger.info(f"Final score: {engine.state.score}/25")

    # Return the final score
    return engine.state.score


if __name__ == "__main__":
    main()
