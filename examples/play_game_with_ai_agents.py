from hanabi_agents.agents.discussion.discussion_manager import DiscussionManager
from hanabi_agents.agents.ai_agent import AIAgent
from hanabi_agents.game.engine import GameEngine
from hanabi_agents.utils import logging as hanabi_logging
from hanabi_agents.utils import game_logger
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

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(log_level)
console_format = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)
root_logger.addHandler(console_handler)

# Create file handler
try:
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(log_level)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)
    print(f"Successfully set up logging to file: {log_file_path}")
except Exception as e:
    print(f"Error setting up file logging: {str(e)}")

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

    # Get the messages from the reasoning graph's last invocation
    if hasattr(agent, 'reasoning_graph') and hasattr(agent, 'checkpointer'):
        # Try to get the latest checkpoint
        try:
            # Get the messages from the agent's memory
            messages = agent.get_memory_from_store("messages", [])

            if messages:
                logger.info(
                    f"Found {len(messages)} messages in agent's memory")

                # Log each message in the reasoning chain
                # Always log the header to the file
                logger.info(
                    f"🧠 REASONING CHAIN FOR AGENT {agent.agent_id} (Turn {turn_count}):")

                if print_to_console:
                    print(
                        f"\n🧠 REASONING CHAIN FOR AGENT {agent.agent_id} (Turn {turn_count}):")
                    print("=" * 80)

                # Track the reasoning steps
                analysis = None
                thoughts = []
                proposed_action = None

                for i, msg in enumerate(messages):
                    if hasattr(msg, 'content'):
                        # Determine the type of message based on content patterns
                        content = msg.content

                        # Log the message with appropriate formatting
                        if i % 2 == 0:  # This is a prompt
                            if "analyze the current game state" in content.lower():
                                step_type = "STATE ANALYSIS PROMPT"
                            elif "generate strategic thoughts" in content.lower():
                                step_type = "THOUGHT GENERATION PROMPT"
                            elif "propose a concrete action" in content.lower():
                                step_type = "ACTION PROPOSAL PROMPT"
                            else:
                                step_type = "PROMPT"

                            # Always log to file
                            logger.info(f"Step {i//2 + 1}: {step_type}")

                            if print_to_console:
                                print(f"\n📝 STEP {i//2 + 1}: {step_type}")
                                print("-" * 40)
                                # Uncomment to print the full prompt (can be very verbose)
                                # print(content)
                                print("-" * 40)
                        else:  # This is a response
                            if i == 1:  # First response is the analysis
                                analysis = content
                                # Log full analysis to file
                                logger.info(
                                    f"Game State Analysis (Turn {turn_count}, Agent {agent.agent_id}):")
                                logger.info(content)

                                if print_to_console:
                                    print(f"\n🔍 ANALYSIS (FULL LLM OUTPUT):")
                                    print("-" * 40)
                                    print(content)
                                    print("-" * 40)
                            elif i == 3:  # Third response is thought generation
                                # Extract thoughts
                                thought_lines = [line.strip() for line in content.split('\n')
                                                 if line.strip() and not line.strip().startswith("Thought") and not line.strip().startswith("#")]
                                thoughts.extend(thought_lines)

                                # Log full thoughts to file
                                logger.info(
                                    f"Generated Thoughts (Turn {turn_count}, Agent {agent.agent_id}):")
                                logger.info(content)

                                # Extract and log thoughts in a more structured way
                                extracted_thoughts = []
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
                                            extracted_thoughts.append(thought)

                                # Log the extracted thoughts
                                if extracted_thoughts:
                                    logger.info(
                                        f"Extracted thoughts for agent {agent.agent_id} (Turn {turn_count}):")
                                    for j, thought in enumerate(extracted_thoughts):
                                        logger.info(
                                            f"  Thought {j+1}: {thought}")

                                if print_to_console:
                                    print(f"\n💭 THOUGHTS (FULL LLM OUTPUT):")
                                    print("-" * 40)
                                    print(content)
                                    print("-" * 40)

                                    if extracted_thoughts:
                                        print(f"\n💭 EXTRACTED THOUGHTS:")
                                        for j, thought in enumerate(extracted_thoughts):
                                            print(f"  {j+1}. {thought}")
                                    else:
                                        print(
                                            "\n⚠️ No thoughts could be extracted from the LLM output")
                            elif i == 5:  # Fifth response is the action proposal
                                proposed_action = content
                                # Log full action proposal to file
                                logger.info(
                                    f"Proposed Action (Turn {turn_count}, Agent {agent.agent_id}):")
                                logger.info(content)

                                if print_to_console:
                                    print(
                                        f"\n🎬 PROPOSED ACTION (FULL LLM OUTPUT):")
                                    print("-" * 40)
                                    print(content)
                                    print("-" * 40)
                            else:
                                # Any other responses
                                # Log other responses to file
                                logger.info(
                                    f"Other LLM Output (Step {i//2 + 1}, Turn {turn_count}, Agent {agent.agent_id}):")
                                logger.info(content)

                                if print_to_console:
                                    print(f"\n📄 LLM OUTPUT (STEP {i//2 + 1}):")
                                    print("-" * 40)
                                    print(content)
                                    print("-" * 40)

                # Validate memory storage
                memory_thoughts = agent.get_memory_from_store(
                    "current_thoughts", [])
                if memory_thoughts:
                    logger.info(
                        f"✅ Validated: {len(memory_thoughts)} thoughts stored in memory")
                    # Explicitly log each thought
                    logger.info("Agent's thoughts:")
                    for i, thought in enumerate(memory_thoughts):
                        logger.info(f"  Thought {i+1}: {thought}")
                    if print_to_console:
                        print(
                            f"\n✅ MEMORY VALIDATION: {len(memory_thoughts)} thoughts stored in memory")
                        print("Agent's thoughts:")
                        for i, thought in enumerate(memory_thoughts):
                            print(f"  Thought {i+1}: {thought}")
                else:
                    logger.warning("❌ No thoughts found in memory store")
                    if print_to_console:
                        print(
                            "\n❌ MEMORY VALIDATION: No thoughts found in memory store")

                if print_to_console:
                    print("=" * 80)
            else:
                logger.warning("No messages found in agent's memory")
                if print_to_console:
                    print("\n⚠️ No reasoning chain available - no messages in memory")
        except Exception as e:
            logger.error(f"Error retrieving reasoning chain: {e}")
            if print_to_console:
                print(f"\n❌ Error retrieving reasoning chain: {e}")
    else:
        logger.warning(
            "Agent does not have reasoning_graph or checkpointer attributes")
        if print_to_console:
            print("\n⚠️ Agent does not have reasoning capabilities")


def main():
    """Run a game of Hanabi with AI agents."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables")
        print("Please set your OPENAI_API_KEY in the .env file or environment")
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
    print("\n🎮 STARTING NEW GAME OF HANABI 🎮")
    print("=" * 50)

    # Log initial game state
    game_logger.log_game_state(engine, print_to_console=True)

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

        # Log turn information
        game_logger.log_turn_info(
            turn_count + 1,
            current_agent.name,
            current_player_id,
            game_state.clue_tokens,
            game_state.max_clue_tokens,
            game_state.fuse_tokens,
            game_state.score,
            print_to_console=True
        )

        # Log detailed game state
        game_logger.log_game_state(engine, print_to_console=True)

        # Modified: Only the active player analyzes the game state and suggests an action
        logger.info("Starting active player analysis phase")
        print("\n--- 🧠 ACTIVE PLAYER ANALYSIS PHASE ---")

        # Get the active player's reasoning
        active_player_id = game_state.current_player
        active_agent = next(
            agent for agent in agents if agent.agent_id == active_player_id)

        print(f"\n👤 PLAYER {active_player_id} (ACTIVE PLAYER) REASONING:")
        print("=" * 80)

        # Empty discussion contributions - only the active player will contribute
        discussion_contributions = []

        # Get the active player's contribution and display raw LLM outputs
        contribution = active_agent.participate_in_discussion(
            game_state, discussion_contributions)

        # Display the raw messages from the agent's memory
        messages = active_agent.get_memory_from_store("messages", [])
        if messages:
            for i, msg in enumerate(messages):
                # Only show LLM responses (odd indices)
                if hasattr(msg, 'content') and i % 2 == 1:
                    content = msg.content
                    print(f"\n📄 LLM OUTPUT (STEP {i//2 + 1}):")
                    print("-" * 40)
                    print(content)
                    print("-" * 40)

                    # Log the content to the log file
                    logger.info(f"LLM OUTPUT (STEP {i//2 + 1}):")
                    logger.info(content)

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

                        # Log the extracted thoughts
                        if thoughts:
                            logger.info(
                                f"Extracted thoughts for agent {active_agent.agent_id} (Turn {turn_count + 1}):")
                            for j, thought in enumerate(thoughts):
                                logger.info(f"  Thought {j+1}: {thought}")
                            print(f"\n💭 EXTRACTED THOUGHTS:")
                            for j, thought in enumerate(thoughts):
                                print(f"  {j+1}. {thought}")
                        else:
                            logger.info(
                                f"No thoughts could be extracted from the LLM output for agent {active_agent.agent_id}")
                            print(
                                "\n⚠️ No thoughts could be extracted from the LLM output")

        # Add the contribution to the discussion history
        discussion_contributions.append({
            "player_id": active_player_id,
            "content": contribution,
            "is_active_player": True
        })

        # Modified: Skip other players' contributions and discussion summary
        # Directly decide on an action based on the active player's analysis
        logger.info("Starting action phase")
        print("\n--- 🎬 Action Phase ---")

        # No discussion summary needed - pass empty string
        action = current_agent.decide_action(game_state, "")
        logger.info(f"Action decided: {action}")

        # Log the agent's reasoning chain
        log_agent_reasoning(current_agent, turn_count + 1)

        # Explicitly log the agent's thoughts
        current_thoughts = current_agent.get_memory_from_store(
            "current_thoughts", [])
        if current_thoughts:
            logger.info(
                f"Agent {current_agent.agent_id}'s thoughts for turn {turn_count + 1}:")
            for i, thought in enumerate(current_thoughts):
                logger.info(f"  Thought {i+1}: {thought}")
            print(f"\n💭 Agent {current_agent.agent_id}'s thoughts:")
            for i, thought in enumerate(current_thoughts):
                print(f"  {i+1}. {thought}")
        else:
            logger.warning(
                f"No thoughts found for agent {current_agent.agent_id} in turn {turn_count + 1}")

        # Display the formatted action
        action_display = game_logger.format_action_for_display(
            action, current_agent.name)
        print(action_display)

        # Execute the action
        result = engine.execute_action(current_player_id, action)
        logger.info(f"Action result: {result}")

        # Log the action result
        game_logger.log_action_result(
            action, result, current_agent.name, print_to_console=True)

        # Notify the agent of the result
        current_agent.notify_action_result(action, result)

        # Log the updated game state after the action
        game_logger.log_game_state(engine, print_to_console=True)

        # Check if the game is over
        game_over = engine.is_game_over()

        # Increment the turn count
        turn_count += 1

    # Log game over information
    game_state = engine.get_game_state()
    game_logger.log_game_over(
        game_state.score,
        engine.get_game_over_reason(),
        print_to_console=True
    )


if __name__ == "__main__":
    main()
