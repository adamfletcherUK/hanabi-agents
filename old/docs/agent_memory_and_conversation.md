# Agent Memory and Conversation Flow in Hanabi

This document explains how agent memory, conversation history, and game state information are managed in the Hanabi AI agent implementation.

## Overview

The Hanabi implementation uses a sophisticated memory management system to maintain conversation history between agents while preserving the hidden information requirements of the game (agents cannot see their own cards). This document outlines how memory is structured, how conversations flow between turns, and how agents access historical information.

## Memory Architecture

### Data Structures

1. **DiscussionEntry**
   - Represents a single contribution to the discussion
   - Contains:
     - `agent_id`: The ID of the agent who made the contribution
     - `timestamp`: When the contribution was made
     - `content`: The actual text of the contribution
     - `round_number`: Which round of discussion within a turn
     - `turn_number`: Which game turn this contribution belongs to

2. **DiscussionManager**
   - Manages two key data structures:
     - `discussion_history`: Current turn's discussion entries
     - `game_history`: Persistent record of all discussions across all turns

3. **Agent Memory**
   - Each agent has a `memory` dictionary that stores:
     - `game_history`: Previous turns' discussions
     - `game_summary`: Condensed information about game state and actions

4. **AgentState** (for LangGraph)
   - Represents the agent's reasoning state
   - Contains:
     - `game_state`: Current game state (filtered to hide the agent's own cards)
     - `discussion_history`: Current turn's discussion
     - `game_history`: Previous turns' discussions
     - `current_thoughts`: Agent's internal reasoning process
     - `proposed_action`: The action the agent plans to take

## Conversation Flow Between Turns

### Turn Transition Process

1. **End of Turn N**
   - Current discussion is added to the persistent game history
   - Game summary is updated with the action taken and resulting state
   - Each agent's memory is updated with this information

2. **Start of Turn N+1**
   - `discussion_history` is reset for the new turn
   - `current_round` is reset to 0
   - `turn_number` is incremented

3. **During Turn N+1**
   - The active player proposes an initial action
   - 2-3 selected agents provide feedback on the proposal
   - The active player makes a final decision based on the feedback
   - Agents have access to both current discussion and previous turns' history

### Discussion Flow

The discussion now follows a more structured approach:

1. **Active Player Proposal**
   - The active player analyzes the game state and proposes an initial action
   - This proposal includes the intended action and reasoning

2. **Feedback Phase**
   - 2-3 other agents are selected to provide feedback
   - These agents can support, suggest modifications, or propose alternatives
   - The selection is based on a simple rotation (next 2-3 agents in turn order)

3. **Final Decision**
   - The active player considers the feedback
   - Makes a final decision on the action to take
   - Executes the action

This approach creates a more focused and efficient discussion while still allowing for collaborative decision-making.

### Information Preservation

When a new turn starts, the following information is preserved:

- Complete history of all previous discussions (in `game_history`)
- Game state changes (in `game_summary`)
- Each agent's internal memory of previous interactions

## Agent Access to Historical Information

### What Agents Can Access

Agents have access to:

1. **Current Turn's Discussion**
   - All contributions from other agents in the current turn
   - Formatted as plain text in prompts

2. **Previous Turns' Discussions**
   - Limited history of previous turns (to prevent token overflow)
   - Formatted with turn and agent information
   - Example: `"Turn 2, Agent 3: Watch teammates closely for subtle cues..."`

3. **Game Summary**
   - Condensed information about game state and actions
   - Example: `"Turn 3: Player 2 give_clue. Score: 1, Clues: 6, Fuses: 3, Clue to Player 4: number 1"`
   - Example: `"Turn 4: Player 3 play_card. Score: 2, Clues: 6, Fuses: 3, Played card at index 2"`

### How Agents Access History

Agents access historical information through their prompts:

1. **State Analysis Prompt**
   - Includes game summary from memory
   - Focuses on current game state analysis

2. **Thought Generation Prompt**
   - Includes both current discussion and previous turns' discussions
   - Limited to last 15 entries to prevent token overflow
   - Encourages agents to reference previous discussions

3. **Action Proposal Prompt**
   - Includes both current discussion and previous turns' discussions
   - Limited to last 10 entries to prevent token overflow
   - Encourages consideration of game progression across turns

## Implementation Details

### Discussion Reset Between Turns

```python
def start_new_discussion(self) -> None:
    """Start a new discussion round for a new turn."""
    # Save current discussion to game history before resetting
    if self.discussion_history:
        self.game_history.extend(self.discussion_history)
    
    # Reset for new turn
    self.discussion_history = []
    self.current_round = 0
    self.current_turn += 1
```

### Processing Discussion History in Agents

```python
# Separate current discussion from previous turns
current_turn = max(entry.turn_number for entry in discussion_history)
current_discussion = [entry for entry in discussion_history if entry.turn_number == current_turn]
previous_discussions = [entry for entry in discussion_history if entry.turn_number < current_turn]

# Convert to strings
discussion_strings = [entry.content for entry in current_discussion]
game_history_strings = [f"Turn {entry.turn_number+1}, Agent {entry.agent_id}: {entry.content}" 
                       for entry in previous_discussions]
```

### Updating Game Summary After Each Turn

```python
# Update each agent's memory with a game summary
game_summary = f"Turn {engine.state.turn_count}: Player {current_player} {action['type']}. "
game_summary += f"Score: {engine.state.score}, Clues: {engine.state.clue_tokens}, Fuses: {engine.state.fuse_tokens}"

# Add details based on action type
if action['type'] == 'play_card':
    # Use the action information and card index
    card_index = action['card_index']
    game_summary += f", Played card at index {card_index}"
elif action['type'] == 'give_clue':
    game_summary += f", Clue to Player {action['target_id']}: {action['clue']['type']} {action['clue']['value']}"
elif action['type'] == 'discard':
    # Use the action information and card index
    card_index = action['card_index']
    game_summary += f", Discarded card at index {card_index}"

# Update each agent's memory
for agent in agents:
    agent.update_memory("game_summary", game_summary)
```

## Memory Management Best Practices

1. **Limit History Size**
   - Prevent token overflow by limiting the number of historical entries
   - Use recency bias (most recent entries are more relevant)

2. **Structured Formatting**
   - Format historical entries with clear turn and agent information
   - Makes it easier for agents to reference specific past discussions

3. **Dual Representation**
   - Maintain both detailed history and summarized context
   - Allows for both specific references and general awareness

4. **Private Information Protection**
   - Ensure that an agent's own cards remain hidden
   - Game state views are filtered for each agent

## Conclusion

The memory and conversation management system in the Hanabi implementation allows agents to maintain context across turns while respecting the hidden information requirements of the game. By preserving conversation history and game state information, agents can make more informed decisions and reference previous discussions, leading to more coherent and strategic gameplay. 