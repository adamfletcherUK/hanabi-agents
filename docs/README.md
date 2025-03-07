# Hanabi Agent Memory System Documentation

This directory contains documentation about the memory and conversation management system used in the Hanabi AI agent implementation.

## Documentation Files

- [**agent_memory_and_conversation.md**](agent_memory_and_conversation.md): Detailed explanation of how agent memory, conversation history, and game state information are managed in the Hanabi implementation.
- [**agent_memory_flow.md**](agent_memory_flow.md): Visual representations (ASCII diagrams) of memory and conversation flow between agents and turns.

## Key Concepts

The Hanabi implementation uses a sophisticated memory management system that:

1. **Maintains conversation history** between turns while hiding each agent's private card information
2. **Preserves game context** across multiple turns
3. **Enables agents to reference previous discussions** when making decisions
4. **Balances memory usage** to prevent token overflow

## How to Use the Memory System

### Accessing Previous Discussions

To access previous discussions in your agent implementation:

```python
# Get game history from the discussion manager
game_history = discussion_manager.get_game_history(last_n_turns=3)

# Pass it to your agent
contribution = agent.participate_in_discussion(
    game_view,
    game_history
)
```

### Updating Agent Memory

To update an agent's memory with new information:

```python
# Update a specific memory key
agent.update_memory("game_summary", game_summary)

# Retrieve memory
stored_summary = agent.get_memory("game_summary")
```

### Customizing Memory Management

You can customize how much history is preserved by adjusting:

1. The number of turns to include when retrieving game history:
   ```python
   # Get only the last 2 turns
   game_history = discussion_manager.get_game_history(last_n_turns=2)
   ```

2. The number of entries to include in prompts:
   ```python
   # In your prompt generation method
   recent_history = game_history[-10:] if len(game_history) > 10 else game_history
   ```

## Preventing Invalid Actions

The system includes mechanisms to prevent invalid actions from being submitted:

### Pre-Validation

Before submitting an action to the game engine, the agent performs pre-validation:

```python
# Validate the action before submitting it
if self._validate_action_before_submission(game_state, action):
    return action
else:
    return self._generate_fallback_action(game_state)
```

### Fallback Actions

When an invalid action is detected, the agent generates a safe fallback action:

```python
def _generate_fallback_action(self, game_state: GameState) -> Dict[str, Any]:
    # Try to give a clue if possible
    if game_state.clue_tokens > 0:
        # Find a target player (not self)
        target_id = (self.agent_id + 1) % len(game_state.hands)
        
        # Choose a simple clue (number 1 is usually safe)
        return {
            "type": "give_clue",
            "target_id": target_id,
            "clue": {"type": "number", "value": 1}
        }
    
    # If we can't give a clue, try to discard
    if game_state.clue_tokens < 8:
        return {
            "type": "discard",
            "card_index": 0  # Discard the oldest card
        }
```

### Enhanced Prompts

The action proposal prompt includes detailed information about valid actions:

```python
valid_actions_info = self._generate_valid_actions_info(game_state)

return f"""
...
VALID ACTIONS INFORMATION:
{valid_actions_info}
...
"""
```

This information helps the agent understand what actions are valid in the current game state.

## Best Practices

1. **Limit History Size**: Prevent token overflow by limiting the number of historical entries
2. **Use Structured Formatting**: Format historical entries with clear turn and agent information
3. **Maintain Dual Representation**: Use both detailed history and summarized context
4. **Protect Private Information**: Ensure that an agent's own cards remain hidden
5. **Validate Actions**: Always validate actions before submitting them to the game engine
6. **Provide Fallbacks**: Have safe fallback actions ready when the primary action is invalid

## Implementation Details

For detailed implementation information, see the source code:

- `src/communication/discussion.py`: Discussion management and history tracking
- `src/agents/ai_agent.py`: Agent memory and reasoning implementation
- `examples/run_game.py`: Game loop and memory update process

## Known Issues and Fixes

- **GameState.last_action Issue**: The original implementation tried to access `engine.state.last_action` which doesn't exist in the `GameState` object. This has been fixed by using the action information directly from the action dictionary instead of trying to access the played/discarded card details from the game state.
- **Invalid Action Loop Issue**: The original implementation could get stuck in a loop when an invalid action was attempted. This has been fixed by always advancing to the next player, even when an action is invalid, and by implementing pre-validation and fallback actions. 