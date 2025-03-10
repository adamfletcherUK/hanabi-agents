# Hanabi Game Engine

A clean implementation of the Hanabi card game engine, designed as a foundation for AI agent development.

## Overview

This repository provides a clean implementation of the Hanabi card game engine, focusing on:

- Clear separation of game logic from agent implementation
- Well-documented code with type hints
- Flexible architecture for future agent development
- Support for future integration with LangGraph, tool calling, and checkpoints

## Game Rules

Hanabi is a cooperative card game where players work together to create five firework stacks (one for each color) in ascending order (1-5). The unique twist is that players cannot see their own cards but can see everyone else's cards. Players must give each other clues to help identify cards, but clues are limited resources.

### Key Game Mechanics

- **Hidden Information**: Players cannot see their own cards but can see all other players' cards
- **Limited Communication**: Players can only give specific types of clues (color or number)
- **Resource Management**: Clue tokens (8) and fuse tokens (3) limit actions
- **Cooperative Goal**: Build five complete firework stacks (1-5 in each color)

## Project Structure

```
hanabi_agents/
├── game/               # Core game logic
│   ├── state.py        # Game state representation
│   └── engine.py       # Game execution engine
└── utils/              # Utility functions
    └── logging.py      # Logging utilities
```

## Game Engine Features

The game engine provides:

- Complete implementation of Hanabi game rules
- Game state management with proper information hiding
- Action validation and execution
- Score tracking and game completion detection
- Comprehensive logging

## Game State

The `GameState` class represents the complete state of a Hanabi game, including:

- Deck of cards
- Player hands
- Firework piles
- Discard pile
- Game resources (clue tokens, fuse tokens)
- Game progress tracking (score, turn count)
- Action history

## Actions

The game supports three types of actions:

1. **Play Card**: Attempt to play a card to a firework pile
   ```python
   {"type": "play_card", "card_index": 0}
   ```

2. **Give Clue**: Give a clue to another player
   ```python
   {"type": "give_clue", "target_id": 1, "clue": {"type": "color", "value": "red"}}
   ```

3. **Discard**: Discard a card to gain a clue token
   ```python
   {"type": "discard", "card_index": 0}
   ```

## Future Plans

- Integration with LangGraph for agent reasoning
- Tool calling for agent actions
- Checkpoints for agent memory
- Advanced agent implementations
- Evaluation framework for comparing agents

## License

[MIT License](LICENSE) 