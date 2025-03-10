# Hanabi AI Agents

A Python implementation of the Hanabi card game played by AI agents using Langgraph. The game features a unique pre-action discussion phase where agents collaborate to make optimal moves.

## Features

- Turn-based Hanabi game with 5 AI agents
- Pre-action discussion phase for collaborative decision-making
- Secure information flow (private vs public information)
- Langgraph-based AI reasoning
- Comprehensive game state management
- Detailed logging and debugging support

## Project Structure

```
hanabi-agents/
├── src/
│   ├── game/           # Core game logic
│   ├── agents/         # AI agent implementations
│   ├── communication/  # Discussion phase management
│   └── utils/          # Utility functions
├── tests/              # Test suite (future)
└── docs/              # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hanabi-agents.git
cd hanabi-agents
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

## Usage

```python
from src.game.engine import GameEngine
from src.agents.ai_agent import AIAgent

# Create agents
agents = [AIAgent(i) for i in range(5)]

# Initialize and run game
engine = GameEngine(agents)
final_score = engine.play_game()
```

## Development

The project uses:
- Python 3.9+
- Langgraph for AI agent management
- Pydantic for data validation
- Rich for terminal output

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 