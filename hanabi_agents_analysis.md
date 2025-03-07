# Hanabi AI Agents: Codebase Analysis

## 1. Introduction

This report provides a comprehensive analysis of the Hanabi AI Agents codebase, which implements a system for 5 AI agents to play the cooperative card game Hanabi. The implementation uses Langgraph for agent reasoning and coordination, with a unique pre-action discussion phase that allows agents to collaborate on decision-making.

## 2. Game Overview

### 2.1 What is Hanabi?

Hanabi is a cooperative card game where players work together to create five firework stacks (one for each color) in ascending order (1-5). The unique twist is that players cannot see their own cards but can see everyone else's cards. Players must give each other clues to help identify cards, but clues are limited resources.

### 2.2 Key Game Mechanics

- **Hidden Information**: Players cannot see their own cards but can see all other players' cards
- **Limited Communication**: Players can only give specific types of clues (color or number)
- **Resource Management**: Clue tokens (8) and fuse tokens (3) limit actions
- **Cooperative Goal**: Build five complete firework stacks (1-5 in each color)

## 3. Codebase Architecture

The codebase is organized into several key components:

### 3.1 Core Game Logic (`src/game/`)

- **`state.py`**: Defines the game state model using Pydantic, including:
  - Card representation with color and number
  - Player hands, firework piles, and discard pile
  - Game resources (clue tokens, fuse tokens)
  - Score tracking and game state validation

- **`engine.py`**: Implements the game engine that:
  - Initializes the game (deck creation, card dealing)
  - Manages turn progression
  - Executes player actions (play card, give clue, discard)
  - Enforces game rules and validates moves
  - Tracks game completion and scoring

### 3.2 Agent System (`src/agents/`)

- **`base.py`**: Defines the abstract Agent interface with:
  - Discussion participation
  - Action decision-making
  - Memory management

- **`ai_agent.py`**: Implements AI agents using Langgraph with:
  - State management for reasoning
  - Integration with LLM models
  - Tool-based action execution

- **Reasoning Components**:
  - `reasoning/graph.py`: Sets up the agent reasoning workflow
  - `reasoning/nodes.py`: Implements reasoning steps (analyze state, generate thoughts, propose actions)
  - `reasoning/router.py`: Manages decision flow between reasoning steps

- **Action Components**:
  - `actions/extractor.py`: Extracts concrete actions from agent reasoning
  - `tools/`: Implements tools for playing cards, giving clues, and discarding

- **Prompting System**:
  - `prompts/state_analysis.py`: Prompts for analyzing game state
  - `prompts/thought_generation.py`: Prompts for generating strategic thoughts
  - `prompts/action_proposal.py`: Prompts for proposing concrete actions

### 3.3 Communication System (`src/communication/`)

- **`discussion.py`**: Manages the pre-action discussion phase:
  - Tracks discussion entries and history
  - Facilitates agent contributions
  - Summarizes discussions for decision-making
  - Manages discussion rounds and consensus detection

## 4. Key Implementation Features

### 4.1 Pre-Action Discussion Phase

One of the most innovative aspects of this implementation is the pre-action discussion phase, which allows agents to collaborate before taking actions. This mimics the way human players might discuss strategy in a cooperative game.

The discussion phase:
1. Allows all agents to contribute thoughts about the current game state
2. Enables the active player to receive feedback from other players
3. Creates a consensus-building mechanism for strategic decisions
4. Maintains a history of discussions for context in future turns

### 4.2 Information Management

The implementation carefully manages information visibility to maintain the core Hanabi mechanic of hidden information:

- Each agent receives a filtered view of the game state where their own cards are hidden
- Clue information is tracked and made available to agents
- The system distinguishes between known information (from clues) and inferences

### 4.3 Agent Reasoning Process

The AI agents follow a structured reasoning process:

1. **State Analysis**: Agents analyze the visible game state
2. **Thought Generation**: Agents generate strategic thoughts based on the analysis
3. **Action Proposal**: Agents propose concrete actions based on their thoughts
4. **Tool Execution**: The system executes the chosen action using appropriate tools

### 4.4 Prompt Engineering

The system uses carefully crafted prompts to guide agent reasoning:

- Prompts emphasize the distinction between known information and inferences
- Agents are instructed to use specific language patterns for certainty vs. uncertainty
- Prompts include comprehensive game state information and valid action constraints
- Discussion contributions are formatted to maintain information boundaries

## 5. Comparison with Human Hanabi Play

### 5.1 Similarities to Human Play

The implementation captures several key aspects of human Hanabi play:

- **Information Management**: The system correctly implements the hidden information mechanic
- **Clue Economy**: Agents must manage limited clue tokens, similar to human players
- **Collaborative Decision-Making**: The discussion phase mimics human table talk
- **Strategic Reasoning**: Agents analyze game state and make inferences, similar to human reasoning

### 5.2 Differences from Human Play

There are some notable differences from typical human play:

- **Explicit Discussion**: Human Hanabi typically prohibits explicit discussion of strategy during play, while this implementation includes a formalized discussion phase
- **Perfect Memory**: AI agents have perfect memory of all past events, unlike humans who may forget details
- **Inference Formalization**: The system formalizes the inference process, while human players often use implicit conventions
- **No Conventions**: Many human players develop conventions (e.g., "finesse" plays), which aren't explicitly built into this system

### 5.3 Advanced Hanabi Concepts

Some advanced Hanabi concepts that human players use are not explicitly implemented:

- **Conventions**: Standard play patterns like "finesse," "bluff," or "save clues"
- **Negative Information**: Reasoning about what cards are not based on absence of clues
- **Priority Plays**: Understanding which plays are most urgent based on game state

## 6. Strengths and Limitations

### 6.1 Strengths

- **Comprehensive Game Logic**: The implementation fully captures Hanabi's rules and mechanics
- **Innovative Discussion Phase**: The pre-action discussion adds a unique collaborative element
- **Structured Agent Reasoning**: The reasoning graph provides a clear, step-by-step approach
- **Information Boundary Enforcement**: The system carefully manages what information each agent can access

### 6.2 Limitations

- **Computational Overhead**: The discussion phase and multi-step reasoning increase computational requirements
- **Limited Conventions**: The system doesn't explicitly implement advanced Hanabi conventions
- **Scalability Concerns**: The discussion phase may become unwieldy with more complex reasoning
- **Deterministic Reasoning**: The structured reasoning approach may limit creative play strategies

## 7. Potential Improvements

### 7.1 Technical Improvements

- **Optimized Reasoning Paths**: Streamline the reasoning graph for efficiency
- **Enhanced Memory Management**: Implement more sophisticated memory structures for tracking game history
- **Improved Inference Engine**: Develop specialized inference mechanisms for Hanabi-specific reasoning
- **Performance Profiling**: Identify and optimize bottlenecks in the reasoning process

### 7.2 Gameplay Improvements

- **Convention Implementation**: Add support for standard Hanabi conventions
- **Adaptive Strategy**: Implement strategy adaptation based on game progress
- **Meta-Learning**: Enable agents to learn from past games and improve over time
- **Hybrid Approaches**: Combine rule-based systems with LLM reasoning for efficiency

## 8. Conclusion

The Hanabi AI Agents codebase represents a sophisticated implementation of AI agents playing the cooperative card game Hanabi. The system successfully captures the core mechanics of Hanabi while adding innovative features like the pre-action discussion phase.

The implementation demonstrates how Langgraph can be used to create structured reasoning processes for complex cooperative games. The careful management of information boundaries and the emphasis on distinguishing between known information and inferences show a deep understanding of Hanabi's unique challenges.

While there are differences between this implementation and human Hanabi play, particularly in the explicit discussion phase and the absence of formalized conventions, the system provides a strong foundation for exploring cooperative AI in partially observable environments.

Future work could focus on implementing more advanced Hanabi concepts, optimizing the reasoning process, and enabling agents to develop and use conventions similar to human players. 