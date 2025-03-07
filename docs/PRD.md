# Product Requirements Document: Hanabi AI Agents Application with Pre-Action Discussion

## Overview
This application simulates a game of Hanabi played by five AI agents using Python and Langgraph. The game proceeds in a turn-based loop, where each turn is defined by one player (the active agent) performing an action. Before the action is finalized, all agents participate in a free-flowing discussion of the available public game information to explore options and reach a consensus on the best move. Once the discussion phase is complete, the active agent synthesizes the input and performs the agreed-upon action.

## Objectives
- **Implement a Turn-Based Game Loop:**  
  Each round is a turn in which one agent acts, ensuring clear structure and progression.
- **Facilitate Pre-Action Collaborative Discussion:**  
  Before any action is taken, all agents communicate openly about the game state, available options, and strategy.
- **Maintain Secure Information Flow:**  
  Ensure that each agent’s private information (its own cards) remains hidden while all public information (other agents’ cards, board state, tokens, and move history) is shared.
- **Leverage Langgraph for AI Reasoning:**  
  Use Langgraph to manage the agents’ chain-of-thought processes and inter-agent communication.

## Key Features

### Game State Management
- **Deck and Cards:**  
  - Maintain a deck with cards in five colors (red, yellow, green, blue, white) numbered 1–5.
  - Deal cards to each of the five agents while ensuring each agent’s own hand is private.
- **Board State:**  
  - Track firework piles for each color.
  - Maintain a discard pile.
- **Tokens:**  
  - Clue Tokens: Used to provide hints.
  - Fuse Tokens: Represent mistakes; the game ends when all are used.

### AI Agent Modeling and Communication
- **Filtered Views:**  
  - Provide each agent with a customized view that hides its own cards but displays all other public information.
- **Pre-Action Discussion Phase:**  
  - For every turn, initiate a discussion session where all agents review the current public game state.
  - Agents exchange reasoning, potential strategies, and suggestions for the best available move.
  - The discussion is iterative, allowing multiple rounds if necessary until a consensus or sufficient clarity is reached.
- **Decision Synthesis:**  
  - The active agent synthesizes the discussion outcomes and selects a final action (play a card, give a clue, or discard a card).
- **Internal Memory:**  
  - Each agent maintains an internal log of clues, previous moves, and discussion contributions to inform future decisions.

### Central Game Controller
- **State Management:**  
  - Maintain the complete game state including all agents’ private and public data.
  - Provide filtered views for each agent via a method like `get_view_for(agent_id)`.
- **Turn-Based Orchestration with Discussion:**  
  - Manage a turn-based loop where each turn starts with a discussion phase.
  - Ensure that the active agent’s action is derived from the collective input of all agents.
- **Action Validation and Execution:**  
  - Validate the active agent’s chosen action using a dedicated rules engine.
  - Update the game state, including tokens and board state, accordingly.
- **Logging:**  
  - Record all moves, agent reasoning, and discussion rounds for transparency and debugging.

### User Interface
- **Display Options:**  
  - Provide a textual or graphical interface showing the current game state, discussion logs, moves, and final score.
- **Final Reporting:**  
  - Present the final score (based on the firework piles) along with a summary of the discussion and decision-making process.

## Functional Requirements

1. **Game Initialization**
   - Create and shuffle a deck.
   - Deal cards to 5 agents.
   - Initialize the board state, tokens, and logging mechanisms.
   - Instantiate agents and distribute initial filtered views.

2. **Agent State and Views**
   - Maintain a complete internal game state while providing each agent with a custom view that hides its own cards.
   - Ensure all discussion among agents is based solely on public information plus each agent’s internal reasoning.

3. **Turn-Based Game Loop with Discussion Phase**
   - **Turn Initialization:**
     - Identify the active agent.
     - Distribute the current public game state to all agents.
   - **Pre-Action Discussion Phase:**
     - All agents discuss the state, potential moves, and strategies.
     - Support iterative rounds of discussion until consensus or a decision threshold is met.
     - Aggregate discussion outcomes for the active agent.
   - **Action Decision and Execution:**
     - The active agent synthesizes the group discussion and selects a final action.
     - The chosen action is validated by the central controller and applied to update the game state.
   - **Post-Action Updates:**
     - Refresh filtered views for all agents.
     - Deal new cards as necessary and update logs.

4. **Move Validation and Rules Enforcement**
   - Validate moves in accordance with Hanabi’s rules.
   - Update fuse tokens on invalid plays and recover clue tokens on discards.

5. **Game Termination and Scoring**
   - End the game when the deck is exhausted (with final turns) or when all fuse tokens have been used.
   - Compute and display the final score along with a summary of the decision-making process.

## Non-Functional Requirements

- **Performance:**  
  - Efficient handling of iterative discussion rounds and real-time state updates.
- **Scalability:**  
  - Modular architecture to support future enhancements, additional agents, or advanced discussion protocols.
- **Maintainability:**  
  - Well-organized, documented code with clear separation of concerns for game logic, AI reasoning, and communication.
- **Security:**  
  - Strictly enforce that agents cannot access their own private card information.
- **Robust Logging:**  
  - Comprehensive logging of game state transitions, discussion content, and agent decisions for debugging and analysis.

## System Architecture

### Components
- **Central Game Controller:**  
  - Manages the complete game state and orchestrates the turn-based game loop with a pre-action discussion phase.
- **AI Agent Nodes/Chains (Langgraph):**  
  - Each agent is implemented as a Langgraph node that receives its filtered view and participates in discussion rounds.
- **Communication Module:**  
  - Facilitates iterative discussion rounds among all agents before the active agent makes a move.
- **Rules Engine:**  
  - Validates moves to ensure compliance with Hanabi’s rules.
- **User Interface Module:**  
  - Displays the current game state, discussion logs, moves, and final results.
- **Logging Module:**  
  - Captures detailed logs of game state changes, inter-agent communication, and decisions.

### Data Flow and Interaction
1. **Initialization Phase:**
   - Build the full game state (deck, board, tokens) and instantiate all agents with their filtered views.
2. **Turn-Based Game Loop:**
   - **Step 1:** Determine the active agent.
   - **Step 2:** Initiate the Pre-Action Discussion Phase:
     - Broadcast the public game state to all agents.
     - Facilitate multiple rounds of free discussion where agents exchange insights and suggestions.
   - **Step 3:** Decision Synthesis:
     - The active agent synthesizes the discussion output to select a final action.
   - **Step 4:** Action Execution:
     - Validate and execute the active agent’s chosen action via the central controller.
   - **Step 5:** Update:
     - Refresh all agents’ views and logs; deal new cards if applicable.
3. **Termination Phase:**
   - Check for game end conditions (deck exhaustion or fuse token depletion).
   - Compute and display the final score with a summary of the decision-making process.

## Tools and Technologies
- **Python:**  
  - Primary programming language.
- **Langgraph:**  
  - Manages the AI chain-of-thought processes and agent communication.
- **Logging Framework:**  
  - Records detailed logs for game state transitions and discussions.
- **Testing Framework:**  
  - Supports unit and integration testing to ensure robust game logic and communication protocols.

## Assumptions & Constraints
- AI agents are cooperative and follow Hanabi rules.
- Each agent’s private view is strictly controlled by the central game controller.
- The discussion phase is designed to be iterative but time-bound to avoid excessive delays.
- The modular design will support future enhancements or adjustments to the discussion protocol.

## Future Enhancements
- **Advanced Discussion Protocols:**  
  - Integrate more sophisticated negotiation or consensus-building mechanisms.
- **Enhanced Decision-Making:**  
  - Employ machine learning techniques to refine agent reasoning and collaborative decision-making.
- **Graphical User Interface (GUI):**  
  - Develop a richer visual representation of both the game state and the discussion process.
- **Support for Game Variants:**  
  - Expand the application to accommodate variants of Hanabi or similar cooperative games.
