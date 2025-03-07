# AI Agent Refactoring Analysis

## Current Structure

The current `ai_agent.py` file is approximately 1,750 lines long and contains multiple responsibilities:

1. Agent state management
2. Reasoning graph setup and execution
3. Tool definitions and implementations
4. Prompt creation and formatting
5. Action parsing and validation
6. Discussion and action decision logic

This monolithic structure makes the code difficult to maintain, test, and extend.

## Refactoring Goals

1. Improve code organization by separating concerns
2. Enhance maintainability by reducing file sizes
3. Increase testability by isolating components
4. Facilitate future extensions by creating modular components
5. Preserve functionality while restructuring

## Proposed File Structure

```
src/
└── agents/
    ├── __init__.py
    ├── base.py                    # Existing base Agent class
    ├── ai_agent.py                # Slimmed down main AIAgent class
    ├── state/
    │   ├── __init__.py
    │   └── agent_state.py         # AgentState and AgentStateDict classes
    ├── reasoning/
    │   ├── __init__.py
    │   ├── graph.py               # Reasoning graph setup and execution
    │   ├── nodes.py               # Node implementations (analyze, generate, propose)
    │   └── router.py              # Routing logic for the graph
    ├── tools/
    │   ├── __init__.py
    │   ├── play_card.py           # Play card tool implementation
    │   ├── give_clue.py           # Give clue tool implementation
    │   ├── discard.py             # Discard tool implementation
    │   └── error_handler.py       # Tool error handling
    ├── prompts/
    │   ├── __init__.py
    │   ├── state_analysis.py      # State analysis prompt creation
    │   ├── thought_generation.py  # Thought generation prompt creation
    │   └── action_proposal.py     # Action proposal prompt creation
    ├── formatters/
    │   ├── __init__.py
    │   ├── game_state.py          # Game state formatting functions
    │   ├── discussion.py          # Discussion formatting functions
    │   └── thoughts.py            # Thoughts formatting functions
    └── actions/
        ├── __init__.py
        ├── parser.py              # Action parsing logic
        └── validator.py           # Action validation logic
```

## Refactored AIAgent Class Structure

The refactored `AIAgent` class would be significantly simplified:

```python
class AIAgent(Agent):
    def __init__(self, agent_id: int, model_name: str = None):
        super().__init__(agent_id)
        
        # Initialize environment and API key
        self._initialize_environment()
        
        # Initialize the model
        self.model = self._initialize_model(model_name)
        
        # Initialize the reasoning graph
        self.reasoning_graph = setup_reasoning_graph(self)
        
        # Store current game state for tool access
        self.current_game_state = None
        
    def _initialize_environment(self):
        # Load environment variables and API key
        # ...
        
    def _initialize_model(self, model_name):
        # Create and configure the LLM model
        # ...
        
    def participate_in_discussion(self, game_state: GameState, discussion_history: list) -> str:
        # Store the game state for tool access
        self.current_game_state = game_state
        
        # Process discussion history
        discussion_strings, game_history_strings = process_discussion_history(
            discussion_history, self.memory)
        
        # Create initial state
        initial_state = create_initial_state(
            game_state, discussion_strings, game_history_strings)
        
        # Run the reasoning graph
        final_state = self.reasoning_graph.invoke(initial_state)
        
        # Generate contribution based on thoughts
        return generate_contribution(final_state, self.model, self.agent_id)
        
    def decide_action(self, game_state: GameState, discussion_history: list) -> Dict[str, Any]:
        # Store the game state for tool access
        self.current_game_state = game_state
        
        # Process discussion history
        discussion_strings, game_history_strings = process_discussion_history(
            discussion_history, self.memory)
        
        # Create initial state with messages for action phase
        initial_state = create_action_state(
            game_state, discussion_strings, game_history_strings)
        
        # Run the reasoning graph
        final_state = self.reasoning_graph.invoke(initial_state)
        
        # Extract action from final state
        return extract_action_from_state(final_state)
```

This refactored class:
1. Delegates complex functionality to imported functions
2. Maintains a clean, high-level interface
3. Focuses on orchestration rather than implementation details
4. Preserves the same public API as the original class

## New Helper Functions

To support the refactored AIAgent class, several new helper functions would need to be created:

### 1. `src/agents/utils/discussion.py`

```python
def process_discussion_history(discussion_history: list, memory: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Process discussion history into strings and extract game history.
    
    Args:
        discussion_history: Raw discussion history (may be objects or strings)
        memory: Agent's memory dictionary
        
    Returns:
        Tuple of (discussion_strings, game_history_strings)
    """
    # Check if discussion_history contains objects or strings
    has_discussion_objects = False
    has_game_history = False
    
    if discussion_history and hasattr(discussion_history[0], 'content'):
        has_discussion_objects = True
        
    # Check for game history (entries with turn_number attribute)
    if discussion_history and hasattr(discussion_history[0], 'turn_number'):
        has_game_history = True
        
    # Convert discussion history to strings
    discussion_strings = []
    game_history_strings = []
    
    # Process based on the type of history objects
    if has_game_history:
        # Extract current turn and previous discussions
        # ...
    elif has_discussion_objects:
        # Extract content from discussion objects
        # ...
    else:
        # Already strings
        discussion_strings = discussion_history
        
    # Get game history from memory if not in discussion
    if not has_game_history and "game_history" in memory:
        game_history_strings = memory["game_history"]
        
    return discussion_strings, game_history_strings
```

### 2. `src/agents/state/state_factory.py`

```python
def create_initial_state(
    game_state: GameState, 
    discussion_strings: List[str], 
    game_history_strings: List[str]
) -> Dict[str, Any]:
    """
    Create the initial state for the reasoning graph (discussion phase).
    
    Args:
        game_state: Current game state
        discussion_strings: Processed discussion history
        game_history_strings: Processed game history
        
    Returns:
        Initial state dictionary for the reasoning graph
    """
    return {
        "game_state": game_state,
        "discussion_history": discussion_strings,
        "game_history": game_history_strings,
        "current_thoughts": []
    }

def create_action_state(
    game_state: GameState, 
    discussion_strings: List[str], 
    game_history_strings: List[str]
) -> Dict[str, Any]:
    """
    Create the initial state for the reasoning graph (action phase).
    
    Args:
        game_state: Current game state
        discussion_strings: Processed discussion history
        game_history_strings: Processed game history
        
    Returns:
        Initial state dictionary for the reasoning graph with messages
    """
    # Create the basic state
    state = create_initial_state(game_state, discussion_strings, game_history_strings)
    
    # Add messages for action phase
    state["messages"] = [HumanMessage(content="It's your turn to take an action in the Hanabi game.")]
    
    return state
```

### 3. `src/agents/actions/extractor.py`

```python
def extract_action_from_state(final_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the action from the final state of the reasoning graph.
    
    Args:
        final_state: Final state from the reasoning graph
        
    Returns:
        Action dictionary
    """
    # Check if we have a tool result
    if "tool_result" in final_state and final_state["tool_result"]:
        return final_state["tool_result"]
        
    # If we don't have a tool result but have messages, check for tool calls
    if "messages" in final_state and final_state["messages"]:
        last_message = final_state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            # Extract action from tool call
            tool_call = last_message.tool_calls[0]
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})
            
            # Map tool names to action types
            if tool_name == "play_card":
                return {
                    "type": "play_card",
                    "card_index": tool_args.get("card_index", 0)
                }
            elif tool_name == "give_clue":
                return {
                    "type": "give_clue",
                    "target_id": tool_args.get("target_id", 0),
                    "clue": {
                        "type": tool_args.get("clue_type", "color"),
                        "value": tool_args.get("clue_value", "")
                    }
                }
            elif tool_name == "discard":
                return {
                    "type": "discard",
                    "card_index": tool_args.get("card_index", 0)
                }
    
    # If we still don't have an action, raise an error
    raise ValueError("Failed to extract any action from the final state")
```

### 4. `src/agents/discussion/contribution.py`

```python
def generate_contribution(
    final_state: Dict[str, Any], 
    model: Any, 
    agent_id: int
) -> str:
    """
    Generate a contribution to the discussion based on the agent's thoughts.
    
    Args:
        final_state: Final state from the reasoning graph
        model: LLM model to use for generation
        agent_id: ID of the agent
        
    Returns:
        Contribution string
    """
    # Extract the generated thoughts
    current_thoughts = final_state.get("current_thoughts", [])
    
    if not current_thoughts:
        return "I'm analyzing the game state and considering our options."
        
    # Create a prompt for generating a contribution
    from ..formatters.thoughts import format_thoughts
    
    prompt = f"""You are Agent {agent_id} in a game of Hanabi, participating in a discussion.

Based on your analysis of the game state, generate a concise contribution to the discussion.

CRITICAL INFORMATION RULES:
1. You MUST distinguish between KNOWN information and INFERENCES.
2. KNOWN information is ONLY what you've been explicitly told through clues.
3. INFERENCES are educated guesses based on game state, but are NOT certainties.
4. You MUST use language like "I believe", "I infer", "likely", "probably", "might be" for inferences.
5. You MUST use language like "I know" ONLY for information directly given through clues.
6. For example, if you received a "green" clue on a card, you can say "I know this card is green" but NOT "I know this is a green 1".
7. You CANNOT claim to know both color AND number of a card unless you've received BOTH clues for that card.
8. You CANNOT claim to know the exact identity of a card based solely on a single clue.

Your thoughts:
{format_thoughts(current_thoughts)}

Generate a concise, strategic contribution that follows the information rules above. Focus on what action you're considering and why, without revealing specific card information you shouldn't know.
"""

    # Generate the contribution
    response = model.invoke([HumanMessage(content=prompt)])
    return response.content.strip()
```

These helper functions encapsulate the logic that was previously embedded in the AIAgent class methods, making the code more modular and easier to test. Each function has a single responsibility and clear inputs/outputs.

## Component Breakdown

### 1. `src/agents/state/agent_state.py`

**Functions to move:**
- `AgentStateDict` (TypedDict)
- `AgentState` class

**Dependencies:**
- Game state types
- TypedDict from typing_extensions

### 2. `src/agents/reasoning/graph.py`

**Functions to move:**
- `_setup_reasoning_graph`

**Dependencies:**
- StateGraph from langgraph
- Tool nodes
- Agent state

### 3. `src/agents/reasoning/nodes.py`

**Functions to move:**
- `_analyze_game_state`
- `_generate_thoughts`
- `_propose_action`

**Dependencies:**
- Prompt creation functions
- LLM model

### 4. `src/agents/reasoning/router.py`

**Functions to move:**
- `should_execute_tools` (currently nested in `_setup_reasoning_graph`)

**Dependencies:**
- None (pure function)

### 5. `src/agents/tools/play_card.py`

**Functions to move:**
- `_play_card_tool`

**Dependencies:**
- Game state

### 6. `src/agents/tools/give_clue.py`

**Functions to move:**
- `_give_clue_tool`

**Dependencies:**
- Game state
- Color enum

### 7. `src/agents/tools/discard.py`

**Functions to move:**
- `_discard_tool`

**Dependencies:**
- Game state

### 8. `src/agents/tools/error_handler.py`

**Functions to move:**
- `_handle_tool_error`

**Dependencies:**
- ToolMessage from langchain

### 9. `src/agents/prompts/state_analysis.py`

**Functions to move:**
- `_create_state_analysis_prompt`

**Dependencies:**
- Formatting functions

### 10. `src/agents/prompts/thought_generation.py`

**Functions to move:**
- `_create_thought_generation_prompt`

**Dependencies:**
- Formatting functions

### 11. `src/agents/prompts/action_proposal.py`

**Functions to move:**
- `_create_action_proposal_prompt`
- `_generate_valid_actions_info`

**Dependencies:**
- Formatting functions
- Game state

### 12. `src/agents/formatters/game_state.py`

**Functions to move:**
- `_format_firework_piles`
- `_format_discard_pile`
- `_format_hand`

**Dependencies:**
- Game state types

### 13. `src/agents/formatters/discussion.py`

**Functions to move:**
- `_format_discussion`

**Dependencies:**
- None

### 14. `src/agents/formatters/thoughts.py`

**Functions to move:**
- `_format_thoughts`

**Dependencies:**
- None

### 15. `src/agents/actions/parser.py`

**Functions to move:**
- `_parse_action_response`
- `_infer_action_from_text`

**Dependencies:**
- Regular expressions
- JSON

### 16. `src/agents/actions/validator.py`

**Functions to move:**
- `_validate_action_format`
- `_validate_action_before_submission`

**Dependencies:**
- Game state

## Main AIAgent Class Refactoring

The main `AIAgent` class in `ai_agent.py` would be significantly slimmed down, primarily containing:

1. Initialization logic
2. Public API methods (`participate_in_discussion`, `decide_action`)
3. Import and composition of the refactored components

## Implementation Strategy

1. **Create the directory structure** first
2. **Extract each component** one at a time, ensuring proper imports
3. **Update the main AIAgent class** to use the new components
4. **Add proper exports** in `__init__.py` files
5. **Test thoroughly** after each component extraction

## Potential Challenges

1. **Circular dependencies**: Some components may have interdependencies that need careful management
2. **State sharing**: Components need access to shared state (like game_state)
3. **Method access**: Private methods becoming module functions may need interface changes
4. **Testing**: Ensuring each component works correctly in isolation and integration

## Benefits of Refactoring

1. **Improved readability**: Each file has a clear, single responsibility
2. **Better maintainability**: Changes to one component don't affect others
3. **Easier testing**: Components can be tested in isolation
4. **Simplified extension**: New features can be added by extending specific components
5. **Better collaboration**: Multiple developers can work on different components

## Next Steps After Refactoring

1. Add comprehensive unit tests for each component
2. Improve error handling and logging
3. Add type hints for better IDE support
4. Consider performance optimizations
5. Document the new architecture

This refactoring will significantly improve the codebase structure while maintaining all existing functionality. 