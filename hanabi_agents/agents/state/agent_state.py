from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
from ...game.state import GameState
from pydantic import BaseModel, Field
import datetime
import uuid


class ActionError(BaseModel):
    """Model for action errors"""
    action: Dict[str, Any] = Field(default_factory=dict)
    error: str = "Unknown error"
    error_reason: str = "unknown_error"
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now().isoformat())
    turn: int = 0


class ActionResult(BaseModel):
    """Model for action results"""
    action: Dict[str, Any] = Field(default_factory=dict)
    result: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now().isoformat())
    turn: int = 0
    unique_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8])
    is_executed: bool = False

    def get_turn(self) -> int:
        """Get the turn number for this action result"""
        return self.turn


class AgentStateDict(TypedDict, total=False):
    """
    State schema for the agent reasoning graph.

    This TypedDict defines the structure of the state that flows through the LangGraph.
    It includes all the information needed for the agent to reason and make decisions.
    """
    # Core game information
    game_state: GameState
    agent_id: int

    # Discussion and history
    discussion_history: List[Dict[str, Any]]
    game_history: List[Dict[str, Any]]

    # Reasoning components
    current_thoughts: List[str]
    card_knowledge: List[Dict[str, Any]]

    # Action components
    proposed_action: Optional[Dict[str, Any]]
    action_result: Optional[Dict[str, Any]]

    # Tool execution
    messages: List[Any]
    proposed_tool_calls: List[Dict[str, Any]]

    # Error handling
    errors: List[Dict[str, Any]]

    # Execution tracking
    execution_path: List[str]

    # Phase tracking
    is_action_phase: bool


class AgentMemory(BaseModel):
    """Enhanced memory management for the agent"""
    # Game state analysis
    game_analysis: Optional[Dict[str, Any]] = None

    # Agent thoughts
    thoughts: List[str] = Field(default_factory=list)

    # Action history
    action_results: List[ActionResult] = Field(default_factory=list)
    action_errors: List[ActionError] = Field(default_factory=list)

    # Discussion history
    discussion_contributions: List[Dict[str, Any]] = Field(
        default_factory=list)

    # Custom memory fields
    custom_memory: Dict[str, Any] = Field(default_factory=dict)

    def get_memory(self, key: str, default: Any = None) -> Any:
        """Get a value from custom memory"""
        return self.custom_memory.get(key, default)

    def store_memory(self, key: str, value: Any) -> None:
        """Store a value in custom memory"""
        # Handle list values specially
        if key in self.custom_memory and isinstance(self.custom_memory[key], list):
            if isinstance(value, list):
                self.custom_memory[key].extend(value)
            else:
                self.custom_memory[key].append(value)
        else:
            self.custom_memory[key] = value

    def add_action_error(self, action: Dict[str, Any], error: str, error_reason: str = "unknown_error", turn: int = 0) -> None:
        """Add an action error to memory"""
        error_record = ActionError(
            action=action,
            error=error,
            error_reason=error_reason,
            timestamp=datetime.datetime.now().isoformat(),
            turn=turn
        )
        self.action_errors.append(error_record)

    def add_action_result(self, action: Dict[str, Any], result: Dict[str, Any], turn: int = 0) -> None:
        """
        Add an action result to memory

        Args:
            action: The action that was taken
            result: The result of the action
            turn: The turn number when the action was taken
        """
        result_record = ActionResult(
            action=action,
            result=result,
            timestamp=datetime.datetime.now().isoformat(),
            turn=turn
        )
        self.action_results.append(result_record)

    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "game_analysis": self.game_analysis,
            "thoughts": self.thoughts,
            "action_results": [result.dict() for result in self.action_results],
            "action_errors": [error.dict() for error in self.action_errors],
            "discussion_contributions": self.discussion_contributions,
            "custom_memory": self.custom_memory
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMemory':
        """Create an AgentMemory instance from a dictionary"""
        # Convert action results and errors back to their proper types
        action_results = [ActionResult(**result)
                          for result in data.get("action_results", [])]
        action_errors = [ActionError(**error)
                         for error in data.get("action_errors", [])]

        return cls(
            game_analysis=data.get("game_analysis"),
            thoughts=data.get("thoughts", []),
            action_results=action_results,
            action_errors=action_errors,
            discussion_contributions=data.get("discussion_contributions", []),
            custom_memory=data.get("custom_memory", {})
        )


class AgentState:
    """
    State class for the agent reasoning graph.

    This class provides a more structured interface for working with the agent state,
    with methods for converting to and from dictionaries for LangGraph compatibility.
    """

    def __init__(
        self,
        game_state: GameState,
        agent_id: int,
        discussion_history: List[Dict[str, Any]] = None,
        game_history: List[Dict[str, Any]] = None
    ):
        """
        Initialize a new agent state.

        Args:
            game_state: Current state of the game
            agent_id: ID of the agent
            discussion_history: History of discussion contributions
            game_history: History of game actions
        """
        self.game_state = game_state
        self.agent_id = agent_id
        self.discussion_history = discussion_history or []
        self.game_history = game_history or []
        self.current_thoughts = []
        self.card_knowledge = []
        self.proposed_action = None
        self.action_result = None
        self.messages = []
        self.proposed_tool_calls = []
        self.errors = []
        self.execution_path = []
        self.is_action_phase = False

    def dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for LangGraph compatibility.

        Returns:
            Dictionary representation of the state
        """
        return {
            "game_state": self.game_state,
            "agent_id": self.agent_id,
            "discussion_history": self.discussion_history,
            "game_history": self.game_history,
            "current_thoughts": self.current_thoughts,
            "card_knowledge": self.card_knowledge,
            "proposed_action": self.proposed_action,
            "action_result": self.action_result,
            "messages": self.messages,
            "proposed_tool_calls": self.proposed_tool_calls,
            "errors": self.errors,
            "execution_path": self.execution_path,
            "is_action_phase": self.is_action_phase
        }

    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> "AgentState":
        """
        Create AgentState from dictionary.

        Args:
            state_dict: Dictionary representation of the state

        Returns:
            New AgentState instance
        """
        agent_state = cls(
            game_state=state_dict["game_state"],
            agent_id=state_dict["agent_id"],
            discussion_history=state_dict.get("discussion_history", []),
            game_history=state_dict.get("game_history", [])
        )
        agent_state.current_thoughts = state_dict.get("current_thoughts", [])
        agent_state.card_knowledge = state_dict.get("card_knowledge", [])
        agent_state.proposed_action = state_dict.get("proposed_action")
        agent_state.action_result = state_dict.get("action_result")
        agent_state.messages = state_dict.get("messages", [])
        agent_state.proposed_tool_calls = state_dict.get(
            "proposed_tool_calls", [])
        agent_state.errors = state_dict.get("errors", [])
        agent_state.execution_path = state_dict.get("execution_path", [])
        agent_state.is_action_phase = state_dict.get("is_action_phase", False)
        return agent_state
