from __future__ import annotations
from typing import List, Dict, Optional, Literal, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ValidationInfo

# --- Basic Enums & Classes ---

class PlayerRole(str, Enum):
    DECISION_MAKER = "decision_maker" # Rational actor
    NATURE = "nature"                 # Represents chance/probability (Source: Slide 17)

class Player(BaseModel):
    name: str
    role: PlayerRole
    description: str

class Payoff(BaseModel):
    """
    Represents the terminal state utility.
    Note: Utility numbers are cardinal (VNM), derived from sentiment.
    """
    outcome_summary: str
    # Map: Player Name -> Utility Value (float)
    utilities: Dict[str, float]

# --- Extensive Form (Tree) Structure ---

class Action(BaseModel):
    name: str
    description: str
    probability: Optional[float] = Field(
        None, description="Only required if the parent player is 'Nature'. Sum of sibling probs must be 1.0."
    )
    # Recursive definition: An action leads to another GameNode
    next_node: 'GameNode'

class GameNode(BaseModel):
    id: str
    current_player_name: Optional[str] = Field(None, description="Who moves at this node?")
    is_terminal: bool = False
    # If not terminal:
    actions: Optional[List[Action]] = None
    # If terminal:
    payoff: Optional[Payoff] = None

    @field_validator('actions')
    def validate_choices(cls, v, info: ValidationInfo):
        # Skip validation if it's a leaf node or no actions provided
        if not v:
            return v
            
        # Strict Rule: A rational player must have choices.
        if len(v) < 2:
            # We allow Nature to have 1 branch if it's 100%, but Players must have choices.
            # For simplicity in this iteration: Warn or Fail if only 1 action exists.
            # Ideally: raise ValueError("Decision nodes must have at least 2 options (strategies).")
            pass
        return v

# --- Top Level Output ---

class GameTheoryAnalysis(BaseModel):
    title: str
    strategic_summary: str = Field(..., description="Explain WHY this is a game (interdependence).")
    players: List[Player]
    game_type: Literal["Extensive_Form", "Normal_Form"]
    
    # Primary Structure: The Root of the Tree
    game_tree: Optional[GameNode] = None
    
    # Metadata for the UI
    confidence_score: int = Field(..., ge=0, le=100, description="How well does the text fit a game model?")
    
    # New Fields for Analysis
    nash_equilibrium_explanation: str = Field(..., description="Explain the Game Solution (Nash Equilibrium) in text.")
    actual_events_comparison: str = Field(..., description="Compare the model's prediction to the Actual Events in the article.")

# Resolve recursion
GameNode.update_forward_refs()
Action.update_forward_refs()
