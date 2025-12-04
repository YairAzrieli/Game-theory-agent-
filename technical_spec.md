# Technical Specification: News-to-GameTheory Agent

## 1. Project Overview & Architecture
**Goal:** Build a "Router Agent" system that ingests narrative text (news articles), determines if a strategic game exists, models it mathematically (Extensive/Normal form), and visualizes the result interactively.

**Tech Stack:**
- **Backend Logic:** Python 3.10+
- **LLM Integration:** Instructor (patched with OpenAI/Anthropic) for strictly structured JSON outputs.
- **Data Validation:** Pydantic (Crucial for enforcing Game Theory rules).
- **Frontend:** Streamlit (Rapid prototyping for UI + Visualization).
- **Visualization:** Graphviz / NetworkX (For rendering Game Trees).
- **Scraping:** Trafilatura or Newspaper3k (For clean text extraction from URLs).

## 2. The Data Contract (Pydantic Schemas)
This is the core of the system. The LLM must adhere to these structures to ensure valid Game Theory logic (VNM Utility, Nature nodes, etc.).

```python
from __future__ import annotations
from typing import List, Dict, Optional, Literal, Union
from enum import Enum
from pydantic import BaseModel, Field

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

# Resolve recursion
GameNode.update_forward_refs()
Action.update_forward_refs()
```

## 3. The Logic Pipeline (Backend)
The processing flow should follow these steps:

1. **Ingest:** Fetch URL content using trafilatura. If fails or empty, use raw text input.
2. **Screener Agent (LLM):**
    - *Input:* Raw text.
    - *Task:* Analyze if the text contains Strategic Interdependence (Player A's payoff depends on Player B's action).
    - *Output:* Boolean (Go/No-Go).
3. **Modeler Agent (LLM with Instructor):**
    - *Input:* Text.
    - *Task:* Map text to `GameTheoryAnalysis` schema.
    - *Crucial Logic for LLM:*
        - **Nature:** If outcomes are uncertain (e.g., "elections might fail"), create a player named "Nature" with estimated probabilities.
        - **Utilities:** Convert qualitative outcomes ("disaster", "victory") into quantitative VNM utilities (e.g., -100, +50). Linear transformations are allowed.
        - **Tree Structure:** Define the sequence of moves strictly.

## 4. The Frontend (Streamlit Skeleton)
Use this code as the starting point for `app.py`.

```python
import streamlit as st
import graphviz
from typing import Dict

# Mock import - replace with actual backend logic
# from backend import analyze_text_to_game

st.set_page_config(layout="wide", page_title="News -> Game Theory Agent")

st.title("🕵️ Game Theory Analyzer Agent")
st.markdown("Turns news narratives into formal Extensive Form Games.")

# --- Sidebar: Input ---
with st.sidebar:
    st.header("Input Source")
    input_type = st.radio("Choose Input:", ["URL", "Raw Text"])
    
    user_content = ""
    if input_type == "URL":
        url = st.text_input("Enter Article URL")
        if st.button("Fetch & Analyze"):
            # TODO: Implement Scraper here
            pass
    else:
        user_content = st.text_area("Paste Article Text Here", height=300)
    
    run_btn = st.button("Generate Game Model", type="primary")

# --- Helper: Visualizer ---
def draw_game_tree(root_node) -> graphviz.Digraph:
    """
    Traverses the Pydantic GameNode structure and builds a Graphviz object.
    """
    dot = graphviz.Digraph()
    dot.attr(rankdir='TB')
    
    def add_nodes_edges(node, parent_id=None, edge_label=""):
        node_id = str(id(node))
        
        # Visual styling based on node type
        if node.is_terminal:
            # Show Payoffs for leaves
            label = f"Outcome:\n{node.payoff.outcome_summary}\n{node.payoff.utilities}"
            dot.node(node_id, label, shape='box', style='filled', color='lightgrey')
        else:
            # Show Player for decision nodes
            label = f"{node.current_player_name}\n(Moves)"
            shape = 'oval' if node.current_player_name != 'Nature' else 'diamond'
            dot.node(node_id, label, shape=shape)
        
        # Connect to parent
        if parent_id:
            dot.edge(parent_id, node_id, label=edge_label)
        
        # Recurse
        if node.actions:
            for action in node.actions:
                label = action.name
                if action.probability:
                    label += f"\n(p={action.probability})"
                add_nodes_edges(action.next_node, node_id, label)

    add_nodes_edges(root_node)
    return dot

# --- Main Interface ---
if run_btn and user_content:
    with st.spinner("Agent is modeling the game..."):
        # 1. Call LLM (Mocked response for dev)
        # response = analyze_text_to_game(user_content) 
        pass 
        
    # Layout: 2 Columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Narrative Analysis")
        # st.write(response.strategic_summary)
        # st.dataframe([p.dict() for p in response.players])
        st.info("Here we explain the players and their incentives.")

    with col2:
        st.subheader("Game Tree Visualization")
        # graph = draw_game_tree(response.game_tree)
        # st.graphviz_chart(graph)
        st.empty() # Placeholder
```

## 5. Development Guidelines & Pitfalls to Avoid
- **Nature Identification:** The LLM often forgets to assign "Nature" when there is luck involved. Force this in the system prompt: "If the outcome depends on chance/external factors not controlled by players, the current player MUST be 'Nature'".
- **Graphviz Dependencies:** Ensure graphviz is installed on the OS level (e.g., `brew install graphviz` or `apt-get install graphviz`), not just the Python library.
- **Recursion Depth:** News articles can be complex. Limit the tree depth (e.g., max 5 levels) in the Prompt to prevent the agent from hallucinating infinite loops.
- **Information Sets:** For this MVP, assume Perfect Information (Players know the history of the game) to simplify the JSON structure. Handling Imperfect Information (dotted lines in trees) requires a much more complex ID-referencing schema.
