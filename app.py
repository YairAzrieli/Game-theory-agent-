import streamlit as st
import graphviz
from typing import Dict
from backend import analyze_text_to_game, fetch_article
from schemas import GameTheoryAnalysis, GameNode

st.set_page_config(layout="wide", page_title="News -> Game Theory Agent")

st.title("🕵️ Game Theory Analyzer Agent")
st.markdown("Turns news narratives into formal Extensive Form Games.")

# --- Sidebar: Input ---
with st.sidebar:
    st.header("Configuration")
    # Check if API key is in environment, otherwise ask user
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.text_input("OpenAI API Key", type="password", help="API key not found in environment. Please enter it here.")
    
    st.divider()
    
    st.header("Input Source")
    input_type = st.radio("Choose Input:", ["URL", "Raw Text"])
    
    # Initialize session state for article text if not present
    if "article_text" not in st.session_state:
        st.session_state.article_text = ""

    if input_type == "URL":
        url = st.text_input("Enter Article URL")
        if st.button("Fetch & Analyze"):
            with st.spinner("Fetching article..."):
                fetched_text = fetch_article(url)
                if fetched_text:
                    st.session_state.article_text = fetched_text
                    st.success("Article fetched successfully!")
                else:
                    st.error("Failed to fetch article. Please check the URL or try raw text.")
        
        # Show content if available
        if st.session_state.article_text:
            st.expander("Show Content").write(st.session_state.article_text[:500] + "...")

    else:
        # For raw text, we bind directly or update state
        st.session_state.article_text = st.text_area("Paste Article Text Here", value=st.session_state.article_text, height=300)
    
    run_btn = st.button("Generate Game Model", type="primary")

# --- Helper: Visualizer ---
import textwrap

def wrap_text(text, width=20):
    """Helper to wrap long text into multiple lines for graph nodes."""
    if not text:
        return ""
    return "\n".join(textwrap.wrap(text, width=width))

def draw_game_tree(root_node: GameNode) -> graphviz.Digraph:
    """
    Traverses the Pydantic GameNode structure and builds a Graphviz object.
    """
    dot = graphviz.Digraph()
    dot.attr(rankdir='TB')  # CHANGED: Top-to-Bottom (Classic Game Theory style)
    dot.attr(splines='ortho')
    dot.attr(nodesep='0.5') # Add space between nodes
    dot.attr(ranksep='1.0') # Add space between levels
    
    def add_nodes_edges(node: GameNode, parent_id=None, edge_label=""):
        node_id = str(id(node))
        
        # Wrap the text so nodes don't get super wide
        # Note: GameNode schema doesn't strictly have 'description' on the node itself in the original schema,
        # but it might be useful to add or just use player name. 
        # Checking schema: GameNode has 'current_player_name'. 
        # Let's use current_player_name.
        
        wrapped_player = wrap_text(node.current_player_name or "Unknown", width=20)
        
        if node.is_terminal:
            # Payoff Node Styling
            outcome_text = ""
            payoff_text = ""
            if node.payoff:
                outcome_text = wrap_text(node.payoff.outcome_summary, width=30)
                payoff_text = "\n".join([f"{k}: {v}" for k,v in node.payoff.utilities.items()])
            
            label = f"Outcome:\n{outcome_text}\n\nPayoffs:\n{payoff_text}"
            dot.node(node_id, label, shape='box', style='filled', fillcolor='#f0f2f6', fontname="Arial", fontsize="10")
        else:
            # Decision Node Styling
            label = f"{wrapped_player}\n(Moves)"
            # Differentiate Nature nodes
            shape = 'diamond' if node.current_player_name and node.current_player_name.lower() == 'nature' else 'oval'
            color = 'lightgrey' if node.current_player_name and node.current_player_name.lower() == 'nature' else 'white'
            dot.node(node_id, label, shape=shape, style='filled', fillcolor=color, fontname="Arial", fontsize="11")
        
        if parent_id:
            # Wrap edge labels too (action names)
            wrapped_edge = wrap_text(edge_label, width=15)
            dot.edge(parent_id, node_id, label=wrapped_edge, fontsize="9")
        
        if node.actions:
            for action in node.actions:
                lbl = action.name
                if action.probability:
                    lbl += f"\n(p={action.probability})"
                add_nodes_edges(action.next_node, node_id, lbl)

    if root_node:
        add_nodes_edges(root_node)
    return dot

# --- Main Interface ---
if run_btn and st.session_state.article_text:
    with st.spinner("Agent is modeling the game..."):
        try:
            response, reasoning = analyze_text_to_game(st.session_state.article_text, api_key=api_key)
            
            if response:
                # Layout: 2 Columns
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Narrative Analysis")
                    st.write(f"**Title:** {response.title}")
                    st.write(f"**Strategic Summary:** {response.strategic_summary}")
                    st.write(f"**Confidence Score:** {response.confidence_score}/100")

                    st.markdown("---")
                    st.subheader("Game Solution")
                    st.info(f"**Nash Equilibrium:** {response.nash_equilibrium_explanation}")
                    st.warning(f"**Model vs. Reality:** {response.actual_events_comparison}")
                    
                    st.subheader("Players")
                    for p in response.players:
                        st.markdown(f"- **{p.name}** ({p.role.value}): {p.description}")

                with col2:
                    st.subheader("Game Tree Visualization")
                    if response.game_tree:
                        graph = draw_game_tree(response.game_tree)
                        st.graphviz_chart(graph, use_container_width=True)
                        
                        with st.expander("🔍 View Raw Text / Zoom Details"):
                            st.info("If the tree is too large, use browser zoom or right-click 'Open Image in New Tab'.")
                    else:
                        st.warning("No game tree generated.")
            else:
                st.warning("No strategic game detected in the text.")
                st.info(f"**Agent Reasoning:** {reasoning}")
                
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
elif run_btn and not st.session_state.article_text:
    st.error("Please provide input text first.")
