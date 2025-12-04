import os
import trafilatura
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, Tuple, Union
from schemas import GameTheoryAnalysis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the client with Instructor
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Don't fail yet, maybe the user will provide it in the UI (we'll handle this in app.py/backend.py)
        client = None
    else:
        client = instructor.from_openai(OpenAI(api_key=api_key))
except Exception as e:
    print(f"Warning: OpenAI client could not be initialized. Error: {e}")
    client = None

class ScreenerOutput(BaseModel):
    is_strategic_game: bool = Field(..., description="Does the text describe a situation with strategic interdependence between players?")
    reasoning: str = Field(..., description="Brief explanation of why this is or isn't a game.")

def fetch_article(url: str) -> Optional[str]:
    """
    Fetches and extracts main text from a URL using Trafilatura.
    """
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        return trafilatura.extract(downloaded)
    return None

def analyze_text_to_game(text: str, api_key: str = None) -> Tuple[Optional[GameTheoryAnalysis], str]:
    """
    Orchestrates the analysis pipeline:
    1. Screen the text for strategic interdependence.
    2. If positive, model it as a Game Theory problem.
    
    Returns:
        tuple: (GameTheoryAnalysis object or None, reasoning string)
    """
    global client
    
    # If client is not initialized, try to initialize it with the provided key
    current_client = client
    if not current_client and api_key:
        try:
            current_client = instructor.from_openai(OpenAI(api_key=api_key))
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client with provided key: {e}")
            
    if not current_client:
        raise RuntimeError("LLM Client not initialized. Please check your API keys.")

    # Step 1: Screener Agent
    screener_response = current_client.chat.completions.create(
        model="gpt-4o", # Or gpt-3.5-turbo, depending on budget/availability
        response_model=ScreenerOutput,
        messages=[
            {"role": "system", "content": "You are a Game Theory Scout. Your job is to identify if a given text contains 'Strategic Interdependence' - where the outcome for one actor depends on the choices of another. Look for conflicts, negotiations, elections, or competitive markets."},
            {"role": "user", "content": f"Analyze this text: {text[:4000]}"} # Truncate to avoid context limits if necessary
        ]
    )

    if not screener_response.is_strategic_game:
        return None, screener_response.reasoning

    # Step 2: Modeler Agent
    # We force the schema defined in schemas.py
    game_analysis = current_client.chat.completions.create(
        model="gpt-4o",
        response_model=GameTheoryAnalysis,
        messages=[
            {"role": "system", "content": """You are an expert Game Theory Modeler. Your goal is to map a news narrative into a mathematical "Extensive Form Game" tree.

### CRITICAL RULES:
1.  **Aggregation (Max 3 Players):** You MUST aggregate real-world entities into maximum 2-3 opposing sides + "Nature".
    * *Example:* Instead of "Hospitals", "Doctors", "HMOs" -> Group them as "Healthcare Providers" vs "Government".
    * *Reason:* Games with >3 players are unreadable.

2.  **The "Choice" Rule:** Every Decision Node (where a player moves) MUST have at least 2 distinct actions.
    * *Bad:* Player A -> "Agrees" -> Next Node. (This is a script, not a game).
    * *Good:* Player A -> Choose ["Agree", "Boycott"].
    * *Constraint:* If a player has no real choice, do not make it a node. Merge it into the previous outcome.

3.  **Nature's Role:** Use "Nature" ONLY for probabilistic events (elections results, court rulings, chance).
    * Nature nodes must have `probabilities` summing to 1.0.

4.  **Utilities:** Assign VNM Cardinal Utilities (-100 to 100) to the leaf nodes.
    * These must reflect the *conflict*. If Player A wins (+50), Player B usually loses (-50), unless it's a cooperative game.

5.  **Root Node:** You MUST assign a specific `current_player_name` to the root node. It cannot be "Unknown".

6.  **Analysis:**
    *   **Nash Equilibrium:** Provide a clear textual explanation of the game's solution (Nash Equilibrium). What is the stable outcome?
    *   **Reality Check:** Compare your model's prediction (the equilibrium) to what actually happened in the news article. Did the players act rationally?
"""},
            {"role": "user", "content": f"Model this text: {text}"}
        ]
    )
    
    return game_analysis, screener_response.reasoning

if __name__ == "__main__":
    # Simple test
    sample_text = "Two companies, A and B, are deciding whether to lower prices. If both lower, they lose profit. If only one lowers, they gain market share."
    try:
        result, reason = analyze_text_to_game(sample_text)
        if result:
            print(result.json(indent=2))
        else:
            print(f"No game detected. Reason: {reason}")
    except Exception as e:
        print(f"Error: {e}")
