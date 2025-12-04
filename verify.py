import os
from backend import analyze_text_to_game
from schemas import GameTheoryAnalysis

def test_backend():
    print("Testing Backend Logic...")
    
    if "OPENAI_API_KEY" not in os.environ:
        print("WARNING: OPENAI_API_KEY not found in environment. Skipping live LLM test.")
        print("To run the full test, set the environment variable and run this script again.")
        return

    sample_text = """
    Two countries, A and B, are in an arms race. 
    If both build nukes, they are safe but poor (-10, -10). 
    If one builds and the other doesn't, the builder dominates (+20, -50). 
    If neither builds, they are rich and safe (+10, +10).
    """
    
    print(f"Input Text: {sample_text.strip()}")
    
    try:
        result, reason = analyze_text_to_game(sample_text)
        if result:
            print("\nSUCCESS: Game Detected!")
            print(f"Title: {result.title}")
            print(f"Summary: {result.strategic_summary}")
            print(f"Players: {[p.name for p in result.players]}")
            print("Tree Root Node ID:", result.game_tree.id if result.game_tree else "None")
        else:
            print(f"\nRESULT: No Game Detected. Reason: {reason}")
            
    except Exception as e:
        print(f"\nERROR during analysis: {e}")

if __name__ == "__main__":
    test_backend()
