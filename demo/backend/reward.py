def score_response(prompt: str, response: str) -> float:
    """Placeholder reward: returns a simple heuristic score.

    In a real setup, you'd run a reward model here and return a float.
    """
    # very naive scoring: longer, polite responses score slightly higher
    score = len(response) / 100.0
    if "please" in response.lower() or "thank" in response.lower():
        score += 0.1
    return min(max(score, 0.0), 1.0)
