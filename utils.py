import json

def load_jsonl(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def generate_candidates(sentence):
    """
    Generate realistic candidates for misspelling detection.
    Args:
        sentence: A sentence object with token annotations.
    Returns:
        Dictionary of candidates for each token.
    """
    candidates = {}
    for token in sentence['annotations']:
        current = token['current_syllable']
        if not token['is_correct']:
            # Use the provided alternative syllables if available
            candidates[current] = token['alternative_syllables']
        else:
            # Add some plausible misspellings for correct tokens (optional)
            candidates[current] = []
    return candidates
