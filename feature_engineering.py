from sklearn.feature_extraction import DictVectorizer

def extract_features_with_context(data, vectorizer=None, fit_vectorizer=False, ngram_range=(1, 2)):
    """Extract n-gram and context-based features."""
    X, y = [], []
    for sentence in data:
        words = [token['current_syllable'] for token in sentence['annotations']]
        labels = [1 if not token['is_correct'] else 0 for token in sentence['annotations']]
        for i, token in enumerate(sentence['annotations']):
            features = {}
            # Add current word features
            features['word'] = token['current_syllable']
            features['is_first'] = int(i == 0)
            features['is_last'] = int(i == len(words) - 1)
            features['length'] = len(token['current_syllable'])
            
            # Add context words (previous and next)
            if i > 0:
                features['prev_word'] = words[i - 1]
            if i < len(words) - 1:
                features['next_word'] = words[i + 1]
            
            X.append(features)
            y.append(labels[i])

    if fit_vectorizer:
        vectorizer = DictVectorizer(sparse=True)
        X = vectorizer.fit_transform(X)
    else:
        X = vectorizer.transform(X)
    return X, y, vectorizer
