from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score

def evaluate_model(model, X, y_true, candidates=None, threshold=0.5):
    """
    Evaluate model performance using precision, recall, F1, and balanced accuracy.
    Args:
        model: The trained model (traditional ML or N-Gram model).
        X: Feature matrix or list of tokens for N-Gram models.
        y_true: True labels (0 for correct, 1 for misspelled).
        candidates: Dictionary of candidate replacements for N-Gram models (optional).
        threshold: Probability threshold for classification (for Naive Bayes or N-Gram).
    Returns:
        precision, recall, f1, accuracy, balanced accuracy: Computed metrics.
    """
    predictions = model.predict(X)

    # Compute metrics
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)
    accuracy = accuracy_score(y_true, predictions)
    balanced_acc = balanced_accuracy_score(y_true, predictions)
    return precision, recall, f1, accuracy, balanced_acc

def evaluate_model_with_threshold(model, X, y_true, threshold=0.5):
    """
    Evaluate model performance with threshold adjustment.
    Args:
        model: Trained traditional model (e.g., Naive Bayes, Random Forest).
        X: Feature matrix.
        y_true: True labels.
        threshold: Probability threshold for classification.
    Returns:
        precision, recall, f1, accuracy, balanced accuracy: Computed metrics.
    """
    proba = model.predict_proba(X)
    predictions = (proba[:, 1] >= threshold).astype(int)

    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)
    accuracy = accuracy_score(y_true, predictions)
    balanced_acc = balanced_accuracy_score(y_true, predictions)
    return precision, recall, f1, accuracy, balanced_acc

def print_metrics(name, precision, recall, f1, accuracy, balanced_acc):
    """
    Print evaluation metrics.
    Args:
        name: Name of the model or dataset (e.g., "logistic (Dev)").
        precision: Precision score.
        recall: Recall score.
        f1: F1 score.
        accuracy: Accuracy score.
        balanced_acc: Balanced accuracy score.
    """
    print(f"Metrics for {name}:")
    print(f"  Precision:          {precision:.4f}")
    print(f"  Recall:             {recall:.4f}")
    print(f"  F1 Score:           {f1:.4f}")
    print(f"  Accuracy:           {accuracy:.4f}")
    print(f"  Balanced Accuracy:  {balanced_acc:.4f}")
