from preprocess import split_data
from feature_engineering import extract_features_with_context
from train_baselines import train_model, save_model
from evaluate import print_metrics, evaluate_model_with_threshold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from utils import load_jsonl
from vsec import VSECModel, VSECDataset



def align_and_flatten(predictions, ground_truths, tokenizer):
    """
    Align predictions and ground truth tokens, skipping sentences with mismatched token lengths.
    """
    y_pred_flat = []
    y_true_flat = []
    for pred, truth in zip(predictions, ground_truths):
        pred_tokens = tokenizer.tokenize(pred)
        truth_tokens = tokenizer.tokenize(truth)
        if len(pred_tokens) == len(truth_tokens):
            y_pred_flat.extend(pred_tokens)
            y_true_flat.extend(truth_tokens)
        else:
            print("Warning: Mismatch in token lengths. Skipping this sentence.")
            print(f"Predicted tokens: {pred_tokens}")
            print(f"Ground truth tokens: {truth_tokens}")
    return y_pred_flat, y_true_flat


def evaluate_token_metrics(y_true, y_pred):
    """
    Evaluate precision, recall, F1, and accuracy for token-level predictions.
    """
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Class-specific metrics (e.g., error class)
    print("Class-Specific Metrics:")
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        print(f"Class {i}: Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")
    
    return precision, recall, f1, accuracy, balanced_acc


def main():
    # Paths
    data_dir = "./data/"
    models_dir = "./models/"
    input_file = f"{data_dir}/VSEC.jsonl"

    # Step 1: Load and Split Data
    print("Loading and splitting data...")
    data = load_jsonl(input_file)
    train_data, dev_data, test_data = split_data(data)

    # Step 2: Feature Engineering
    print("Extracting features with context...")
    X_train, y_train, vectorizer = extract_features_with_context(train_data, fit_vectorizer=True)
    X_dev, y_dev, _ = extract_features_with_context(dev_data, vectorizer=vectorizer)
    X_test, y_test, _ = extract_features_with_context(test_data, vectorizer=vectorizer)

    # Step 3: Train and Evaluate Traditional Models
    for model_name in ["logistic", "random_forest", "naive_bayes"]:
        print(f"Training {model_name}...")
        model = train_model(X_train, y_train, model_type=model_name, class_weight="balanced")
        save_model(model, model_name, models_dir)

        # Dev set evaluation
        precision, recall, f1, accuracy, balanced_acc = evaluate_model_with_threshold(
            model, X_dev, y_dev, threshold=0.3
        )
        print_metrics(f"{model_name} (Dev)", precision, recall, f1, accuracy, balanced_acc)

        # Test set evaluation
        precision, recall, f1, accuracy, balanced_acc = evaluate_model_with_threshold(
            model, X_test, y_test, threshold=0.3
        )
        print_metrics(f"{model_name} (Test)", precision, recall, f1, accuracy, balanced_acc)

    # Step 4: Train and Evaluate VSEC Model
    print("Training VSEC model...")
    train_sentences = [" ".join([token['current_syllable'] for token in sentence['annotations']]) for sentence in train_data]
    train_labels = [" ".join([token['alternative_syllables'][0] if not token['is_correct'] and token['alternative_syllables'] else token['current_syllable'] for token in sentence['annotations']]) for sentence in train_data]
    dev_sentences = [" ".join([token['current_syllable'] for token in sentence['annotations']]) for sentence in dev_data]
    dev_labels = [" ".join([token['alternative_syllables'][0] if not token['is_correct'] and token['alternative_syllables'] else token['current_syllable'] for token in sentence['annotations']]) for sentence in dev_data]
    test_sentences = [" ".join([token['current_syllable'] for token in sentence['annotations']]) for sentence in test_data]
    test_labels = [" ".join([token['alternative_syllables'][0] if not token['is_correct'] and token['alternative_syllables'] else token['current_syllable'] for token in sentence['annotations']]) for sentence in test_data]

    vsec_model = VSECModel()
    train_dataset = VSECDataset(train_sentences, train_labels, vsec_model.tokenizer)
    dev_dataset = VSECDataset(dev_sentences, dev_labels, vsec_model.tokenizer)
    test_dataset = VSECDataset(test_sentences, test_labels, vsec_model.tokenizer)

    vsec_model.train(train_dataset, dev_dataset, epochs=3)

    print("Evaluating VSEC model...")
    predictions, ground_truth = vsec_model.evaluate(test_dataset)

    # Align and flatten tokens
    y_pred_flat, y_true_flat = align_and_flatten(predictions, ground_truth, vsec_model.tokenizer)

    # Evaluate token-level metrics
    precision, recall, f1, accuracy, balanced_acc = evaluate_token_metrics(y_true_flat, y_pred_flat)
    print_metrics("VSEC (Test)", precision.mean(), recall.mean(), f1.mean(), accuracy, balanced_acc)


if __name__ == "__main__":
    main()

