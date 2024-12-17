from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import joblib

def train_model(X_train, y_train, model_type="logistic", class_weight=None):
    """
    Train a classification model (Logistic Regression, Random Forest, Naive Bayes).
    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        model_type: Type of model to train ("logistic", "random_forest", "naive_bayes").
        class_weight: Class weights for imbalanced datasets (only applicable to some models).
    Returns:
        Trained model object.
    """
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weight)
    elif model_type == "random_forest":
        model = RandomForestClassifier(random_state=42, class_weight=class_weight)
    elif model_type == "naive_bayes":
        model = MultinomialNB()
    else:
        raise ValueError("Invalid model type!")

    model.fit(X_train, y_train)
    return model


def save_model(model, model_name, output_dir="./models/"):
    """
    Save the trained model to a file.
    Args:
        model: Trained model object.
        model_name: Name of the model (e.g., "logistic").
        output_dir: Directory where the model will be saved.
    """
    joblib.dump(model, f"{output_dir}/{model_name}.pkl")

