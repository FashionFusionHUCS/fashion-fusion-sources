from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

def compute_metrics(all_labels, all_preds, threshold=0.5):
    """
    Compute accuracy, precision, recall, and F1 score for binary classification.
    
    Parameters:
    - all_labels: numpy array of true labels
    - all_preds: numpy array of predicted probabilities
    - threshold: threshold to convert predicted probabilities to binary predictions
    
    Returns:
    - metrics: dictionary containing AUC, accuracy, precision, recall, and F1 score
    """
    # Convert probabilities to binary predictions
    binary_preds = (all_preds >= threshold).astype(int)
    
    # Calculate metrics
    auc = roc_auc_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, binary_preds)
    precision = precision_score(all_labels, binary_preds)
    recall = recall_score(all_labels, binary_preds)
    f1 = f1_score(all_labels, binary_preds)
    
    # Store metrics in a dictionary
    metrics = {
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics