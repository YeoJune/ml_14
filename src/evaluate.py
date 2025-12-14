"""
Evaluation and analysis utilities
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support
)
import os


ACTION_NAMES = ['fold', 'check_call', 'raise_small', 'raise_medium', 'raise_large', 'all_in']


def evaluate_model(predictions, labels, class_names=ACTION_NAMES, verbose=True):
    """
    Comprehensive model evaluation
    
    Args:
        predictions: Array of predicted labels
        labels: Array of true labels
        class_names: List of class names
        verbose: Whether to print detailed report
        
    Returns:
        metrics: Dict containing evaluation metrics
    """
    # Overall accuracy
    accuracy = np.mean(predictions == labels) * 100
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(len(class_names))), zero_division=0
    )
    
    # Macro and weighted F1
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    weighted_f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    }
    
    if verbose:
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"\nOverall Accuracy: {accuracy:.2f}%")
        print(f"Macro F1-Score: {macro_f1:.4f}")
        print(f"Weighted F1-Score: {weighted_f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(
            labels, predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0
        ))
    
    return metrics


def plot_confusion_matrix(predictions, labels, class_names=ACTION_NAMES, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        predictions: Array of predicted labels
        labels: Array of true labels
        class_names: List of class names
        save_path: Optional path to save figure
    """
    cm = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy)
    
    Args:
        history: Dict with keys 'train_loss', 'train_acc', 'test_loss', 'test_acc'
        save_path: Optional path to save figure
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['test_acc'], 'r-', label='Test Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training history to {save_path}")
    
    plt.show()


def plot_class_distribution(labels, class_names=ACTION_NAMES, save_path=None):
    """
    Plot class distribution
    
    Args:
        labels: Array of labels
        class_names: List of class names
        save_path: Optional path to save figure
    """
    unique, counts = np.unique(labels, return_counts=True)
    percentages = counts / len(labels) * 100
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(unique)), counts, alpha=0.7)
    
    # Color bars
    colors = sns.color_palette("husl", len(unique))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('Action Class')
    plt.ylabel('Count')
    plt.title('Class Distribution in Dataset')
    plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45, ha='right')
    
    # Add percentage labels
    for i, (count, pct) in enumerate(zip(counts, percentages)):
        plt.text(i, count, f'{count}\n({pct:.1f}%)', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved class distribution to {save_path}")
    
    plt.show()


def compare_models(results_dict, metric='weighted_f1', save_path=None):
    """
    Compare multiple models
    
    Args:
        results_dict: Dict mapping model names to their metrics dicts
        metric: Metric to compare ('accuracy', 'macro_f1', 'weighted_f1')
        save_path: Optional path to save figure
    """
    model_names = list(results_dict.keys())
    values = [results_dict[name][metric] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(model_names)), values, alpha=0.7)
    
    # Color bars
    colors = sns.color_palette("Set2", len(model_names))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('Model')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Model Comparison: {metric.replace("_", " ").title()}')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    
    # Add value labels
    for i, value in enumerate(values):
        plt.text(i, value, f'{value:.4f}', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved model comparison to {save_path}")
    
    plt.show()


def analyze_errors(predictions, labels, features=None, class_names=ACTION_NAMES, n_examples=5):
    """
    Analyze common misclassifications
    
    Args:
        predictions: Array of predicted labels
        labels: Array of true labels
        features: Optional array of features for detailed analysis
        class_names: List of class names
        n_examples: Number of examples to show per error type
    """
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    
    # Get misclassified indices
    errors = predictions != labels
    n_errors = np.sum(errors)
    
    print(f"\nTotal misclassifications: {n_errors} / {len(labels)} ({n_errors/len(labels)*100:.2f}%)")
    
    # Confusion pairs
    confusion_pairs = {}
    for true_label, pred_label in zip(labels[errors], predictions[errors]):
        pair = (true_label, pred_label)
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    
    # Sort by frequency
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
    
    print("\nMost common confusion pairs:")
    print(f"{'True Label':<15} {'Predicted':<15} {'Count':<10}")
    print("-" * 40)
    for (true_label, pred_label), count in sorted_pairs[:10]:
        print(f"{class_names[true_label]:<15} {class_names[pred_label]:<15} {count:<10}")


def save_evaluation_report(
    predictions, 
    labels, 
    metrics, 
    output_dir='outputs',
    experiment_name='baseline'
):
    """
    Save comprehensive evaluation report
    
    Args:
        predictions: Array of predicted labels
        labels: Array of true labels
        metrics: Metrics dict from evaluate_model()
        output_dir: Directory to save report
        experiment_name: Name of experiment
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, f'{experiment_name}_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"EVALUATION REPORT: {experiment_name}\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Overall Accuracy: {metrics['accuracy']:.2f}%\n")
        f.write(f"Macro F1-Score: {metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}\n\n")
        
        f.write("Classification Report:\n")
        f.write(classification_report(
            labels, predictions,
            labels=list(range(len(ACTION_NAMES))),
            target_names=ACTION_NAMES,
            zero_division=0
        ))
    
    print(f"✓ Saved evaluation report to {report_path}")


def top_k_accuracy(predictions_probs, labels, k=3):
    """
    Calculate top-k accuracy
    
    Args:
        predictions_probs: Array of prediction probabilities (N, num_classes)
        labels: Array of true labels (N,)
        k: Top-k value
        
    Returns:
        top_k_acc: Top-k accuracy
    """
    top_k_preds = np.argsort(predictions_probs, axis=1)[:, -k:]
    correct = np.any(top_k_preds == labels[:, None], axis=1)
    top_k_acc = np.mean(correct) * 100
    
    return top_k_acc