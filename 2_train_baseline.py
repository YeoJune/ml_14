"""
Step 2: Train Baseline Model
Train MLP model on game state features only
"""
import sys
sys.path.append('src')

import torch
import numpy as np
from dataset import load_processed_data, get_dataloaders
from models import get_model
from train import train_model
from evaluate import (
    evaluate_model, 
    plot_confusion_matrix, 
    plot_training_history,
    plot_class_distribution,
    save_evaluation_report
)

# Configuration
CONFIG = {
    'batch_size': 64,
    'learning_rate': 0.0005,
    'n_epochs': 10,
    'hidden_dims': [1024, 512, 256],
    'dropout': 0.2,
    'test_size': 0.2,
    'random_seed': 42,
    'checkpoint_dir': 'checkpoints',
    'output_dir': 'outputs'
}

def main():
    print("="*60)
    print("STEP 2: TRAIN BASELINE MODEL")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Set random seed
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    
    # Load processed data
    print("\n1. Loading processed data...")
    features, labels = load_processed_data('data/processed')
    
    # Show class distribution
    print("\n2. Visualizing class distribution...")
    plot_class_distribution(labels, save_path='outputs/class_distribution.png')
    
    # Create dataloaders
    print("\n3. Creating dataloaders...")
    train_loader, test_loader, class_weights = get_dataloaders(
        features, labels,
        batch_size=CONFIG['batch_size'],
        test_size=CONFIG['test_size'],
        random_seed=CONFIG['random_seed']
    )
    
    # Create model
    print("\n4. Creating baseline model...")
    model = get_model(
        model_type='baseline',
        device=device,
        input_dim=377,
        hidden_dims=CONFIG['hidden_dims'],
        output_dim=6,
        dropout=CONFIG['dropout']
    )
    
    # Train
    print("\n5. Training model...")
    history = train_model(
        model,
        train_loader,
        test_loader,
        n_epochs=CONFIG['n_epochs'],
        learning_rate=CONFIG['learning_rate'],
        class_weights=class_weights,
        device=device,
        checkpoint_dir=CONFIG['checkpoint_dir'],
        model_name='baseline',
        multimodal=False
    )
    
    # Plot training history
    print("\n6. Plotting training history...")
    plot_training_history(history, save_path='outputs/baseline_training_history.png')
    
    # Final evaluation
    print("\n7. Final evaluation on test set...")
    from train import evaluate
    
    # Load best model
    best_checkpoint = f"{CONFIG['checkpoint_dir']}/baseline_best.pt"
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    test_loss, test_acc, predictions, true_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    # Detailed metrics
    metrics = evaluate_model(predictions, true_labels, verbose=True)
    
    # Confusion matrix
    print("\n8. Plotting confusion matrix...")
    plot_confusion_matrix(predictions, true_labels, save_path='outputs/baseline_confusion_matrix.png')
    
    # Save report
    print("\n9. Saving evaluation report...")
    save_evaluation_report(
        predictions, true_labels, metrics,
        output_dir=CONFIG['output_dir'],
        experiment_name='baseline'
    )
    
    print("\n" + "="*60)
    print("✓ BASELINE TRAINING COMPLETE")
    print("="*60)
    print(f"\nBest Test Accuracy: {checkpoint['test_acc']:.2f}%")
    print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")

if __name__ == '__main__':
    main()
