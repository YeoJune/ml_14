"""
Test all module imports
"""
import sys
sys.path.append('src')

print("Testing imports...")

try:
    print("  - dataset module...", end=' ')
    from dataset import (
        PokerDataset,
        MultimodalPokerDataset,
        download_pluribus_files,
        process_phh_files,
        get_dataloaders
    )
    print("✓")
    
    print("  - models module...", end=' ')
    from models import (
        PokerMLP,
        MultimodalPokerModel,
        TextEncoder,
        get_model
    )
    print("✓")
    
    print("  - train module...", end=' ')
    from train import (
        train_epoch,
        train_multimodal_epoch,
        evaluate,
        evaluate_multimodal,
        train_model
    )
    print("✓")
    
    print("  - evaluate module...", end=' ')
    from evaluate import (
        evaluate_model,
        plot_confusion_matrix,
        plot_training_history,
        compare_models
    )
    print("✓")
    
    print("  - generate_text module...", end=' ')
    from generate_text import (
        generate_dialogues_vllm,
        generate_dialogues_rule_based,
        load_dialogues
    )
    print("✓")
    
    print("\n✓ All imports successful!")
    
except ImportError as e:
    print(f"\n✗ Import error: {e}")
    sys.exit(1)
