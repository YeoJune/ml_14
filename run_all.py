"""
Run Complete Pipeline
Execute all steps in sequence
"""
import subprocess
import sys

STEPS = [
    ('1_preprocess_data.py', 'Data Preprocessing'),
    ('2_train_baseline.py', 'Train Baseline Model'),
    ('3_generate_dialogues.py', 'Generate Dialogues'),
    ('4_train_multimodal.py', 'Train Multimodal Model'),
]

def run_step(script, description):
    """Run a single step"""
    print("\n" + "="*70)
    print(f"Running: {description}")
    print("="*70 + "\n")
    
    result = subprocess.run([sys.executable, script], cwd='.')
    
    if result.returncode != 0:
        print(f"\n✗ Error in {script}")
        return False
    
    print(f"\n✓ Completed: {description}")
    return True

def main():
    print("="*70)
    print("POKER AI MULTIMODAL TRAINING PIPELINE")
    print("="*70)
    
    for script, description in STEPS:
        success = run_step(script, description)
        if not success:
            print(f"\n✗ Pipeline stopped at: {description}")
            sys.exit(1)
    
    print("\n" + "="*70)
    print("✓ ALL STEPS COMPLETED SUCCESSFULLY")
    print("="*70)
    
    print("\nResults saved in:")
    print("  - checkpoints/   (model checkpoints)")
    print("  - outputs/       (evaluation reports and plots)")
    print("  - data/          (processed data and dialogues)")

if __name__ == '__main__':
    main()
