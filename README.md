# Poker AI: Multimodal Decision Making

Multimodal AI agent for Texas Hold'em poker combining game state and dialogue information.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. HuggingFace Token Setup (Optional)

For using gated models like Llama:

**Option A: Environment Variable (Recommended)**

```bash
export HF_TOKEN="your_huggingface_token"
```

**Option B: Login via CLI**

```bash
huggingface-cli login
# Enter your token when prompted
```

**Option C: Set in Config**
Edit `3_generate_dialogues.py`:

```python
CONFIG = {
    'hf_token': 'your_token_here',
    ...
}
```

**Getting a Token:**

1. Go to https://huggingface.co/settings/tokens
2. Create a new token (Read permission is enough)
3. For Llama models, request access at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

**Using Open Models (No Token Required):**
The default config uses `microsoft/Phi-3-mini-4k-instruct` which requires no authentication.

## Project Structure

```
poker_project/
├── data/
│   ├── raw/              # Pluribus dataset cache
│   ├── processed/        # Processed features/labels
│   └── text/             # Generated dialogues
├── src/
│   ├── dataset.py        # Data loading and preprocessing
│   ├── models.py         # Model architectures
│   ├── train.py          # Training loops
│   ├── evaluate.py       # Evaluation utilities
│   └── generate_text.py  # Dialogue generation
├── checkpoints/          # Saved model checkpoints
├── outputs/              # Evaluation reports and plots
├── 1_preprocess_data.py
├── 2_train_baseline.py
├── 3_generate_dialogues.py
├── 4_train_multimodal.py
├── run_all.py
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
python run_all.py
```

This executes all 4 steps sequentially.

### 3. Run Individual Steps

#### Step 1: Preprocess Data

```bash
python 1_preprocess_data.py
```

Downloads Pluribus dataset and extracts 377-dim features.

#### Step 2: Train Baseline Model

```bash
python 2_train_baseline.py
```

Trains MLP on game state features only.

#### Step 3: Generate Dialogues

```bash
python 3_generate_dialogues.py
```

Generates poker dialogues using LLM (vLLM or rule-based).

#### Step 4: Train Multimodal Model

```bash
python 4_train_multimodal.py
```

Trains model combining game state + text embeddings.

## Configuration

Each script has a `CONFIG` dict at the top for easy customization:

```python
CONFIG = {
    'batch_size': 64,
    'learning_rate': 0.0005,
    'n_epochs': 10,
    ...
}
```

## Model Architectures

### Baseline (Game Only)

```
Input (377) -> [1024, 512, 256] -> ReLU + Dropout -> Output (6)
```

### Multimodal (Game + Text)

```
Game (377) -> [512, 256] -> [512]
                                   -> Concat [768] -> [384, 192] -> Output (6)
Text -> DistilBERT -> PCA -> [256]
```

## Output Files

- `checkpoints/baseline_best.pt` - Best baseline model
- `checkpoints/multimodal_best.pt` - Best multimodal model
- `outputs/baseline_report.txt` - Baseline evaluation
- `outputs/multimodal_report.txt` - Multimodal evaluation
- `outputs/*_confusion_matrix.png` - Confusion matrices
- `outputs/*_training_history.png` - Training curves

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Macro F1**: Unweighted average F1 across classes
- **Weighted F1**: Support-weighted average F1
- **Per-class Precision/Recall/F1**: Detailed per-action metrics

## Action Classes

0. Fold
1. Check/Call
2. Raise Small (< 0.5× pot)
3. Raise Medium (0.5-1.5× pot)
4. Raise Large (> 1.5× pot)
5. All-in

## GPU Requirements

- **Baseline Training**: ~4GB VRAM
- **Multimodal Training**: ~8GB VRAM
- **Dialogue Generation (vLLM)**: ~12-16GB VRAM
  - For 3B parameter model in FP16

Tested on A10G 24GB GPU.

## Notes

- Data is cached after first download
- Dialogue generation can use rule-based fallback if LLM unavailable
- All scripts support resuming from checkpoints
- Class imbalance handled via weighted loss

## Citation

Dataset: Pluribus Poker Hand Histories
https://github.com/uoftcprg/phh-dataset
