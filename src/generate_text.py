"""
Text generation for poker dialogues using LLM
"""
import torch
import numpy as np
from tqdm import tqdm
import json
import os


# ============================================================================
# Prompt Templates
# ============================================================================

SYSTEM_PROMPT = """You are a professional poker player. Generate realistic poker dialogue based on the game situation and action taken. The dialogue should be concise (1-2 sentences max) and match the player's action and hand strength."""

def create_dialogue_prompt(state, action_label):
    """
    Create prompt for dialogue generation
    
    Args:
        state: Game state dict
        action_label: Action class (0-5)
        
    Returns:
        prompt: String prompt for LLM
    """
    action_names = ['fold', 'check/call', 'small raise', 'medium raise', 'large raise', 'all-in']
    action = action_names[action_label]
    
    street = state.get('street', 'preflop')
    pot = state.get('pot', 0)
    stack = state.get('stack', 0)
    bet_to_call = state.get('bet_to_call', 0)
    
    # Describe hole cards (without revealing actual cards for privacy)
    hole_cards = state.get('hole_cards', [])
    if len(hole_cards) == 2:
        # Describe hand strength qualitatively
        hand_desc = "strong hand" if action_label in [3, 4, 5] else "marginal hand" if action_label == 1 else "weak hand"
    else:
        hand_desc = "unknown hand"
    
    prompt = f"""Game Situation:
- Street: {street}
- Pot: {pot} chips
- Your stack: {stack} chips
- Bet to call: {bet_to_call} chips
- Hand: {hand_desc}
- Action taken: {action}

Generate a realistic poker dialogue (1-2 sentences) that this player might say:"""
    
    return prompt


# ============================================================================
# vLLM-based Generation (Fast Batch Inference)
# ============================================================================

def generate_dialogues_vllm(
    states,
    action_labels,
    model_name='meta-llama/Llama-3.2-3B-Instruct',
    max_tokens=50,
    temperature=0.7,
    batch_size=32,
    output_file='data/text/dialogues.jsonl',
    use_cache=True,
    hf_token=None
):
    """
    Generate dialogues using vLLM for fast batch inference
    
    Args:
        states: List of game state dicts
        action_labels: List of action labels
        model_name: HuggingFace model name
        max_tokens: Max tokens per generation
        temperature: Sampling temperature
        batch_size: Batch size for generation
        output_file: Path to save generated dialogues
        use_cache: Whether to use cached results
        hf_token: HuggingFace token for gated models (optional)
        
    Returns:
        dialogues: List of generated dialogue strings
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Check cache
    if use_cache and os.path.exists(output_file):
        print(f"Loading cached dialogues from {output_file}")
        dialogues = []
        with open(output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                dialogues.append(data['dialogue'])
        print(f"✓ Loaded {len(dialogues)} cached dialogues")
        return dialogues
    
    print(f"Generating dialogues using vLLM...")
    print(f"Model: {model_name}")
    print(f"Total samples: {len(states)}")
    
    # Setup HuggingFace token if provided
    if hf_token is None:
        import os as os_module
        hf_token = os_module.environ.get('HF_TOKEN')
    
    if hf_token:
        print("  Using HuggingFace token for authentication")
    
    try:
        from vllm import LLM, SamplingParams
        
        # Initialize vLLM with token if available
        llm_kwargs = {
            'model': model_name,
            'tensor_parallel_size': 1,
            'dtype': 'float16',
            'gpu_memory_utilization': 0.9,
            'max_model_len': 2048
        }
        
        if hf_token:
            llm_kwargs['tokenizer_mode'] = 'auto'
            llm_kwargs['trust_remote_code'] = True
            # Set environment variable for huggingface_hub
            import os as os_module
            os_module.environ['HF_TOKEN'] = hf_token
        
        llm = LLM(**llm_kwargs)
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9
        )
        
        # Create prompts
        prompts = []
        for state, action_label in tqdm(zip(states, action_labels), desc='Creating prompts', total=len(states)):
            prompt = create_dialogue_prompt(state, action_label)
            full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
            prompts.append(full_prompt)
        
        # Generate in batches
        print("\nGenerating dialogues...")
        all_outputs = llm.generate(prompts, sampling_params)
        
        # Extract generated texts
        dialogues = []
        for output in all_outputs:
            generated_text = output.outputs[0].text.strip()
            dialogues.append(generated_text)
        
        # Save to file
        with open(output_file, 'w') as f:
            for i, dialogue in enumerate(dialogues):
                data = {
                    'index': i,
                    'dialogue': dialogue,
                    'action': int(action_labels[i])
                }
                f.write(json.dumps(data) + '\n')
        
        print(f"✓ Generated {len(dialogues)} dialogues")
        print(f"✓ Saved to {output_file}")
        
        return dialogues
        
    except ImportError:
        print("vLLM not available. Falling back to HuggingFace transformers (slower)...")
        return generate_dialogues_hf(
            states, action_labels, model_name, max_tokens, 
            temperature, batch_size, output_file
        )


# ============================================================================
# HuggingFace Transformers-based Generation (Fallback)
# ============================================================================

def generate_dialogues_hf(
    states,
    action_labels,
    model_name='microsoft/Phi-3-mini-4k-instruct',
    max_tokens=50,
    temperature=0.7,
    batch_size=8,
    output_file='data/text/dialogues.jsonl',
    use_cache=True,
    hf_token=None
):
    """
    Generate dialogues using HuggingFace transformers (fallback if vLLM unavailable)
    
    Args:
        states: List of game state dicts
        action_labels: List of action labels
        model_name: HuggingFace model name
        max_tokens: Max tokens per generation
        temperature: Sampling temperature
        batch_size: Batch size for generation
        output_file: Path to save generated dialogues
        use_cache: Whether to use cached results
        hf_token: HuggingFace token for gated models
        
    Returns:
        dialogues: List of generated dialogue strings
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Check cache
    if use_cache and os.path.exists(output_file):
        print(f"Loading cached dialogues from {output_file}")
        dialogues = []
        with open(output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                dialogues.append(data['dialogue'])
        print(f"✓ Loaded {len(dialogues)} cached dialogues")
        return dialogues
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print(f"Generating dialogues using HuggingFace transformers...")
    print(f"Model: {model_name}")
    print(f"Total samples: {len(states)}")
    
    # Setup HuggingFace token if provided
    if hf_token is None:
        import os as os_module
        hf_token = os_module.environ.get('HF_TOKEN')
    
    # Load model and tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    if device == 'cuda':
        print(f"  GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    tokenizer_kwargs = {}
    model_kwargs = {
        'torch_dtype': torch.float16 if device == 'cuda' else torch.float32,
    }
    
    # Use device_map='auto' for GPU, this automatically handles multi-GPU
    if device == 'cuda':
        model_kwargs['device_map'] = 'auto'
        model_kwargs['low_cpu_mem_usage'] = True
    
    if hf_token:
        tokenizer_kwargs['token'] = hf_token
        model_kwargs['token'] = hf_token
    
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    
    print("  Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Explicitly move to GPU if not using device_map
    if device == 'cuda' and 'device_map' not in model_kwargs:
        model = model.to(device)
        print(f"  ✓ Model moved to {device}")
    elif device == 'cuda':
        print(f"  ✓ Model loaded with device_map=auto")
    else:
        model = model.to(device)
        print(f"  ⚠️  Using CPU (this will be slow)")
    
    # Set padding token if not exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dialogues = []
    
    # Generate in batches
    print("  Generating dialogues...")
    print("  Note: First batch may take longer due to model compilation")
    for batch_idx, i in enumerate(tqdm(range(0, len(states), batch_size), desc='Generating')):
        batch_states = states[i:i+batch_size]
        batch_labels = action_labels[i:i+batch_size]
        
        # Create prompts
        prompts = []
        for state, action_label in zip(batch_states, batch_labels):
            prompt = create_dialogue_prompt(state, action_label)
            full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
            prompts.append(full_prompt)
        
        # Tokenize
        inputs = tokenizer(
            prompts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        for j, output in enumerate(outputs):
            # Get only the generated part (skip input prompt)
            input_length = inputs['input_ids'][j].shape[0]
            generated_ids = output[input_length:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Clean up
            generated_text = generated_text.strip()
            # Take only first sentence if multiple
            if '.' in generated_text:
                generated_text = generated_text.split('.')[0] + '.'
            if '\n' in generated_text:
                generated_text = generated_text.split('\n')[0]
            
            dialogues.append(generated_text)
    
    # Save to file
    with open(output_file, 'w') as f:
        for i, dialogue in enumerate(dialogues):
            data = {
                'index': i,
                'dialogue': dialogue,
                'action': int(action_labels[i])
            }
            f.write(json.dumps(data) + '\n')
    
    print(f"✓ Generated {len(dialogues)} dialogues")
    print(f"✓ Saved to {output_file}")
    
    return dialogues


# ============================================================================
# Rule-based Fallback Templates
# ============================================================================

RULE_BASED_TEMPLATES = {
    0: [  # Fold
        "I'm out.",
        "Too rich for my blood.",
        "Not this time.",
        "I'll pass.",
        "You got it."
    ],
    1: [  # Check/Call
        "Let's see it.",
        "I'm in.",
        "Call.",
        "I'll check.",
        "Okay."
    ],
    2: [  # Raise Small
        "Let's make it interesting.",
        "I'll bump it up a bit.",
        "Small raise.",
        "Adding a little pressure.",
        "Let's go."
    ],
    3: [  # Raise Medium
        "Time to raise.",
        "I'm raising.",
        "Let's see who's serious.",
        "Building the pot.",
        "I like my hand."
    ],
    4: [  # Raise Large
        "Big raise!",
        "Let's play for real money.",
        "I'm putting you to the test.",
        "All or nothing.",
        "You better have something."
    ],
    5: [  # All-in
        "All in!",
        "I'm all in!",
        "Let's settle this now.",
        "Everything I got.",
        "This is it!"
    ]
}

def generate_dialogues_rule_based(action_labels, output_file='data/text/dialogues_rule_based.jsonl'):
    """
    Generate dialogues using simple rule-based templates (fast fallback)
    
    Args:
        action_labels: List of action labels
        output_file: Path to save generated dialogues
        
    Returns:
        dialogues: List of generated dialogue strings
    """
    print(f"Generating dialogues using rule-based templates...")
    print(f"Total samples: {len(action_labels)}")
    
    dialogues = []
    rng = np.random.RandomState(42)
    
    for action_label in tqdm(action_labels, desc='Generating'):
        templates = RULE_BASED_TEMPLATES[action_label]
        dialogue = rng.choice(templates)
        dialogues.append(dialogue)
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for i, dialogue in enumerate(dialogues):
            data = {
                'index': i,
                'dialogue': dialogue,
                'action': int(action_labels[i])
            }
            f.write(json.dumps(data) + '\n')
    
    print(f"✓ Generated {len(dialogues)} dialogues")
    print(f"✓ Saved to {output_file}")
    
    # Also save to the default filename for compatibility
    if 'rule_based' in output_file:
        default_file = output_file.replace('_rule_based', '')
        with open(default_file, 'w') as f:
            for i, dialogue in enumerate(dialogues):
                data = {
                    'index': i,
                    'dialogue': dialogue,
                    'action': int(action_labels[i])
                }
                f.write(json.dumps(data) + '\n')
        print(f"✓ Also saved to {default_file} for compatibility")
    
    return dialogues


# ============================================================================
# Loading Utilities
# ============================================================================

def load_dialogues(input_file='data/text/dialogues.jsonl'):
    """
    Load generated dialogues from file
    
    Args:
        input_file: Path to dialogue file
        
    Returns:
        dialogues: List of dialogue strings
        actions: List of action labels
    """
    # Try primary file first
    if not os.path.exists(input_file):
        # Try rule-based fallback
        fallback_file = input_file.replace('.jsonl', '_rule_based.jsonl')
        if os.path.exists(fallback_file):
            print(f"Primary file not found, using fallback: {fallback_file}")
            input_file = fallback_file
        else:
            raise FileNotFoundError(f"Dialogue file not found: {input_file}")
    
    dialogues = []
    actions = []
    
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            dialogues.append(data['dialogue'])
            actions.append(data['action'])
    
    print(f"✓ Loaded {len(dialogues)} dialogues from {input_file}")
    return dialogues, actions