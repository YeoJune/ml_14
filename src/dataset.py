"""
Dataset and preprocessing utilities for Texas Hold'em poker
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from typing import List, Dict, Tuple
import re
import requests


class PokerDataset(Dataset):
    """Standard PyTorch Dataset for poker decision points"""
    
    def __init__(self, features, labels):
        """
        Args:
            features: np.ndarray of shape (N, 377)
            labels: np.ndarray of shape (N,)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MultimodalPokerDataset(Dataset):
    """Dataset for multimodal poker (game state + text)"""
    
    def __init__(self, features, labels, texts, text_embeddings=None):
        """
        Args:
            features: np.ndarray of shape (N, 377)
            labels: np.ndarray of shape (N,)
            texts: List[str] of length N
            text_embeddings: Optional pre-computed embeddings of shape (N, text_dim)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.texts = texts
        self.text_embeddings = text_embeddings
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.text_embeddings is not None:
            text_emb = torch.FloatTensor(self.text_embeddings[idx])
            return self.features[idx], text_emb, self.labels[idx]
        else:
            return self.features[idx], self.texts[idx], self.labels[idx]


# ============================================================================
# Data Loading from Pluribus Dataset
# ============================================================================

CACHE_FILE = 'pluribus_data_cache.pkl'

def download_pluribus_files(n_files=None, use_cache=True, cache_path='data/raw'):
    """
    Download Pluribus poker hand histories with caching
    
    Args:
        n_files: Optional limit on number of files
        use_cache: Whether to use cached data
        cache_path: Directory for cache file
        
    Returns:
        List of PHH text strings
    """
    cache_file = os.path.join(cache_path, CACHE_FILE)
    
    # Check cache
    if use_cache and os.path.exists(cache_file):
        print(f"Loading from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        print(f"✓ Loaded {len(cached_data)} files from cache")
        
        if n_files is not None:
            return cached_data[:n_files]
        return cached_data
    
    # Download
    print("No cache found. Downloading from GitHub...")
    base_url = "https://api.github.com/repos/uoftcprg/phh-dataset/contents/data/pluribus"
    response = requests.get(base_url)
    
    phh_texts = []
    
    if response.status_code == 200:
        folders = response.json()
        
        for folder in folders:
            if folder['type'] != 'dir':
                continue
            
            folder_name = folder['name']
            folder_url = f"{base_url}/{folder_name}"
            folder_response = requests.get(folder_url)
            
            if folder_response.status_code == 200:
                files = folder_response.json()
                
                if len(files) == 0:
                    continue
                
                print(f"  Downloading from {folder_name}/ ({len(files)} files)...")
                
                for file_item in files:
                    file_url = f"https://raw.githubusercontent.com/uoftcprg/phh-dataset/master/data/pluribus/{folder_name}/{file_item['name']}"
                    file_response = requests.get(file_url)
                    
                    if file_response.status_code == 200:
                        phh_texts.append(file_response.text)
                        
                        if len(phh_texts) % 500 == 0:
                            print(f"    → Total downloaded: {len(phh_texts)} files")
                        
                        if n_files is not None and len(phh_texts) >= n_files:
                            print(f"  Reached limit of {n_files} files")
                            break
                
                if n_files is not None and len(phh_texts) >= n_files:
                    break
    
    # Save cache
    if len(phh_texts) > 0:
        os.makedirs(cache_path, exist_ok=True)
        print(f"\nSaving to cache: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(phh_texts, f)
        print(f"✓ Cached {len(phh_texts)} files")
    
    return phh_texts


# ============================================================================
# PHH Parsing
# ============================================================================

def parse_phh_hand(phh_text):
    """Parse PHH text into structured data"""
    hand = {}
    
    variant_match = re.search(r"variant = '(\w+)'", phh_text)
    if variant_match:
        hand['variant'] = variant_match.group(1)
    
    antes_match = re.search(r"antes = \[([\d, ]+)\]", phh_text)
    if antes_match:
        hand['antes'] = [int(x.strip()) for x in antes_match.group(1).split(',')]
    
    blinds_match = re.search(r"blinds_or_straddles = \[([\d, ]+)\]", phh_text)
    if blinds_match:
        hand['blinds'] = [int(x.strip()) for x in blinds_match.group(1).split(',')]
    
    stacks_match = re.search(r"starting_stacks = \[([\d, ]+)\]", phh_text)
    if stacks_match:
        hand['starting_stacks'] = [int(x.strip()) for x in stacks_match.group(1).split(',')]
    
    actions_match = re.search(r"actions = \[(.*?)\]", phh_text, re.DOTALL)
    if actions_match:
        actions_str = actions_match.group(1)
        action_list = re.findall(r"'([^']+)'", actions_str)
        hand['actions'] = action_list
    
    players_match = re.search(r"players = \[(.*?)\]", phh_text)
    if players_match:
        players_str = players_match.group(1)
        hand['players'] = [p.strip().strip("'") for p in players_str.split(',')]
    
    return hand


def parse_cards(card_str):
    """Parse card string into list of cards (e.g., 'TcQc' -> ['Tc', 'Qc'])"""
    cards = []
    i = 0
    while i < len(card_str):
        if i + 1 < len(card_str):
            cards.append(card_str[i:i+2])
            i += 2
        else:
            i += 1
    return cards


# ============================================================================
# Feature Extraction
# ============================================================================

def card_to_index(card):
    """Convert card to 0-51 index"""
    if len(card) != 2:
        return -1
    
    rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, 
                '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
    suit_map = {'c': 0, 'd': 1, 'h': 2, 's': 3}
    
    rank = card[0]
    suit = card[1]
    
    if rank in rank_map and suit in suit_map:
        return rank_map[rank] * 4 + suit_map[suit]
    return -1


def state_to_features(state):
    """
    Convert game state to 377-dim feature vector
    
    Feature breakdown:
    - Hole cards: 104 dim (52 * 2 one-hot)
    - Board cards: 260 dim (52 * 5 one-hot)
    - Position: 6 dim (one-hot)
    - Street: 4 dim (one-hot)
    - Numeric: 3 dim (pot, stack, bet_to_call normalized)
    
    Returns:
        np.ndarray of shape (377,)
    """
    features = []
    
    # Hole cards (104 dim)
    hole_cards_vec = np.zeros(104)
    hole_cards = state['hole_cards']
    for i, card in enumerate(hole_cards[:2]):
        idx = card_to_index(card)
        if idx >= 0:
            hole_cards_vec[i * 52 + idx] = 1
    features.extend(hole_cards_vec)
    
    # Board cards (260 dim)
    board_cards_vec = np.zeros(260)
    board_cards = state['board_cards']
    for i, card in enumerate(board_cards[:5]):
        idx = card_to_index(card)
        if idx >= 0:
            board_cards_vec[i * 52 + idx] = 1
    features.extend(board_cards_vec)
    
    # Position (6 dim)
    position_vec = np.zeros(6)
    if 0 <= state['position'] < 6:
        position_vec[state['position']] = 1
    features.extend(position_vec)
    
    # Street (4 dim)
    street_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
    street_vec = np.zeros(4)
    if state['street'] in street_map:
        street_vec[street_map[state['street']]] = 1
    features.extend(street_vec)
    
    # Numeric features (3 dim) - normalized by 10000 chips
    pot_norm = state['pot'] / 10000.0
    stack_norm = state['stack'] / 10000.0
    bet_to_call_norm = state['bet_to_call'] / 10000.0
    features.extend([pot_norm, stack_norm, bet_to_call_norm])
    
    return np.array(features, dtype=np.float32)


# ============================================================================
# Action Labeling
# ============================================================================

ACTION_NAMES = ['fold', 'check_call', 'raise_small', 'raise_medium', 'raise_large', 'all_in']

def categorize_action(action_type, action_amount, bet_to_call, pot, stack):
    """
    Categorize action into discrete classes:
    0: fold
    1: check_call
    2: raise_small (< 0.5x pot)
    3: raise_medium (0.5-1.5x pot)
    4: raise_large (> 1.5x pot)
    5: all_in
    """
    if action_type == 'f':
        return 0
    elif action_type == 'cc':
        return 1
    elif action_type == 'cbr':
        raise_amount = action_amount - bet_to_call
        
        if raise_amount >= stack * 0.95:
            return 5
        
        if raise_amount < pot * 0.5:
            return 2
        elif raise_amount < pot * 1.5:
            return 3
        else:
            return 4
    else:
        return -1


# ============================================================================
# Decision Point Extraction
# ============================================================================

def extract_decision_points(hand, target_player='p1'):
    """Extract all decision points for a target player"""
    actions = hand.get('actions', [])
    n_players = len(hand.get('players', []))
    starting_stacks = hand.get('starting_stacks', [])
    blinds = hand.get('blinds', [])
    
    current_stacks = starting_stacks.copy()
    current_pot = sum(blinds)
    hole_cards = {}
    board_cards = []
    current_bets = blinds.copy()
    street = 'preflop'
    
    decision_points = []
    
    for action in actions:
        parts = action.split()
        
        if parts[0] == 'd':
            if parts[1] == 'dh':
                player = parts[2]
                if len(parts) > 3:
                    cards_str = parts[3]
                    hole_cards[player] = parse_cards(cards_str)
            elif parts[1] == 'db':
                if len(parts) > 2:
                    cards_str = parts[2]
                    new_cards = parse_cards(cards_str)
                    
                    prev_board_len = len(board_cards)
                    board_cards.extend(new_cards)
                    
                    if prev_board_len < 3 and len(board_cards) >= 3:
                        street = 'flop'
                        current_bets = [0] * n_players
                    elif prev_board_len < 4 and len(board_cards) >= 4:
                        street = 'turn'
                        current_bets = [0] * n_players
                    elif prev_board_len < 5 and len(board_cards) >= 5:
                        street = 'river'
                        current_bets = [0] * n_players
        
        elif len(parts) >= 2:
            player = parts[0]
            action_type = parts[1]
            
            # Get action amount
            action_amount = 0
            if len(parts) >= 3:
                try:
                    action_amount = int(parts[2])
                except (ValueError, IndexError):
                    action_amount = 0
            
            # Get player index
            try:
                player_idx = int(player[1]) - 1
            except (ValueError, IndexError):
                continue
            
            if player_idx < 0 or player_idx >= n_players:
                continue
            
            # Only record decision points for target player
            if player == target_player and player in hole_cards:
                bet_to_call = max(current_bets) - current_bets[player_idx]
                
                # Create state
                state = {
                    'street': street,
                    'hole_cards': hole_cards[player],
                    'board_cards': board_cards.copy(),
                    'pot': current_pot,
                    'stack': current_stacks[player_idx],
                    'bet_to_call': bet_to_call,
                    'position': player_idx,
                }
                
                label = categorize_action(
                    action_type, 
                    action_amount, 
                    bet_to_call, 
                    current_pot, 
                    current_stacks[player_idx]
                )
                
                if label >= 0:
                    decision_points.append((state, label))
            
            # Update game state for ALL players
            if action_type == 'f':
                pass
            elif action_type == 'cc':
                bet_to_call = max(current_bets) - current_bets[player_idx]
                call_amount = min(bet_to_call, current_stacks[player_idx])
                current_stacks[player_idx] -= call_amount
                current_bets[player_idx] += call_amount
                current_pot += call_amount
            elif action_type == 'cbr':
                additional = action_amount - current_bets[player_idx]
                additional = min(additional, current_stacks[player_idx])
                current_stacks[player_idx] -= additional
                current_bets[player_idx] += additional
                current_pot += additional
    
    return decision_points


def process_phh_files(phh_texts, target_player='p1'):
    """
    Process PHH files into features and labels
    
    Returns:
        features: np.ndarray of shape (N, 377)
        labels: np.ndarray of shape (N,)
    """
    all_features = []
    all_labels = []
    
    for phh_text in phh_texts:
        hand = parse_phh_hand(phh_text)
        decision_points = extract_decision_points(hand, target_player)
        
        for state, label in decision_points:
            features = state_to_features(state)
            all_features.append(features)
            all_labels.append(label)
    
    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    
    return features, labels


# ============================================================================
# Data Loading Utilities
# ============================================================================

def get_dataloaders(features, labels, batch_size=64, test_size=0.2, random_seed=42):
    """
    Create train/test dataloaders with stratified split
    
    Args:
        features: np.ndarray of shape (N, 377)
        labels: np.ndarray of shape (N,)
        batch_size: Batch size for dataloaders
        test_size: Fraction of data for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        train_loader, test_loader, class_weights
    """
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, 
        test_size=test_size, 
        stratify=labels, 
        random_state=random_seed
    )
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights = torch.FloatTensor(class_weights)
    
    # Create datasets
    train_dataset = PokerDataset(X_train, y_train)
    test_dataset = PokerDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=0
    )
    
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    print(f"Class weights: {class_weights}")
    
    return train_loader, test_loader, class_weights


def save_processed_data(features, labels, output_path='data/processed'):
    """Save processed features and labels"""
    os.makedirs(output_path, exist_ok=True)
    
    np.save(os.path.join(output_path, 'features.npy'), features)
    np.save(os.path.join(output_path, 'labels.npy'), labels)
    
    print(f"✓ Saved processed data to {output_path}")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")


def load_processed_data(input_path='data/processed'):
    """Load processed features and labels"""
    features = np.load(os.path.join(input_path, 'features.npy'))
    labels = np.load(os.path.join(input_path, 'labels.npy'))
    
    print(f"✓ Loaded processed data from {input_path}")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    return features, labels