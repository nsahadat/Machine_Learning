# chess_agent_model.py
import torch
import torch.nn as nn
import chess
import random

# --- Model Definition ---
class ChessNet(nn.Module):
    def __init__(self, num_moves):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_moves)
        )

    def forward(self, x):
        return self.model(x)

# --- Piece encoding ---
piece_to_num = {
    'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,
    'P': 7, 'N': 8, 'B': 9, 'R': 10, 'Q': 11, 'K': 12,
}

def encode_fen(fen):
    board = chess.Board(fen)
    board_tensor = torch.zeros(8, 8, dtype=torch.float32)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            row = 7 - (i // 8)
            col = i % 8
            board_tensor[row, col] = piece_to_num.get(piece.symbol(), 0)
    return board_tensor.flatten()  # (64,)

# --- Move mapping ---
def load_move_map(csv_path="rl_training_data.csv"):
    import pandas as pd
    data = pd.read_csv(csv_path)
    all_moves = sorted(list(set(data['action'])))
    move_to_idx = {move: idx for idx, move in enumerate(all_moves)}
    idx_to_move = {idx: move for move, idx in move_to_idx.items()}
    return move_to_idx, idx_to_move

# --- Agent move function ---
def get_agent_move(fen, model, move_to_idx, idx_to_move):
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [m.uci() for m in legal_moves]

    x = encode_fen(fen).unsqueeze(0)  # (1, 64)
    with torch.no_grad():
        logits = model(x)[0]

    # Mask illegal moves
    legal_indices = [move_to_idx[m] for m in legal_moves_uci if m in move_to_idx]
    if not legal_indices:
        return random.choice(legal_moves)

    legal_logits = torch.tensor([logits[i] for i in legal_indices])
    best_idx = legal_indices[torch.argmax(legal_logits).item()]
    best_move_uci = idx_to_move[best_idx]
    return chess.Move.from_uci(best_move_uci)

# --- Load model for inference ---
def load_trained_model(model_path="chess_rl_model.pth", csv_path="rl_training_data_both_players.csv"):
    move_to_idx, idx_to_move = load_move_map(csv_path)
    model = ChessNet(len(move_to_idx))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, move_to_idx, idx_to_move

def generate_all_possible_moves():
    # board = chess.Board()
    squares = [chess.square_name(i) for i in range(64)]
    promotions = ['q', 'r', 'b', 'n']
    
    all_moves = []
    for from_sq in squares:
        for to_sq in squares:
            move = from_sq + to_sq
            all_moves.append(move)
            # Add promotions only for 7th to 8th or 2nd to 1st ranks
            if (from_sq[1] == '7' and to_sq[1] == '8') or (from_sq[1] == '2' and to_sq[1] == '1'):
                for p in promotions:
                    all_moves.append(move + p)

    return sorted(list(set(all_moves)))  # Ensure uniqueness
