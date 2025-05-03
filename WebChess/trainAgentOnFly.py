import chess
import sqlite3
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import os
    

piece_to_num = {
    'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,
    'P': 7, 'N': 8, 'B': 9, 'R': 10, 'Q': 11, 'K': 12,
}

def piece_value(piece):
    values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }
    return values.get(piece.piece_type, 0)

def add_step_rewards(df):
    df = df.copy()
    df["reward"] = 0.0

    for session_id, group in df.groupby("session_id"):
        board = chess.Board()
        rewards = []

        result = group["result"].iloc[0]
        moves = group["move_uci"].tolist()
        players = group["player"].tolist()

        prev_material_balance = sum(
            piece_value(p) for p in board.piece_map().values()
        )

        for i, move_uci in enumerate(moves):
            move = chess.Move.from_uci(move_uci)
            prev_board = board.copy()
            reward = 0

            # 1. Capture reward
            if prev_board.is_capture(move):
                captured_piece = prev_board.piece_at(move.to_square)
                if captured_piece:
                    reward += piece_value(captured_piece)

            # 2. Promotion reward
            if move.promotion:
                reward += 0.8

            board.push(move)

            # 3. Check reward
            if board.is_check():
                reward += 0.1

            # 4. Material loss penalty (based on material balance diff)
            new_material_balance = sum(
                piece_value(p) for p in board.piece_map().values()
            )
            material_diff = new_material_balance - prev_material_balance

            if players[i] == result:  # Current move made by the winner
                reward += max(material_diff, 0) * 0.2  # Reward for maintaining advantage
            else:
                reward += min(material_diff, 0) * 0.2  # Penalty for losing material

            prev_material_balance = new_material_balance

            # 5. End-of-game bonus or penalty
            if board.is_game_over():
                if board.is_checkmate():
                    if result == players[i]:
                        reward += 10  # Win
                    else:
                        reward -= 10  # Loss
                elif board.is_stalemate():
                    reward += 0  # Neutral

            # 6. Slight time penalty per move
            reward -= 0.01

            rewards.append(reward)

        df.loc[group.index, "reward"] = rewards

    return df

def encode_fen(fen):
    board = chess.Board(fen)
    board_tensor = torch.zeros(8, 8, dtype=torch.float32)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            row = 7 - (i // 8)
            col = i % 8
            board_tensor[row, col] = piece_to_num.get(piece.symbol(), 0)
    return board_tensor.flatten()

def encode_fen_conv(fen):
    board = chess.Board(fen)
    board_tensor = torch.zeros((8, 8), dtype=torch.float32)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            row = 7 - (i // 8)
            col = i % 8
            board_tensor[row, col] = piece_to_num.get(piece.symbol(), 0)
    return board_tensor  # Shape: (8, 8)

def one_hot_encode_fen(fen: str) -> torch.Tensor:
    piece_to_idx = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    
    board = chess.Board(fen)
    tensor = torch.zeros(64, 12)  # 64 squares, 12 possible pieces
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            idx = piece_to_idx[piece.symbol()]
            tensor[square][idx] = 1.0
    
    return tensor.view(8, 8, 12).permute(2, 0, 1)  # Shape: (12, 8, 8)

def prepare_dataset(df):
    dataset = []
    for session_id, group in df.groupby("session_id"):
        group = group.sort_values("move_number").reset_index(drop=True)
        for i in range(len(group)):
            current_row = group.iloc[i]

            if i == 0:
                prev_fen = chess.STARTING_FEN  # Start of the game
            else:
                prev_fen = group.iloc[i - 1]["fen"]

            dataset.append({
                "state": prev_fen,
                "action": current_row["move_uci"],
                "reward": current_row["reward"],
                "next_state": current_row["fen"]
            })
    return dataset

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
        self.num_moves = num_moves

    def forward(self, x):
        return self.model(x)
    
    def save(self, path="chess_rl_model.pth"):
        """Save the model state."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_moves': self.num_moves
        }, path)

    @classmethod
    def load(cls, path="chess_rl_model.pth"):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model = cls(num_moves=checkpoint['num_moves'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

class ConvChessNet(nn.Module):
    def __init__(self, num_moves):
        super().__init__()
        self.num_moves = num_moves

        self.conv1 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_moves)

    def forward(self, x):
        # x shape: (batch_size, 12, 8, 8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def save(self, path="conv_chess_model.pth"):
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_moves': self.num_moves
        }, path)

    @classmethod
    def load(cls, path="conv_chess_model.pth"):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model = cls(num_moves=checkpoint['num_moves'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model



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

def train_and_save_model(db_path="chess_games.db", 
                         model_path="chess_rl_model.pth", 
                         epochs=300, 
                         batch_size=64, 
                         gamma=0.99,
                         target_update_freq=100):
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    query = """
    SELECT 
        moves.session_id,
        moves.move_number,
        moves.move_uci,
        moves.player,
        moves.fen,
        moves.timestamp,
        sessions.result
    FROM moves
    JOIN sessions ON moves.session_id = sessions.session_id
    --WHERE player = result
    ORDER BY moves.session_id, moves.move_number
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Preprocessing
    df = add_step_rewards(df)

    # normalize the reward
    df["reward"] = (df["reward"] - df["reward"].mean()) / (df["reward"].std() + 1e-6)
    df.dropna(subset=["reward"], inplace=True)
    print(f"Number of sessions in the dataset: {df.session_id.nunique()}")

    #keep only one player now
    # df = df[df.player == df.result].reset_index(drop=True)

    dataset = prepare_dataset(df)

    ALL_MOVES = generate_all_possible_moves()
    all_moves = sorted(list(set(ALL_MOVES)))
    move_to_idx = {move: idx for idx, move in enumerate(all_moves)}
    idx_to_move = {idx: move for move, idx in move_to_idx.items()}

    num_moves = len(move_to_idx)

    try:
        # model = ChessNet.load()
        model = ConvChessNet.load()
    except:
        # model = ChessNet(num_moves)
        model = ConvChessNet(num_moves)

    target_model = copy.deepcopy(model)
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert dataset to tensors
    # state_tensors = torch.stack([encode_fen(d["state"]) for d in dataset])
    # next_state_tensors = torch.stack([encode_fen(d["next_state"]) for d in dataset])
    # actions = torch.tensor([move_to_idx[d["action"]] for d in dataset], dtype=torch.long)
    # rewards = torch.tensor([d["reward"] for d in dataset], dtype=torch.float32)

    state_tensors = torch.stack([one_hot_encode_fen(d["state"]) for d in dataset])
    next_state_tensors = torch.stack([one_hot_encode_fen(d["next_state"]) for d in dataset])
    actions = torch.tensor([move_to_idx[d["action"]] for d in dataset], dtype=torch.long)
    rewards = torch.tensor([d["reward"] for d in dataset], dtype=torch.float32)

    for epoch in range(epochs):
        permutation = torch.randperm(state_tensors.size(0))
        for i in range(0, state_tensors.size(0), batch_size):
            idx = permutation[i:i+batch_size]
            s = state_tensors[idx]
            a = actions[idx]
            r = rewards[idx]
            s_next = next_state_tensors[idx]

            # Compute Q-values
            q_values = model(s)
            q_value = q_values.gather(1, a.unsqueeze(1)).squeeze()

            # Compute max Q(s', a') for next state
            with torch.no_grad():
                next_q_values = target_model(s_next)
                max_next_q = next_q_values.max(dim=1)[0]

            q_target = r + gamma * max_next_q
            q_target = q_target.detach()

            # Loss: Mean Squared Error between predicted and target Q-values
            loss = F.mse_loss(q_value, q_target)
            # print(f"reward: {r}, q_value: {q_value}, max_next_q: {max_next_q}")


            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if epoch % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())
            model.save()

        if epoch%10==0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    
    # Save model
    model.save()
    print(f"✅ Model saved as {model_path}")

