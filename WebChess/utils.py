import sqlite3
import torch
from torch.utils.data import Dataset
import chess
import numpy as np
import pandas as pd


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    board_tensor = np.zeros(773, dtype=np.float32)

    piece_map = board.piece_map()
    offset = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}

    for sq, piece in piece_map.items():
        idx = sq * 12 + offset[piece.symbol().upper()]
        if piece.color == chess.BLACK:
            idx += 6
        board_tensor[idx] = 1

    board_tensor[768] = board.turn
    board_tensor[769] = board.has_kingside_castling_rights(chess.WHITE)
    board_tensor[770] = board.has_queenside_castling_rights(chess.WHITE)
    board_tensor[771] = board.has_kingside_castling_rights(chess.BLACK)
    board_tensor[772] = board.has_queenside_castling_rights(chess.BLACK)

    return torch.tensor(board_tensor).unsqueeze(0)  # shape: (1, 773)

class ChessCSVDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.samples = self.df[["fen", "white_result"]].dropna()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fen = self.samples.iloc[idx]["fen"]
        result = self.samples.iloc[idx]["white_result"]

        # Convert FEN to tensor
        board = chess.Board(fen)
        x = board_to_tensor(board).squeeze(0)  # shape: (773,) or your chosen encoding

        # Label: win = 1, everything else = 0
        y = 1.0 if result == "win" else 0.0
        y = torch.tensor(y, dtype=torch.float32)

        return x, y

class ChessValueDataset(Dataset):
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.samples = self._load_data()

    def _load_data(self):
        cursor = self.conn.cursor()
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
        ORDER BY moves.session_id, moves.move_number
        """
        return cursor.execute(query).fetchall()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        session_id, move_number, move_uci, player, fen, timestamp, result = self.samples[idx]
        board = chess.Board(fen)
        x = board_to_tensor(board).squeeze(0)  # shape: (773,)
        y = torch.tensor(1.0 if result == "player" else 0.0, dtype=torch.float32)
        return x, y
