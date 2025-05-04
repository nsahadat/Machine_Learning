import chess
import chess.engine
import math
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *


def simulate_random_game(board):
    limit = 100  # move limit to prevent infinite loops
    for _ in range(limit):
        if board.is_game_over():
            break
        moves = list(board.legal_moves)
        move = random.choice(moves)
        board.push(move)

    result = board.result()
    if result == '1-0':
        return 1 if board.turn == chess.BLACK else 0
    elif result == '0-1':
        return 1 if board.turn == chess.WHITE else 0
    else:
        return 0.5  # draw

def evaluate_with_nn(board, model):
    with torch.no_grad():
        input_tensor = board_to_tensor(board)
        value = model(input_tensor).item()
        return value


class ChessValueNet(nn.Module):
    def __init__(self):
        super(ChessValueNet, self).__init__()
        self.fc1 = nn.Linear(773, 256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.out(x))  # win probability
    
class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_moves = list(board.legal_moves)

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_param=1.4):
        choices = []
        for child in self.children:
            if child.visits == 0:
                score = float('inf')
            else:
                exploitation = child.value / child.visits
                exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
                score = exploitation + exploration
            choices.append((score, child))
        return max(choices, key=lambda x: x[0])[1]

    def expand(self):
        move = self.untried_moves.pop()
        new_board = self.board.copy()
        new_board.push(move)
        child_node = MCTSNode(new_board, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def is_terminal(self):
        return self.board.is_game_over()

class MCTS:
    def __init__(self, value_net, simulations=100):
        self.value_net = value_net
        self.simulations = simulations

    # def board_to_tensor(self, board):
    #     # You should replace this with your actual input encoding
    #     return board_to_tensor(board.fen()).unsqueeze(0)  # shape: [1, C, 8, 8]

    def evaluate(self, board):
        with torch.no_grad():
            x = board_to_tensor(board)
            value = self.value_net(x)
        return value.item()

    def backpropagate(self, node, value):
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    def default_policy(self, board):
        return self.evaluate(board)

    def search(self, board):
        root = MCTSNode(board.copy())

        for _ in range(self.simulations):
            node = root

            # Selection
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child()

            # Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            # Simulation
            value = self.default_policy(node.board)

            # Backpropagation
            self.backpropagate(node, value)

        # Choose the move of the most visited child
        best_move = max(root.children, key=lambda c: c.visits).move
        return best_move
