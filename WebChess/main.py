from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import chess
import random
import uuid
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Optional
import torch
import random
from trainAgentOnFly import *
from fastapi import BackgroundTasks

ALL_MOVES = generate_all_possible_moves()
MOVE_TO_INDEX = {move: i for i, move in enumerate(ALL_MOVES)}
INDEX_TO_MOVE = {i: move for move, i in MOVE_TO_INDEX.items()}

EPSILON = 0.1  # 10% exploration; tweak as needed

model = ConvChessNet.load()
# model = ChessNet.load()

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chess board state
board = chess.Board()
session_id = str(uuid.uuid4())
move_count = 0
session_start_time = datetime.now()

# Database setup
DATABASE_URL = "sqlite:///./chess_games.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

Base = declarative_base()

class MoveRecord(Base):
    __tablename__ = "moves"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    move_number = Column(Integer)
    move_uci = Column(String)
    player = Column(String)  # "player" or "agent"
    fen = Column(String)
    timestamp = Column(DateTime)

class SessionRecord(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    start_time = Column(DateTime)
    end_time = Column(DateTime, nullable=True)
    result = Column(String, nullable=True)  # "player", "agent", or "draw"

Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class MoveRequest(BaseModel):
    from_: str
    to: str
    promotion: Optional[str] = None

@app.post("/move")
def player_move(move: MoveRequest, background_tasks: BackgroundTasks):
    global board, session_id, move_count

    db = SessionLocal()

    # Handle promotion if provided
    if move.promotion:
        uci_move = move.from_ + move.to + move.promotion.lower()
    else:
        uci_move = move.from_ + move.to

    player_move_obj = chess.Move.from_uci(uci_move)

    if player_move_obj in board.legal_moves:
        board.push(player_move_obj)
        move_count += 1
        save_move(db, session_id, move_count, player_move_obj.uci(), "player", board.fen())
    else:
        db.close()
        return {"error": "Illegal move"}

    # Check after player's move
    if board.is_game_over():
        winner = finalize_session(db, background_tasks)
        db.close()
        return {
            "game_over": True,
            "winner": winner
        }

    # Agent move
    agent_move = pick_agent_move(board)
    board.push(agent_move)
    move_count += 1
    save_move(db, session_id, move_count, agent_move.uci(), "agent", board.fen())

    if board.is_game_over():
        winner = finalize_session(db, background_tasks)
        db.close()
        return {
            "game_over": True,
            "winner": winner
        }

    db.close()
    return {
        "from": agent_move.uci()[:2],
        "to": agent_move.uci()[2:],
        "fen": board.fen(),
        "game_over": False
    }

def move_to_index(move_str: str) -> int:
    return MOVE_TO_INDEX.get(move_str, -1)

def index_to_move(index: int) -> str:
    return INDEX_TO_MOVE.get(index, None)

def pick_agent_move(board):
    fen = board.fen()
    
    # input_tensor = encode_fen(fen)
    # input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    input_tensor = one_hot_encode_fen(fen)
    input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)

    legal_moves = list(board.legal_moves)
    legal_move_uci = [move.uci() for move in legal_moves]

    legal_indices = [move_to_index(uci) for uci in legal_move_uci if move_to_index(uci) is not None]

    # Mask illegal moves
    probs = torch.zeros(logits.shape[-1])
    probs[legal_indices] = logits[0][legal_indices]

    if probs.sum() == 0:
        return random.choice(legal_moves)  # fallback

    best_index = torch.argmax(probs).item()
    best_uci = index_to_move(best_index)

    try:
        if random.random()< EPSILON:
            print("agent made the move, exploration...")
            return random.choice(legal_moves)
        else:
            print("agent made the move, exploitation...")
            return chess.Move.from_uci(best_uci)
    except:
        print("random move made ...")
        return random.choice(legal_moves)

def save_move(db, session_id, move_number, move_uci, player, fen):
    move_record = MoveRecord(
        session_id=session_id,
        move_number=move_number,
        move_uci=move_uci,
        player=player,
        fen=fen,
        timestamp=datetime.now()
    )
    db.add(move_record)
    db.commit()

def finalize_session(db, background_tasks: BackgroundTasks):
    global session_id

    result = board.result()
    if result == "1-0":
        winner = "player"
    elif result == "0-1":
        winner = "agent"
    else:
        winner = "draw"

    # Update the session record
    session_record = db.query(SessionRecord).filter(SessionRecord.session_id == session_id).first()
    if session_record:
        session_record.end_time = datetime.now()
        session_record.result = winner
        db.commit()

    # ✅ Trigger model training in background
    background_tasks.add_task(train_and_save_model, epochs=300, target_update_freq=100)

    # defining global then reloading the model
    global model
    model = ConvChessNet.load()
    # model = ChessNet.load()

    return winner

@app.post("/restart")
def restart_game():
    global board, session_id, move_count, session_start_time

    board.reset()
    session_id = str(uuid.uuid4())
    move_count = 0
    session_start_time = datetime.now()

    db = SessionLocal()
    # Create a new session record
    new_session = SessionRecord(
        session_id=session_id,
        start_time=session_start_time,
        end_time=None,
        result=None
    )
    db.add(new_session)
    db.commit()
    db.close()

    return {"message": "Game restarted"}