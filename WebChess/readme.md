# ♟️ WebChess: Trainable Reinforcement Learning Chess Agent

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains both the **frontend** and **backend** to run a web-based chess game powered by a **Q-learning RL agent**. The agent improves over time by learning from its games against you. The backend is powered by **FastAPI**, and the frontend runs directly in your browser.

---

## 🚀 Features

- ✅ Play chess in your browser
- ✅ Reinforcement learning agent (Q-learning)
- ✅ FastAPI backend handles move logic and training
- ✅ SQLite database stores each game session
- ✅ The agent updates and retrains after every completed game
- ✅ The more you play, the smarter it gets!

---

## 📦 Requirements

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Make sure to create and activate a virtual environment first:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

---

## 🧠 How It Works

- Each time a game is played, the agent logs:
  - Board state
  - Move taken
  - Reward signal
  - Resulting state
- This data is stored in an SQLite database.
- After each session, the agent trains using these transitions.
- The updated model (`chess_rl_model.pth`) is saved and reloaded before the next game.

---

## 🖥️ Running the Application

Open **two terminals**, navigate to your project folder, and do the following:

### 🧩 Step 1: Start the FastAPI Backend

```bash
uvicorn main:app --reload
```

This serves the RL agent API.

---

### 🌐 Step 2: Start the Frontend Server

In another terminal window:

```bash
python -m http.server 8000
```

---

### 🔗 Step 3: Open the App in Your Browser

Go to:

```
http://localhost:8000/frontend.html
```

Start playing — and observe how the agent improves over time!

---

## 📁 Project Structure

```
.
├── main.py                  # FastAPI backend (API, agent logic)
├── trainAgentOnFly.py       # RL training code
├── frontend.html            # Chess UI in the browser
├── chess_rl_model.pth       # Saved model (optional, created after training)
├── game_sessions.db         # SQLite DB of move history and rewards
├── requirements.txt         # Required dependencies
└── README.md                # You're here!
```

---

## 🤝 Contributing

Contributions, bug reports, and feature suggestions are welcome! Fork this repo, open an issue, or submit a pull request.

---

## 📜 License

This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).

---

## 🧪 Future Enhancements

- ✅ Smarter reward functions
- ⏳ Better state encodings (e.g., using board embeddings)
- ⏳ MCTS or deep Q-networks
- ⏳ Frontend polish with drag-and-drop UI
