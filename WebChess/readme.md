## Step 1: Install requirements.txt file
- create a custom environment and install all required packages from requirements.txt
- activate the same environment to run fastapi command
- run pip install command:

pip install -r requirements.txt

## Information: Two parts need to be run using the command window
- from the command window browse to the directory of the files

## Step 2: Start first server for fastapi by running following command
uvicorn main:app --reload

## Step 3: Start python http server by running following command
python -m http.server 8000


## Step 4: open your web browser by going to this url
http://localhost:8000/frontend.html

### Step 5: Start playing with the agent
- it will create database and save the moves, actions, timestamps, session ids to the database
- After each game it will access the database, create torch formatted state, action, reward, next_state to train a RL agent.
- Agent will be saved and later fastapi will load the agent and use it to make the moves
