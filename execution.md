## HOW TO RUN BATTLE SNAKE

You can read through the quikstart guide on how to create battlesnake games [here](https://docs.battlesnake.com/quickstart). This markdown file only covers how to run the server.


### Option 1: Use the Render.com URL

The battle snake is already hosted on Render.com, you can access it by clicking [here](https://cs152battlesnake.onrender.com).

You can add it to any BattleSnake Game by using the ID `IanWafulaCs152Snake`

### Option 2: Run the Battle Snake Server Locally

You can run the server locally by following the steps below:

1. Install dependencies by running `pip install -r requirements.txt`
2. Start the server by running `python3 main.py`
3. You might need to forward a port to that your local server is accessible publicly and can add it to any game.

### Option 3: Run using Docker

You can run the server using Docker by following the steps below:
1. Build the Docker image by running `docker build -t battlesnake .`
2. Run the Docker image by running `docker run -p 8080:8080 battlesnake`
3. You might need to forward a port to that your local server is accessible publicly and can add it to any game.

### Option 4: Run Snake on local Battle Snake

Instructions on how to run this are in the README.md file in the root directory of this project.