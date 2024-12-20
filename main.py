# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file can be a nice home for your Battlesnake logic and helper functions.
#
# To get you started we've included code to prevent your Battlesnake from moving backwards.
# For more info see docs.battlesnake.com

import random
import typing
import heapq
import pprint
import json


# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "iamwafula",  # Username
        "color": "#123524",  # Pthalo Green
        "head": "nr-rocket",  # Rocket Head
        "tail": "coffee",  # Coffee Tail
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")


def simple_move_logic(start_x, start_y, target_x, target_y):
    moves = []

    if start_x < target_x:
        moves.append("right")
    elif start_x > target_x:
        moves.append("left")

    if start_y < target_y:
        moves.append("up")
    elif start_y > target_y:
        moves.append("down")

    return moves


# reduces latency by caching heuristic values
heuristic_cache = {}


def a_star(start_x, start_y, target_x, target_y, grid):
    """
    A* algorithm to find the shortest path between two points on a grid.
    :param start_x: x-coordinate of the starting point.
    :param start_y: y-coordinate of the starting point.
    :param target_x: x-coordinate of the target point.
    :param target_y: y-coordinate of the target point.
    :param grid: 2D list representing the board.
    :return: List of moves to reach the target point.
    """

    # Heuristic function (Manhattan distance in this case)

    def heuristic(x, y):
        if (x, y) in heuristic_cache:
            return heuristic_cache[(x, y)]

        # Manhattan distance
        heuristic_cache[(x, y)] = abs(x - target_x) + abs(y - target_y)
        return heuristic_cache[(x, y)]

    # Possible moves
    directions = [("right", 1, 0), ("left", -1, 0), ("up", 0, 1), ("down", 0, -1)]

    # Priority queue for A* (min-heap)
    open_list = []
    heapq.heappush(
        open_list, (0 + heuristic(start_x, start_y), 0, start_x, start_y, [])
    )  # (f, g, x, y, path)

    # Set to track visited nodes
    visited = set()

    while open_list:
        _, g, x, y, path = heapq.heappop(open_list)

        if (x, y) == (target_x, target_y):
            return path  # Return the path to the target

        if (x, y) not in visited:
            visited.add((x, y))

            # Explore neighbors
            for direction, dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (
                    0 <= new_x < len(grid[0])
                    and 0 <= new_y < len(grid)
                    and (new_x, new_y) not in visited
                ):
                    new_g = g + 1  # Assuming uniform cost (each move has the same cost)

                    if grid[len(grid[0]) - 1 - new_y][new_x] == 1:
                        new_g += 100_000  # Penalize if the path goes through a point marked 1 (some snake body, self or opponent)

                    heapq.heappush(
                        open_list,
                        (
                            new_g + heuristic(new_x, new_y),
                            new_g,
                            new_x,
                            new_y,
                            path + [direction],
                        ),
                    )

    return []  # If no path is found


def get_moves_towards_target(x, y, target_x, target_y):
    """
    Get the moves to reach the target point from the current point.
    This is a simple heuristic that moves towards the target point, it will fail if there are obstacles in the way.

    :param x: x-coordinate of the current point.
    :param y: y-coordinate of the current point.
    :param target_x: x-coordinate of the target point.
    :param target_y: y-coordinate of the target point.
    :return: List of moves to reach the target point.
    """

    moves = []

    if x < target_x:
        moves.append("right")
    elif x > target_x:
        moves.append("left")

    if y < target_y:
        moves.append("up")
    elif y > target_y:
        moves.append("down")

    return moves


def is_reachable(grid, start, target):
    """
    Check if the target is reachable from the start in the given grid.
    Used in the get_closest_food function to check if the food is reachable.
    This is because we're trying to select the island with the food that is reachable.

    :param grid: 2D list representing the board.
    :param start: Tuple (x, y) representing the start point.
    :param target: Tuple (x, y) representing the target point.
    :return: Boolean indicating whether the target is reachable.
    """
    from collections import deque

    width = len(grid[0])
    height = len(grid)
    visited = set()
    queue = deque([start])

    while queue:
        x, y = queue.popleft()

        if (x, y) == target:
            return True

        if (x, y) in visited:
            continue
        visited.add((x, y))

        # Explore neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < width and 0 <= new_y < height and grid[new_y][new_x] == 0:
                queue.append((new_x, new_y))

    return False


def get_closest_food(game_state):
    """
    Get the closest food to the head of the snake.
    :param game_state: The game state.
    :return: The closest food to the head of the snake.
    """

    my_head = game_state["you"]["body"][0]
    food_sources = game_state["board"]["food"]
    closest_food = food_sources[0]
    closest_distance = abs(my_head["x"] - closest_food["x"]) + abs(
        my_head["y"] - closest_food["y"]
    )

    # Generate a grid representation of the board
    grid = generate_grid(
        game_state["board"]["width"],
        game_state["board"]["height"],
        game_state["you"]["body"],
        opponents=game_state["board"]["snakes"],
    )
    islands = get_islands(grid)

    # If there is more than one island, they are ordered by size and we'll start with the largest
    # that is also reachable.
    if len(islands) > 1:
        for island in islands:
            for food in food_sources:
                food_pos = (food["x"], food["y"])
                if food_pos in island["coordinates"] and is_reachable(
                    grid, (my_head["x"], my_head["y"]), food_pos
                ):
                    closest_food = food
                    return closest_food
    else:
        # If there is only one island, we'll just find the closest food that is reachable.
        for food in food_sources:
            distance = abs(my_head["x"] - food["x"]) + abs(my_head["y"] - food["y"])
            food_pos = (food["x"], food["y"])
            if distance < closest_distance and is_reachable(
                grid, (my_head["x"], my_head["y"]), food_pos
            ):
                closest_food = food
                closest_distance = distance

    return closest_food


def get_next_coordinate(x_val, y_val, next_move):
    """
    Helper function to get the next coordinate given the current coordinate and the next move
    :param x_val: x-coordinate of the current point.
    :param y_val: y-coordinate of the current point.
    :param next_move: The next move to make.
    :return: Tuple (x, y) representing the next coordinate.
    """

    if next_move == "up":
        return x_val, y_val + 1
    elif next_move == "down":
        return x_val, y_val - 1
    elif next_move == "left":
        return x_val - 1, y_val
    elif next_move == "right":
        return x_val + 1, y_val


def predict_filter_moves(safe_moves, game_state: typing.Dict):
    """
    Ensure that after this move, we still have safe moves left
    """

    filtered_safe_moves = []
    for move in safe_moves:
        # Simulate the move
        # Check that the move leaves options for the next turn
        new_x, new_y = (
            game_state["you"]["body"][0]["x"],
            game_state["you"]["body"][0]["y"],
        )

        new_x, new_y = get_next_coordinate(new_x, new_y, move)

        # Check that after the move, there are still safe moves left
        is_move_safe = {"up": True, "down": True, "left": True, "right": True}

        # Check for Collision with self
        my_body = game_state["you"]["body"]
        my_head = game_state["you"]["body"][0]
        for body_part in my_body[1:]:
            if my_head["x"] + 1 == body_part["x"] and my_head["y"] == body_part["y"]:
                is_move_safe["right"] = False
            elif my_head["x"] - 1 == body_part["x"] and my_head["y"] == body_part["y"]:
                is_move_safe["left"] = False
            elif my_head["y"] + 1 == body_part["y"] and my_head["x"] == body_part["x"]:
                is_move_safe["up"] = False
            elif my_head["y"] - 1 == body_part["y"] and my_head["x"] == body_part["x"]:
                is_move_safe["down"] = False

        # Check for Collision with walls
        board_width = game_state["board"]["width"]
        board_height = game_state["board"]["height"]

        if new_x == 0:
            is_move_safe["left"] = False
        if new_x == board_width - 1:
            is_move_safe["right"] = False
        if new_y == 0:
            is_move_safe["down"] = False
        if new_y == board_height - 1:
            is_move_safe["up"] = False

        # Check if there are any safe moves left
        if any(is_move_safe.values()):
            filtered_safe_moves.append(move)

    return filtered_safe_moves


def flood_fill(grid, cells, visited, start_x, start_y):
    """
    Flood-fill algorithm to explore the grid and find connected cells.
    :param grid: 2D list representing the board.
    :param cells: List to store the connected cells.
    :param visited: 2D list to keep track of explored cells.
    :param start_x: x-coordinate of the starting point.
    :param start_y: y-coordinate of the starting point.

    """
    # Directions to move in the grid: right, left, up, down
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    stack = [(start_x, start_y)]
    visited[start_x][start_y] = True

    while stack:
        x, y = stack.pop()
        cells.append((x, y))

        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if (
                0 <= new_x < len(grid)
                and 0 <= new_y < len(grid[0])
                and not visited[new_x][new_y]
                and grid[new_x][new_y] == 0
            ):
                visited[new_x][new_y] = True
                stack.append((new_x, new_y))


def get_islands(grid):
    """
    Get the islands of connected cells in the grid.
    :param grid: 2D list representing the board.
    :return: List of islands.
    """

    # Create a visited array to keep track of explored squares
    visited = [[False for _ in range(len(grid[0]))] for _ in range(len(grid))]

    # Island data structure to store the islands
    # island { "size": 0, "coordinates": [] }
    islands = []

    # Explore the grid and count islands
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if not visited[i][j] and grid[i][j] == 0:
                # create new island, fill in cells
                # Start a flood-fill from this unvisited and unblocked square
                new_island = {"size": 0, "coordinates": []}
                cells = []
                flood_fill(grid, cells, visited, i, j)
                new_island["size"] = len(cells)
                new_island["coordinates"] = cells
                islands.append(new_island)

    # Sort the islands by size
    islands.sort(key=lambda x: x["size"], reverse=True)

    return islands


def generate_grid(width, height, snake_body, opponents=[]):
    # Initialize the grid with empty spaces (0)
    grid = [[0 for _ in range(width)] for _ in range(height)]

    # Mark the snake's body with 1s
    for segment in snake_body:
        x, y = segment["x"], segment["y"]
        if 0 <= x < width and 0 <= y < height:
            grid[height - 1 - y][x] = 1  # Snake body part at position (x, y)

    # Mark the opponent snakes' bodies with 1s
    for opponent in opponents:
        for segment in opponent["body"]:
            x, y = segment["x"], segment["y"]
            if 0 <= x < width and 0 <= y < height:
                grid[height - 1 - y][
                    x
                ] = 1  # Opponent snake body part at position (x, y)

    return grid


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
def move(game_state: typing.Dict) -> typing.Dict:

    is_move_safe = {"up": True, "down": True, "left": True, "right": True}

    food_source = get_closest_food(game_state)

    grid_representation = generate_grid(
        game_state["board"]["width"],
        game_state["board"]["height"],
        game_state["you"]["body"],
        opponents=game_state["board"]["snakes"],
    )

    # We've included code to prevent your Battlesnake from moving backwards
    my_head = game_state["you"]["body"][0]  # Coordinates of your head
    my_neck = game_state["you"]["body"][1]  # Coordinates of your "neck"

    if my_neck["x"] < my_head["x"]:  # Neck is left of head, don't move left
        is_move_safe["left"] = False

    elif my_neck["x"] > my_head["x"]:  # Neck is right of head, don't move right
        is_move_safe["right"] = False

    elif my_neck["y"] < my_head["y"]:  # Neck is below head, don't move down
        is_move_safe["down"] = False

    elif my_neck["y"] > my_head["y"]:  # Neck is above head, don't move up
        is_move_safe["up"] = False

    # Prevent Battlesnake from moving out of bounds
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]

    if my_head["x"] == 0:
        is_move_safe["left"] = False

    if my_head["x"] == board_width - 1:
        is_move_safe["right"] = False

    if my_head["y"] == 0:
        is_move_safe["down"] = False

    if my_head["y"] == board_height - 1:
        is_move_safe["up"] = False

    # Prevent your Battlesnake from colliding with itself
    my_body = game_state["you"]["body"]

    # Check if any body part is in the same position as the head
    for body_part in my_body[1:]:
        # If the body part is in the same position as the head, don't move there
        if my_head["x"] + 1 == body_part["x"] and my_head["y"] == body_part["y"]:
            is_move_safe["right"] = False
        elif my_head["x"] - 1 == body_part["x"] and my_head["y"] == body_part["y"]:
            is_move_safe["left"] = False
        elif my_head["y"] + 1 == body_part["y"] and my_head["x"] == body_part["x"]:
            is_move_safe["up"] = False
        elif my_head["y"] - 1 == body_part["y"] and my_head["x"] == body_part["x"]:
            is_move_safe["down"] = False

    # Prevent your Battlesnake from colliding with other Battlesnakes
    opponents = []
    if len(game_state["board"]["snakes"]) > 1:
        for snake in game_state["board"]["snakes"][1:]:
            if snake["id"] != game_state["you"]["id"]:
                opponents.append(snake)

    for opponent in opponents:

        opponent_head = opponent["body"][0]
        # Ensure that where we avoid squares next to the opponents head to avoid head-on collisions
        # Lets also avoid where the opponents head could be next turn
        possible_moves = ["up", "down", "left", "right"]
        for move in possible_moves:

            # Get the next coordinate of the opponent's head based on possible move
            opponent_next_x, opponent_next_y = get_next_coordinate(
                opponent_head["x"], opponent_head["y"], move
            )

            if my_head["x"] + 1 == opponent_next_x and my_head["y"] == opponent_next_y:
                is_move_safe["right"] = False
            elif (
                my_head["x"] - 1 == opponent_next_x and my_head["y"] == opponent_next_y
            ):
                is_move_safe["left"] = False
            elif (
                my_head["y"] + 1 == opponent_next_y and my_head["x"] == opponent_next_x
            ):
                is_move_safe["up"] = False
            elif (
                my_head["y"] - 1 == opponent_next_y and my_head["x"] == opponent_next_x
            ):
                is_move_safe["down"] = False

        # Avoid the rest of the opponents body
        for body_part in opponent["body"]:
            if my_head["x"] + 1 == body_part["x"] and my_head["y"] == body_part["y"]:
                is_move_safe["right"] = False
            elif my_head["x"] - 1 == body_part["x"] and my_head["y"] == body_part["y"]:
                is_move_safe["left"] = False
            elif my_head["y"] + 1 == body_part["y"] and my_head["x"] == body_part["x"]:
                is_move_safe["up"] = False
            elif my_head["y"] - 1 == body_part["y"] and my_head["x"] == body_part["x"]:
                is_move_safe["down"] = False

    # Ensure that after this move, we still have safe moves left
    # This is similar to running the move function in the future to check if the move is safe
    # A more efficient version fo this would be to check moves to second food source instead
    safe_moves = predict_filter_moves(is_move_safe, game_state)

    # Are there any safe moves left?
    safe_moves = []
    for move, isSafe in is_move_safe.items():
        if isSafe:
            safe_moves.append(move)

    if len(safe_moves) == 0:
        print(f"MOVE {game_state['turn']}: No safe moves detected! Moving down")
        return {"move": "down"}

    # Move towards food instead of random, to regain health and survive longer
    move_list = a_star(
        my_head["x"],
        my_head["y"],
        food_source["x"],
        food_source["y"],
        grid_representation,
    )

    # choose first safe move in the list
    next_move = move_list[0] if move_list[0] in safe_moves else safe_moves[0]

    # Priority is getting a food item, so try to find a safe move that is in the move_list
    for move in move_list:
        if move in safe_moves:
            next_move = move
            break

    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
