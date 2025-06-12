import numpy as np
import heapq  # For priority queue
from time import time
import matplotlib.pyplot as plt

class Node:
    def __init__(self, value, is_start=False):
        self.value = value
        self.dist = value if is_start else np.inf
        self.backtrack = -1 if is_start else None
        self.visited = False  # New attribute to mark visited nodes

class Grid:
    def __init__(self, grid, start):
        self.grid = np.empty(grid.shape, dtype=object)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                self.grid[i, j] = Node(grid[i, j], is_start=(i, j) == tuple(start))

class Dijkstra:
    def __init__(self, grid, start, goal):
        self.grid = grid.grid
        self.start = start
        self.goal = goal
        self.found_path = False

        # Priority queue for unvisited nodes
        self.unvisited = []
        heapq.heappush(self.unvisited, (0, tuple(start)))

    def find_path(self):
        while self.unvisited:
            # Pop the node with the smallest distance
            current_dist, current_node = heapq.heappop(self.unvisited)
            x, y = current_node

            # Skip if the node is already visited
            if self.grid[x, y].visited:
                continue

            # Mark the node as visited
            self.grid[x, y].visited = True

            # Check if we reached the goal
            if current_node == tuple(self.goal):
                self.found_path = True
                break

            # Process neighbors
            for nx, ny in self.get_neighbours(x, y):
                if not self.grid[nx, ny].visited:
                    tentative_dist = self.grid[x, y].dist + self.grid[nx, ny].value
                    if tentative_dist < self.grid[nx, ny].dist:
                        self.grid[nx, ny].dist = tentative_dist
                        self.grid[nx, ny].backtrack = (x, y)
                        heapq.heappush(self.unvisited, (tentative_dist, (nx, ny)))

    def get_neighbours(self, x, y):
        max_x, max_y = self.grid.shape
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
        return [
            (x + dx, y + dy)
            for dx, dy in directions
            if 0 <= x + dx < max_x and 0 <= y + dy < max_y
        ]

    def backtrack_path(self):
        if not self.found_path:
            return None
        path = []
        current = tuple(self.goal)
        while current != -1:
            path.append(current)
            current = self.grid[current[0], current[1]].backtrack
        return path[::-1]  # Reverse to get the path from start to goal

def plot_map(array):
    current_cmap = plt.cm.Blues
    current_cmap.set_bad(color='red')
    fig, ax = plt.subplots(figsize=(40,28)) #costmap
    ax.matshow(array,cmap=plt.cm.Blues, vmin=0, vmax = 2)
    plt.show()

if __name__ == "__main__":
    grid_data = np.load('../assets/cavasos_costmap_final.npy', allow_pickle=True)
    start = [519,505]
    goal = [186,506]

    start_time = time()
    grid = Grid(grid_data, start)
    dijkstra_solver = Dijkstra(grid, start, goal)
    dijkstra_solver.find_path()

    path = dijkstra_solver.backtrack_path()
    print("Execution time:", time() - start_time)
    if path:
        print("Path found:", path)
    else:
        print("No path found.")
    
    for i in path:
        grid_data[i[0],i[1]] = 2
    plot_map(grid_data)