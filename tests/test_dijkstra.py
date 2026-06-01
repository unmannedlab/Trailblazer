import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _load_dijkstra_module():
    if "matplotlib" not in sys.modules:
        matplotlib_stub = types.ModuleType("matplotlib")
        matplotlib_stub.__path__ = []
        pyplot_stub = types.ModuleType("matplotlib.pyplot")
        matplotlib_stub.pyplot = pyplot_stub
        sys.modules["matplotlib"] = matplotlib_stub
        sys.modules["matplotlib.pyplot"] = pyplot_stub

    module_path = Path(__file__).resolve().parents[1] / "scripts" / "dijkstra.py"
    spec = importlib.util.spec_from_file_location("dijkstra", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grid_initializes_start_node_metadata():
    dijkstra = _load_dijkstra_module()
    grid_data = np.array([[3.0, 1.0], [1.0, 1.0]])
    grid = dijkstra.Grid(grid_data, start=[0, 0])

    start_node = grid.grid[0, 0]
    other_node = grid.grid[1, 1]
    assert start_node.dist == 3.0
    assert start_node.backtrack == -1
    assert other_node.dist == np.inf
    assert other_node.backtrack is None


def test_get_neighbours_respects_bounds_and_diagonals():
    dijkstra = _load_dijkstra_module()
    grid = dijkstra.Grid(np.ones((3, 3)), start=[0, 0])
    solver = dijkstra.Dijkstra(grid, start=[0, 0], goal=[2, 2])

    neighbours = set(solver.get_neighbours(0, 0))
    assert neighbours == {(0, 1), (1, 0), (1, 1)}


def test_find_path_avoids_high_cost_cells():
    dijkstra = _load_dijkstra_module()
    grid_data = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 100.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    grid = dijkstra.Grid(grid_data, start=[0, 0])
    solver = dijkstra.Dijkstra(grid, start=[0, 0], goal=[2, 2])

    solver.find_path()
    path = solver.backtrack_path()
    assert solver.found_path is True
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert (1, 1) not in path


def test_backtrack_path_returns_none_when_goal_not_found():
    dijkstra = _load_dijkstra_module()
    grid = dijkstra.Grid(np.ones((2, 2)), start=[0, 0])
    solver = dijkstra.Dijkstra(grid, start=[0, 0], goal=[5, 5])

    solver.find_path()
    assert solver.found_path is False
    assert solver.backtrack_path() is None
