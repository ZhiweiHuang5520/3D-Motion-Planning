import numpy as np

from src.rrt.rrt import RRT
from src.rrt.rrt_star import RRTStar
from src.search_space.search_space import SearchSpace


class Planner:

    Q = np.array([(1, 26)])  # length of tree edges
    r = 0.01  # length of smallest edge to check for intersection with obstacles
    max_samples = 100000  # max number of samples to take before timing out
    prc = 0.3  # probability of checking for a connection to goal
    rewire_count = 16  # optional, number of nearby branches to rewire

    def __init__(self, boundary, blocks):
        self.boundary = boundary
        self.X_dimensions = np.zeros((3,2))
        self.X_dimensions[:,0] = self.boundary[:,0:3]
        self.X_dimensions[:, 1] = self.boundary[:,3:6]
        self.Obstacles = blocks[:,0:6]

    def plan(self, start, goal):
        x_init = tuple(start)  # starting location
        x_goal = tuple(goal)
        X = SearchSpace(self.X_dimensions, self.Obstacles)
        # rrt = RRT(X, Planner.Q, x_init, x_goal, Planner.max_samples, Planner.r, Planner.prc)
        # path = rrt.rrt_search()
        rrt = RRTStar(X, Planner.Q, x_init, x_goal, Planner.max_samples, Planner.r, Planner.prc, Planner.rewire_count)
        path = rrt.rrt_star()
        return np.array(path)
