#!/usr/bin/env python3
"""
Module containing the agent class for the coverage control algorithm
"""
# Standard libraries
# Third-party libraries
import numpy as np
# Local libraries
import voronoi

class Agent(object):
    """
    Agent class.
    """
    def __init__(self, position):
        self.position = position
        self.voronoi_cell = None

    def get_voronoi_cell(self, map_vertices, neighbours):
        """
        Call Voronoi algorithm for getting cell surrounding agent.
        """
        sites = np.vstack((self.position, neighbours))
        self.voronoi_cell = voronoi.get_voronoi_map(map_vertices, sites, 0)
        return self.voronoi_cell

    def move(self):
        """
        Calculate new position of agent based on specified algorithm.
        """
        pass
        return self.position

