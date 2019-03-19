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
        self.centroid = None

    def get_voronoi_cell(self, map_vertices, neighbours):
        """
        Call Voronoi algorithm for getting cell surrounding agent.

        :param np.array map_vertices: 2xN array with the coordinates of
         the N vetices of the map.
        :param np.array neighbours: 2xM array with the coordinates of
         the M neighbouring sites in the map.
        :returns: 2xV array containing the coordinates of the V vertices
         of the Voronoi cell surrounding the agent.
        """
        sites = np.vstack((self.position, neighbours))
        self.voronoi_cell = voronoi.get_voronoi_cell(map_vertices, sites, 0)
        self.centroid = self.voronoi_cell.sum(axis=0) / len(self.voronoi_cell)
        return self.voronoi_cell

    def move(self):
        """
        Calculate new position of agent based on coverage control.
        """
        self.position = self.centroid
        return self.position

