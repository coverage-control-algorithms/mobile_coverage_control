#!/usr/bin/env python3
"""
Module containing the agent class for the coverage control algorithm
"""
# Standard libraries
# Third-party libraries
import numpy as np
# Local libraries
import geometry
import voronoi

class Agent(object):
    """
    Agent class.
    """
    def __init__(self, position, id):
        self.position = position
        self.z = None
        self.id = id
        self.voronoi_cell = None
        self.centroid = None
        self.velocity = np.array([0, 0])
        self.u = np.array([0, 0])

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
        self.centroid = geometry.get_polygon_centroid(self.voronoi_cell)
        self.centroid = self.voronoi_cell.sum(axis=0) / len(self.voronoi_cell)
        return self.voronoi_cell

    def get_control_variable(self, Kp=1):
        """
        Get the proportional control variable of the agent.
        
        :param float Kp: Proportional constant of the controller
        :returns: 1xN float array, where N is the number of system
         dimensions. Each element corresponds to the velocity in its
         corresponding dimension.
        """
        dist_to_centroid = self.centroid - self.position
        self.u = dist_to_centroid * Kp
        return self.u

    def get_velocity(self):
        """
        Use control variable for getting agent velocity
        
        :return: 1xN float array, where each element is the velocity in
         each of the N dimensions.
        """
        self.velocity = self.u
        return self.velocity

    def move(self, delta_t):
        """
        Calculate new position of agent based on coverage control.

        :param float delta_t: time step between the previous iteration
         and the current one.
        :returns: 1xN float array with the N new coordinates of the
         agent position.
        """
        # Apply control rule for getting the agent velocity.
        self.get_control_variable()
        vel = self.get_velocity()
        # Get the new position integrating the velocity through time.
        self.position += vel*delta_t
        return self.position

