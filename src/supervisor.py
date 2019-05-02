#!/usr/bin/env python3
"""
Module containing the main supervisor for the coverage control algorithm
"""
# Standard libraries
import math
# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
import yaml
# Local libraries
import agent
import voronoi

class Supervisor(object):
    """
    Supervisor class.
    """
    def __init__(self, config_file="./config/simulation.yaml"):
        self.map, self.sites, function = self.parse_config_file(config_file)
        self.vertices = [0] * len(self.sites)
        self.triangles = [0] * len(self.sites)
        # Create list containing N agent objects.
        self.agents = []
        for index, site in enumerate(self.sites):
            print(site)
            self.agents.append(agent.Agent(site, index))
        # Map density function
        if function=="uniform":
            self.density_func = lambda x,y: 1
        elif function=="gaussian":
            x0 = 30
            y0 = 10
            sigma=100
            self.density_func = lambda x,y: math.exp(-((x-x0)**2)/sigma
                                                     -((y-y0)**2)/sigma)
        elif function=="ellipse":
            k = 0.001
            a = 0.14
            b = 0.06
            r = 1.0
            x_0 = 20.0
            y_0 = 20.0
            self.density_func = lambda x,y: math.exp(-k*(a*(x-x_0)**2
                                                     + b*(y-y_0)**2 - r)**2)
        else:
            raise(ValueError, "Invalid function name '{}'".format(function))

    def parse_config_file(self, filename):
        """
        Read the configuration file and parse map and sites coordinates.
        
        :param str filename: Relative path to the configuration file.
        :returns: (map_vertices, sites) - an array with the M vertices
         of the map, and another array with the coordinates of the N
         sites in the system.
        """
        with open(filename, "r") as f: 
            config = yaml.load(f)
        # Create an array with map coordinates.
        map_dict = config["Map"]
        map_vertices = np.zeros([len(map_dict), 2])
        for key, value in enumerate(map_dict):
            map_vertices[key] = np.array(map_dict[value])
        # Create an array with the sites coordinates.
        sites_dict = config["Sites"]
        sites = np.zeros([len(sites_dict), 2])
        for key, value in enumerate(sites_dict):
            sites[key] = np.array(sites_dict[value])
        function = config["Function"]
        return (map_vertices, sites, function)

    def new_iteration(self, time_step=0.1):
        """
        Iterate the system once.
        """
        # Get the Voronoi cell of each of the agents.
        for index, agent in enumerate(self.agents):
            neighbours = np.delete(self.sites, index, axis=0)
            agent.get_voronoi_cell(self.map, neighbours)
        # Make each of the agents to move
        for index, agent in enumerate(self.agents):
            self.sites[index] = agent.move(time_step, self.density_func)
        # Get the Voronoi map of all of the agents
        for i, agent in enumerate(self.agents):
            self.vertices[i] = agent.voronoi_cell
            self.triangles[i] = agent.tri.simplices
        return


if __name__=="__main__":
    supervisor = Supervisor()
    # Create figure and axes
    fig, ax = plt.subplots(1)
    for i in range(10000):
        supervisor.new_iteration()
        voronoi.plot_voronoi(ax, supervisor.map, supervisor.sites,
                             supervisor.vertices, supervisor.triangles)
