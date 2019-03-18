#!/usr/bin/env python3
"""
Module containing the main supervisor for the coverage control algorithm
"""
# Standard libraries
# Third-party libraries
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
        self.map, self.sites = self.parse_config_file(config_file)
        self.vertices = [0] * len(self.sites)
        # Create list containing N agent objects.
        self.agents = []
        for site in self.sites:
            self.agents.append(agent.Agent(site))

    def parse_config_file(self, filename):
        """
        Read the configuration file and parse map and sites coordinates.
        
        :param str filename: Relative path to the configuration file.
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
        return (map_vertices, sites)

    def new_iteration(self):
        """
        Iterate the system once.
        """
        # Get the Voronoi cell of each of the agents.
        for index, agent in enumerate(self.agents):
            neighbours = np.delete(self.sites, index, axis=0)
            agent.get_voronoi_cell(self.map, neighbours)
        # Make each of the agents to move
        for index, agent in enumerate(self.agents):
            self.sites[index] = agent.move()
        # Get the Voronoi map of all of the agents
        for i, agent in enumerate(self.agents):
            self.vertices[i] = voronoi.get_voronoi_map(self.map, self.sites, i)
        return


if __name__=="__main__":
    supervisor = Supervisor()
    supervisor.new_iteration()
    voronoi.plot_voronoi(supervisor.map, supervisor.sites, supervisor.vertices)
