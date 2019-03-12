#!/usr/bin/env python3
"""
Library for generating Voronoi diagrams
"""
# Third-party libraries
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Polygon
import matplotlib.pyplot as plt
import numpy as np
# Local libraries


def plot_voronoi(map_vertices, agents_coords, voronoi_vertices):
    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.add_patch(Polygon(map_vertices, True, fill=False, edgecolor="k",
                         linewidth=2))
    for coords in agents_coords:
        ax.add_patch(Circle(coords, 0.2, color='#FF9900'))
    for voronoi in voronoi_vertices:
        ax.add_patch(Circle(voronoi, 0.2, color='#0022FF'))
    ax.set_xlim(-5, 15)
    ax.set_ylim(-5, 15)
    plt.grid()
    plt.show()
    return


def get_voronoi_vertices(map_vertices, sites, index):
    """
    Create a Voronoi diagram given a bounding map and points coordinates

    :param np.array map_vertices: 2xN array with the coordinates of all
     of the vetices of the map.
    :param np.array sites: 2xM array with the coordinates of all of the
     agents(sites) in the system.
    :param int index: index of the agent(site) whose Voronoi cell is
    wanted to be found.

    NOTE: the arrays are 0-indexed.
    """
    map_sides = len(map_vertices)
    n_sites = len(sites)
    vertices = np.array(())

    # Get the explicit form of all of the possible Voronoi segments
    # around the agent. y = m*x + a
    # Get the explicit form of the polygon sides
    segments = np.zeros((map_sides+(n_sites-1), 2))
    diffs = map_vertices - np.roll(map_vertices, -1, axis=0)
    distances = np.linalg.norm(diffs, axis=1).reshape(map_sides, 1)
    unit_vectors = diffs / distances
    m = unit_vectors[:,1]/unit_vectors[:,0]
    a = map_vertices[:,1] - m*map_vertices[:,0]
    segments[:map_sides, 0] = m
    segments[:map_sides, 1] = a

    # Get the explicit form of the segments separating the agents.
    agent_coords = sites[index]
    sites = np.delete(sites, index, 0)
    mid_points = agent_coords + (sites-agent_coords) / 2
    diffs = sites - agent_coords
    distances = np.linalg.norm(diffs, axis=1).reshape(n_sites-1, 1)
    unit_vectors = diffs / distances
    normal_vectors = np.zeros_like(unit_vectors)
    normal_vectors[:, 0] = unit_vectors[:, 1]
    normal_vectors[:, 1] = -unit_vectors[:, 0]
    m = normal_vectors[:,1]/normal_vectors[:,0]
    a = mid_points[:,1] - m*mid_points[:,0]
    segments[map_sides:, 0] = m
    segments[map_sides:, 1] = a
    print(segments)
    # Get the intersection points
    for index, segment in enumerate(segments):
        print(index)
        for _, segment2 in enumerate(segments[index+1:]):
            x_int = (segment2[1]-segment[1]) / (segment[0]-segment2[0])
            y_int = segment[0]*x_int + segment[1]
            intersection = np.array((x_int, y_int))
            agent_dist = np.linalg.norm(agent_coords-intersection)
            if (np.linalg.norm(site-intersection)>agent_dist for site in sites):
                if not vertices.size:
                    vertices = intersection
                else:
                    vertices = np.vstack((vertices, intersection))
    print(vertices)

    return vertices


def get_voronoi_segments(map_vertices, sites, index):
    """
    Get Voronoi segments defined by their parametric equations.
    """
    map_sides = len(map_vertices)
    n_sites = len(sites)
    vertices = np.array(())
    # Initialize the segments list. First element of a segment is a point S_0,
    # and the second element is its unit vector.
    segments = np.zeros((map_sides+(n_sites-1), 2, 2))
    # Add border segments to the segments variable
    diffs = map_vertices - np.roll(map_vertices, 1, axis=0)
    distances = np.linalg.norm(diffs, axis=1).reshape(map_sides, 1)
    unit_vectors = diffs / distances
    segments[:map_sides, 0] = map_vertices
    segments[:map_sides, 1] = unit_vectors

    # Get current agent coords and remove it from the input list
    agent_coords = sites[index]
    sites = np.copy(sites)
    sites = np.delete(sites, index, 0)
    # Add paramters of segments separating agent from other sites.
    mid_points = agent_coords + (sites-agent_coords)/2
    diffs = sites - agent_coords
    distances = np.linalg.norm(diffs, axis=1).reshape(n_sites-1, 1)
    unit_vectors = diffs / distances
    normal_vectors = np.zeros_like(unit_vectors)
    normal_vectors[:, 0] = unit_vectors[:, 1]
    normal_vectors[:, 1] = -unit_vectors[:, 0]
    segments[map_sides:, 0] = mid_points
    segments[map_sides:, 1] = normal_vectors
    return segments


def get_intersections(segments):
    """
    Return all the intersections between the provided segments.

    More info at http://geomalgorithms.com/a05-_intersect-1.html
    """
    intersections = np.array([])
    for index, segment in enumerate(segments):
        v = segment[1]
        v_perp = np.array([[-v[1], v[0]]])
        for index_2, segment_2 in enumerate(segments[index+1:]):
            u = segment_2[1]
            # Ignore calculations if segments are parallel.
            if u[0]*v[1] == u[1]*v[0]:
                continue
            # Get intersection between both segments.
            w = segment_2[0]-segment[0]
            num = (-np.dot(v_perp, w))
            den = (np.dot(v_perp, u))
            s_I = num / den
            intersection = segment_2[0] + u*s_I
            if not len(intersections):
                intersections = intersection
            else:
                intersections = np.vstack((intersections, intersection))
    return intersections


def get_voronoi_map(map_vertices, sites, index=0):
    voronoi_verts = []
    voronoi_segments = get_voronoi_segments(map_vertices, sites, index)
    voronoi_verts = get_intersections(voronoi_segments)
    print(voronoi_segments)
    return voronoi_verts


if __name__ == "__main__":
    map_vertices = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    agents_coords = np.array([[3, 5], [7, 6], [1, 1]])
    voronoi_vertices = get_voronoi_map(map_vertices, agents_coords, 0)

    plot_voronoi(map_vertices, agents_coords, voronoi_vertices)
