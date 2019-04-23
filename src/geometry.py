#!/usr/bin/env python3
"""
Module containing geometric functions
"""
# Standard libraries
# Third-party libraries
import numpy as np


def get_polygon_centroid(vertices):
    """
    Get the X,Y coordinates of the centroid, given the vertices

    This function applies only to convex polygons. The vertices must be
    ordered  according their ocurrence along the perimeter of the
    polygon.
    """
    n_verts = vertices.shape[0]
    signed_area = 0
    centroid = np.array([.0, .0])

    # Iterate through the n vertices and compute the centroid coordinates.
    for i in range(n_verts):
        x0 = vertices[i, 0]
        y0 = vertices[i, 1]
        if i==(n_verts-1):
            x1 = vertices[0, 0]
            y1 = vertices[0, 1]
        else:
            x1 = vertices[i+1, 0]
            y1 = vertices[i+1, 1]
        signed_area += x0*y1 - x1*y0
        centroid[0] += (x0 + x1) * (x0*y1 - x1*y0)
        centroid[1] += (y0 + y1) * (x0*y1 - x1*y0)

    signed_area *= 0.5
    centroid /= (6.0*signed_area)
    return centroid


def sort_polygon_vertices(vertices):
    """
    Sort the vertices in order of appearance alongside the perimeter.
    """
    # compute centroid and sort vertices by polar angle.
    inner_point = vertices.sum(axis=0) / len(vertices)
    diff = vertices-inner_point
    order = np.argsort(np.arctan2(diff[:,1], diff[:, 0]))
    ordered_vertices = vertices[order]
    return ordered_vertices
