#!/usr/bin/env python3
"""
Module containing geometric functions
"""
# Standard libraries
# Third-party libraries
import numpy as np
import scipy.integrate
import scipy.spatial


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


def get_mass(vertices, function):
    """
    Get the mass of a shape, given the density function.

    :param np.array vertices: Nx2 array with the coordinates of all the
     vertices defining the shape.
    :param callable function: 2 arguments function representing the
     density function of the shape.
    :returns: float representing the mass of the shape.
    """
    # Divide the cell in triangles, so the density function can be 
    # integrated through the whole shape.
    triangulation = scipy.spatial.Delaunay(self.voronoi_cell)
    mass = 0
    # Calculate the mass of each triangle and add them up.
    for indices in triangulation.simplices:
        tri = vertices[indices]
        a = tri[0]
        b = tri[1]
        c = tri[2]
        jacobian = (b[0]-a[0]) * (c[1]-a[1]) - (c[0]-a[0]) * (b[1]-a[1])
        jacobian = np.abs(jacobian)
        f_pullback = lambda u,v: function(a[0] + u*(b[0]-a[0]) + v*(c[0]-a[0])),
                                          a[1] + u*(b[1]-a[1]) + v*(c[1]-a[1]))
        mass += scipy.integrate.dblquad(f_pullback, 0, 1, 0, lambda u: 1-u)
    return mass

def get_centre_of_mass(vertices, function, mass):
    """
    Get the centroid of a triangle given its density and mass.

    :param np.array vertices: Nx2 array with the coordinates of all the
     vertices defining the shape.
    :param callable function: 2 arguments function representing the
     density function of the shape.
    :param float mass: mass of the shape.
    :returns: (x, y) coordinates of the centre of mass of the shape
    """
    centre = np.array([0, 0])
    return centre
