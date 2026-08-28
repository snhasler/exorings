'''
"Ray casting" methods for exorings model
S. Hasler
'''

import numpy as np

def normalize(v):
    return v / np.linalg.norm(v)

def intersect_sphere_distance(ray_origin, ray_dir, sphere_center, sphere_radius):
    """
    Ray–sphere intersection.
    Returns nearest positive intersection distance t, or None.

    Parameters
    ----------
    ray_origin : ndarray
        Origin of the ray
    ray_dir : ndarray
        Direction of the ray
    sphere_center : ndarray
        Center of the sphere
    sphere_radius : float
        Radius of the sphere
    
    Returns
    -------
    t : float or None
        Nearest positive intersection distance or None if no intersection
    """
    ray_dir = normalize(ray_dir)

    oc = ray_origin - sphere_center
    b = 2.0 * np.dot(oc, ray_dir)
    c = np.dot(oc, oc) - sphere_radius**2
    disc = b*b - 4*c

    if disc < 0:
        return None

    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / 2.0
    t2 = (-b + sqrt_disc) / 2.0

    ts = [t for t in (t1, t2) if t > 0]
    return min(ts) if ts else None


def intersect_ring_distance(ray_origin, ray_dir,
                            ring_center, ring_normal,
                            r_inner, r_outer):
    """
    Ray–ring intersection via ring plane + radial bounds.
    Make sure all distances have the same units.

    Parameters
    ----------
    ray_origin : ndarray
        Origin of the ray
    ray_dir : ndarray
        Direction of the ray
    ring_center : ndarray
        Center of the ring plane
    ring_normal : ndarray
        Normal vector of the ring plane
    r_inner : float
        Inner radius of the ring
    r_outer : float
        Outer radius of the ring
    
    Returns
    -------
    t : float or None
        Nearest positive intersection distance or None if no intersection
    """
    ray_dir = normalize(ray_dir)
    ring_normal = normalize(ring_normal)

    denom = np.dot(ray_dir, ring_normal)
    if np.abs(denom) < 1e-8:
        return None

    t = np.dot(ring_center - ray_origin, ring_normal) / denom
    if t <= 0:
        return None

    hit = ray_origin + t * ray_dir
    r = np.linalg.norm(hit - ring_center)

    if r_inner <= r <= r_outer:
        return t

    return None

