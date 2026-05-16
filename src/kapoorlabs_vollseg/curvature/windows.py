"""Sliding-window iteration over boundary / surface points.

Two flavours:

- **2D**: boundary points form an ordered 1D loop, so the window is
  ``n_window`` consecutive points along the contour, advanced by
  ``stride``. Closed contours wrap around.
- **3D**: a vertex on a triangle mesh has no natural ordering, so the
  window is the ``n_window`` nearest neighbours — either by **geodesic
  distance** along the mesh (default, topology-aware) or **Euclidean
  distance** in 3D (faster but ignores topology).
"""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Iterator, Sequence

import numpy as np
from scipy.spatial import cKDTree


# ============================================================== 2D


def slide_windows_2d(
    contour: np.ndarray,
    *,
    n_window: int,
    stride: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(centre_idx, window_indices)`` for a closed 2D contour.

    The window is ``n_window`` consecutive contour points centred on
    ``centre_idx``, wrapping around the loop. ``stride`` steps the
    centre forward by ``stride`` indices per yield.

    Requires ``n_window >= 3`` (a 3-point circle fit needs ≥ 3 points)
    and that ``contour`` has at least ``n_window`` points.
    """
    n_pts = len(contour)
    if n_pts < n_window:
        return
    if n_window < 3:
        raise ValueError(f"n_window must be ≥ 3 for circle fit, got {n_window}")
    half = n_window // 2
    for centre in range(0, n_pts, stride):
        idx = (np.arange(centre - half, centre - half + n_window) % n_pts).astype(
            np.int64
        )
        yield centre, idx


# ============================================================== 3D


def bfs_geodesic_neighbors(
    adjacency: Sequence[set[int]],
    start_idx: int,
    n_neighbors: int,
) -> np.ndarray:
    """BFS on the mesh adjacency graph to find ``n_neighbors`` nearest hops.

    Uses unweighted BFS — distance is measured in *hops* along mesh
    edges, which is fast (O(n_neighbors · mean_degree)) and a good
    approximation of geodesic distance for locally near-uniform meshes
    (which is what ``marching_cubes`` produces). For exact geodesic
    distance use :func:`dijkstra_geodesic_neighbors` instead.

    The starting vertex itself is excluded from the returned set.
    """
    visited = {start_idx}
    queue: deque[int] = deque([start_idx])
    result: list[int] = []
    while queue and len(result) < n_neighbors:
        v = queue.popleft()
        for nb in adjacency[v]:
            if nb not in visited:
                visited.add(nb)
                result.append(nb)
                queue.append(nb)
                if len(result) >= n_neighbors:
                    break
    return np.asarray(result, dtype=np.int64)


def dijkstra_geodesic_neighbors(
    vertices: np.ndarray,
    adjacency: Sequence[set[int]],
    start_idx: int,
    n_neighbors: int,
) -> np.ndarray:
    """Exact geodesic neighbours via Dijkstra weighted by Euclidean edge length.

    Slower than the BFS approximation but correct on meshes with
    irregular edge lengths (e.g. after mesh decimation). Excludes the
    starting vertex itself.
    """
    n_v = len(vertices)
    dist = np.full(n_v, np.inf, dtype=np.float64)
    dist[start_idx] = 0.0
    heap: list[tuple[float, int]] = [(0.0, start_idx)]
    visited = np.zeros(n_v, dtype=bool)
    collected: list[tuple[float, int]] = []

    while heap and len(collected) < n_neighbors:
        d, v = heapq.heappop(heap)
        if visited[v]:
            continue
        visited[v] = True
        if v != start_idx:
            collected.append((d, v))
            if len(collected) >= n_neighbors:
                break
        for nb in adjacency[v]:
            if visited[nb]:
                continue
            new_d = d + float(np.linalg.norm(vertices[nb] - vertices[v]))
            if new_d < dist[nb]:
                dist[nb] = new_d
                heapq.heappush(heap, (new_d, nb))

    return np.asarray([idx for _, idx in collected], dtype=np.int64)


def euclidean_neighbors(
    tree: cKDTree,
    query_point: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    """Nearest-neighbour selection in 3D Euclidean space via KD-tree.

    Fast but ignores surface topology — can pull in points from the
    opposite side of a thin cell neck. Use only when you've verified
    your shapes are convex enough that this doesn't matter.
    """
    _, idx = tree.query(query_point, k=n_neighbors + 1)
    return np.asarray(idx[1:], dtype=np.int64)  # drop self


def slide_windows_3d(
    vertices: np.ndarray,
    adjacency: Sequence[set[int]],
    *,
    n_window: int,
    stride: int,
    geodesic: bool = True,
    geodesic_method: str = "bfs",
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(centre_idx, window_indices)`` for a 3D mesh.

    Picks every ``stride``-th vertex as a window centre and selects
    ``n_window`` neighbours by geodesic (default, ``bfs`` or
    ``dijkstra``) or Euclidean distance.
    """
    if n_window < 4:
        raise ValueError(f"n_window must be ≥ 4 for sphere fit, got {n_window}")
    n_v = len(vertices)
    if n_v < n_window + 1:
        return

    tree = cKDTree(vertices) if not geodesic else None

    for centre in range(0, n_v, stride):
        if geodesic:
            if geodesic_method == "dijkstra":
                idx = dijkstra_geodesic_neighbors(vertices, adjacency, centre, n_window)
            elif geodesic_method == "bfs":
                idx = bfs_geodesic_neighbors(adjacency, centre, n_window)
            else:
                raise ValueError(f"Unknown geodesic_method: {geodesic_method!r}")
        else:
            idx = euclidean_neighbors(tree, vertices[centre], n_window)
        if len(idx) < n_window:
            continue
        yield centre, idx
