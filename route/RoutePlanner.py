"""
route_planner.py
------------------
Intelligent Route Planner using A* search with landmark-based heuristics.

Locations are modelled as graph nodes carrying (x, y) planar coordinates
(used only to derive an admissible straight-line heuristic and to make the
example map easy to reason about -- the coordinates are NOT real-world
geographic data). Roads are weighted, undirected edges whose weight is the
travel cost between two locations.

Three search strategies are implemented on the same graph so they can be
compared directly:
    1. Dijkstra / Uniform Cost Search   -- baseline, h(n) = 0 everywhere.
    2. A* with straight-line distance   -- basic admissible heuristic.
    3. A* with landmark heuristic (ALT) -- tighter admissible heuristic
       built from precomputed distances to a handful of "landmark" nodes,
       using the triangle inequality: for landmark L,
           h(n) = max(|d(L, n) - d(L, goal)|)   over all landmarks L
       This is admissible because true shortest-path distances obey the
       triangle inequality, and it is usually more informed than raw
       straight-line distance because it "knows" about real road layout,
       not just Euclidean geometry.

Author: Devyansh Singh
"""

import heapq
import itertools
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

Node = str


# --------------------------------------------------------------------------
# Graph environment: 14 locations, weighted undirected roads, multiple
# alternative routes between several location pairs.
# --------------------------------------------------------------------------

# (x, y) planar coordinates for each location -- used for the straight-line
# heuristic and to keep the example map spatially coherent.
COORDINATES: Dict[Node, Tuple[float, float]] = {
    "Airport":       (0, 10),
    "CityCenter":    (6, 8),
    "TechPark":      (12, 9),
    "OldTown":       (4, 4),
    "Riverside":     (9, 3),
    "Hilltop":       (2, 12),
    "University":    (7, 12),
    "Harbor":        (13, 1),
    "IndustrialZone":(15, 5),
    "Suburbia":      (3, 0),
    "LakeView":      (10, 13),
    "Market":        (5, 6),
    "Stadium":       (9, 7),
    "Museum":        (2, 7),
}

# Weighted, undirected roads. Weights are travel cost/time in arbitrary
# units. Several nodes (e.g. CityCenter, Market, Stadium) sit on multiple
# alternative paths, so both Dijkstra and A* have real choices to make.
        # NOTE: every weight below is >= the straight-line distance between
# its two endpoints. This is what guarantees the straight-line heuristic
# is admissible (and consistent): a road can never be shorter than a
# straight line between the places it connects, so Euclidean distance is
# always a safe lower bound on true travel cost.
ROADS: List[Tuple[Node, Node, float]] = [
    ("Airport", "Hilltop", 5),
    ("Airport", "CityCenter", 9),
    ("Airport", "Museum", 7),
    ("Hilltop", "University", 5),
    ("Hilltop", "Museum", 6),
    ("CityCenter", "University", 5),
    ("CityCenter", "Market", 3),
    ("CityCenter", "TechPark", 8),
    ("CityCenter", "Stadium", 4),
    ("University", "LakeView", 4),
    ("Museum", "OldTown", 4),
    ("Museum", "Market", 5),
    ("Market", "OldTown", 3),
    ("Market", "Stadium", 6),
    ("OldTown", "Suburbia", 5),
    ("OldTown", "Riverside", 7),
    ("Stadium", "TechPark", 4),
    ("Stadium", "Riverside", 5),
    ("TechPark", "LakeView", 5),
    ("TechPark", "IndustrialZone", 6),
    ("LakeView", "IndustrialZone", 11),
    ("Riverside", "Harbor", 6),
    ("Riverside", "Suburbia", 8),
    ("Harbor", "IndustrialZone", 5),
    ("Suburbia", "Harbor", 12),
]


class RoadGraph:
    """Adjacency-list representation of the weighted, undirected road network."""

    def __init__(self, coordinates: Dict[Node, Tuple[float, float]],
                 roads: List[Tuple[Node, Node, float]]):
        self.coordinates = coordinates
        self.adjacency: Dict[Node, List[Tuple[Node, float]]] = {n: [] for n in coordinates}
        for a, b, w in roads:
            self.adjacency[a].append((b, w))
            self.adjacency[b].append((a, w))

    def neighbors(self, node: Node) -> List[Tuple[Node, float]]:
        return self.adjacency[node]

    def nodes(self) -> List[Node]:
        return list(self.coordinates.keys())

    def straight_line_distance(self, a: Node, b: Node) -> float:
        """Euclidean distance between two nodes' coordinates -- admissible
        because no road can be shorter than a straight line between the
        two points it connects."""
        (x1, y1), (x2, y2) = self.coordinates[a], self.coordinates[b]
        return math.hypot(x1 - x2, y1 - y2)


# --------------------------------------------------------------------------
# Baseline: Dijkstra / Uniform Cost Search
# --------------------------------------------------------------------------

@dataclass
class SearchResult:
    found: bool
    path: List[Node]
    cost: float
    nodes_expanded: int
    nodes_generated: int
    time_taken: float


def dijkstra(graph: RoadGraph, start: Node, goal: Node) -> SearchResult:
    """
    Uniform Cost Search / Dijkstra's algorithm: explores nodes purely by
    accumulated path cost g(n), with no heuristic guidance (h(n) = 0).
    Used here as the ground-truth baseline for path cost and as a point of
    comparison for how many nodes A* needs to expand.
    """
    start_time = time.perf_counter()
    counter = itertools.count()
    frontier = [(0.0, next(counter), start)]
    best_g: Dict[Node, float] = {start: 0.0}
    came_from: Dict[Node, Optional[Node]] = {start: None}
    visited = set()
    nodes_expanded = 0
    nodes_generated = 1

    while frontier:
        g, _, node = heapq.heappop(frontier)
        if node in visited:
            continue
        visited.add(node)
        nodes_expanded += 1

        if node == goal:
            return SearchResult(True, _reconstruct_path(came_from, goal), g,
                                 nodes_expanded, nodes_generated,
                                 time.perf_counter() - start_time)

        for neighbor, weight in graph.neighbors(node):
            new_g = g + weight
            if new_g < best_g.get(neighbor, math.inf):
                best_g[neighbor] = new_g
                came_from[neighbor] = node
                nodes_generated += 1
                heapq.heappush(frontier, (new_g, next(counter), neighbor))

    return SearchResult(False, [], math.inf, nodes_expanded, nodes_generated,
                         time.perf_counter() - start_time)


def _reconstruct_path(came_from: Dict[Node, Optional[Node]], goal: Node) -> List[Node]:
    path = [goal]
    while came_from[path[-1]] is not None:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


# --------------------------------------------------------------------------
# Landmark preprocessing (ALT: A*, Landmarks, Triangle inequality)
# --------------------------------------------------------------------------

class LandmarkHeuristic:
    """
    Precomputes true shortest-path distances from a small set of landmark
    nodes to every other node (one Dijkstra run per landmark). At query
    time, the heuristic for reaching `goal` from any node `n` is:

        h(n) = max over landmarks L of |d(L, n) - d(L, goal)|

    This is admissible (it never overestimates the remaining distance,
    by the triangle inequality) and is generally more informed than plain
    straight-line distance, because it is derived from the graph's actual
    shortest-path structure rather than raw Euclidean geometry.
    """

    def __init__(self, graph: RoadGraph, landmarks: List[Node]):
        self.graph = graph
        self.landmarks = landmarks
        self.distance_from_landmark: Dict[Node, Dict[Node, float]] = {}
        for landmark in landmarks:
            self.distance_from_landmark[landmark] = self._single_source_shortest_paths(landmark)

    def _single_source_shortest_paths(self, source: Node) -> Dict[Node, float]:
        dist = {source: 0.0}
        counter = itertools.count()
        frontier = [(0.0, next(counter), source)]
        visited = set()
        while frontier:
            d, _, node = heapq.heappop(frontier)
            if node in visited:
                continue
            visited.add(node)
            for neighbor, weight in self.graph.neighbors(node):
                nd = d + weight
                if nd < dist.get(neighbor, math.inf):
                    dist[neighbor] = nd
                    heapq.heappush(frontier, (nd, next(counter), neighbor))
        return dist

    def estimate(self, node: Node, goal: Node) -> float:
        best = 0.0
        for landmark in self.landmarks:
            d_ln = self.distance_from_landmark[landmark][node]
            d_lg = self.distance_from_landmark[landmark][goal]
            best = max(best, abs(d_ln - d_lg))
        return best


def choose_landmarks(graph: RoadGraph, count: int = 3) -> List[Node]:
    """
    Pick landmarks spread around the map's perimeter, which is a simple
    and effective heuristic for landmark selection (corner/extreme nodes
    give the tightest triangle-inequality bounds). We pick nodes with
    extreme x or y coordinates.
    """
    nodes = graph.nodes()
    candidates = set()
    candidates.add(max(nodes, key=lambda n: graph.coordinates[n][0]))  # rightmost
    candidates.add(min(nodes, key=lambda n: graph.coordinates[n][0]))  # leftmost
    candidates.add(max(nodes, key=lambda n: graph.coordinates[n][1]))  # topmost
    candidates.add(min(nodes, key=lambda n: graph.coordinates[n][1]))  # bottommost
    return list(candidates)[:count] if len(candidates) >= count else list(candidates)


# --------------------------------------------------------------------------
# A* search, parameterised by a heuristic function h(node, goal) -> cost
# --------------------------------------------------------------------------

def astar(graph: RoadGraph, start: Node, goal: Node, h) -> SearchResult:
    """
    A* search: f(n) = g(n) + h(n). `h` is any function (node, goal) -> cost
    estimate. Correctness (optimality) depends on h being admissible.
    """
    start_time = time.perf_counter()
    counter = itertools.count()
    frontier = [(h(start, goal), next(counter), 0.0, start)]
    best_g: Dict[Node, float] = {start: 0.0}
    came_from: Dict[Node, Optional[Node]] = {start: None}
    visited = set()
    nodes_expanded = 0
    nodes_generated = 1

    while frontier:
        f, _, g, node = heapq.heappop(frontier)
        if node in visited:
            continue
        visited.add(node)
        nodes_expanded += 1

        if node == goal:
            return SearchResult(True, _reconstruct_path(came_from, goal), g,
                                 nodes_expanded, nodes_generated,
                                 time.perf_counter() - start_time)

        for neighbor, weight in graph.neighbors(node):
            new_g = g + weight
            if new_g < best_g.get(neighbor, math.inf):
                best_g[neighbor] = new_g
                came_from[neighbor] = node
                nodes_generated += 1
                new_f = new_g + h(neighbor, goal)
                heapq.heappush(frontier, (new_f, next(counter), new_g, neighbor))

    return SearchResult(False, [], math.inf, nodes_expanded, nodes_generated,
                         time.perf_counter() - start_time)


# --------------------------------------------------------------------------
# Display helpers
# --------------------------------------------------------------------------

def display_result(label: str, result: SearchResult) -> None:
    print(f"\n-- {label} --")
    if not result.found:
        print("No route found.")
        return
    print(f"Route: {' -> '.join(result.path)}")
    print(f"Total cost: {result.cost}")
    print(f"Nodes expanded: {result.nodes_expanded} | Nodes generated: {result.nodes_generated} "
          f"| Time: {result.time_taken:.6f}s")


def print_comparison_table(rows: List[Tuple[str, SearchResult]]) -> None:
    header = f"{'Strategy':<28}{'Cost':<10}{'Expanded':<11}{'Generated':<11}{'Time(s)':<10}"
    print(header)
    print("-" * len(header))
    for label, res in rows:
        cost_str = f"{res.cost:.1f}" if res.found else "-"
        print(f"{label:<28}{cost_str:<10}{res.nodes_expanded:<11}{res.nodes_generated:<11}{res.time_taken:<10.6f}")


# --------------------------------------------------------------------------
# Interactive / demo driver
# --------------------------------------------------------------------------

def run_query(graph: RoadGraph, landmark_h: LandmarkHeuristic, start: Node, goal: Node) -> None:
    print("\n" + "=" * 70)
    print(f"ROUTE QUERY: {start} -> {goal}")
    print("=" * 70)

    dijkstra_result = dijkstra(graph, start, goal)
    straight_line_result = astar(graph, start, goal, graph.straight_line_distance)
    landmark_result = astar(graph, start, goal, landmark_h.estimate)

    display_result("Dijkstra / Uniform Cost Search (baseline)", dijkstra_result)
    display_result("A* with straight-line distance heuristic", straight_line_result)
    display_result("A* with landmark-based (ALT) heuristic", landmark_result)

    print("\nComparison:")
    print_comparison_table([
        ("Dijkstra (baseline)", dijkstra_result),
        ("A* straight-line", straight_line_result),
        ("A* landmark (ALT)", landmark_result),
    ])


def interactive_mode(graph: RoadGraph, landmark_h: LandmarkHeuristic) -> None:
    print("\nAvailable locations:")
    for n in sorted(graph.nodes()):
        print(f"  - {n}")
    start = input("\nEnter start location: ").strip()
    goal = input("Enter goal location: ").strip()
    if start not in graph.coordinates or goal not in graph.coordinates:
        print("Unknown location entered. Please pick from the list above.")
        return
    run_query(graph, landmark_h, start, goal)


def run_demo() -> None:
    graph = RoadGraph(COORDINATES, ROADS)
    landmarks = choose_landmarks(graph, count=3)
    print(f"Selected landmarks for ALT heuristic: {landmarks}")
    landmark_h = LandmarkHeuristic(graph, landmarks)

    test_queries = [
        ("Airport", "IndustrialZone"),
        ("Suburbia", "LakeView"),
        ("Hilltop", "Harbor"),
        ("Museum", "TechPark"),
    ]

    for start, goal in test_queries:
        run_query(graph, landmark_h, start, goal)


if __name__ == "__main__":
    import sys
    graph = RoadGraph(COORDINATES, ROADS)
    landmarks = choose_landmarks(graph, count=3)
    landmark_h = LandmarkHeuristic(graph, landmarks)

    if "--interactive" in sys.argv:
        interactive_mode(graph, landmark_h)
    else:
        run_demo()