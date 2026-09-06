"""
test_route_planner.py
------------------------
Test suite for the route planner: graph integrity, heuristic admissibility,
and cross-checking that Dijkstra, straight-line A*, and landmark A* always
agree on the optimal path cost. Run with: python3 test_route_planner.py
"""

import itertools

from route_planner import (
    RoadGraph, COORDINATES, ROADS, choose_landmarks, LandmarkHeuristic,
    dijkstra, astar,
)


def check(condition: bool, message: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        raise AssertionError(message)


def test_graph_size():
    graph = RoadGraph(COORDINATES, ROADS)
    check(len(graph.nodes()) >= 12, f"Graph has at least 12 locations ({len(graph.nodes())} found)")
    check(len(ROADS) >= 12, f"Graph has multiple alternative routes ({len(ROADS)} roads)")


def test_graph_connected():
    """Every location should be reachable from every other location."""
    graph = RoadGraph(COORDINATES, ROADS)
    start = graph.nodes()[0]
    for target in graph.nodes():
        result = dijkstra(graph, start, target)
        check(result.found, f"{target} reachable from {start}")


def test_heuristic_admissibility():
    """A heuristic is admissible if it never exceeds the true shortest cost."""
    graph = RoadGraph(COORDINATES, ROADS)
    landmarks = choose_landmarks(graph, 3)
    landmark_h = LandmarkHeuristic(graph, landmarks)

    for start, goal in itertools.permutations(graph.nodes(), 2):
        true_cost = dijkstra(graph, start, goal).cost
        sld = graph.straight_line_distance(start, goal)
        lmk = landmark_h.estimate(start, goal)
        check(sld <= true_cost + 1e-9, f"Straight-line heuristic admissible for {start}->{goal}")
        check(lmk <= true_cost + 1e-9, f"Landmark heuristic admissible for {start}->{goal}")


def test_all_strategies_agree_on_optimal_cost():
    graph = RoadGraph(COORDINATES, ROADS)
    landmarks = choose_landmarks(graph, 3)
    landmark_h = LandmarkHeuristic(graph, landmarks)

    mismatches = 0
    for start, goal in itertools.permutations(graph.nodes(), 2):
        d = dijkstra(graph, start, goal)
        a_sld = astar(graph, start, goal, graph.straight_line_distance)
        a_lmk = astar(graph, start, goal, landmark_h.estimate)
        if not (abs(d.cost - a_sld.cost) < 1e-9 and abs(d.cost - a_lmk.cost) < 1e-9):
            mismatches += 1
    check(mismatches == 0, f"All strategies agree on optimal cost across every node pair ({mismatches} mismatches)")


def test_landmark_heuristic_at_least_as_efficient_on_average():
    """
    Not guaranteed for every single query, but on average across many
    queries the landmark heuristic should expand no more nodes than the
    straight-line heuristic, since it is derived from real path structure.
    """
    graph = RoadGraph(COORDINATES, ROADS)
    landmarks = choose_landmarks(graph, 3)
    landmark_h = LandmarkHeuristic(graph, landmarks)

    sld_total, lmk_total = 0, 0
    for start, goal in itertools.permutations(graph.nodes(), 2):
        sld_total += astar(graph, start, goal, graph.straight_line_distance).nodes_expanded
        lmk_total += astar(graph, start, goal, landmark_h.estimate).nodes_expanded

    check(lmk_total <= sld_total,
          f"Landmark heuristic expands fewer total nodes on average "
          f"(straight-line: {sld_total}, landmark: {lmk_total})")


if __name__ == "__main__":
    test_graph_size()
    test_graph_connected()
    test_heuristic_admissibility()
    test_all_strategies_agree_on_optimal_cost()
    test_landmark_heuristic_at_least_as_efficient_on_average()
    print("\nAll tests passed.")