"""
eight_puzzle_solver.py
-----------------------
Intelligent 8-Puzzle Solver using A* Search.

A 3x3 sliding puzzle is represented as a flat tuple of 9 integers, where 0
marks the blank tile. The board is laid out row-major:

        0 1 2
        3 4 5
        6 7 8

The program can:
    * validate and generate puzzle states,
    * check whether a state is solvable,
    * generate legal successor states,
    * run A* search with a choice of heuristic function,
    * compare the performance (nodes expanded, time, path length) of
      different heuristics on the same puzzle instances.

Author: Devyansh Singh
"""

import heapq
import itertools
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

State = Tuple[int, ...]

GOAL_STATE: State = (1, 2, 3, 4, 5, 6, 7, 8, 0)
BOARD_SIZE = 3


# --------------------------------------------------------------------------
# Puzzle environment
# --------------------------------------------------------------------------

def index_to_rc(index: int) -> Tuple[int, int]:
    """Convert a flat index (0-8) to (row, col) on the 3x3 board."""
    return divmod(index, BOARD_SIZE)


def rc_to_index(row: int, col: int) -> int:
    """Convert a (row, col) pair back to a flat index."""
    return row * BOARD_SIZE + col


def is_valid_state(state: State) -> bool:
    """A state is valid if it is a permutation of 0..8 (one blank, tiles 1-8)."""
    return sorted(state) == list(range(9))


def get_blank_position(state: State) -> int:
    """Return the flat index of the blank (0) tile."""
    return state.index(0)


def get_successors(state: State) -> List[Tuple[State, str]]:
    """
    Generate all legal successor states by sliding the blank
    Up, Down, Left or Right, along with a human-readable move label.
    """
    successors = []
    blank_idx = get_blank_position(state)
    row, col = index_to_rc(blank_idx)

    # (row_delta, col_delta, move_name) -- move name describes the blank's
    # direction of travel, which is the intuitive way puzzles are narrated.
    moves = [(-1, 0, "Up"), (1, 0, "Down"), (0, -1, "Left"), (0, 1, "Right")]

    for d_row, d_col, name in moves:
        new_row, new_col = row + d_row, col + d_col
        if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
            new_idx = rc_to_index(new_row, new_col)
            new_state = list(state)
            new_state[blank_idx], new_state[new_idx] = new_state[new_idx], new_state[blank_idx]
            successors.append((tuple(new_state), name))

    return successors


def count_inversions(state: State) -> int:
    """
    Count inversions in the state ignoring the blank tile. Two tiles
    (a, b) form an inversion if a appears before b but a > b.
    """
    tiles = [t for t in state if t != 0]
    inversions = 0
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            if tiles[i] > tiles[j]:
                inversions += 1
    return inversions


def is_solvable(state: State) -> bool:
    """
    For a 3x3 (odd width) puzzle, a configuration is solvable if and only
    if the number of inversions (ignoring the blank) is even. This is the
    standard parity argument for sliding-tile puzzles.
    """
    return count_inversions(state) % 2 == 0


def generate_random_state(solvable: bool = True, shuffle_moves: int = 200) -> State:
    """
    Generate a random start state. By default we shuffle the goal state
    with `shuffle_moves` legal moves, which guarantees solvability by
    construction (much cheaper than rejection-sampling permutations).
    If solvable=False is requested, one extra tile swap is applied to
    flip parity, producing a deliberately unsolvable puzzle for testing.
    """
    state = GOAL_STATE
    for _ in range(shuffle_moves):
        state = random.choice(get_successors(state))[0]

    if not solvable:
        # Swap two non-blank tiles to flip parity -> guaranteed unsolvable.
        lst = list(state)
        i, j = 0, 1
        if lst[i] == 0:
            i, j = 1, 2
        lst[i], lst[j] = lst[j], lst[i]
        state = tuple(lst)

    return state


# --------------------------------------------------------------------------
# Heuristic functions -- each takes (state, goal) and returns an estimate
# of the number of moves remaining. All three are admissible (never
# overestimate), which is what guarantees A* returns an optimal solution.
# --------------------------------------------------------------------------

def heuristic_misplaced_tiles(state: State, goal: State = GOAL_STATE) -> int:
    """Number of tiles (excluding blank) not in their goal position."""
    return sum(1 for i in range(9) if state[i] != 0 and state[i] != goal[i])


def heuristic_manhattan_distance(state: State, goal: State = GOAL_STATE) -> int:
    """
    Sum, over every non-blank tile, of the Manhattan (grid) distance
    between its current position and its goal position. Strictly
    dominates misplaced-tiles (always >= it), so it is at least as
    informed, which is why we expect it to expand fewer nodes.
    """
    goal_pos = {tile: index_to_rc(i) for i, tile in enumerate(goal)}
    total = 0
    for i, tile in enumerate(state):
        if tile == 0:
            continue
        cur_row, cur_col = index_to_rc(i)
        goal_row, goal_col = goal_pos[tile]
        total += abs(cur_row - goal_row) + abs(cur_col - goal_col)
    return total


def heuristic_manhattan_plus_linear_conflict(state: State, goal: State = GOAL_STATE) -> int:
    """
    Manhattan distance augmented with the "linear conflict" correction:
    if two tiles are in their correct row (or column) but in reversed
    order relative to each other, they must each move out of the way and
    back in, costing at least 2 extra moves per conflicting pair. This
    stays admissible while being a tighter (more informed) bound than
    plain Manhattan distance.
    """
    base = heuristic_manhattan_distance(state, goal)
    goal_pos = {tile: index_to_rc(i) for i, tile in enumerate(goal)}
    conflicts = 0

    # Row conflicts
    for row in range(BOARD_SIZE):
        tiles_in_row = [state[rc_to_index(row, c)] for c in range(BOARD_SIZE)]
        for a in range(len(tiles_in_row)):
            for b in range(a + 1, len(tiles_in_row)):
                t1, t2 = tiles_in_row[a], tiles_in_row[b]
                if t1 == 0 or t2 == 0:
                    continue
                if goal_pos[t1][0] == row and goal_pos[t2][0] == row:
                    if goal_pos[t1][1] > goal_pos[t2][1]:
                        conflicts += 1

    # Column conflicts
    for col in range(BOARD_SIZE):
        tiles_in_col = [state[rc_to_index(r, col)] for r in range(BOARD_SIZE)]
        for a in range(len(tiles_in_col)):
            for b in range(a + 1, len(tiles_in_col)):
                t1, t2 = tiles_in_col[a], tiles_in_col[b]
                if t1 == 0 or t2 == 0:
                    continue
                if goal_pos[t1][1] == col and goal_pos[t2][1] == col:
                    if goal_pos[t1][0] > goal_pos[t2][0]:
                        conflicts += 1

    return base + 2 * conflicts


HEURISTICS: Dict[str, Callable[[State, State], int]] = {
    "misplaced": heuristic_misplaced_tiles,
    "manhattan": heuristic_manhattan_distance,
    "manhattan_linear_conflict": heuristic_manhattan_plus_linear_conflict,
}


# --------------------------------------------------------------------------
# A* search
# --------------------------------------------------------------------------

@dataclass(order=True)
class _PQEntry:
    """Priority-queue entry ordered by f = g + h, tie-broken by insertion order."""
    priority: int
    count: int
    state: State = field(compare=False)
    g: int = field(compare=False)
    path: Tuple[str, ...] = field(compare=False)


@dataclass
class SearchResult:
    solved: bool
    path_moves: List[str]
    path_states: List[State]
    cost: int
    nodes_expanded: int
    nodes_generated: int
    max_frontier_size: int
    time_taken: float


def solve_astar(start: State, goal: State, heuristic_name: str = "manhattan") -> SearchResult:
    """
    Run A* search from `start` to `goal` using the named heuristic.
    f(n) = g(n) + h(n), where g(n) is the number of moves taken so far
    and h(n) is the heuristic estimate of moves remaining.
    """
    if heuristic_name not in HEURISTICS:
        raise ValueError(f"Unknown heuristic '{heuristic_name}'. Options: {list(HEURISTICS)}")
    h_func = HEURISTICS[heuristic_name]

    start_time = time.perf_counter()

    if not is_solvable(start):
        return SearchResult(False, [], [], -1, 0, 0, 0, time.perf_counter() - start_time)

    counter = itertools.count()  # tie-breaker for equal-priority entries
    frontier: List[_PQEntry] = []
    start_h = h_func(start, goal)
    heapq.heappush(frontier, _PQEntry(start_h, next(counter), start, 0, ()))

    best_g: Dict[State, int] = {start: 0}
    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        entry = heapq.heappop(frontier)

        # Skip stale entries (a cheaper path to this state was already found)
        if entry.g > best_g.get(entry.state, float("inf")):
            continue

        nodes_expanded += 1

        if entry.state == goal:
            path_states = [start]
            state = start
            for move in entry.path:
                state = [s for s, m in get_successors(state) if m == move][0]
                path_states.append(state)
            return SearchResult(
                True, list(entry.path), path_states, entry.g,
                nodes_expanded, nodes_generated, max_frontier_size,
                time.perf_counter() - start_time,
            )

        for succ_state, move in get_successors(entry.state):
            new_g = entry.g + 1
            if new_g < best_g.get(succ_state, float("inf")):
                best_g[succ_state] = new_g
                nodes_generated += 1
                f = new_g + h_func(succ_state, goal)
                heapq.heappush(
                    frontier,
                    _PQEntry(f, next(counter), succ_state, new_g, entry.path + (move,)),
                )

    return SearchResult(False, [], [], -1, nodes_expanded, nodes_generated, max_frontier_size,
                         time.perf_counter() - start_time)


# --------------------------------------------------------------------------
# Display helpers
# --------------------------------------------------------------------------

def format_state(state: State) -> str:
    """Pretty-print a state as a 3x3 grid, using '_' for the blank."""
    rows = []
    for r in range(BOARD_SIZE):
        row_tiles = state[r * BOARD_SIZE:(r + 1) * BOARD_SIZE]
        rows.append(" ".join(str(t) if t != 0 else "_" for t in row_tiles))
    return "\n".join(rows)


def display_solution(result: SearchResult) -> None:
    if not result.solved:
        print("No solution (puzzle is unsolvable or search failed).")
        return
    print(f"Solved in {result.cost} moves: {' -> '.join(result.path_moves)}\n")
    for i, state in enumerate(result.path_states):
        label = "Start" if i == 0 else (f"Goal (move {i}: {result.path_moves[i-1]})"
                                         if i == len(result.path_states) - 1
                                         else f"Move {i}: {result.path_moves[i-1]}")
        print(f"-- {label} --")
        print(format_state(state))
        print()


# --------------------------------------------------------------------------
# Heuristic comparison experiment
# --------------------------------------------------------------------------

def compare_heuristics(start: State, goal: State = GOAL_STATE) -> Dict[str, SearchResult]:
    """Run A* on the same puzzle instance with every registered heuristic."""
    results = {}
    for name in HEURISTICS:
        results[name] = solve_astar(start, goal, name)
    return results


def print_comparison_table(all_results: Dict[str, Dict[str, SearchResult]]) -> None:
    """
    Print a comparison table across multiple puzzle instances and
    heuristics: solution length, nodes expanded, nodes generated, time.
    """
    header = f"{'Puzzle':<16}{'Heuristic':<28}{'Solved':<8}{'Moves':<8}{'Expanded':<11}{'Generated':<11}{'Time(s)':<10}"
    print(header)
    print("-" * len(header))
    for puzzle_label, heur_results in all_results.items():
        for heur_name, res in heur_results.items():
            print(f"{puzzle_label:<16}{heur_name:<28}{str(res.solved):<8}"
                  f"{res.cost if res.solved else '-':<8}{res.nodes_expanded:<11}"
                  f"{res.nodes_generated:<11}{res.time_taken:<10.5f}")


# --------------------------------------------------------------------------
# Demo / manual test driver
# --------------------------------------------------------------------------

def run_demo() -> None:
    random.seed(42)  # reproducible test instances

    print("=" * 70)
    print("PART A DEMO: single puzzle solved and displayed step by step")
    print("=" * 70)
    demo_state: State = (1, 2, 3, 4, 0, 6, 7, 5, 8)  # 2 moves from goal
    print("Initial state:")
    print(format_state(demo_state))
    print(f"\nSolvable? {is_solvable(demo_state)}\n")
    result = solve_astar(demo_state, GOAL_STATE, "manhattan")
    display_solution(result)

    print("=" * 70)
    print("UNSOLVABLE PUZZLE CHECK")
    print("=" * 70)
    unsolvable_state = generate_random_state(solvable=False, shuffle_moves=50)
    print("State:")
    print(format_state(unsolvable_state))
    print(f"Solvable? {is_solvable(unsolvable_state)}")
    res = solve_astar(unsolvable_state, GOAL_STATE, "manhattan")
    print(f"Search result -> solved: {res.solved}\n")

    print("=" * 70)
    print("PART B: heuristic comparison across several puzzle instances")
    print("=" * 70)
    test_cases = {
        "Easy (5)": (1, 2, 3, 4, 5, 6, 0, 7, 8),
        "Medium (12)": generate_random_state(shuffle_moves=12),
        "Hard (25)": generate_random_state(shuffle_moves=25),
        "Very Hard (35)": generate_random_state(shuffle_moves=35),
    }

    all_results = {}
    for label, state in test_cases.items():
        print(f"\n{label} start state:")
        print(format_state(state))
        all_results[label] = compare_heuristics(state)

    print()
    print_comparison_table(all_results)


if __name__ == "__main__":
    run_demo()