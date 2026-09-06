# AI Assignment: Heuristic Search — 8-Puzzle Solver & Route Planner

**Author:** Devyansh Singh  
**Course:** BCA (Semester V/VI) — Artificial Intelligence  
**Submission Date:** September 2026

***

This repository has my two Python programs for the AI heuristic search assignment. Both use A* search but with different heuristic functions to show how a better heuristic can make search much faster without changing the answer.

```
.
├── puzzle/
│   ├── EightPuzzleSolver.py   # Q1: A* 8-puzzle solver
│   ├── testEightPuzzle.py     # Unit tests for Q1
│   └── sample_output.txt        # Demo run output
├── route/
│   ├── RoutePlanner.py         # Q2: A* route planner with landmarks
│   ├── testRoutePlanner.py    # Unit tests for Q2
│   └── sample_output.txt        # Demo run output
└── README.md
```

***

## Question 1 — 8-Puzzle Solver

### What I built

I made an A* solver for the classic 8-puzzle (3×3 board with tiles 1–8 and one blank). The goal is always:

```
1 2 3
4 5 6
7 8 _
```

The main file is `puzzle/eight_puzzle_solver.py`. Here’s what each part does:

- **State representation:** A state is just a 9-tuple like `(1,2,3,4,5,6,7,8,0)` where `0` is the blank. Index `i` maps to board position `(i//3, i%3)`.
- **`get_successors`:** Generates all legal moves (Up/Down/Left/Right) by sliding the blank.
- **`is_solvable`:** Checks solvability using inversion count parity. For 3×3, even inversions = solvable.
- **`generate_random_state`:** Creates random **solvable** start states by shuffling from the goal using legal moves. Can also flip parity to test the unsolvable case.
- **Heuristics:** I implemented three admissible heuristics:
  - `heuristic_misplaced_tiles` — counts tiles not in goal position
  - `heuristic_manhattan_distance` — sum of row+col distances for each tile
  - `heuristic_manhattan_plus_linear_conflict` — Manhattan + extra penalty for linear conflicts (still admissible but tighter)
- **`solve_astar`:** Standard A* with `f(n) = g(n) + h(n)`, using `heapq` for the priority queue and a `best_g` dict to track cheapest path to each state (handles re-discovered states correctly).
- **`display_solution` / `format_state`:** Prints the solution path step-by-step.
- **`compare_heuristics`:** Runs all three heuristics on the same puzzle and prints a table with nodes expanded/generated and time taken.

### How to run

```bash
cd puzzle
python3 eight_puzzle_solver.py      # runs the demo
python3 test_eight_puzzle.py        # runs unit tests
```

The demo uses a fixed random seed (42) so results are reproducible for grading. No user input needed. If you want to test your own puzzle, just import the module and call `solve_astar(your_state, GOAL_STATE, "manhattan")` from Python, or edit `demo_state` in `run_demo()`.

### Assumptions

- Goal state is fixed as `1 2 3 / 4 5 6 / 7 8 _`.
- Each move costs 1, so `g(n)` = number of moves so far.
- I check solvability before running A* — if a puzzle is unsolvable, it’s reported immediately instead of wasting time searching.
- All three heuristics are admissible, so A* always returns the shortest solution. I verified this in the tests — all three give the same solution cost on every puzzle.

### Results (from `sample_output.txt`, seed = 42)

| Puzzle | Heuristic | Moves | Nodes expanded | Nodes generated | Time (s) |
|--------|-----------|-------|----------------|-----------------|----------|
| Easy (5 shuffles) | Misplaced tiles | 2 | 3 | 5 | 0.00003 |
| Easy (5 shuffles) | Manhattan | 2 | 3 | 5 | 0.00003 |
| Easy (5 shuffles) | Manhattan + linear conflict | 2 | 3 | 5 | 0.00006 |
| Medium (12 shuffles) | Misplaced tiles | 4 | 5 | 10 | 0.00003 |
| Medium (12 shuffles) | Manhattan | 4 | 5 | 10 | 0.00010 |
| Medium (12 shuffles) | Manhattan + linear conflict | 4 | 5 | 10 | 0.00016 |
| Hard (25 shuffles) | Misplaced tiles | 7 | 14 | 27 | 0.00008 |
| Hard (25 shuffles) | Manhattan | 7 | 10 | 20 | 0.00011 |
| Hard (25 shuffles) | Manhattan + linear conflict | 7 | 10 | 20 | 0.00028 |
| Very Hard (35 shuffles) | Misplaced tiles | 19 | **2555** | 4100 | 0.02579 |
| Very Hard (35 shuffles) | Manhattan | 19 | **603** | 977 | 0.00472 |
| Very Hard (35 shuffles) | Manhattan + linear conflict | 19 | **390** | 630 | 0.00656 |

### What I observed

- All three heuristics found the same optimal solution length every time — confirms they’re all admissible.
- On easy puzzles, there’s barely any difference — the search space is small enough that any heuristic works fine.
- The real difference shows up on harder puzzles. On the hardest test case:
  - Misplaced tiles expanded **2555** nodes
  - Manhattan cut that to **603** (about 4.2× fewer)
  - Manhattan + linear conflict went down to **390** (about 6.5× fewer than misplaced tiles)
- This makes sense: Manhattan distance is always ≥ misplaced tiles (since every misplaced tile adds at least 1 to the Manhattan sum), so it’s more informed and prunes more of the search tree. Linear conflict adds an extra correction that’s still admissible but even tighter — it just takes a bit more time per node, which is worth it when node count gets large.
- The unsolvable check worked correctly — it rejected a parity-flipped state before running any search.

***

## Question 2 — Route Planner with Landmark Heuristics

### What I built

This is an A* route planner on a small graph of 14 locations (like Airport, Market, Stadium, etc.) with 25 undirected weighted roads. The main file is `route/route_planner.py`.

Key parts:

- **`COORDINATES`, `ROADS`:** 14 named locations with (x, y) coords and 25 weighted roads. Multiple paths exist between many pairs (e.g., via Market/Stadium/CityCenter).
- **`RoadGraph`:** Adjacency-list graph + `straight_line_distance` helper.
- **`dijkstra`:** Baseline Uniform Cost Search (h(n) = 0 everywhere).
- **`LandmarkHeuristic` + `choose_landmarks`:** Precomputes true shortest-path distances from a few perimeter “landmark” nodes to all other nodes. Then estimates remaining distance using the triangle inequality:  
  `h(n) = max_L |d(L,n) − d(L,goal)|`  
  (This is the ALT algorithm — A*, Landmarks, Triangle-inequality.)
- **`astar`:** Generic A* that takes any `h(node, goal)` function — used for both straight-line and landmark heuristics.
- **`run_query` / `print_comparison_table`:** Runs all three strategies on one start/goal pair and prints a table with cost, nodes expanded/generated, and time.
- **`interactive_mode`:** Lets you type in a start and goal location from the console.

### How to run

```bash
cd route
python3 route_planner.py              # runs 4 built-in demo queries
python3 route_planner.py --interactive # prompts for start/goal
python3 test_route_planner.py         # runs unit tests
```

### Assumptions

- Roads are undirected (same cost both ways).
- Coordinates are just illustrative planar positions — not real GPS — used only for the straight-line heuristic and to make the map geometry easy to reason about.
- **Every road’s weight is ≥ the straight-line distance between its endpoints.** I checked this for all 25 roads — it’s what guarantees the straight-line heuristic is admissible.
- Landmarks are chosen automatically as the graph’s extreme nodes (leftmost/rightmost/topmost) — a simple but standard ALT strategy.
- I verified correctness across all 182 ordered start/goal pairs in the test file, not just the 4 demo queries.

### Results (from `sample_output.txt`)

| Query | Strategy | Cost | Nodes expanded | Nodes generated |
|-------|----------|------|----------------|-----------------|
| Airport → IndustrialZone | Dijkstra | 23.0 | 13 | 16 |
| Airport → IndustrialZone | A* straight-line | 23.0 | 10 | 14 |
| Airport → IndustrialZone | A* landmark (ALT) | 23.0 | **5** | 11 |
| Suburbia → LakeView | Dijkstra | 20.0 | 14 | 15 |
| Suburbia → LakeView | A* straight-line | 20.0 | 9 | 16 |
| Suburbia → LakeView | A* landmark (ALT) | 20.0 | 9 | 13 |
| Hilltop → Harbor | Dijkstra | 23.0 | 14 | 16 |
| Hilltop → Harbor | A* straight-line | 23.0 | 12 | 15 |
| Hilltop → Harbor | A* landmark (ALT) | 23.0 | **9** | 15 |
| Museum → TechPark | Dijkstra | 15.0 | 11 | 15 |
| Museum → TechPark | A* straight-line | 15.0 | 6 | 12 |
| Museum → TechPark | A* landmark (ALT) | 15.0 | **4** | 9 |

Averaged across all 182 ordered pairs: straight-line A* expanded **737** total nodes vs. **655** for landmark A* about 11% fewer nodes on top of what straight-line already saves over Dijkstra.

### What I observed

- All three strategies always agree on the optimal path cost (verified on all 182 pairs) as expected since both heuristics are admissible.
- Dijkstra (h = 0) is the least efficient — it expands nodes purely by distance-so-far with no sense of direction toward the goal.
- Straight-line heuristic consistently reduces node expansions vs. Dijkstra because it gives A* a sense of geographic direction.
- Landmark-based (ALT) heuristic is usually even tighter because it’s derived from actual shortest-path distances over the road network, not just straight-line geometry. It “knows” about detours the road layout forces even when Euclidean distance can’t see them. This shows up clearly on `Airport → IndustrialZone` (5 vs. 10 nodes) and `Hilltop → Harbor` (9 vs. 12).
- It’s not strictly better on every single query (see `Suburbia → LakeView`, where both expanded 9 nodes) landmark quality depends on how well-placed the landmarks are relative to the specific query, which is why the comparison is made on average across many queries.

***

