# ev-charging-assignment-exact
Assign electric vehicles (EVs) to charging stations to minimize total travel and waiting time while respecting each station’s capacity constraints. Solutions use exact methods (OR-Tools).

Lines 1–2: numpy and sys for numerical operations and command-line arguments.
Line 3: DataLoader (custom module) to load data from files.
Line 4: OR-Tools MIP solver (pywraplp).
Line 5: OR-Tools CP-SAT solver (cp\_model).
Line 6: time to measure execution duration.

\## 7–50: Exact method using MIP (ev\_allocator)
Lines 8–10: Initialize solver, count vehicles and stations.
Lines 14–17: Create binary decision variables x\[i,j] for assignments.
Lines 20–21: Constraint 1 — each EV assigned exactly once.
Lines 24–25: Constraint 2 — station capacities are respected.
Lines 28–29: Define the objective: minimize total assignment cost.
Lines 32–40: Solve the problem, print each assignment and total cost.

\## 52–102: Exact method using CP-SAT (ev\_allocator\_cp\_sat)
Lines 52–55: Initialize CP-SAT model and counts.
Line 57: Create Boolean variables for each EV-station assignment.
Lines 59–60: Each EV is assigned to exactly one station.
Lines 62–63: Each station capacity constraint.
Line 65: Minimize total assignment cost.
Lines 68–78: Solve CP-SAT, print assignments and total cost.

\## 104–128: Cost update function
Lines 104–112: Adds charging time/cost to base assignment costs.
\## 130–146: Data loading
Lines 130–139: Load demand, cost, capacities, and optionally update costs with charging times.
\## 148–156: Main execution
Lines 148–156: Read data file path, load data, run solver, measure execution time, print results.

\## Summary
Lines 7–50: MIP solver exact method.
Lines 52–102: CP-SAT solver exact method.
Lines 104–128: Update cost function.
Lines 130–146: Load and process data.
Lines 148–156: Execute script, print assignments, total cost, execution time.
