# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 22:34:46 2022

@author: chaim
"""

import numpy as np
import sys
from dataloader import DataLoader
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
import time

def ev_allocator(costs: np.ndarray, cap : np.ndarray) -> None:
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')
    n_evs = costs.shape[0]
    n_cs = len(cap)
    # x[i, j] is an array of 0-1 variables, which will be 1
    # if ev i is assigned to cs j.
    x = {}
    for i in range(n_evs):
       for j in range(n_cs):
          x[i, j] = solver.IntVar(0, 1, '')
    # Adding constraints
    # Each ev is assigned to at most 1 cs.
    for i in range(n_evs):
       solver.Add(solver.Sum([x[i, j] for j in range(n_cs)]) == 1)
    # Each cs cap should not be exceeded.
    for j in range(n_cs):
       solver.Add(solver.Sum([x[i, j] for i in range(n_evs)]) <= cap[j])
       
    objective_terms = []
    for i in range(n_evs):
       for j in range(n_cs):
          objective_terms.append(costs[i][j] * x[i, j])
    solver.Minimize(solver.Sum(objective_terms))
    # invoke the solver
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
       for i in range(n_evs):
          for j in range(n_cs):
             # Test if x[i,j] is 1 (with tolerance for floating point arithmetic).
             if x[i, j].solution_value() > 0.5:
                print('Worker %d assigned to task %d.  Cost = %d' %(i, j, costs[i][j]))
       print('Total cost = ', solver.Objective().Value(), '\n')
       
def ev_allocator_cp_sat(costs: np.ndarray, cap: np.ndarray) -> None:
    # Model
    model = cp_model.CpModel()

    # Variables
    n_evs = costs.shape[0]
    n_cs = costs.shape[1]
    x = []
    for i in range(n_evs):
        t = []
        for j in range(n_cs):
            t.append(model.NewBoolVar(f'x[{i},{j}]'))
        x.append(t)

    # Constraints
    # Each ev is assigned to at most one cs.
    for i in range(n_evs):
        model.Add(sum(x[i][j] for j in range(n_cs)) == 1)

    # Each cs capacity should not be exceeded.
    for j in range(n_cs):
        model.Add(sum(x[i][j] for i in range(n_evs)) <= cap[j])

    # Objective
    objective_terms = []
    for i in range(n_evs):
        for j in range(n_cs):
            objective_terms.append(costs[i][j] * x[i][j])
    model.Minimize(sum(objective_terms))

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # Print solution.
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for i in range(n_evs):
            for j in range(n_cs):
                if solver.BooleanValue(x[i][j]):
                    print(
                        f'Worker {i} assigned to task {j} Cost = {costs[i][j]}')
        print(f'Total cost = {solver.ObjectiveValue()}')
    else:
        print('No solution found.')
        

def update_costs(costs: np.ndarray, charging_cost: np.ndarray, n_cars: int, n_parking: int) -> np.ndarray:
    # charging_cost = np.random.randint(30, 960, (n_cars, ))
    # print("charging cosgt = ", charging_cost[:10])
    total_cost = np.zeros((n_cars, n_parking))
    for i in range(n_cars):
        for j in range(n_parking):
            total_cost[i, j] = costs[i, j] + charging_cost[i]
    return total_cost


def test_simulated_data_updated(filename: str, n_cars, n_parking, COST_UPDATED: bool=True):
    data_loader = DataLoader(filename, n_cars, n_parking)
    src, trg, cap = data_loader.load_src_trg_cap()
    demand, costs = data_loader.load_demand_cost()
    if COST_UPDATED:
        cost_folder = "D:\workspace\dev\smart_parking\PAP\Feasible\9000\900030_1.txt"
        charging_cost = data_loader.load_charging_time(cost_folder)
        # print(charging_cost.shape)
        new_costs = update_costs(costs, charging_cost, n_cars, n_parking)
        return demand, new_costs
    return demand, costs
    
if __name__ == '__main__':
    datafile  = sys.argv[1]
    # data_loader = DataLoader(datafile, 1000, 10)
    # cap, costs = data_loader.load_demand_cost()
    cap, costs = test_simulated_data_updated(datafile, 9000, 30)
    ## ev_allocator(costs, cap)
    start = time.time()
    ev_allocator_cp_sat(costs, cap)
    end = time.time()
    print("Solving took {} seconds.".format(end-start))