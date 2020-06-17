#!/usr/bin/python3

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
import copy
import pandas as pd


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour.  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of solution, 
        time spent to find solution, number of permutations tried during search, the 
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
        This is the entry point for the greedy solver, which you must implement for 
        the group project (but it is probably a good idea to just do it for the branch-and
        bound project as a way to get your feet wet).  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found, the best
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    def greedy(self, time_allowance=60.0):
        pass

    def reducedCostMatrix(self, cost_matrix, ncities):
        lower_bound = 0
        row_cost_matrix = np.copy(cost_matrix)
        # Get the minimum index in each row
        min_row_indices = np.argmin(cost_matrix, axis=1)

        # Reduce each row to contain a zero
        for i in range(ncities):
            lower_bound += cost_matrix[i][min_row_indices[i]]
            for j in range(ncities):
                if not np.isinf(cost_matrix[i][j]):
                    row_cost_matrix[i][j] -= cost_matrix[i][min_row_indices[i]]

        # Get the minimum index in each column
        min_col_indices = np.argmin(row_cost_matrix, axis=0)
        reduced_cost_matrix = np.copy(row_cost_matrix)
        # Reduce each column that does not contain a zero to have a zero
        for i in range(ncities):
            if row_cost_matrix[min_col_indices[i]][i] != 0:
                lower_bound += row_cost_matrix[min_col_indices[i]][i]
                for j in range(ncities):
                    if not np.isinf(row_cost_matrix[j][i]):
                        reduced_cost_matrix[j][i] -= row_cost_matrix[min_col_indices[i]][i]
        return lower_bound, reduced_cost_matrix

    def partialPathSearch(self, stack, ncities, bssf, start_time, time_allowance):
        num_of_solutions = 0
        pruned_states = 0
        total_states = 0
        max_stack_size = 0
        best_sln = bssf

        while len(stack) != 0 and time.time() - start_time < time_allowance:
            if len(stack) > max_stack_size:
                max_stack_size = len(stack)
            active_state = stack.pop()
            parent_cost_matrix = np.copy(active_state[1])
            parent_partial_sln = copy.deepcopy(active_state[2])
            parent_lower_bound = int(active_state[0])
            """Prune the parent state if its lower bound is greater than best solution's lower bound"""
            if parent_lower_bound >= best_sln[0]:
                pruned_states += 1
                continue
            # Set i to be the last node in the partial solution
            i = parent_partial_sln[-1]
            breaker = False
            # Loop through for each possible state of unvisited cities (j=column)
            for j in range(len(parent_cost_matrix)):
                if time.time() - start_time < time_allowance and j != i and j not in parent_partial_sln:
                    child_lower_bound = int(active_state[0])
                    child_cost_matrix = np.copy(active_state[1])
                    child_partial_sln = copy.deepcopy(active_state[2])
                    child_partial_sln.append(j)

                    """Add the path cost to the lower bound"""
                    if child_cost_matrix[i][j] == float("inf"):
                        continue
                    else:
                        child_lower_bound += parent_cost_matrix[i][j]

                    """Set to infinity"""
                    # Set row i equal to infinity
                    for k in range(len(child_cost_matrix)):
                        child_cost_matrix[i][k] = float("inf")
                    # Set column j equal to infinity
                    for k in range(len(child_cost_matrix)):
                        child_cost_matrix[k][j] = float("inf")
                    # Set back edge (j,i) to infinity
                    child_cost_matrix[j][i] = float("inf")

                    """If the cost does not equal zero, reduce matrix to contain zeros in every row and column"""
                    if parent_cost_matrix[i][j] != 0:
                        """Set each row to zero"""
                        # Get the minimum index in each row
                        min_row_indices = np.argmin(child_cost_matrix, axis=1)
                        # For each row in min_row_indices (k=row)
                        for k in range(len(min_row_indices)):
                            if child_cost_matrix[k][min_row_indices[k]] != 0 and k != i:
                                # if the row minimum is infinity and not in the partial solution, not a viable state
                                if child_cost_matrix[k][min_row_indices[k]] == float("inf"):
                                    is_row_n_partial_sln = False
                                    # Determine if infinity row has already been visited (in the partial solution)
                                    for visited_node in parent_partial_sln:
                                        if visited_node == k:
                                            is_row_n_partial_sln = True
                                            break
                                    # a row of infinities was introduced making the node unreachable
                                    if not is_row_n_partial_sln:
                                        breaker = True
                                        break
                                # The minimum in a row is greater than 0
                                else:
                                    # Reduce the row to contain a zero
                                    child_lower_bound += child_cost_matrix[k][min_row_indices[k]]
                                    # For each column in the provided row of min_row_indices (l=column), update the value
                                    for l in range(len(child_cost_matrix)):
                                        if not np.isinf(child_cost_matrix[k][l]) and l != min_row_indices[k]:
                                            child_cost_matrix[k][l] -= child_cost_matrix[k][min_row_indices[k]]
                                    child_cost_matrix[k][min_row_indices[k]] -= child_cost_matrix[k][min_row_indices[k]]
                        # break out of second for loop to continue on to another state
                        if breaker:
                            breaker = False
                            continue
                        """Set each column to zero"""
                        # Get the minimum index in each column
                        min_col_indices = np.argmin(child_cost_matrix, axis=0)
                        # For each column in min_col_indices (k=col)
                        for k in range(len(min_col_indices)):
                            if child_cost_matrix[min_col_indices[k]][k] != 0 and k != j:
                                # if the column minimum is infinity and not in the partial solution, not a viable state
                                if child_cost_matrix[min_col_indices[k]][k] == float("inf"):
                                    is_col_in_partial_sln = False
                                    len_of_child_partial_sln = len(child_partial_sln)
                                    # Determine if infinity column has already been visited (in the partial solution)
                                    for h in range(1, len_of_child_partial_sln):
                                        if child_partial_sln[h] == k:
                                            is_col_in_partial_sln = True
                                            break
                                    # a row of infinities was introduced making the node unreachable
                                    if not is_col_in_partial_sln:
                                        breaker = True
                                        break
                                # the minimum in a column is greater than 0
                                else:
                                    # Reduce the row to contain a zero
                                    child_lower_bound += child_cost_matrix[min_col_indices[k]][k]
                                    # For each column in the provided row of min_row_indices (l=row), update the value
                                    for l in range(len(child_cost_matrix)):
                                        if not np.isinf(child_cost_matrix[l][k]) and l != min_col_indices[k]:
                                            child_cost_matrix[l][k] -= child_cost_matrix[min_col_indices[k]][k]
                                    child_cost_matrix[min_col_indices[k]][k] -= child_cost_matrix[min_col_indices[k]][k]
                        # break out of second for loop to continue on to another state
                        if breaker:
                            breaker = False
                            continue
                    """When in the reduced cost matrix form"""
                    total_states += 1
                    # If all cities have been visited
                    if len(child_partial_sln) == ncities:
                        # Add the cost from the last node to node 0
                        child_lower_bound += child_cost_matrix[child_partial_sln[-1]][0]
                        # if the lower bound is better than the best solution's lower bound, update the best solution
                        if child_lower_bound < best_sln[0]:
                            best_sln = (child_lower_bound, child_partial_sln)
                            num_of_solutions += 1
                    # If the partial solution is less than the best solution's lower bound, add to the stack
                    elif child_lower_bound < best_sln[0]:
                        stack.append((child_lower_bound, child_cost_matrix, child_partial_sln))
                    # Else prune the state
                    else:
                        pruned_states += 1
                    breaker = False
        return best_sln, num_of_solutions, pruned_states, total_states, max_stack_size

    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints: 
        max queue size, total number of states created, and number of pruned states.</returns> 
    '''
    def branchAndBound(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        cost_matrix = np.zeros((ncities, ncities), dtype=float)
        stack = []

        results_default = self.defaultRandomTour(60.0)
        partial_sln = []
        # Append index of city to partial solution
        for i in range(len(results_default['soln'].route)):
            partial_sln.append(results_default['soln'].route[i]._index)
        best_sln = (results_default['cost'], partial_sln)

        start_time = time.time()
        """Populate the cost_matrix"""
        for i in range(ncities):
            for j in range(ncities):
                cost_matrix[i][j] = cities[i].costTo(cities[j])

        """Obtain the reduced cost matrix"""
        lower_bound, reduced_cost_matrix = self.reducedCostMatrix(cost_matrix, ncities)
        """Put starting node on the stack"""
        partial_sln = [0]
        stack.append((lower_bound, reduced_cost_matrix, partial_sln))
        """Get the best solution so far"""
        best_sln, num_of_solutions, pruned_states, total_states, max_stack_size = self.partialPathSearch(stack,
            ncities, best_sln, start_time, time_allowance)

        route = []
        """Build the route of cities"""
        for i in range(ncities):
            route.append(cities[best_sln[1][i]])
        bssf = TSPSolution(route)
        if bssf.cost < np.inf:
            # Found a valid route
            foundTour = True
        end_time = time.time()
        """Add statistics to results and return results"""
        if results_default['cost'] == bssf.cost:
            results['cost'] = 0
        else:
            results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = num_of_solutions
        results['soln'] = bssf
        results['max'] = max_stack_size
        results['total'] = total_states
        results['pruned'] = pruned_states
        return results

    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found during search, the 
        best solution found.  You may use the other three field however you like.
        algorithm</returns> 
    '''

    def fancy(self, time_allowance=60.0):
        pass
