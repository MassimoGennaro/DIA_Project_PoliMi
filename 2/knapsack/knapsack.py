import math
import numpy as np

# minus infinity value
m_inf = -math.inf


def knapsack_optimizer(table):
    """
    Given a budgets/sub-campaigns matrix, resolve the Knapsack problem optimization.
    :param table: budgets/sub-campaigns matrix, each cell contains the number of clicks

        EXAMPLE
        table = [
            [m_inf, 90, 100, 105, 110, m_inf, m_inf, m_inf],
            [0, 82, 90, 92, m_inf, m_inf, m_inf, m_inf],
            [0, 80, 83, 85, 86, m_inf, m_inf, m_inf],
            [m_inf, 90, 110, 115, 118, 120, m_inf, m_inf],
            [m_inf, 111, 130, 138, 142, 148, 155, m_inf]
        ]

    :return: list of budget-indexes correspondent to each sub-campaign
    """

    rows = len(table)
    cols = len(table[0])

    # optimization table
    opt_table = [[] for row in range(rows)]

    # pointer matrix
    opt_indexes = [[] for row in range(rows - 1)]

    # copy the value of the first sub-campaign in the optimization table
    for col in range(cols):
        opt_table[0].append(table[0][col])

    # optimization algorithm
    for row in range(1, rows):
        for col in range(cols):
            temp = []

            for col2 in range(col + 1):
                temp.append(table[row][col2] + opt_table[row - 1][col - col2])

            max_value = max(temp)

            # update tables of values and pointers
            opt_table[row].append(max_value)
            opt_indexes[row - 1].append(temp.index(max_value))

    # optimal cumulative number of clicks
    opt_value = max(opt_table[rows - 1])

    # pointer to the optimal budget column
    opt_col = opt_table[rows - 1].index(opt_value)

    # list of budget-pointers for each sub-campaign
    assignments = [0 for r in range(rows)]

    for row in range(rows - 1, 0, -1):
        # index of the optimal sub-campaign budget
        subc_col = opt_indexes[row - 1][opt_col]

        assignments[row] = subc_col
        opt_col -= subc_col

    # assign the index of the first sub-campaign
    assignments[0] = opt_col

    #return list(enumerate(budgets[a] for a in assignments))
    return list(enumerate(assignments))


class Knapsack2:
    def __init__(self, budget, values, arms=None):
        self.subcampaigns_number = len(values) + 1
        self.subcampaigns_list = list(range(len(values)))
        step = budget / (len(values[0]) - 1)

        self.budgets = [i * step for i in range(len(values[0]))]  # list(range(0, budget + step_1, step_1))
        if arms != None:
            self.budgets = arms

        self.budget_value = budget
        self.combinations = []

        # It is a matrix: values[subcamp_id][budget_id] = value
        self.values = values

    def optimize(self):
        all_zero = self.all_zero_result()
        if all_zero[0]:
            return all_zero[1]

        results = [[0] * len(self.budgets) for _ in range(self.subcampaigns_number)]
        # self.values = self.make_values_feasibles(self.values)
        temp_l = []

        # Perform knapsack optimization
        # self.knapsack_optimization(results, 1, 1, 1, len(self.budgets), len(self.budgets), (0, 0), temp_l)

        res, self.combinations = self.knapsack_optimization_2()

        # Compute the assignment from the knapsack optimization results
        return self.compute_assignment(self.combinations[-1][-1], self.combinations.copy())

    def make_values_feasibles(self, values):
        for row in values:
            row[0] = -math.inf
        return values

    def knapsack_optimization_2(self):
        numerical_results = [[0] * len(self.budgets)]
        indices = []

        for current_row in self.values:
            results_row = []
            indices_row = []
            for i in range(len(current_row)):
                best_value = 0
                best_indices = (i, 0)
                for old_index in range(0, i + 1):
                    index = i - old_index
                    if (current_row[index] + numerical_results[-1][old_index] > best_value):
                        best_value = current_row[index] + numerical_results[-1][old_index]
                        best_indices = (index, old_index)
                results_row.append(best_value)
                indices_row.append(best_indices)
            numerical_results.append(results_row)
            indices.append(indices_row)

        return (numerical_results, indices)

    def knapsack_optimization(self, results, ind_value_row, ind_value_col, ind_res_row, ind_res_col, ind_res_col_curr,
                              best_budget_comb, temp_l):
        # Base case: all the subcampaings have been evaluated
        if ind_res_row == self.subcampaigns_number - 1 and ind_res_col_curr == 0 and ind_res_col == 1:
            temp_l.append((0, 0))
            self.combinations.append(list(reversed(temp_l)))
            return results

        # Happens when the optimization step of the current subcampaing have been terminated
        elif ind_res_col_curr == 0 and ind_res_col == 1:
            temp_l.append((0, 0))
            self.combinations.append(list(reversed(temp_l)))
            return self.knapsack_optimization(results, ind_value_row + 1, 1, ind_res_row + 1,
                                              len(self.budgets), len(self.budgets), (0, 0), [])

        # Happens when we need to perform the optimization for the next budget in a given subcampaing
        elif ind_res_col_curr == 0:
            temp_l.append(best_budget_comb)
            return self.knapsack_optimization(results, ind_value_row, 1, ind_res_row, ind_res_col - 1, ind_res_col - 1,
                                              (ind_res_col - 2, 0), temp_l)

        # Perform the optimization for a given budget
        composed_value = self.values[ind_value_row - 1][ind_value_col - 1] + results[ind_res_row - 1][
            ind_res_col_curr - 1]

        if composed_value > results[ind_res_row][ind_res_col - 1]:
            best_budget_comb = (ind_value_col - 1, ind_res_col_curr - 1)
        results[ind_res_row][ind_res_col - 1] = max(composed_value, results[ind_res_row][ind_res_col - 1])

        return self.knapsack_optimization(results, ind_value_row, ind_value_col + 1, ind_res_row,
                                          ind_res_col, ind_res_col_curr - 1, best_budget_comb, temp_l)

    '''
        Returns a list of tuple of the following kind: (index of the sub-campaing, budget to assign)
    '''

    def compute_assignment(self, last_sub, combinations, assignment=None):

        if assignment is None:
            assignment = []

        assignment.append((len(combinations) - 1, self.budgets[last_sub[0]]))
        combinations.pop()

        if len(combinations) == 0:
            return assignment

        last_sub = combinations[-1][last_sub[1]]

        return self.compute_assignment(last_sub, combinations, assignment)

    def all_zero_result(self):
        final_sum = 0
        for subcamp in self.values:
            final_sum += sum(x > 0 for x in subcamp)

        if final_sum > 0:
            return (False, self.values)

        else:
            budget = self.budget_value
            res = []
            num = self.subcampaigns_number - 2
            # step = budget / (len(self.values[0]) - 1)
            # budgets = [i * step for i in range(len(self.values[0]))]  # list(range(0, budget + step_1, step_1))
            while budget > 0 and num >= 0:
                step = budget / (len(self.values[0]) - 1)
                budgets = [i * step for i in range(len(self.values[0]))]
                budget_ass = np.random.choice(budgets, replace=True)
                res.append((num, budget_ass))
                num -= 1
                budget -= budget_ass

            if len(res) <= self.subcampaigns_number - 2:
                while num >= 0:
                    res.append((num, 0))
                    num -= 1

            # I assign to the last campaign all the budget remaining for consistency reasons
            elif budget != 0:
                res[-1] = (res[-1][0], res[-1][1] + budget)

            return (True, res)
