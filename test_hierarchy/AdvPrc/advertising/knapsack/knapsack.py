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

    # set negative values to -inf
    for row in range(rows):
        for col in range(cols):
            if table[row][col] < 0:
                table[row][col] = m_inf

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

    # return list(enumerate(assignments))
    return assignments
