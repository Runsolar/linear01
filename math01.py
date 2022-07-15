def shape(a):
    res = [len(a)]
    b = a
    while type(b[0]) == list:
        res.append(len(b[0]))
        b = b[0]
    return res


def row(matrix, row_index):
    return matrix[row_index]


def column(matrix, column_index):
    return [matrix[row_index][column_index] for row_index in range(len(matrix))]


def cross(m1, m2):
    result_row_count = len(m1)
    result_column_count = len(m2[0])
    result = [
        [0] * result_column_count
        for _ in range(result_row_count)
    ]
    for result_row_index in range(result_row_count):
        for result_column_index in range(result_column_count):
            result[result_row_index][result_column_index] = sum(
                elem1 * elem2 for elem1, elem2 in zip(row(m1, result_row_index),
                                                      column(m2, result_column_index))
            )
    return result


def multiply(matrix, num):
    return [
        [num * matrix[row_index][column_index] for column_index in range(len(matrix[0]))]
        for row_index in range(len(matrix))
    ]


def dot(a, b):
    if type(a) != list and type(b) != list:
        return a*b
    if (type(a) == list and type(b) != list) or (type(b) == list and type(a) != list):  # xor
        if type(a) == list:
            if len(shape(a)) == 1:
                return [b * i for i in a]
            else:
                return multiply(a, b)
        if type(b) == list:
            if len(shape(b)) == 1:
                return [a * i for i in b]
            else:
                return multiply(b, a)
    if type(a) == list and type(b) == list:
        if len(shape(a)) == len(shape(b)) == 1:
            return sum(elem1 * elem2 for elem1, elem2 in zip(a, b))
        elif len(shape(a)) == 1 and len(shape(b)) > 1:
            if shape(a)[0] == shape(b)[0]:
                a = [[a[i]] for i in range(len(a))]
                return cross(a, b)
            else:
                raise ValueError
        elif len(shape(b)) == 1 and len(shape(a)) > 1:
            b = [[b[i]] for i in range(len(b))]
            return cross(a, b)
        elif len(shape(a)) == 2 and len(shape(b)) == 2:
            try:
                return cross(a, b)
            except TypeError:
                return cross(b, a)
            else:
                raise ValueError


def where(inpt, x, y):
    if type(inpt) == list:
        res = inpt.copy()
        for i in range(len(inpt)):
            for j in range(len(inpt[0])):
                if res[i][j] >= 0:
                    res[i][j] = x
                else:
                    res[i][j] = y
    elif inpt >= 0:
        res = x
    else:
        res = y
    return res


def summ(x, y):
    res = x.copy()
    if len(shape(x)) == 2:
        for i in range(len(x)):
            for j in range(len(x[0])):
                res[i][j] += y
    if len(shape(x)) == 1:
        for j in range(len(x)):
            res[j] += y
    return res
