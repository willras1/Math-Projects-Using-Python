from time import perf_counter
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

'''The goal is to compute determinants using different algorithms and compare the time it takes to compute the determinant of a given matrix. 
Then plot the data to see the relationship of the dimension of a matrix and the time it takes to compute the determinant'''

def zero_matrix(rows, cols):
    '''return a zero matrix'''
    M = []
    while len(M) < rows:
        M.append([])
        while len(M[-1]) < cols:
            M[-1].append(0.00)
    return M


def identity_matrix(n):
    '''return the Identity matrix'''
    identity = zero_matrix(n, n)
    for i in range(n):
        identity[i][i] = 1
    return identity


def copy_matrix(M):
    '''return a copy of a given matrix'''
    rows = len(M)
    cols = len(M[0])
    MC = zero_matrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            MC[i][j] = M[i][j]
    return MC


def transpose(M):
    '''compute the transpose of a given matrix'''
    rows = len(M)
    cols = len(M[0])
    transpose = zero_matrix(cols, rows)
    for i in range(rows):
        for j in range(cols):
            transpose[i][j] = M[j][i]
    return transpose


def cf_determinant(A, total = 0): 
    '''compute determinant by cofactor expansion'''
    # store indices in list for row referencing 
    indices = list(range(len(A)))

    # start with 2x2 matrix
    if len(A) == 2 and len(A[0]) == 2:
        val = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        return val
    
    # find submatrix
    for fc in indices: 
        As = copy_matrix(A) 
        As = As[1:]
        height = len(As)
        for i in range(height):
            As[i] = As[i][0:fc] + As[i][fc + 1:]
        sign = (-1) ** (fc % 2)
        sub_determinant = cf_determinant(As)
        total += sign * A[0][fc] * sub_determinant
    return total


def determinant_ut(A): 
    '''compute determinant by creating upper traingular matrix, then finding the product of diagonal elements'''
    n = len(A)
    AM = copy_matrix(A)
    # row reduce until A is upper triangular, focus diagonal = fd
    for fd in range(n):
        if AM[fd][fd] == 0:
            AM[fd][fd] = 1.0e-18
        for i in range(fd+1, n):
            crScalar = AM[i][fd] / AM[fd][fd]
            for j in range(n):
                AM[i][j] = AM[i][j] - crScalar * AM[fd][j]
    # now A is upper triangular 
    product = 1.0
    for i in range(n):
        product *= AM[i][i]
    return product


def determinant_d(A):
    '''calculate determinant by adding product of diagonals'''
    n = len(A)
    AM = copy_matrix(A)
    for d in range(n):
        if AM[d][d] == 0:
            AM[d][d] = 1.0e-19
        for i in range(d+1, n):
            for j in range(n):
                p = AM[i][i] * AM[j][j]


def random_matrix(rows, cols):
    '''create a random matrix'''
    matrix = np.random.randint(10, size = (rows, cols))
    return matrix


def test(n):
    '''compare time to compute the determinant using cofactor expansion and numpy's determinat command'''
    A = random_matrix(n, n)

    start = perf_counter()
    cf_determinant(A)
    end = perf_counter()
    total_cf = end - start

    matrix = np.array(A)
    start1 = perf_counter()
    np.linalg.det(matrix)
    end1 = perf_counter()
    total_np = end1 - start1

    start2 = perf_counter()
    determinant_ut(A)
    end2 = perf_counter()
    total_ut = end2 - start2

    return [total_cf, total_np, n, total_ut]


def np_v_ut(n):
    '''compare time to compute the determinant using numpy's determinant command and using an upper triangular matrix'''
    A = random_matrix(n, n)

    matrix = np.array(A)
    start1 = perf_counter()
    np.linalg.det(matrix)
    end1 = perf_counter()
    total_np = end1 - start1

    start2 = perf_counter()
    determinant_ut(A)
    end2 = perf_counter()
    total_ut = end2 - start2

    return [total_np, n, total_ut]


def plot_np_v_ut(a,b):
    '''plot the time on the x-axis and the dimension of the matrix on the y-axis and compare numpy's determinant command and the upper triangular algorithm
    plot a trendline
    print a trend line for each line, as well as the r^2 value to see how well the line fits the data'''
    x = []
    y_np = []
    y_ut = []
    for i in range(a,b):
        values = np_v_ut(i)
        x.append(values[1])
        y_np.append(values[0])
        y_ut.append(values[2])

    figure1, ax = plt.subplots(figsize = (12,6))
    ax.plot(x, y_np, label='Numpy', color = 'green')
    ax.plot(x, y_ut, label='Upper Triangular', color = 'blue')
    plt.xlabel('dim(A)')
    plt.ylabel('Time (seconds)')
    z = np.polyfit(x, y_np, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), label='Numpy trend', color = 'lightgreen')
    u = np.polyfit(x, y_ut, 2)
    t = np.poly1d(u)
    plt.plot(x, t(x), label = 'Upper Tri trend', color = 'lightblue')
    plt.suptitle("numpy trendline: y=%.6fx+%.6f"%(z[0],z[1])) 
    plt.title('Upper triangular trendline: y=%.6fx^2+%.6fx+%.6f'%(u[0],u[1], u[2])) # y=%.6fx+%.6f'%(u[0],u[1]))
    plt.xlim([2, int(b)])
    max_np = max(y_np)
    max_ut = max(y_ut)
    max1 = max(max_np, max_ut) + (max(max_np, max_ut) * .05)
    plt.ylim([0, max1])
    plt.legend()

    actual_ut = y_ut
    predict_ut = list(t(x))
    corr_matrix_ut = np.corrcoef(actual_ut, predict_ut)
    corr_ut = corr_matrix_ut[0,1]
    r_square_ut = corr_ut**2
    print(f'coefficienct of determination of upper triangular trend line: {r_square_ut}')

    actual_np = y_np
    predict_np = list(p(x))
    corr_matrix_np = np.corrcoef(actual_np, predict_np)
    corr_np = corr_matrix_np[0,1]
    r_square_np = corr_np**2
    print(f'coefficienct of determination of Numpy trend line: {r_square_np}')

    print('\n')
    print('equation for numpy trendline: y=%.6fx+%.6f'%(z[0],z[1]))
    print('equation for upper triangular trendline: y=%.6fx^2+%.6fx+%.6f'%(u[0],u[1], u[2]))

    # plt.show()

plot_np_v_ut(2,200)


def plot_values(a,b):
    '''plot the time on the x-axis and the dimension of the matrix on the y-axis and compare time to compute the determinant using cofactor expansion and an upper triangular matrix
    plot a trendline
    print a trend line for each line, as well as the r^2 value to see how well the line fits the data'''
    x = []
    y_cf = []
    y_np = []
    y_ut = []
    for i in range(a,b):
        values = test(i)
        x.append(values[2])
        y_cf.append(values[0])
        y_np.append(values[1])
        y_ut.append(values[3])

    figure2, ax = plt.subplots(figsize = (12,6))
    ax.plot(x, y_cf, label='Cofactor expansion', color = 'red')
    # plt.plot(x, y_np, label='Numpy', color = 'green')
    ax.plot(x, y_ut, label='Upper Triangular', color = 'blue')
    z = np.polyfit(x, y_cf, 2)
    p = np.poly1d(z)
    u = np.polyfit(x, y_ut, 2)
    t = np.poly1d(u)
    plt.plot(x, p(x), label='Cofactor trend', color = 'lightcoral')
    plt.xlabel('dim(A)')
    plt.ylabel('Time (seconds)')
    plt.plot(x, t(x), label = 'Upper Tri trend', color = 'lightblue')
    plt.suptitle("cofactor trendline: y=%.6fx^2+%.6fx+%.6f"%(z[0],z[1], z[2])) 
    plt.title('Upper triangular trendline: y=%.6fx^2+%.6fx+%.6f'%(u[0],u[1], u[2]))
    plt.xlim([2, int(b)])
    max_np = max(y_cf)
    max_ut = max(y_ut)
    max1 = max(max_np, max_ut) + (max(max_np, max_ut) * .1)
    plt.ylim([0, max1])

    actual_ut = y_ut
    predict_ut = list(t(x))
    corr_matrix_ut = np.corrcoef(actual_ut, predict_ut)
    corr_ut = corr_matrix_ut[0,1]
    r_square_ut = corr_ut**2
    print('\n')
    print(f'coefficienct of determination of upper triangular trend line: {r_square_ut}')

    actual_cf = y_cf
    predict_cf = list(p(x))
    corr_matrix_cf = np.corrcoef(actual_cf, predict_cf)
    corr_np = corr_matrix_cf[0,1]
    r_square_cf = corr_np**2
    print(f'coefficienct of determination of cofactor trend line: {r_square_cf}')

    print('\n')
    print('equation for cofactor trendline: y=%.6fx^2+%.6fx+%.6f'%(z[0],z[1], z[2]))
    print('equation for upper triangular trendline: y=%.6fx^2+%.6fx+%.6f'%(u[0],u[1], u[2]))

    plt.legend()
    plt.show()


plot_values(2, 12) # (2,13) is highest it can go for finding the determinant with cofactor expansion


# overall findings: cofactor expansion is by far the slowest algorithm. Computing a 13x13 matrix is generally the limit of what the computer can handle, anything larger takes too long. 
# Computing a 15x15 matrix will take over 8 minutes, 20x20 will take over 18 minutes, whereas using an upper triangular matrix will take about 0.01 seconds to compute a 20x20 matrix. 
# Using upper triangular matrices, a 200x200 matrix can be computed in under 1.5 seconds, whereas using cofactor expansion will take over 50 hours. Cofactor expansion is computationally infeasible 
# Numpy's command computes the determinant the fastest, by far