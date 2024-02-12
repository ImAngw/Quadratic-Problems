import my_functions as mf
import numpy as np
import time
import matplotlib.pyplot as plt


# If you want to collect the performances of the optimum finder you can execute these lines

dimension = 1000

new = mf.QuadraticProblem(dimension)

# First random initialization of x
x = np.random.randn(dimension, 1)

# Equality constraint:  x_1 + x_2 + ... + x_n = 0
x[dimension - 1][0] = -(np.sum(x) - x[dimension - 1][0])

# Different selection possibilities
valid_inputs = ["r", "rp", "g", "eg1", "eg2", "ra"]
labels = ['Random', 'Random Li', 'Greedy', 'Greedy Li (GS-1)', 'Greedy Li (GS-q)', 'Greedy Li (Ratio)']

k = 0

print('DIMENSION: ', dimension)
print('CYCLES: ', new.MAX_ITERATION)
print()
for choice in valid_inputs:
    start = time.time()
    x_plot, y_plot = new.solver(choice, x)
    stop = time.time()
    plt.plot(x_plot, y_plot, label=labels[k])
    print(labels[k], 'done in ', round(stop - start, 3), 's!')
    k += 1

plt.title(f'Very-different Li (dimension={dimension})')
plt.xlabel('Iterations')
plt.ylabel('f(x)')
plt.yscale('log')
plt.legend()
plt.show()


# If you want to collect times you can execute these lines

'''
# Different size of matrix A
different_size = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# Different selection possibilities
valid_inputs = ["r", "rp", "g", "eg1", "eg2", "ra"]
labels = ['Random', 'Random Li', 'Greedy', 'Greedy Li (GS-1)', 'Greedy Li (GS-q)', 'Greedy Li (Ratio)']


time_points = {}
k = 0


for size in different_size:
    new = mf.QuadraticProblem(size)

    # First random initialization of x
    x = np.random.randn(size, 1)
    x[size - 1][0] = -(np.sum(x) - x[size - 1][0])

    for choice in valid_inputs:
        start = time.time()
        x_plot, y_plot = new.solver(choice, x)
        stop = time.time()

        if choice not in time_points.keys():
            time_points[choice] = [stop - start]
        else:
            time_points[choice].append(stop - start)

    print('Size ', size, 'completed!')

for choice in valid_inputs:
    plt.plot(different_size, time_points[choice], label=labels[k])
    k += 1


plt.title('Time for 1000 cycles')
plt.xlabel('Dimension of (A^T)*A')
plt.ylabel('Time [s]')
plt.yscale('log')
plt.legend()
plt.show()
'''
