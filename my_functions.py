import numpy as np


class QuadraticProblem:

    MAX_ITERATION = 1000

    def __init__(self, dimension, a=None, b=None):
        '''
        Initialize the quadratic problem:
        y = 1/2 * || ax - b ||^2

        :param dimension: Vector dimension
        :param a: Matrix associated with the quadratic problem
        :param b: Bias vector

        If a and b are not initialized, it returns a random initialized version.
        '''

        self.dimension = dimension

        # Matrix 'a' initialization
        if a is None:
            self.a = np.random.randn(dimension, dimension)
        else:
            self.a = a

        # You need this operation if you want to obtain very different Lipshitz constants Li
        for k in range(self.dimension):
            my = np.random.randn(1, 1)
            for t in range(self.dimension):
                self.a[t][k] *= my[0][0]

        # Squared matrix a^T * a
        self.q = self.a.T @ self.a

        # Bias initialization
        # Bias vector is obtained by: b = a * x_true
        # where x_true is the true optimum of the problem
        # with the restriction: x1 + x2 + ....... + xn = 0
        if b is None:
            x_true = np.random.randn(dimension, 1)
            #x_true[self.dimension - 1][0] = -(np.sum(x_true) - x_true[self.dimension - 1][0])
            self.x_true = x_true
            self.b = np.dot(self.a, self.x_true)
        else:
            self.x_true = None
            self.b = b

    def function(self, x):
        '''
        It returns the value of the function evaluated in x
        :param x: position vector
        :return: y = 1/2 * || ax - b ||^2
        '''
        return 0.5 * np.power(np.linalg.norm(np.dot(self.a, x) - self.b), 2)

    def grad_solver(self, x, i):
        '''
        It returns the derivate y' evaluated in x
        :param x: position vector
        :param i: index you want to derive
        :return: y' = Ri * x - Ci * b
                where:
                    Ri = i-th    row of    q = a^T * a
                    Ci = i-th column of    a
        '''

        d = 0
        for k in range(self.dimension):
            d += self.q[i][k] * x[k][0] - self.a[k][i] * self.b[k][0]
        return d

    def random_selection(self):
        '''
        It returns random indexes i and j in range [0, dimension]
        :return: i, j
        '''

        i = np.random.randint(low=0, high=self.dimension)
        j = np.random.randint(low=0, high=self.dimension)

        while i == j:
            j = np.random.randint(low=0, high=self.dimension)

        return i, j

    @staticmethod
    def greedy_selection(gradient):
        '''
        It returns i and j index in range [0, dimension] where:
        i = index of max gradient component
        j = index of min gradient component

        :param gradient: gradient vector
        :return: i, j
        '''
        i = np.argmax(gradient)
        j = np.argmin(gradient)

        return i, j

    def random_prob_selection(self, lip_const):
        '''
        It returns i and j indexes in range [0, dimension].
        Each k index has a probability that is proportional to its Li contant

        :param lip_const: array of lipshitz constants Li
        :return: i, j
        '''

        tot = np.sum(lip_const)
        prob = lip_const / tot
        values = np.arange(0, self.dimension)

        i = np.random.choice(values, p=prob)
        j = np.random.choice(values, p=prob)

        while i == j:
            j = np.random.choice(values, p=prob)

        return i, j

    @staticmethod
    def ratio_approximation_selection(gradient, lip_const):
        '''
        It returns i and j indexes in range [0, dimension], where:

        i = arg max {f_k'(x) / sqrt (L_k)}
                k

        i = arg min {f_k'(x) / sqrt (L_k)}
                k

        :param gradient: gradient array
        :param lip_const: lipshitz constants array
        :return: i, j
        '''

        weighted_gradient = gradient / np.sqrt(lip_const)
        i = np.argmax(weighted_gradient)
        j = np.argmin(weighted_gradient)

        return i, j

    def gradient_update(self, gradient, i, j, d):
        '''
        It updates gradient of a quadratic function

        :param gradient: gradient array
        :param i: first index to update
        :param j: second index to update
        :param d: step value
        :return: gradient
        '''

        for k in range(self.dimension):
            gradient[k] -= self.q[k][i] * d
            gradient[k] += self.q[k][j] * d

        return gradient

    def lips_matrix(self, metric):
        '''
        If metric == 1, it returns:

        | sqrt(L_1) + sqrt(L_1)     sqrt(L_1) + sqrt(L_2)  ..........  sqrt(L_1) + sqrt(L_n) |
        |           .                         .                                              |
        |           .                                           .                            |
        |           .                                                            .           |
        | sqrt(L_n) + sqrt(L_1)     sqrt(L_n) + sqrt(L_2)  ..........  sqrt(L_n) + sqrt(L_n) |



        If metric == 2, it returns:

        | sqrt(L_1 + L_1)     sqrt(L_1 + L_2)  ..........  sqrt(L_1 + L_n) |
        |         .                   .                                    |
        |         .                                 .                      |
        |         .                                                .       |
        | sqrt(L_n + L_1)     sqrt(L_n + L_2)  ..........  sqrt(L_n + L_n) |


        :param metric: metric you want to use (allowed values are 1 or 2)
                        1:  sqrt(L_i) + sqrt(L_j)
                        2: sqrt(L_i + L_j)

        :return: None           if metric is not allowed
                 lip_matrix     if metric is allowed
        '''

        allowed_metrics = [1, 2]
        if metric not in allowed_metrics:
            print("FATAL ERROR: not allowed choice!")
            return None

        lip_const = np.diagonal(self.q)
        if metric == 1:
            lip_const = np.sqrt(lip_const)

        same_row = np.tile(lip_const, (self.dimension, 1))
        col = lip_const.reshape((-1, 1))
        same_col = np.repeat(col, self.dimension, axis=1)

        lip_matrix = same_col + same_row

        if metric == 2:
            lip_matrix = np.sqrt(lip_matrix)

        return lip_matrix

    @staticmethod
    def gradient_matrix(gradient, dimension):
        '''
        It returns:

                        | f'_1 - f'_1     f'_1 - f'_2  ..........  f'_1 - f'_n |
                        |      .               .                               |
        grad_matrix  =  |      .                            .                  |
                        |      .                                        .      |
                        | f'_n - f'_1     f'_n - f'_2  ..........  f'_n - f'_n |


        :param gradient: gradient array
        :param dimension: dimension of the array
        :return: grad_matrix
        '''

        gradient = np.array(gradient)
        same_row = np.tile(gradient, (dimension, 1))
        col = gradient.reshape((-1, 1))
        same_col = np.repeat(col, dimension, axis=1)

        grad_matrix = same_col - same_row

        return grad_matrix

    @staticmethod
    def exact_greedy_selection(gradient_matrix, lip_matrix):
        '''
        It returns i and j such that:

        arg max {(f'_i(x) - f'_j(x)) / sqrt(L_i + L_j)}
          i,j

                                OR

        arg max {(f'_i(x) - f'_j(x)) / (sqrt(L_i) + sqrt(L_j))}
          i,j


        The correct formula depends on the metric of lip_matrix

        :param gradient_matrix:
        :param lip_matrix:
        :return: i, j
        '''

        matrix = gradient_matrix / lip_matrix

        max_val = np.argmax(matrix)
        i, j = np.unravel_index(max_val, matrix.shape)

        return i, j

    def solver(self, choice, y):
        '''
        :param choice:  selection rule. You can use:
                             r:     random selection
                            rp:     random selection proportional to Li
                             g:     greedy selection
                            ra:     ration approximation
                           eg1:     exact greedy rule with metric 1
                           eg2:     exact greedy rule with metric 2

        :param y:       starting point
        :return:        x_iteration = array with the number of iterations
                        y_f         = array with the values of the function after each iteration
        '''

        valid_inputs = ["r", "rp", "g", "eg1", "eg2", "ra"]
        x = y.copy()
        lip_const = np.diagonal(self.q)
        lip_matrix = []
        gradient = []
        x_iteration = []
        y_f = []
        i = 0
        j = 0

        if choice not in valid_inputs:
            print("FATAL ERROR: not allowed choice!")
            return None

        # Gradient initialization
        for k in range(self.dimension):
            gradient.append(self.grad_solver(x, k))

        # Lipshitz matrix initialization, only if it is required (for eg1 and eg2)
        if choice == 'eg1':
            lip_matrix = self.lips_matrix(1)

        if choice == 'eg2':
            lip_matrix = self.lips_matrix(2)

        iteration = 0

        #print(choice, "START", self.function(x))

        while iteration < self.MAX_ITERATION:

            if choice == 'r':
                i, j = self.random_selection()

            elif choice == 'rp':
                i, j = self.random_prob_selection(lip_const)

            elif choice == 'g':
                i, j = self.greedy_selection(gradient)

            elif choice == 'eg2' or choice == 'eg1':
                # Gradient matrix initialization, only for eg1 and eg2
                grad_matrix = self.gradient_matrix(gradient, self.dimension)
                i, j = self.exact_greedy_selection(grad_matrix, lip_matrix)

            elif choice == 'ra':
                i, j = self.ratio_approximation_selection(gradient, lip_const)

            # Step evaluation
            d = (gradient[i] - gradient[j]) / (lip_const[i] + lip_const[j])

            # Position update
            x[i] = x[i] - d
            x[j] = x[j] + d

            # Gradient update
            gradient = self.gradient_update(gradient, i, j, d)

            # Collect points to represent
            x_iteration.append(iteration)
            y_f.append(self.function(x))

            iteration += 1

        #print(choice, "STOP", self.function(x))

        return x_iteration, y_f
