import numpy as np
import time
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import statsmodels as sm
from sklearn.datasets import make_spd_matrix
from time import time


def question_8():
    """
    You are requested to empirically compare the performances of the (sub)gradient method for non-smooth optimization
    (with decaying step-sizes) gradient descent for smooth convex optimization,
    and the accelerated gradient method on the linear regression optimization task.

    Generate the data as follows:
        1. take A to be a random matrix (random as you choose) with fixed values of σmax(A), σmin(A) (of your choosing).
        2. Choose the following parameters:
            @:param A: random matrix (random as you choose) with fixed values of σmax(A), σmin(A) (of your choosing)
            @:param x∗: solution
            @:param ξ: random noise of low magnitude.
            @:param b: b = Ax∗ + ξ
        3. Compare the convergence rate of the algorithms (i.e,. function value vs. number of iterations).
        4. Experiment both in the case in which A>A is not positive definite and in the case in which it is
        (then the problem is strongly convex).
        You may set the parameters (D, G, β, α) based directly on the data (A, b) (though this is not likely in real-life).
        5. Since data is random, plot the average of several i.i.d. experiments.
        6. Briefly discuss your observations of the experiment and contrast with the theory we have developed.
    """
    matrix_size = 3
    n, num_cols, num_rows = matrix_size, matrix_size, matrix_size
    sigma_max, sigma_min = 2, 0.0
    x_star = np.array([1, 2, 3])
    noise_magnitude = 0.1
    if positive_definite:
        A, b, noise_vec = create_data(matrix_size, sigma_max, sigma_min, x_star, noise_magnitude, "Positive definite")
    elif positive_semi_definite:
        A, b, noise_vec = create_data(matrix_size, sigma_max, sigma_min, x_star, noise_magnitude,
                                      "Positive Semi-definite")
    else:
        A, b, noise_vec = create_data(matrix_size, sigma_max, sigma_min, x_star, noise_magnitude,
                                      "Positive Semi-definite")
    f_opt = calc_f(A, b, x_star)


def create_data(matrix_size, sigma_max, sigma_min, x_star, noise_magnitude, matrix_definition='Positive'):
    """
    :param matrix_size:
    :param sigma_max:
    :param sigma_min:
    :param x_star:
    :param noise_magnitude:
    :param matrix_definition:
    :return:  (A,b,ξ), where -
                @:param A: random matrix (random as you choose) with fixed values of σmax(A), σmin(A) (of your choosing)
                @:param x∗: solution
                @:param ξ: random noise of low magnitude,created by the least-squares solution named noise_vec.
                @:param b: b = Ax∗ + ξ
    """
    A = make_spd_matrix(n_dim=matrix_size, random_state=42)
    ATA = A.T @ A
    eig_values, eig_vectors = np.linalg.eig(ATA)
    sigma_min, sigma_max = np.sqrt(np.min(eig_values)), np.sqrt(np.max(eig_values))
    if matrix_definition != "Positive" and sigma_min > 0:
        eig_values[np.argmin(eig_values)] = 0
        ATA = eig_vectors @ np.diag(eig_values) @ eig_vectors.T
        sigma_min, sigma_max = np.sqrt(np.min(eig_values)), np.sqrt(np.max(eig_values))
    noise_vec = np.random.normal(loc=0, scale=noise_magnitude, size=(matrix_size,))
    b = np.dot(A, x_star) + noise_vec
    print(f"A matrix : {A}")
    print(f"A eigen values: {np.around(np.linalg.eigvals(A), decimals=3)}")
    print(f"ATA matrix : {ATA}")
    print(f"ATA eigen values: {np.around(np.linalg.eigvals(ATA), decimals=3)}")
    print(f"x* : {x_star}")
    print(f"Noise vector : {noise_vec}")
    print(f"b = Ax* + ξ : {b}")
    print(f"sigma_min,sigma_max : {sigma_min},{sigma_max}")
    return A, b, noise_vec


def calc_f(A, b, x):
    f_x = lambda x: 0.5 * np.square(np.linalg.norm(A @ x - b))
    return f_x(x)


def calc_grad(A, b, x):
    grad_x = lambda x: np.linalg.norm(A.T @ (A @ x - b))
    return grad_x(x)


def calc_G(A, b, x):
    grad_x = calc_grad(A, b, x)
    G = np.max(grad_x)
    return G


def calc_hyperparameters(A, x_opt, b, x_1, sigma_max, sigma_min):
    """
    G = np.norm(x_1)                -> Since GD methods for minimize a smooth-convex function
                                       sets a Converse series {x_t}t>1
    D = np.linalg.norm(x_1 - x_opt) -> Since GD methods for minimize a smooth-convex function
                                       sets a Converse series {x_t}t>1
    beta = np.square(sigma_max) ->   Since from Rayleigh inequality,
                                     the largest eigen value of ATA is an upper-bound
                                     for the hessian.
    alpha = np.square(sigma_min) ->  Since from Rayleigh inequality,
                                     the smallest eigen value of ATA is a lower-bound
                                     for the hessian.
    :return: G, D, beta, alpha
    """
    G = np.norm(x_1)
    beta = np.square(sigma_max)
    D = np.linalg.norm(x_1 - x_opt)
    alpha = np.square(sigma_min)
    return G, D, beta, alpha


def compare_performances():
    """
    Empirical comparison of the performances of the methods on the linear regression optimization task.
    """
    pass


def sub_gradient_non_smooth():
    """
    Sub gradient method for non-smooth optimization problem.
    The SG for non-smooth optimization algorithm preforms the following steps:
    - x_1 <- arbitrary point in K
    - for each t>=1 : x_t+1 <- Π_κ(x_t - η_t*sub_g(x_t))
    """
    pass


def gradient_descent_smooth_convex(A, b, D, G, beta, alpha, f_opt, n):
    """
    Gradient Descent for smooth and convex optimization problem.
    Let min f(x) over x belongs to K, where K is compact and convex and f is dif over K be in optimization task.
    Let {η_t}_t>=1 contained in R be a sequence of step-sizes.
    The PGD algorithm preforms the following steps:
    - x_1 <- arbitrary point in K
    - for each t>=1 :
            x_t+1 <- Π_κ(x_t - (1/β)*grad(f(x_t))
    """
    converge = False
    epsilon = -1 * f_opt * np.exp(-0.25 * alpha / beta)  # FixMe
    x_1 = np.random.rand(n, )
    x_new = x_1
    iterations_durations = []
    num_iterations = 0
    start = time.time()
    iter_time = start
    while not converge:
        num_iterations += 1
        x_prev = x_new
        x_new = x_prev - (1 / beta) * calc_grad(A, b, x_prev)
        converge = calc_f(A, b, x_new) - f_opt < epsilon
        iterations_durations.append(time.time() - iter_time)
        iter_time = time.time()
    return np.mean(iterations_durations), num_iterations


def accelerated_gradient(A, b, D, G, beta, alpha, f_opt, n):
    """
    Accelerated Gradient Method.
    The AGM algorithm preforms the following steps:
    - x_1 = y_1 <- arbitrary point in K
    - for each t>=1 :
            z_t+1 <- (1-η_t)*y_t + η_t*x_t
            x_t+1 <- <x,(grad(f(z_t+1))> + (β/2)*η_t*||x-x_t||^2
            y_t+1 <-  (1-η_t)*y_t + η_t*x_t+1
    """
    converge = False
    epsilon = 32 * beta * np.square(D) / 9  # FixMe
    x_1 = y_1 = np.random.rand(n, )
    x_new = y_new = x_1
    iterations_durations = []
    num_iterations = 0
    start = time.time()
    iter_time = start
    eta_new = 1
    while not converge:
        num_iterations += 1
        x_prev = x_new
        y_prev = y_new
        eta_prev = eta_new
        z_new = (1 - eta_prev) * y_prev + eta_prev * x_prev
        x_new = 0  # FixMe
        y_new = (1 - eta_prev) * y_prev + eta_prev * x_new
        eta_new = 0.5 * (-1 * np.square(eta_prev) + np.sqrt(np.power(eta_prev, 4) + 4 * np.square(eta_prev)))
        converge = calc_f(A, b, y_new) - f_opt < epsilon
        iterations_durations.append(time.time() - iter_time)
        iter_time = time.time()
    return np.mean(iterations_durations), num_iterations


if __name__ == '__main__':
    methods = ["Sub-Gradient method for non-smooth optimization", "Gradient-Descent for smooth convex optimization",
               "Accelerated Gradient method"]
    task = "Linear regression optimization task"
    matrix_definition = ["Positive definite,Not-Positive definite"]
    positive_definite, positive_semi_definite = True, False
    question_8()
