import numpy as np

from methods.basic_mehtods import *
from methods.configuration_methods import *
from methods.configuration_methods import n, rows, cols, s_max, x_initial, x_star, u, sig, vt, sigma_min, sigma_max, \
    new_sigma, A_pds, A_pd, b_pd, b_psd, noise_vec, num_epochs, num_iters, num_experiments, num_batches, \
    POSITIVE_DEFINITE as n, rows, cols, s_max, x_initial, x_star, u, sig, vt, sigma_min, sigma_max, \
    new_sigma, A_pds, A_pd, b_pd, b_psd, noise_vec, num_epochs, num_iters, num_experiments, num_batches, \
    POSITIVE_DEFINITE


def sgd():
    """
    1. Set x to x_initial
    Next, for each epoch step:
        In each iteration:
            1a_1. Select an index from Uniform[1,...,n]
            1a_2. Compute the stochastic gradient of f_index at x
            1a_3. Update x using the SGD update
    :return: SGD method output
    """
    num_samples = rows
    x = x_initial.copy()
    A = A_pd if POSITIVE_DEFINITE else A_pds
    b = b_pd if POSITIVE_DEFINITE else b_psd
    optimal_xs = []
    indices = np.arange(1, num_samples)
    sgd_results = []
    for epoch in range(num_epochs):
        np.random.shuffle(indices)
        for i in indices:
            stochastic_gradient = calc_grad_fi(A, b, i, x)
            x -= eta * stochastic_gradient
            sgd_results.append(x)
        optimal_x = np.mean(sgd_results)
        optimal_xs.append(optimal_x)
        x = optimal_x.copy()
    return x


def minibatch_sgd():
    num_samples = rows
    num_features = cols
    x = x_initial.copy()
    A = A_pd if POSITIVE_DEFINITE else A_pds
    b = b_pd if POSITIVE_DEFINITE else b_psd
    optimal_xs = []
    indices = np.arange(1, num_samples)
    num_batches = num_samples // batch_size
    for epoch in range(num_epochs):
        np.random.shuffle(indices)
        A_shuffled = A[indices]
        b_shuffled = b[indices]

        for i in range(0, num_batches):
            A_batch = A_shuffled[batch_size * i:batch_size * (i + 1)]
            b_batch = b_shuffled[batch_size * i:batch_size * (i + 1)]

            gradient = np.zeros((num_features, 1))
            for j in range(len(A_batch)):
                gradient += calc_grad_fi(A_batch, b_batch, j, x)

            x -= learning_rate * gradient / batch_size  # Do we need to take the average of x in each epoch?
        optimal_xs.append(x)
    return x


def svrg():
    num_samples = rows
    y = x_initial.copy()
    A = A_pd if POSITIVE_DEFINITE else A_pds
    b = b_pd if POSITIVE_DEFINITE else b_psd
    optimal_xs = []
    indices = np.arange(1, num_samples)
    num_batches = num_samples // batch_size
    for epoch in range(num_epochs):
        x = y.copy()
        full_gradient_y = calc_grad_F(A, b, y)
        np.random.shuffle(indices)
        A_shuffled = A[indices]
        b_shuffled = b[indices]

        for i in range(num_batches):
            batch_indices = indices[batch_size * i:batch_size * (i + 1)]
            j = np.random.sample(batch_indices)
            A_batch = A_shuffled[batch_indices]
            b_batch = b_shuffled[batch_indices]

            stochastic_gradient_y = calc_grad_fi(A_batch, b_batch, x, j)
            stochastic_gradient_x = calc_grad_fi(A_batch, b_batch, x, j)

            x -= step_size * (stochastic_gradient_x - stochastic_gradient_y + full_gradient_y)
            optimal_xs.append(x)
        y = np.mean(optimal_xs)
    x = y.copy()
    return x


def convex_nonsmooth(**kwargs):
    """
    Sub gradient method for non-smooth optimization problem.
    Considering 2 cases - f is α-convex or a-strongly-convex over K.
    The SG for non-smooth optimization algorithm preforms the following steps:
        - x_1 <- arbitrary point in K
        - for each t>=1 :
                        x_t+1 <- Π_κ(x_t - η_t*sub_g(x_t))
    Note the iterated {x_t}_t>=1 themselves do not converge.
    Note that in each step-size choice we normalize the subgradient by G to disregard its magnitude.
    Furthermore, we take η_t ~ ε to force convergence.
    """
    converge = False
    x_new = x_1
    print(x_new)
    fx = calc_f(x_new)
    iterations_durations = []
    f_values = [fx]
    start = time.time()
    iter_time = start
    converged_at = '-1'
    for i in range(1, num_iters):
        x_prev = x_new
        step_size = D / (G * np.sqrt(i))
        x_new = x_prev - step_size * calc_grad(x_prev)
        fx = calc_f(x_new)
        f_values.append(fx)
        converge = np.abs(fx - f_opt) < epsilon
        if converge:
            converged_at = str(i)
        iterations_durations.append(time.time() - iter_time)
        iter_time = time.time()
    return f_values, iterations_durations, converged_at


def convex_nonsmooth(f, df, x_start, x_optimal, G, R, iterations):
    x = x_start
    D = 2 * R

    log = []
    for i in range(iterations):
        step_size = D / (G * np.sqrt(i + 1))
        x = x - step_size * df(x)
        if norm(x) > R:
            x *= (1 / norm(x)) * R
        log.append(np.abs(f(x) - f(x_optimal)))
    return np.array(log), x


def strongly_convex_nonsmooth(*args):
    """
    Sub gradient method for non-smooth optimization problem.
    Considering 2 cases - f is α-convex or a-strongly-convex over K.
    The SG for non-smooth optimization algorithm preforms the following steps:
        - x_1 <- arbitrary point in K
        - for each t>=1 :
                        x_t+1 <- Π_κ(x_t - η_t*sub_g(x_t))
    Note the iterated {x_t}_t>=1 themselves do not converge.
    Note that in each step-size choice we normalize the subgradient by G to disregard its magnitude.
    Furthermore, we take η_t ~ ε to force convergence.
    """
    return convex_nonsmooth()


def convex_smooth(*args):
    """
    Gradient Descent for smooth and convex optimization problem.
    Let min f(x) over x belongs to K, where K is compact and convex and f is dif over K be in optimization task.
    Let {η_t}_t>=1 contained in R be a sequence of step-sizes.
    The PGD algorithm preforms the following steps:
        - x_1 <- arbitrary point in K
        - for each t>=1 :
                        x_t+1 <- Π_κ(x_t - (1/β)*grad(f(x_t))
    Note the iterated {x_t}_t>=1 themselves do converge.
    Note that in each step-size choice we normalize the subgradient by G to disregard its magnitude.
    Furthermore, we take η_t ~ ε to force convergence.
    """
    converge = False
    x_new = x_1
    fx = calc_f(x_new)
    iterations_durations = []
    f_values = [fx]
    step_size = 1 / beta
    start = time.time()
    iter_time = start
    converged_at = '-1'
    for i in range(1, num_iters):
        x_prev = x_new
        x_new = x_prev - step_size * calc_grad(x_prev)
        fx = calc_f(x_new)
        f_values.append(fx)
        converge = np.abs(fx - f_opt) < epsilon
        if converge:
            converged_at = str(i)
        iterations_durations.append(time.time() - iter_time)
        iter_time = time.time()
    return f_values, iterations_durations, converged_at


def strongly_convex_smooth(*args):
    """
    Gradient Descent for smooth and convex optimization problem.
    Let min f(x) over x belongs to K, where K is compact and convex and f is dif over K be in optimization task.
    Let {η_t}_t>=1 contained in R be a sequence of step-sizes.
    The PGD algorithm preforms the following steps:
        - x_1 <- arbitrary point in K
        - for each t>=1 :
                        x_t+1 <- Π_κ(x_t - (1/β)*grad(f(x_t))
    Note the iterated {x_t}_t>=1 themselves do converge.
    Note that in each step there is no need to calculate f(x_new), but to check the convergence condition based on the
    theory in Lec.2 page 18.
    """
    return convex_smooth()


def accelerated_gradient(*args):
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
    x_new = x_1
    y_new = x_1
    fx = calc_f(x_new)
    step_size = 1
    iterations_durations = []
    f_values = [fx]
    step_size = 1 / beta
    start = time.time()
    iter_time = start
    converged_at = '-1'
    for i in range(1, num_iters):
        new_step_size = (-1 * np.square(step_size) + np.sqrt(
            np.power(step_size, 4) + 4 * np.square(step_size))) / 2
        step_size = new_step_size
        x = x_new
        y = y_new
        z_new = (1 - step_size) * y + step_size * x
        x_new = np.dot(x_star, short_GD(z_new)) + beta * step_size * 0.5 * np.square(np.norm(x_star - x))
        y_new = (1 - step_size) * y + step_size * x_new
        fx = calc_f(x_new)
        f_values.append(fx)
        converge = np.abs(fx - f_opt) < epsilon
        if converge:
            converged_at = str(i)
        iterations_durations.append(time.time() - iter_time)
        iter_time = time.time()
    return f_values, iterations_durations, converged_at


def strongly_accelerated_gradient(*args):
    """
    Accelerated Gradient Method for Strongly Convex F.
    The AGM algorithm preforms the following steps:
    - x_1 = y_1 <- arbitrary point in K
    - for each t>=1 :
            z_t+1 <- (1-η_t)*y_t + η_t*x_t
            x_t+1 <- calc_fi(x) by the Algorithm given at the lecture for strongly convex
            y_t+1 <-  (1-η_t)*y_t + η_t*x_t+1
    """
    converge = False
    x_new = x_1
    y_new = x_1
    fx = calc_f(x_new)
    step_size = 1
    iterations_durations = []
    f_values = [fx]
    step_size = 1 / beta
    start = time.time()
    iter_time = start
    converged_at = '-1'
    for i in range(1, num_iters):
        new_step_size = (-1 * np.square(step_size) + np.sqrt(
            np.power(step_size, 4) + 4 * np.square(step_size))) / 2
        step_size = new_step_size
        x = x_new
        y = y_new
        z_new = (1 - step_size) * y + step_size * x
        x_new = calc_fi(x, z_new, step_size)
        y_new = (1 - step_size) * y + step_size * x_new
        fx = calc_f(x_new)
        f_values.append(fx)
        converge = np.abs(fx - f_opt) < epsilon
        if converge:
            converged_at = str(i)
        iterations_durations.append(time.time() - iter_time)
        iter_time = time.time()
    return f_values, iterations_durations, converged_at
