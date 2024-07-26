import numpy as np

from methods.basic_mehtods import *
from methods.basic_mehtods import calc_b
global n, rows, cols, s_max, x_initial, x_star, u, sig, vt, sigma_min, sigma_max, new_sigma, A_pds, A_pd, b_pd, \
    b_psd, noise_vec

global POSITIVE_DEFINITE
global num_epochs, num_iters, num_experiments
global batch_sizes
global num_batches


def generate_data(prints=False):
    global n, rows, cols, s_max, x_initial, x_star, u, sig, vt, sigma_min, sigma_max, new_sigma, \
        A_pds, A_pd, b_pd, b_psd, noise_vec
    n = 50
    rows = 50
    cols = 50
    s_max = n * 4
    A = np.random.rand(rows, cols)
    x_initial = np.zeros((cols, 1))
    x_star = np.array([np.pi] * cols).reshape(cols, 1)
    u, sig, vt = np.linalg.svd(A, full_matrices=True)
    sigma_min, sigma_max = np.min(sig), np.max(sig)
    new_sigma = s_max * (sig - sigma_min) / (sigma_max - sigma_min)
    A_psd = u @ np.diag(new_sigma) @ vt
    new_sigma[-1] += .5
    A_pd = u @ np.diag(new_sigma) @ vt
    noise_vec = np.random.normal(0, 0.2, (rows, 1))
    b_pd = A_pd @ x_star + np.random.normal(0, 0.2, (rows, 1))
    b_psd = A_psd @ x_star + np.random.normal(0, 0.2, (rows, 1))
    if prints:
        print(f"|| Optimal x (First 5 values) ||\n{x_star[:5]}")
        print(f"|| Positive Definite A (First 5 X 5 block) ||\n{A_pd[:5, :5]}\n Matching b: \n{b_pd[:5]} ")
        print(f"|| Positive Semi-Definite A (First 5 X 5 block) || \n{A_psd[:5, :5]}\n Matching b: \n{b_psd[:5]}")
        print(f"|| Noise Vector (First 5 values) ||\n{noise_vec[:5]}")
        print(f"| Min Singular Value: {sigma_min} |\n| Max Singular Value : {sigma_max} |")
    return n, rows, cols, s_max, x_initial, x_star, u, sig, vt, sigma_min, sigma_max, new_sigma, \
           A_psd, A_pd, b_pd, b_psd, noise_vec


def compare_performances():
    """
    Empirical comparison of the performances of the methods on the linear regression optimization task.
    """
    generate_data()
    convex_nonsmooth_values, strongly_convex_nonsmooth_values = [], []
    convex_smooth_values, strongly_convex_smooth_values = [], []
    accelerated_gradient_values, strongly_accelerated_gradient_values = [], []

    generate_data()
    for _ in range(num_experiments):

        calc_params(x_1)
        if not positive_definite_case:
            convex_nonsmooth_value = convex_nonsmooth()
            convex_smooth_value = convex_smooth()
            accelerated_gradient_value = accelerated_gradient()

            convex_nonsmooth_values.append(convex_nonsmooth_value)
            convex_smooth_values.append(convex_smooth_value)
            accelerated_gradient_values.append(accelerated_gradient_value)
        else:
            strongly_convex_nonsmooth_value = strongly_convex_nonsmooth()
            strongly_convex_smooth_value = strongly_convex_smooth()
            strongly_accelerated_gradient_value = strongly_accelerated_gradient()

            strongly_convex_nonsmooth_values.append(strongly_convex_nonsmooth_value)
            strongly_convex_smooth_values.append(strongly_convex_smooth_value)
            strongly_accelerated_gradient_values.append(strongly_accelerated_gradient_value)

    if positive_definite_case:
        pos_subgradient_avg_values = np.mean(strongly_convex_nonsmooth_values, axis=0)
        pos_convex_smooth_avg_values = np.mean(strongly_convex_smooth_values, axis=0)
        pos_accelerated_gradient_avg_values = np.mean(strongly_accelerated_gradient_values, axis=0)

        plt.plot(pos_subgradient_avg_values, label='Subgradient Nonsmooth & Strongly-Convex')
        plt.plot(pos_convex_smooth_avg_values, label='Gradient Descent Smooth & Strongly-Convex')
        plt.plot(pos_accelerated_gradient_avg_values, label='Accelerated Gradient & Strongly-Convex')

        plt.title('Linear Regression Optimization : Positive-Definite')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Function Value')
        plt.legend()
        plt.show()
    else:
        subgradient_avg_values = np.mean(convex_nonsmooth_values, axis=0)
        convex_smooth_avg_values = np.mean(convex_smooth_values, axis=0)
        accelerated_gradient_avg_values = np.mean(accelerated_gradient_values, axis=0)

        plt.plot(subgradient_avg_values, plt.title('GD_Smooth'))
        plt.plot(convex_smooth_avg_values, label='Gradient Descent Smooth & Convex')
        plt.plot(accelerated_gradient_avg_values, label='Accelerated Gradient')

        plt.title('Linear Regression Optimization : Positive-Semi-Definite')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Function Value')
        plt.legend()
        plt.show()
