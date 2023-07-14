import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib
import torch.optim.lr_scheduler as lr_scheduler
from methods.plotting_methods import *
from methods.basic_mehtods import *
from methods.opt_methods_hw2 import *

global POSITIVE_DEFINITE
global num_epochs
global num_batches
global num_experiments

"""
If the plots of the SVRG (Stochastic Variance-Reduced Gradient) method resemble a Gaussian shape, it could indicate that the algorithm is converging efficiently and following a smooth trajectory towards the optimal solution.
The SVRG algorithm is an optimization method that iteratively updates the solution by combining stochastic gradients with a full gradient correction. It aims to reduce the variance of the stochastic gradients to achieve faster convergence.
When the SVRG method is applied to a well-behaved objective function, it tends to converge smoothly towards the optimal solution. The trajectory of the algorithm's iterations, as shown in the plot, can exhibit a smooth, bell-shaped pattern resembling a Gaussian distribution.
This Gaussian-like shape arises due to several factors:
Convergence to a local minimum: The SVRG method strives to find a local minimum of the objective function. As the iterations progress, the algorithm explores the function landscape, and the stochastic gradients guide it towards a minimum. The trajectory of the algorithm can resemble a smooth curve with a Gaussian-like shape as it converges to the optimal solution.
Reduction of variance: The SVRG algorithm utilizes a full gradient correction term to reduce the variance of the stochastic gradients. This reduction in variance leads to a smoother and more consistent update process, which can contribute to the Gaussian-like appearance of the plot.
Smoothness of the objective function: If the objective function itself exhibits a smooth and well-behaved shape, the SVRG method is likely to follow a smoother trajectory during convergence. Smoothness in the objective function can result in a Gaussian-like pattern in the plot.
It's worth noting that the exact shape of the plot can vary depending on the specific characteristics of the objective function, the SVRG hyperparameters, and the initialization of the algorithm. Additionally, noise, numerical approximations, or other factors can introduce variations in the plot. Therefore, it's essential to consider the overall trend and convergence behavior rather than focusing solely on the visual resemblance to a Gaussian distribution.
"""


def generate_data(prints=False):
    global n, rows, cols, s_max, x_initial, x_star, u, sig, vt, sigma_min, sigma_max, new_sigma, \
        A_pds, A_pd, b_pd, b_psd, noise_vec
    n = 10  # 50
    rows = 10  # 50
    cols = 10  # 50
    s_max = 1
    s_min = 0.5
    A = np.random.rand(rows, cols)  # Generate a random matrix with values between 0 and 1
    # U, S, Vt = np.linalg.svd(A)  # Perform singular value decomposition (SVD)
    S, U = np.linalg.eigh(A)
    S = np.linspace(s_min, s_max, min(rows, cols))  # Adjust singular values to achieve desired condition number
    A_pd = U @ np.diag(S) @ U.T  # Vt  # Reconstruct matrix A
    x_initial = np.zeros((cols, 1))
    x_star = np.array([np.pi] * cols).reshape(cols, 1)
    # u, sig, vt = np.linalg.svd(A_pd, full_matrices=True)
    sig = S
    sigma_min, sigma_max = np.min(sig), np.max(sig)
    # new_sigma = s_max * (sig - sigma_min) / (sigma_max - sigma_min)
    # A_psd = u @ np.diag(new_sigma) @ vt
    # new_sigma[-1] += .075
    # A_pd = u @ np.diag(new_sigma) @ vt
    noise_variance = np.sqrt(1.0 / (rows + cols))
    noise_vec = np.random.normal(0, noise_variance, (rows, 1))
    b_pd = A_pd @ x_star + noise_vec
    u, sig, vt = np.linalg.svd(A_pd, full_matrices=True)
    A_psd = u @ np.diag(sig) @ vt
    b_psd = A_psd @ x_star + noise_vec
    new_sigma = sigma_min
    if prints:
        print(f"|| Optimal x (First 5 values) ||\n{x_star[:5]}")
        print(f"|| Positive Definite A (First 5 X 5 block) ||\n{A_pd[:5, :5]}\n Matching b: \n{b_pd[:5]} ")
        print(f"A conditioning: {np.linalg.cond(A_pd)}")
        # print(f"|| Positive Semi-Definite A (First 5 X 5 block) || \n{A_psd[:5, :5]}\n Matching b: \n{b_psd[:5]}")
        print(f"|| Noise Vector (First 5 values) ||\n{noise_vec[:5]}")
        print(f"| Min Singular Value: {sigma_min} |\n| Max Singular Value : {sigma_max} |")
    return n, rows, cols, s_max, x_initial, x_star, u, sig, vt, sigma_min, sigma_max, new_sigma, \
           A_psd, A_pd, b_pd, b_psd, noise_vec


def sgd(num_epochs=100):
    def plot_single(sgd_results):
        matplotlib.use('TkAgg', force=False)
        epochs = np.arange(1, len(sgd_results) + 1, 1)
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, sgd_results, linewidth=2, marker='o', markersize=5, label='SGD')
        plt.xlabel('Epoch')
        plt.ylabel('Objective Function Value')
        plt.title(
            f'SGD Results\nStep-size : {step_size} || Total Epochs : {num_epochs}  || Total Batches : {num_batches}')
        best_epoch = np.argmin(sgd_results)
        best_value = sgd_results[best_epoch]
        plt.text(num_epochs, best_value, f'Best F(x) : {best_value:.2f}', ha='right', va='top')
        plt.legend()
        plt.savefig(f'SGD_stepSize{step_size}_10rank.png')
        plt.show()

    x = np.zeros((cols, 1)).reshape((-1,)).copy()
    x_sgd = []
    sgd_results = []
    num_samples = rows
    num_batches = num_samples
    A = A_pd if POSITIVE_DEFINITE else A_psd
    b = b_pd if POSITIVE_DEFINITE else b_psd
    optimal_xs = []
    indices = np.arange(1, num_samples)
    for epoch in range(num_epochs):
        for i in indices:
            stochastic_gradient = calc_grad_fi(A=A, b=b, x=x, i=i)
            x -= step_size * stochastic_gradient
            x_sgd.append(x)
        optimal_x = np.mean(x_sgd, axis=0)
        optimal_xs.append(optimal_x)
        x = optimal_x.copy()
        f_value = calc_F(A, b, x)
        sgd_results.append(f_value)
    plot_single(sgd_results)
    return x, optimal_xs, sgd_results


def minibatch_sgd(batch_size):
    def plot_single(miniB_sgd_results):
        matplotlib.use('TkAgg', force=False)
        epochs = np.arange(1, len(miniB_sgd_results) + 1, 1)
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, miniB_sgd_results, linewidth=2, marker='o', markersize=5, label='Mini Batch SGD')
        plt.xlabel('Epoch')
        plt.ylabel('Objective Function Value')
        plt.title(
            f'Mini-Batch SGD Results\nStep-size : {step_size} || Total Epochs : {num_epochs}  || Batch Size : {batch_size}')
        best_epoch = np.argmin(miniB_sgd_results)
        best_value = miniB_sgd_results[best_epoch]
        plt.text(num_epochs, best_value, f'Best F(x) : {best_value:.2f}', ha='right', va='top')
        plt.legend()
        plt.savefig(f'MiniBatchSGD_stepSize{step_size}_batchSize{batch_size}_10rank.png')
        plt.show()

    num_samples = rows
    num_features = cols
    x = np.zeros((cols, 1)).reshape((-1,)).copy()
    A = A_pd if POSITIVE_DEFINITE else A_psd
    b = b_pd if POSITIVE_DEFINITE else b_psd
    optimal_xs = []
    miniB_sgd_results = []
    x_miniB_sgd = []
    indices = np.arange(1, num_samples)
    arr = np.random.permutation(indices)
    num_batches = num_samples // batch_size
    for epoch in range(num_epochs):
        A_shuffled = A[arr]
        b_shuffled = b[arr]
        A_batches = np.array_split(A_shuffled, num_batches)
        b_batches = np.array_split(b_shuffled, num_batches)
        for A_batch, b_batch in zip(A_batches, b_batches):
            # gradient = np.zeros((rows, cols))
            gradients = []
            for j in range(len(A_batch)):
                iter_grad = calc_grad_fi(A=A_batch, b=b_batch, i=j, x=x)
                gradients.append(iter_grad)
            batch_gradient = np.mean(gradients, axis=0)  # gradient / len(A_batch)
            x -= step_size * batch_gradient
            x_miniB_sgd.append(x)
        x_miniB = np.mean(x_miniB_sgd, axis=0)
        x = x_miniB.copy()
        optimal_xs.append(x)
        f_value = calc_F(A_shuffled, b_shuffled, x)
        miniB_sgd_results.append(f_value)
    plot_single(miniB_sgd_results)
    return x, optimal_xs, miniB_sgd_results


def svrg(num_epochs: int = 50, num_batches: int = 20):
    def plot_single(svrg_results):
        matplotlib.use('TkAgg', force=False)
        epochs = np.arange(1, len(svrg_results) + 1, 1)
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, svrg_results, linewidth=2, marker='o', markersize=5, label='SVRG')
        plt.xlabel('Epoch')
        plt.ylabel('Objective Function Value')
        plt.title(
            f'SVRG Results\nStep-size : {step_size} || Total Epochs : {num_epochs}  || Total Batches : {num_batches}')
        best_epoch = np.argmin(svrg_results)
        best_value = svrg_results[best_epoch]
        plt.text(num_epochs, best_value, f'Best F(x) : {best_value:.2f}', ha='right', va='top')
        plt.legend()
        plt.show()
        plt.savefig(f'SVRG_stepSize{step_size}_numBatches{num_batches}_10rank.png')

    svrg_results = []
    num_samples = rows
    y = x_initial.reshape((-1,)).copy()
    A = A_pd if POSITIVE_DEFINITE else A_psd
    b = b_pd if POSITIVE_DEFINITE else b_psd
    optimal_xs = []
    optimal_ys = []
    indices = np.arange(1, num_samples)
    arr = np.random.permutation(indices)
    for epoch in range(num_epochs):
        x = y.copy()
        full_gradient_y = calc_grad_F(A, b, y)
        A_shuffled = A[arr]
        b_shuffled = b[arr]
        A_batches = np.array_split(A_shuffled, num_batches)
        b_batches = np.array_split(b_shuffled, num_batches)
        for A_batch, b_batch in zip(A_batches, b_batches):
            j = np.random.choice(range(0, A_batch.shape[0]), size=1).item(0)
            stochastic_gradient_y = calc_grad_fi(A=A_batch, b=b_batch, i=j, x=y)
            stochastic_gradient_x = calc_grad_fi(A=A_batch, b=b_batch, i=j, x=x)
            update_step = step_size * (stochastic_gradient_x - stochastic_gradient_y + full_gradient_y)
            x -= update_step
            optimal_xs.append(x)
        y = np.mean(optimal_xs, axis=0)
        optimal_ys.append(y)
        f_value = calc_F(A_shuffled, b_shuffled, y)
        svrg_results.append(f_value)
    print(svrg_results)
    plot_single(svrg_results)
    x = y.copy()

    return x, optimal_ys, svrg_results


def question_3():
    n, rows, cols, s_max, x_initial, x_star, u, sig, vt, sigma_min, sigma_max, new_sigma, \
    A_psd, A_pd, b_pd, b_psd, noise_vec = generate_data(prints=False)
    global POSITIVE_DEFINITE
    global num_epochs
    global num_batches
    global num_experiments
    global step_size
    global best_batch_size
    svrg_results_ = []
    sgd_results_ = []
    minibatch_sgd_results_ = []
    eigenvalues = np.linalg.eigvalsh(A_pd)
    alpha = np.min(eigenvalues) / 2
    beta = np.linalg.norm(A_pd, ord=2)
    POSITIVE_DEFINITE = True
    num_experiments = 10
    num_epochs = 100
    svrg_num_batches = int(2 * (beta / alpha))
    svrg_step_size = round(1 / (10 * beta), 6)
    step_sizes_svrg = np.arange(start=0.0005, stop=svrg_step_size, step=(svrg_step_size - 0.0005) / 5)
    num_batches_svrg = [1, svrg_num_batches, 15, 25, 50]
    best_step_size = 0.01
    best_svrg_F = None
    for step_size in step_sizes_svrg:
        _, svrg_x, svrg_F = svrg(num_epochs=num_epochs, num_batches=svrg_num_batches)
        svrg_results_.append(svrg_F)
        if best_svrg_F is None:
            best_svrg_F = svrg_F.copy()
            best_step_size = step_size
        elif np.min(svrg_F) < np.min(best_svrg_F):
            best_svrg_F = svrg_F.copy()
            best_step_size = step_size
        else:
            best_svrg_F = best_svrg_F.copy()
            best_step_size = best_step_size
    matplotlib.use('TkAgg', force=False)
    plt.figure(figsize=(10, 10))
    plt.plot(svrg_results_[0], label=f'Step Size ={step_sizes_svrg[0]}')
    plt.plot(svrg_results_[1], label=f'Step Size ={step_sizes_svrg[1]}')
    plt.plot(svrg_results_[2], label=f'Step Size ={step_sizes_svrg[2]}')
    plt.plot(svrg_results_[3], label=f'Step Size ={step_sizes_svrg[3]}')
    plt.plot(svrg_results_[4], label=f'Step Size ={step_sizes_svrg[4]}')
    plt.title('SVRG F(x) Results by Step Size')
    plt.legend(loc='upper right')
    plt.savefig('SVRG_F(x)_Results_by_StepSize_10rank.png')
    # plt.show()
    _, sgd_x, sgd_F = sgd(num_epochs=num_epochs)
    best_batch_size = 5
    best_minibatch_F = None
    mbsgd_batch_sizes = [3, 5, 6, 8, 9]
    for batch_size in mbsgd_batch_sizes:
        _, minibatch_sgd_x, minibatch_sgd_F = minibatch_sgd(batch_size=batch_size)
        minibatch_sgd_results_.append(minibatch_sgd_F)
        if best_minibatch_F is None:
            best_minibatch_F = minibatch_sgd_F.copy()
            best_batch_size = batch_size
        elif np.min(minibatch_sgd_F) < np.min(best_minibatch_F):
            best_minibatch_F = minibatch_sgd_F.copy()
            best_batch_size = batch_size
        else:
            best_minibatch_F = best_minibatch_F.copy()
            best_batch_size = best_batch_size
    matplotlib.use('TkAgg', force=False)
    plt.figure(figsize=(10, 10))
    plt.plot(minibatch_sgd_results_[0], label=f'Batch Size ={mbsgd_batch_sizes[0]}')
    plt.plot(minibatch_sgd_results_[1], label=f'Batch Size ={mbsgd_batch_sizes[1]}')
    plt.plot(minibatch_sgd_results_[2], label=f'Batch Size ={mbsgd_batch_sizes[2]}')
    plt.plot(minibatch_sgd_results_[3], label=f'Batch Size ={mbsgd_batch_sizes[3]}')
    plt.plot(minibatch_sgd_results_[4], label=f'Batch Size ={mbsgd_batch_sizes[4]}')
    plt.title('Mini Batch SGD F(x) Results by Batch Size')
    plt.legend(loc='upper right')
    plt.savefig('MiniBatch_SGD_F(x)_Results_by_Batch_Size_10rank.png')
    # plt.show()
    pd_results = {'SVRG': best_svrg_F, 'SGD': sgd_F, 'Mini Batch SGD': best_minibatch_F}
    plot_summary(_range=range(0, max(len(sgd_F), len(best_minibatch_F), len(best_svrg_F))),
                 results=pd_results,
                 title="Experiments Results - Comparison")


def plot_summary(_range: range, results: dict, title: str):
    matplotlib.use('TkAgg', force=False)
    plt.figure(figsize=(10, 10))
    iters = np.arange(0, len(_range), 1)
    label1, label2, label3 = results.keys()
    plt.plot(iters, results[label3], label=f'{label3} - Batch Size = {best_batch_size}')
    plt.plot(iters, results[label1], label=label1)
    plt.plot(iters, results[label2], label=label2)
    plt.xlabel('Epoch')
    plt.ylabel('F(x)')
    plt.title(title)
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig(f'{title}rank_10.png')


if __name__ == '__main__':
    hw1, hw2 = False, True
    if hw1:
        question_8()
    if hw2:
        question_3()
