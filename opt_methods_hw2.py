import numpy as np

from methods.basic_mehtods import *
from methods.configuration_methods import *


def sgd():
    num_samples = rows
    x = x_initial.copy()
    A = A_pd if POSITIVE_DEFINITE else A_psd
    b = b_pd if POSITIVE_DEFINITE else b_psd
    optimal_xs = []
    indices = np.arange(1, num_samples)
    sgd_results = []
    for epoch in range(num_epochs):
        np.random.shuffle(indices)
        for i in indices:
            stochastic_gradient = calc_grad_fi(A=A_batch, b=b_batch, x=x, i=i)
            x -= eta * stochastic_gradient
            sgd_results.append(x)
        optimal_x = np.mean(sgd_results)
        optimal_xs.append(optimal_x)
        x = optimal_x.copy()
    return x, optimal_xs


def minibatch_sgd():
    num_samples = rows
    num_features = cols
    x = x_initial.copy()
    A = A_pd if POSITIVE_DEFINITE else A_psd
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
                gradient += calc_grad_fi(A=A_batch, b=b_batch, x=x, i=j)

            x -= learning_rate * gradient / batch_size  # Do we need to take the average of x in each epoch?
        optimal_xs.append(x)
    return x, optimal_xs


def svrg():
    num_samples = rows
    y = x_initial.copy()
    A = A_pd if POSITIVE_DEFINITE else A_psd
    b = b_pd if POSITIVE_DEFINITE else b_psd
    optimal_xs = []
    optimal_ys = []
    indices = np.arange(1, num_samples)
    num_batches = cnfg.num_batches
    batch_size = num_samples // num_batches
    arr = np.random.permutation(indices)
    for epoch in range(num_epochs):
        x = y.copy()
        full_gradient_y = calc_grad_F(A, b, y)
        A_shuffled = A[arr]
        b_shuffled = b[arr]
        A_batches = np.array_split(A_shuffled, num_batches)
        b_batches = np.array_split(b_shuffled, num_batches)
        for A_batch, b_batch in zip(A_batches, b_batches):
            # for i in range(num_batches):
            # batch_indices = arr[batch_size * i:batch_size * (i + 1)]
            j = np.random.choice(range(0, batch_size), size=1).item(0)
            # A_batch = A_shuffled[batch_size * i:batch_size * (i + 1)]  # A_shuffled[batch_indices]
            # b_batch = b_shuffled[batch_size * i:batch_size * (i + 1)]  # [batch_indices]
            stochastic_gradient_y = calc_grad_fi(A=A_batch, b=b_batch, i=j, x=x)
            stochastic_gradient_x = calc_grad_f(A=A_batch, b=b_batch, i=j, x=x)

            x -= step_size * (stochastic_gradient_x - stochastic_gradient_y + full_gradient_y)
            optimal_xs.append(x)
        y = np.mean(optimal_xs)
        optimal_ys.append(y)
    x = y.copy()
    return x, optimal_ys
