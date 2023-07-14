import numpy as np


def calc_F(A, b, x):
    fx = 0.5 * np.linalg.norm(A @ x - b) ** 2
    return fx


def calc_grad_F(A, b, x):
    x = x.reshape((-1,))
    b = b.reshape((-1,))
    grad_x = A.T @ (A @ x - b)
    return grad_x.reshape((-1,))


def calc_fi(A, b, i, x):
    A_i = A[i]
    b_i = b[i]
    fi_x = 0.5 * np.linalg.norm(np.dot(A_i, x) - b_i) ** 2
    return fi_x


def calc_grad_fi(A, b, i, x):
    A_i = A[i]
    b_i = b[i]
    df_i = (np.dot(A_i, x) - b_i) * A_i
    return df_i.reshape((-1,))


def calc_R(A):
    R = norm(np.linalg.pinv(A) @ b) + 1
    return R


def calc_G(A, b):
    sigma_max = np.sqrt(max(np.linalg.eigvals(A.T @ A)))
    G = (sigma_max ** 2) * R(A) + sigma_max * norm(b)
    return G


def calc_beta(A):
    beta = max(np.linalg.eigvals(A)) ** 2
    return beta


def log():
    return np.log(x)


def calc_b(A):
    b = A @ x_star + np.random.normal(0, 0.2, (rows, 1))
    return b
