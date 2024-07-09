import numpy as np
import numba as nb

def calculate(X: np.array, i: int, j: int, beta= None):
    def calculate_cond_prob(X, Ar, arj):
        return len(X[np.where(X[:, Ar] == arj)]) / len(X[np.where(X[:, Ar] != np.nan)])

    def calculate_rev_cond_prob(X, Ar, arj):
        return (len(X[np.where(X[:, Ar] == arj)])-1) / (len(X[np.where(X[:, Ar] != np.nan)])-1)

    def calculate_inter_prob(X, Ar, arj, Al, alk):
        return len(X[np.where(np.all((X[:, Ar] == arj, X[:, Al] == alk), axis= 0))]) / len(X[np.where(np.all((X[:, Ar] != np.nan, X[:, Al] != np.nan), axis= 0))])

    def calculate_rev_inter_prob(X, Ar, arj, Al, alk):
        return (len(X[np.where(np.all((X[:, Ar] == arj, X[:, Al] == alk), axis= 0))])-1) / (len(X[np.where(np.all((X[:, Ar] != np.nan, X[:, Al] != np.nan), axis= 0))])-1)

    def calculate_delta(xir, xjr):
        # Implement Eq. (2) here to calculate δ(xir , xjr)
        return 0 if xir == xjr else 1

    def calculate_ps(X, Ar):
        # Implement Eq. (27) here to calculate ps(Ar)
        return sum([calculate_cond_prob(X, Ar, arj) * calculate_rev_cond_prob(X, Ar, arj) for arj in np.unique(X[:, Ar])])

    def calculate_pf(X, Ar):
        # Implement Eq. (28) here to calculate pf(Ar)
        return 1 - calculate_ps(X, Ar)

    def calculate_I(X):
        # Implement Eq. (33) here to calculate I(Ar; Al)
        I = np.zeros((X.shape[1], X.shape[1]))
        for Ar in range(X.shape[1]):
            for Al in range(X.shape[1]):
                I[Ar, Al] = sum(map(lambda x: calculate_inter_prob(X, Ar, x[0], Al, x[1]) * np.nan_to_num(np.log(calculate_inter_prob(X, Ar, x[0], Al, x[1]) / (calculate_cond_prob(X, Ar, x[0]) * calculate_cond_prob(X, Al, x[1])))), np.concatenate(np.meshgrid(np.unique(X[:, Ar]), np.unique(X[:, Al]), indexing='ij'), axis= 1)))
        return I

    def calculate_H(X):
        # Implement Eq. (38) here to calculate H(Ar; Al)
        H = np.zeros((X.shape[1], X.shape[1]))
        for Ar in range(X.shape[1]):
            for Al in range(X.shape[1]):
                H[Ar, Al] = -sum(map(lambda x: calculate_inter_prob(X, Ar, x[0], Al, x[1]) * np.nan_to_num(np.log(calculate_inter_prob(X, Ar, x[0], Al, x[1]))), np.concatenate(np.meshgrid(np.unique(X[:, Ar]), np.unique(X[:, Al]), indexing='ij'), axis= 1)))
        return H

    def calculate_R(X):
        # Implement Eq. (37) here to calculate R(Ar; Al)
        I = calculate_I(X)
        H = calculate_H(X)
        R = I / H
        return R

    d = X.shape[1]
    # 3: Calculate ps(Ar) and p f(Ar) for each attribute Ar according to Eq.(27) and Eq.(28).
    ps = [calculate_ps(X, Ar) for Ar in range(d)]
    pf = [calculate_pf(X, Ar) for Ar in range(d)]
    # 4: For each pair of attributes(Ar, Al)(r, l ∈{1, 2, ..., d}), calculate R(Ar; Al) according to Eq.(37).
    # 5: Construct the relationship matrix R.
    R = calculate_R(X)
    np.fill_diagonal(R, 1)
    # calculate beta
    if beta is None:
        beta = (1 / d ** 2) * sum(sum(R))
    # 6: Get the index set Sr for each attribute Ar by Sr ={l | R(r, l) > β, 1 ≤ l ≤ d}.
    S = [np.where(R[Ar, :] > beta)[0] for Ar in range(d)]

    # 7: Choose two objects xi and xj from X.
    xi = X[i, :]
    xj = X[j, :]
    # 8: Let D(xi, xj ) = 0 and w = 0.
    D = 0
    w = 0
    # 9: for r = 1 to d do
    for r in range(len(xi)):
        # 10: if xir = x jr then
        if xi[r] != xj[r]:
            # 11: D(xir, xjr) = ∑ l∈Sr R(r, l) p((xir, xil) = (x jr, x jl))
            Dr = sum([R[r, l] * (calculate_inter_prob(X, i, xi[r], l , xi[l]) *
                                 calculate_rev_inter_prob(X, i, xi[r], l, xi[l]) +
                                 calculate_inter_prob(X, j, xj[r], l, xj[l]) *
                                 calculate_rev_inter_prob(X, j, xj[r], l, xj[l]))
                      for l in S[r]])
            # 12: wr = ps(Ar)
            wr = ps[r]
        # 13: else
        else:
            # 14: D(xir, xjr) = ∑ l∈Sr R(r, l) δ(xil, xjl) p((xir, xil) = (x jr, x jl))
            Dr = sum([R[r, l] * calculate_delta(xi[l], xj[l]) *
                                (calculate_inter_prob(X, r, xi[r], l, xi[l]) *
                                 calculate_rev_inter_prob(X, r, xi[r], l, xi[l]) +
                                 calculate_inter_prob(X, r, xj[r], l, xj[l]) *
                                 calculate_rev_inter_prob(X, r, xj[r], l, xj[l]))
                      for l in S[r]])
            # 15: wr = pf(Ar)
            wr = pf[r]
        # 16: end if
        # 17: w = w + wr
        w += wr
        # 18: D(xi, xj ) = D(xi, xj ) + wrD(xir, xjr)
        D += wr * Dr
    # 19: end for
    # 20: D(xi, xj ) = D(xi, xj ) / w
    D = D/w
    return D

def distance_matrix(X: np.array, parallel=True):
    calculate_nb = nb.njit(calculate, fastmath=True, inline='always')
    def cust_dot_T(X):
        out = np.empty((X.shape[0], X.shape[0]), dtype=X.dtype)
        for i in nb.prange(X.shape[0]):
            for j in range(X.shape[0]):
                out[i, j] = calculate_nb(X[i], X[j])
        return out
    if parallel == True:
        return nb.njit(cust_dot_T, fastmath=True, parallel=True)(X)
    else:
        return nb.njit(cust_dot_T, fastmath=True, parallel=False)(X)

if __name__ == '__main__':
    dummy_array = np.array([[0, 0, 0, 0, 0],
                            [1, 0, 1, 0, 1],
                            [0, 0, 0, 0, 1],
                            [1, 1, 1, 0, 0]])
    calculate(dummy_array, 0, 1)
    dist_matrix = distance_matrix(dummy_array)
    print(dist_matrix)
