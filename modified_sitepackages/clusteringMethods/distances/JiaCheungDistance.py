from .base import distance

import numpy as np
import numba as nb
import pandas as pd


class jiangCheungDistance(distance):
    def __init__(self):
        super().__init__(need_complete_data= True)
        self.ps = None
        self.pf = None
        self.R = None
        self.S = None
        self.beta = None

    def reset_global_vars(self):
        self.pf = None
        self.ps = None
        self.S = None
        self.R = None
        self.beta = None

    def calculate_cond_prob(self, X, Ar, arj):
        return len(X[np.where(X[:, Ar] == arj)]) / len(X[np.where(X[:, Ar] != np.nan)])

    def calculate_rev_cond_prob(self, X, Ar, arj):
        return (len(X[np.where(X[:, Ar] == arj)])-1) / (len(X[np.where(X[:, Ar] != np.nan)])-1)

    def calculate_inter_prob(self, X, Ar, arj, Al, alk):
        return len(X[np.where(np.all((X[:, Ar] == arj, X[:, Al] == alk), axis= 0))]) / len(X[np.where(np.all((X[:, Ar] != np.nan, X[:, Al] != np.nan), axis= 0))])

    def calculate_rev_inter_prob(self, X, Ar, arj, Al, alk):
        return (len(X[np.where(np.all((X[:, Ar] == arj, X[:, Al] == alk), axis= 0))])-1) / (len(X[np.where(np.all((X[:, Ar] != np.nan, X[:, Al] != np.nan), axis= 0))])-1)

    def calculate_delta(self, xir, xjr):
        # Implement Eq. (2) here to calculate δ(xir , xjr)
        return 0 if xir == xjr else 1

    def calculate_ps(self, X, Ar):
        # Implement Eq. (27) here to calculate ps(Ar)
        return sum([self.calculate_cond_prob(X, Ar, arj) * self.calculate_rev_cond_prob(X, Ar, arj) for arj in np.unique(X[:, Ar])])

    def calculate_pf(self, X, Ar):
        # Implement Eq. (28) here to calculate pf(Ar)
        return 1 - self.calculate_ps(X, Ar)

    def calculate_I(self, X):
        # Implement Eq. (33) here to calculate I(Ar; Al)
        I = np.zeros((X.shape[1], X.shape[1]))
        for Ar in range(X.shape[1]):
            for Al in range(X.shape[1]):
                I[Ar, Al] = sum(map(lambda x: self.calculate_inter_prob(X, Ar, x[0], Al, x[1]) * np.nan_to_num(np.log(self.calculate_inter_prob(X, Ar, x[0], Al, x[1]) / (self.calculate_cond_prob(X, Ar, x[0]) * self.calculate_cond_prob(X, Al, x[1])))), np.array([(xi, yi) for xi in np.unique(X[:, Ar]) for yi in np.unique(X[:, Al])])))
        return I

    def calculate_H(self, X):
        # Implement Eq. (38) here to calculate H(Ar; Al)
        H = np.zeros((X.shape[1], X.shape[1]))
        for Ar in range(X.shape[1]):
            for Al in range(X.shape[1]):
                H[Ar, Al] = -sum(map(lambda x: self.calculate_inter_prob(X, Ar, x[0], Al, x[1]) * np.nan_to_num(np.log(self.calculate_inter_prob(X, Ar, x[0], Al, x[1]))), np.array([(xi, yi) for xi in np.unique(X[:, Ar]) for yi in np.unique(X[:, Al])])))
        return H

    def calculate_R(self, X):
        # Implement Eq. (37) here to calculate R(Ar; Al)
        I = self.calculate_I(X)
        H = self.calculate_H(X)
        R = I / H
        np.nan_to_num(R, copy= False)
        return R

    def calculate_global_vars(self, X):
        d = X.shape[1]
        # 3: Calculate ps(Ar) and p f(Ar) for each attribute Ar according to Eq.(27) and Eq.(28).
        self.ps = [self.calculate_ps(X, Ar) for Ar in range(d)]
        self.pf = [self.calculate_pf(X, Ar) for Ar in range(d)]
        # 4: For each pair of attributes(Ar, Al)(r, l ∈{1, 2, ..., d}), calculate R(Ar; Al) according to Eq.(37).
        # 5: Construct the relationship matrix R.
        self.R = self.calculate_R(X)
        np.fill_diagonal(self.R, 1)
        # calculate beta
        if self.beta is None:
            self.beta = (1 / d ** 2) * sum(sum(self.R))
        # 6: Get the index set Sr for each attribute Ar by Sr ={l | R(r, l) > β, 1 ≤ l ≤ d}.
        self.S = [np.where(self.R[Ar, :] > self.beta)[0] for Ar in range(d)]

    def calculate(self, xi: np.array, xj: np.array, X= None, beta= None, recalculateGlobalVars= False):
        if beta is not None:
            self.beta = beta
        if X is not None:
            self.X = X
        elif self.X is None:
            raise ValueError('X is not set. Either provide X as an argument or set it with preset_x(X)')

        if (None in (self.pf, self.ps, self.S) or self.R is None) or recalculateGlobalVars:
            self.calculate_global_vars(self.X)

        # 7: Choose two objects xi and xj from X.
        #xi = X[i, :]
        #xj = X[j, :]
        # 8: Let D(xi, xj ) = 0 and w = 0.
        D = 0
        w = 0
        # 9: for r = 1 to d do
        for r in range(len(xi)):
            # 10: if xir = x jr then
            if xi[r] != xj[r]:
                # 11: D(xir, xjr) = ∑ l∈Sr R(r, l) p((xir, xil) = (x jr, x jl))
                Dr = sum([self.R[r, l] * (self.calculate_inter_prob(self.X, r, xi[r], l , xi[l]) *
                                     self.calculate_rev_inter_prob(self.X, r, xi[r], l, xi[l]) +
                                     self.calculate_inter_prob(self.X, r, xj[r], l, xj[l]) *
                                     self.calculate_rev_inter_prob(self.X, r, xj[r], l, xj[l]))
                          for l in self.S[r]])
                # 12: wr = ps(Ar)
                wr = self.ps[r]
            # 13: else
            else:
                # 14: D(xir, xjr) = ∑ l∈Sr R(r, l) δ(xil, xjl) p((xir, xil) = (x jr, x jl))
                Dr = sum([self.R[r, l] * self.calculate_delta(xi[l], xj[l]) *
                                    (self.calculate_inter_prob(self.X, r, xi[r], l, xi[l]) *
                                     self.calculate_rev_inter_prob(self.X, r, xi[r], l, xi[l]) +
                                     self.calculate_inter_prob(self.X, r, xj[r], l, xj[l]) *
                                     self.calculate_rev_inter_prob(self.X, r, xj[r], l, xj[l]))
                          for l in self.S[r]])
                # 15: wr = pf(Ar)
                wr = self.pf[r]
            # 16: end if
            # 17: w = w + wr
            w += wr
            # 18: D(xi, xj ) = D(xi, xj ) + wrD(xir, xjr)
            D += wr * Dr
        # 19: end for
        # 20: D(xi, xj ) = D(xi, xj ) / w
        D = D/w
        return D

if __name__ == '__main__':
    jc_distance = jiangCheungDistance()
    dummy_array = np.array([[0, 0, 0, 0, 0],
                            [1, 0, 1, 0, 1],
                            [0, 0, 0, 0, 1],
                            [1, 1, 1, 0, 0]])
    dummy_distance = jc_distance.calculate(dummy_array, dummy_array[0], dummy_array[1])
    print(dummy_distance)
