# Copyright (c) 2019 Pablo Moreno-Munoz
# Universidad Carlos III de Madrid
#
# The Bayesian CPD computation is based on
# the original code by R. P. Adams (2006).
#
# INFINITE HIERARCHICAL CHANGE-POINT DETECTION

import numpy as np
from scipy.stats import norm
import random
import matplotlib.pyplot as plt

class infiniteHierCPD():
    def __init__(self, X, hyp_lambda=None, hyp_alpha=None, name='infiniteHierCPD'):

        self.X = X
        self.T = X.shape[0]
        self.Xnan = np.isnan(self.X)

        self.z_t = np.zeros((self.T,1))
        self.r_map = np.zeros((self.T + 1, 1))

        # Initialization of hyperparameters
        if hyp_lambda is not None:
            self.hyp_lambda = hyp_lambda
        else:
            self.hyp_lambda = 1e6

        if hyp_alpha is not None:
            self.hyp_alpha = hyp_alpha
        else:
            self.hyp_alpha = 1.0

        # Initialization of the infinite latent-class model
        self.K_t = 1
        self.Kmax = 1000
        self.m_k = np.zeros((self.Kmax, 1))
        mu_0 = np.random.randn()
        var_0 = np.random.rand()
        self.mu_tk = np.zeros((self.T, self.Kmax))
        self.var_tk = np.ones((self.T, self.Kmax))
        self.mu_tk[0, 0] = mu_0
        self.var_tk[0, 0] = var_0


        # Initialization of the CP detector
        self.R = np.zeros((self.T+1,1))
        self.R[0,0] = 1
        self.mk0 = np.zeros((1, self.Kmax))
        self.mkT = self.mk0

    def continual_detection(self):

        p_z = np.zeros((self.K_t + 1, 1))
        maxes = np.empty((self.T + 1, 1))
        maxes[0] = 0

        lr_mean = 1.0 * np.ones((self.Kmax, 1))  # learning rate for mean
        lr_var = 0.02 * np.ones((self.Kmax, 1))  # learning rate for variance
        adaptive_rule = 0.95

        for t in range(self.T-1):
            if t == 0:
                # First step
                self.z_t[t] = 1
                self.m_k[t] += 1
                self.mu_tk[t+1,0] = self.mu_tk[t,0] - lr_mean[0]*(self.X[t]-self.mu_tk[t,0])/self.var_tk[t,0]
                self.var_tk[t+1,0] = self.var_tk[t,0] + lr_var[0]*(0.5*(self.X[t] - self.mu_tk[t,0])**2/(self.var_tk[t,0]**2) - (0.5/self.var_tk[t,0]))
                self.mu_tk[t+1, self.K_t] = np.random.randn()
                self.var_tk[t+1, self.K_t] = np.random.rand()
            else:
                for k in range(self.K_t + 1):
                    # Likelihood Evaluation
                    p_z[k] = norm.pdf(self.X[t], self.mu_tk[t, k], 1 / np.sqrt(self.var_tk[t, k]))

                # Expectation
                p_z_cond = self.crp_expectation(t+1)
                p_z = p_z / p_z.sum()
                p_z = p_z[:self.K_t + 1]

                # Stochastic Maximization
                for k in range(int(self.K_t)):
                    self.mu_tk[t+1, k] = self.mu_tk[t, k] + lr_mean[k]*(self.X[t+1] - self.mu_tk[t,k])/self.var_tk[t,k]
                    self.var_tk[t+1, k] = self.var_tk[t, k] + lr_var[k]*(0.5*(self.X[t+1] - self.mu_tk[t,k]) ** 2 / (self.var_tk[t,k]**2) - (0.5/self.var_tk[t,k]))

                # MAP Estimate
                expected_k = np.argmax(p_z) + 1
                self.z_t[t] = expected_k

                k_new = np.random.binomial(n=1, p=p_z_cond[-1])
                if k_new == 1:
                    self.mu_tk[t+1, self.K_t] = self.X[t+1]
                    self.var_tk[t+1, self.K_t] = 0.25

                    p_z = np.vstack((p_z, np.zeros((1, 1))))
                    self.m_k[expected_k-1] += 1
                    self.K_t = np.sum(self.m_k > 0)

                else:
                    self.m_k[expected_k - 1] += 1
                    lr_mean[expected_k - 1] = lr_mean[expected_k - 1] * adaptive_rule
                    lr_var[expected_k - 1] = lr_var[expected_k - 1] * adaptive_rule

            # INFINITE HierCPD
            zbin = self.onehot_infinite(self.z_t[t])
            pred = self.crp_predictive(zbin, t)
            H = self.constant_hazard(np.arange(t + 1) + 1)
            self.R[0] = np.sum(self.R[0:t + 1] * pred * H)
            self.R[1:t + 2] = self.R[0:t + 1] * pred * (1 - H)
            self.R[:] = self.R[:] / np.sum(self.R[:])
            mkT0 = np.vstack((self.mk0, self.mkT + np.tile(zbin, (self.mkT.shape[0], 1))))
            self.mkT = mkT0
            maxes[t + 1] = np.where(self.R[1:, 0] == np.max(self.R[1:, 0]))
            self.r_map[t] = maxes[t + 1]

            # DISPLAY
            print('t=' + str(t + 1) + '  z=' + str(int(self.z_t[t, 0])) + '   K_t=' + str(self.K_t + 1))

    def crp_expectation(self, t):
        # Chinese Restaurant Process (CRP) predictive distribution (Latent Variable Model)
        expectations = np.zeros((self.K_t + 1, 1))
        denominator = t - 1 + self.hyp_alpha
        expectations[self.K_t, 0] = self.hyp_alpha / denominator
        expectations[0:self.K_t, 0] = self.m_k[0:self.K_t, 0] / denominator
        return expectations


    def onehot_infinite(self, z):
        # One-hot encoding of Categorical data with an unbounded numbers of classes
        Z_onehot = np.zeros((1, self.Kmax))
        for k in range(self.Kmax):
            Z_onehot[:, k, None] = (z == k + 1).astype(np.int)
        return Z_onehot


    def constant_hazard(self, r_vector):
        # Probability of CP per unit of time
        H = (1. / self.hyp_lambda) * np.ones((r_vector.shape[0], 1))
        return H


    def crp_predictive(self, zbin, t):
        # CRP predictive equations (Detector)
        predictive = self.mkT[:, zbin[0, :] == 1] / (np.arange(t + 1)[:, None] + self.hyp_alpha)
        predictive[predictive == 0] = self.hyp_alpha / (t + self.hyp_alpha)
        return predictive


    def plot_detection(self):
        plt.figure(figsize=[15, 6])
        colors = np.random.rand(self.Kmax, 3)
        time = np.arange(0, self.X.shape[0])
        plt.subplot(211)
        plt.plot(self.X, 'black', linewidth=0.5, alpha=0.5)
        for k in range(self.K_t):
            ix_k = self.z_t == k + 1
            plt.plot(time[ix_k.flatten()], self.X[ix_k.flatten(), 0], color=tuple(colors[k, :]), marker='x', linestyle='',
                     markersize=2.0)

        plt.title(r'Infinite Hierarchical Change-Point Detection')
        plt.ylabel(r'Nuclear Response')
        plt.xlim(0, self.T)

        plt.subplot(212)
        plt.plot(self.r_map[:-2], color='red', linewidth=2.0)
        plt.xlim(0, self.T-1)
        plt.ylabel(r'Run length $r_t$')
        plt.xlabel(r'Time')
        plt.show()