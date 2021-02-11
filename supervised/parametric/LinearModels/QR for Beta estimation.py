# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 15:03:02 2021

@author: jkcle
"""
import numpy as np
from scipy.stats import norm
from datetime import datetime


size = 20000
dims = 1000

beta = norm(loc=10, scale=10).rvs(dims)

X = np.repeat(1, repeats=size)

for i in range(1, dims):
    X = np.concatenate([X, norm(0., scale=5.).rvs(size)])

X = np.transpose(X.reshape(dims, size))
y = np.matmul(X, beta) + norm(loc=0, scale=3).rvs(size)

startTime = datetime.now()
Q, R = np.linalg.qr(X)
z = np.matmul(np.transpose(Q), y)
beta_hat_qr = np.linalg.solve(R, z)
print(datetime.now() - startTime)

startTime = datetime.now()
XtX = np.matmul(np.transpose(X), X)
XtX_inv = np.linalg.solve(XtX, np.eye(dims))
XtX_inv_Xt = np.matmul(XtX_inv, np.transpose(X))
beta_hat_naive = np.matmul(XtX_inv_Xt, y)
print(datetime.now() - startTime)