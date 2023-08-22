# TDT4171, Assignment 4, Karl Edvin Undheim

import numpy as np

# I will use newtons method to approximate R.
# This requires f and f'
# How I found f and fi is shown on paper.

def fn(R):
    return np.exp(-500/R)-2*np.exp(-100/R)+1

def dfn(R):
    return 5*np.exp(-500/R)-2*np.exp(-100/R)


# Newtons method with R0 and tolerance as inputs.
def newtons(R0, tol):
    R = R0
    while True:
        # x_n+1 = x_n - f/f'
        R_next = R - fn(R)/dfn(R)
        if abs(R_next-R)<tol:
            return R_next
        R = R_next

R_approx = newtons(100, 1e-3)

print("")
print("Approximated value of R: {:.3f}".format(R_approx))
print("")