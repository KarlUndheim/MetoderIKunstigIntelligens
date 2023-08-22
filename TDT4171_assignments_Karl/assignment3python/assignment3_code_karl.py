# TDT4171, Assignment 3, Karl Edvin Undheim

import numpy as np

# Dynamic model described in exercise 1. 
T = np.asarray([[0.7, 0.3],
                [0.3, 0.7]])

# The two matrices for the observation model
# O_u is when the umbrella is seen, O_not_u is when it is not seen
O_u = np.asarray([[0.9, 0.0], 
                  [0.0, 0.2]])

O_not_u = np.asarray([[0.1, 0.0], 
                      [0.0, 0.8]])

# This is just a list for connecting index 0 to O_not_u and index 1 to O_u.
sensorModel = [O_not_u, O_u]
# This way I can access the desired observation model 
# for each step of the recursion in the forward algorithm.

# The evidence for the first task. Umbrella is seen on both days.
evidence = [1,1]

# The forward operation is implemented as shown in the textbook; by recursion.
def forward(t):
    # On day 0 I assume that there is a 50% chance of rain, as no other info is known.
    if t == 0:
        return np.array([0.5, 0.5])
    else:
        # Note the constant is implemented afterwards so that the probabilities sum to 1.
        return sensorModel[evidence[t-1]].dot(T.T).dot(forward(t-1))

def normalized_forward(t):
    # Probabilities, sum != 1
    probability = forward(t)
    # Constant to multiply the result with so that the probabilities sum to 1.
    const = 1/(sum(probability))
    # Probabilities, sum = 1
    probability = probability*const
    return probability

print("")
print("Task 1")
print("Probability of rain at day 2: {:.3f}".format(normalized_forward(2)[0]))
print("")
print("")


print("Task 2")
# Task 2, new evidence
evidence = [1,1,0,1,1]
for i in range(6):
    print("Normalized forward message day {}:".format(i))
    print(normalized_forward(i))
    print("")
print("Probability of rain at day 5: {:.3f}".format(normalized_forward(5)[0]))
print("")
