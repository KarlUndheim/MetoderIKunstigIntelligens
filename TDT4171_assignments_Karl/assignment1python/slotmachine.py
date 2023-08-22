import math
import random
import numpy as np

symbols = ["BAR", "BELL", "LEMON", "CHERRY"]

def spin(saldo):
    spins = 0
    while saldo-1>=0:
        saldo-=1
        i,j,k = [random.choice(symbols) for a in range(3)]
        payout = 0

        if len(set([i,j,k]))==1:
            if i=="BAR":
                saldo+=20
            elif i=="BELL":
                saldo+=15
            elif i=="LEMON":
                saldo+=5
            else:
                saldo+=3
            continue
        if (i=="CHERRY" and j=="CHERRY"):
            saldo+=2
        elif (i=="CHERRY"):
            saldo+=1
        spins+=1
    return spins
        
total = [spin(10) for i in range(10000)]
print(np.average(total))
print(np.median(total))