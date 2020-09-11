# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

loss=[]
with open('loss.log') as f:
    for l in f:
        loss.append(float(l.strip().split(',')[1].split()[0]))
plt.plot(loss[100:])
plt.xlabel("Iteration number")
plt.ylabel("loss")
plt.show()