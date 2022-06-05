import numpy as np
import matplotlib.pyplot as plt

traceFile = np.genfromtxt('taskTraceFile.csv', delimiter=',')
traceFile[:, 0] = traceFile[:, 0]/1000000
traceFile[:, 1] = traceFile[:, 1]/1000000
# print(traceFile)
fig, axs = plt.subplots(1, 1)
# print(traceFile[traceFile[:, 2] == 0, :])
localTasks = traceFile[traceFile[:, 5] == 0, :]
stolenTasks = traceFile[traceFile[:, 5] == 1, :]
priStolenTasks = traceFile[traceFile[:, 5] == 2, :]
# print(localTasks)
axs.barh(localTasks[:, 2], localTasks[:, 1], left=localTasks[:, 0], height=1, edgecolor='#000000', facecolor='b')
axs.barh(stolenTasks[:, 2], stolenTasks[:, 1], left=stolenTasks[:, 0], height=1, edgecolor='#000000', facecolor='r')
axs.barh(priStolenTasks[:, 2], priStolenTasks[:, 1], left=priStolenTasks[:, 0], height=1, edgecolor='#000000', facecolor='g')

plt.show()

