import numpy as np
import matplotlib.pyplot as plt

traceFile = np.genfromtxt('taskTraceFile.csv', delimiter=',')
traceFile[:, 0] = traceFile[:, 0]/1000000
traceFile[:, 1] = traceFile[:, 1]/1000000
# print(traceFile)
fig, axs = plt.subplots(1, 1)
# print(traceFile[traceFile[:, 2] == 0, :])

maxWorkerNum = np.max(traceFile[:, 2])
print("Workers: ", int(maxWorkerNum))

localTasks = traceFile[traceFile[:, 5] == 0, :]
localTasks = localTasks[localTasks[:, 2] != 0, :]
stolenTasks = traceFile[traceFile[:, 5] == 1, :]
stolenTasks = stolenTasks[stolenTasks[:, 2] != 0, :]
priStolenTasks = traceFile[traceFile[:, 5] == 2, :]
priStolenTasks = priStolenTasks[priStolenTasks[:, 2] != 0, :]

mtWrapperInstances = traceFile[traceFile[:, 2] == 0, :]
print("MTWrapper Instances: ", mtWrapperInstances.shape[0])

# print(localTasks)
axs.barh(mtWrapperInstances[:, 2], mtWrapperInstances[:, 1], left=mtWrapperInstances[:, 0], align='edge', height=(maxWorkerNum+1), edgecolor='#000000', facecolor='#CCCCCC')
axs.barh(localTasks[:, 2], localTasks[:, 1], left=localTasks[:, 0], height=1, edgecolor='#000000', facecolor='b')
axs.barh(stolenTasks[:, 2], stolenTasks[:, 1], left=stolenTasks[:, 0], height=1, edgecolor='#000000', facecolor='r')
axs.barh(priStolenTasks[:, 2], priStolenTasks[:, 1], left=priStolenTasks[:, 0], height=1, edgecolor='#000000', facecolor='g')
for instance in mtWrapperInstances:
    axs.text((instance[0]+(instance[1]/2)), 0, int(instance[4]), horizontalalignment='center', verticalalignment='top', fontsize=6, rotation=270)

axs.set_xlabel("Program Execution Time")
axs.set_ylabel("Worker Number (Starts at 1)")
axs.set_title("DAPHNE Task Trace Visualization")
axs.set_ylim([(maxWorkerNum/4*-1), maxWorkerNum+2])
axs.set_yticks(list(range(1, int(maxWorkerNum)+1)))

plt.show()

