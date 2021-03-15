import numpy as np 



array x
array y

meanX = np.full(x.shape, np.mean(x))
meanY = np.full(y.spahe, np.mean(y))

standardX = np.subtract(x, meanX)
standardY = np.subtract(y, meanY)

return np.dot(standardX, standardY) /  sqrt(np.sum(np.square(StandradX))) * np.sum(np.square(StandardY))))