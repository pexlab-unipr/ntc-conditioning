import numpy as np
a = np.ones(3)
b = np.ones((3,2))
c = a[:, np.newaxis] + b
d = a + b
