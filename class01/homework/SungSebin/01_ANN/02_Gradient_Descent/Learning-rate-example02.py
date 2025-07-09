import numpy as np
import matplotlib.pylab as plt

def f(x):
  return x**2 - 4*x + 6

NumberOfPoints = 101
x = np.linspace(-5,5,NumberOfPoints)
fx = f(x)

xid = np.argmin(fx)
xopt = x[xid]
print(xopt, f(xopt))

plt.plot(x,f(x))
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')

plt.plot(xopt, f(xopt), 'xr')
plt.show()