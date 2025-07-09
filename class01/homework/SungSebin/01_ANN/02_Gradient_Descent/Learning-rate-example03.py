import numpy as np
import matplotlib.pylab as plt

def f(x):
  return x**2 - 4*x + 6

def grad_fx(x):
  return 2*x - 4

def steepest_descent(func, grad_func, x0, learning_rate=0.01, MaxIter=10, verbose=True):
    paths = []
    for i in range(MaxIter):
        x1 = x0 - learning_rate * grad_func(x0)
        if verbose:
            print('{:03d}: x = {:1.4f}, f(x) = {:4.2E}'.format(i, x1, func(x1)))
        paths.append(x0)
        x0 = x1
    return x0, func(x0), paths

xopt, fopt, paths = steepest_descent(f, grad_fx, 0.0, learning_rate=1.2)

x = np.linspace(0.5, 2.5, 1000)
paths = np.array(paths)  # ← 이건 경사하강법 실행 결과

plt.plot(x, f(x))
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')
plt.plot(paths, f(paths), 'o-')
plt.show()

plt.plot(f(paths), 'o-')
plt.grid()
plt.xlabel('x')
plt.ylabel('cost')
plt.title('plot of cost')
plt.show()