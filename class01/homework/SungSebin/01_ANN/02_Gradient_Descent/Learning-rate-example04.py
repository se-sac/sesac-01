import numpy as np
import matplotlib.pyplot as plt

# 1. 함수 정의 (비용 함수와 그 기울기)
def f(x):
    return x**2 - 4*x + 6

def grad_fx(x):
    return 2*x - 4

# 2. 경사하강법 함수 정의
def steepest_descent(func, grad_func, x0, learning_rate=0.01, MaxIter=10, verbose=True):
    paths = []
    for i in range(MaxIter):
        x1 = x0 - learning_rate * grad_func(x0)
        if verbose:
            print('{:03d}: x = {:1.4f}, f(x) = {:4.2E}'.format(i, x1, func(x1)))
        paths.append(x0)
        x0 = x1
    return x0, func(x0), paths

# 3. 경사하강법 실행
xopt, fopt, paths = steepest_descent(f, grad_fx, 1.0, learning_rate=1)

# 4. 결과 시각화 - 함수와 이동 경로
x = np.linspace(0.5, 3.5, 1000)
paths = np.array(paths)

plt.plot(x, f(x))                          # 전체 함수 그래프
plt.plot(paths, f(paths), 'o-', label='steps')  # 이동한 경로 시각화
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')
plt.legend()
plt.show()

# 5. 결과 시각화 - cost 값의 변화
plt.plot(f(paths), 'o-')
plt.grid()
plt.xlabel('step')
plt.ylabel('cost')
plt.title('plot of cost')
plt.show()