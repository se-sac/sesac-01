import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

def contour(f, x, y, level = np.logspace(0, 5, 35)):
	fig, ax = plt.subplots(figsize=(8, 8))
	ax.contour(x, y, f(x,y), levels=level, norm=LogNorm(), cmap=plt.cm.jet)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')
	plt.show()

def contour_with_quiver(f, x, y, grad_x, grad_y, norm=LogNorm(), level = np.logspace(0, 5, 35),
	minima=None):
	dz_dx = grad_x(x,y)
	dz_dy = grad_y(x,y)
	fig, ax = plt.subplots(figsize=(6, 6))
	ax.contour(x, y, f(x,y), levels=level, norm=norm, cmap=plt.cm.jet)
	if minima is not None:
		ax.plot(*minima, 'r*', markersize=18)
	ax.quiver(x, y, -dz_dx, -dz_dy, alpha=.5)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')
	plt.show()

def surf(f, x, y, norm=LogNorm(), minima=None):
	fig = plt.figure(figsize=(8, 5))
	ax = plt.axes(projection='3d', elev=50, azim=-50)
	ax.plot_surface(x, y, f(x,y), norm=norm, rstride=1, cstride=1,
	                edgecolor='none', alpha=.8, cmap=plt.cm.jet)
	if minima is not None:
		ax.plot(*minima, f(*minima), 'r*', markersize=10)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')
	ax.set_zlabel('$z$')
	plt.show()

def contour_with_path(f, x, y, paths, norm=LogNorm(), level=np.logspace(0, 5, 35), minima=None):
	fig, ax = plt.subplots(figsize=(6, 6))

	ax.contour(x, y, f(x,y), levels=level, norm=norm, cmap=plt.cm.jet)
	ax.quiver(paths[0,:-1], paths[1,:-1], paths[0,1:]-paths[0,:-1], paths[1,1:]-paths[1,:-1], scale_units='xy', angles='xy', scale=1, color='k')
	if minima is not None:
		ax.plot(*minima, 'r*', markersize=18)

	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')

  # ax.set_xlim((xmin, xmax))
	# ax.set_ylim((ymin, ymax))

	plt.show()

# 1. 그래프 그리기 위한 격자 설정
xmin, xmax, xstep = -4.0, 4.0, 0.25
ymin, ymax, ystep = -4.0, 4.0, 0.25

x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep),
                   np.arange(ymin, ymax + ystep, ystep))

# 2. 목적 함수 f(x, y) 정의
f = lambda x, y: (x - 2)**2 + (y - 2)**2
z = f(x, y)
minima = np.array([2., 2.])  # 실제 최소값
print(f(*minima))            # 최소 지점에서 함수값 확인

# 3. 시각화용 기준
minima_ = minima.reshape(-1, 1)
surf(f, x, y, minima=minima - minima)  # 중심이 (0,0)인 그래프

# 4. 도함수 정의 (편미분)
grad_f_x = lambda x, y: 2 * (x - 2)
grad_f_y = lambda x, y: 2 * (y - 2)

# 5. Gradient Descent 함수 정의
def steepest_descent_twod(func, gradx, grady, x0, MaxIter=20, learning_rate=0.1, verbose=True):
    paths = [x0]
    fval_paths = [func(x0[0], x0[1])]

    for i in range(MaxIter):
        grad = np.array([gradx(*x0), grady(*x0)])
        x1 = x0 - learning_rate * grad
        fval = func(*x1)
        if verbose:
            print(f"{i:03d}: x = {x1}, f(x) = {fval:.4E}")
        x0 = x1
        paths.append(x0)
        fval_paths.append(fval)

    paths = np.array(paths)
    fval_paths = np.array(fval_paths)
    return x0, fval, paths, fval_paths

# 6. 실행
x0 = np.array([-2., -2.])
xopt, fopt, paths, fval_paths = steepest_descent_twod(f, grad_f_x, grad_f_y, x0)
print(xopt)
print(fopt)
print(paths)
# 7. 시각화

contour_with_quiver(f, x, y, grad_f_x, grad_f_y, minima=minima)
contour_with_path(f, x, y, paths.T, minima=np.array([[2], [2]]))
