import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.ticker as mtick

# Constants
DPI = 150
FILE = True

def gradientDescent(X, y, theta, alpha, iterations):
    m, p = X.shape

    for _ in range(iterations):
        theta = theta - alpha / m * X.T @ (X @ theta - y)

    return theta

def computeCost(X, y, theta):
    m, p = X.shape
    pred = X @ theta
    return (1 / (2*m)) * ((pred - y) @ (pred - y).T)[0][0]

def plotCostEv(X, y, theta, alpha, iterations, file=False):
    costs = [computeCost(X, y, theta)]

    # Getting the values of the cost for the different iterations
    for _ in range(iterations):
        theta = gradientDescent(X, y, theta, alpha, 1)
        costs.append(computeCost(X, y, theta))

    fig = plt.figure(figsize=(4,2.75), dpi=DPI)
    ax = fig.add_subplot()
    ax.plot(costs)
    ax.set_ylim([0, max(costs)])
    ax.set_xlim([0, iterations])
    ax.title.set_text('Evolución del coste en el entrenamiento')
    ax.set_ylabel('$J(w, b)$')
    ax.set_xlabel('Nº de iteraciones')
    plt.tight_layout()

    # Outputing the plot to an image or showning it
    if file:
        plt.savefig('costEvolution.png', dpi=DPI)
    else:
        fig.show()

    plt.close(fig)

def plotThetaEv(X, y, theta, alpha, iterations, file=False):
    regCoeficients = [theta]

    # Getting the values of the cost for the different iterations
    for _ in range(iterations):
        theta = gradientDescent(X, y, theta, alpha, 1)
        regCoeficients.append(theta)

    fig = plt.figure(figsize=(6, 4), dpi=DPI)
    ax = fig.add_subplot()
    ax.scatter(X.T[0], y)
    for i, coeficients in enumerate(regCoeficients):
        line = X.dot(coeficients)
        ax.plot(X.T[0], line, color='r', alpha=(i / len(regCoeficients)))
    ax.title.set_text('Evolución de la regresión en el entrenamiento')
    ax.set_ylabel('Price')
    ax.set_xlabel('Engine Size')
    ax.set_xlim([0, max(X.T[0]) * 1.1])
    ax.set_ylim([0, max(y) * 1.1])
    plt.tight_layout()

    # Outputing the plot to an image or showning it
    if file:
        plt.savefig('thetaEvolution.png', dpi=DPI)
    else:
        fig.show()

    plt.close(fig)

def plot3dGradientEv(X, y, theta, alpha, iterations, file=False):
    regCoeficients = [[theta[0][0], theta[1][0], computeCost(X, y, theta)]]

    # Getting the values of the cost for the different iterations
    for _ in range(iterations):
        theta = gradientDescent(X, y, theta, alpha, 1)
        regCoeficients.append([theta[0][0], theta[1][0], (computeCost(X, y, theta) + 10)])

    t0 = np.arange(regCoeficients[0][0] - 20, regCoeficients[-1][0] + 20, 2)
    t1 = np.arange(regCoeficients[0][1] - 20, regCoeficients[-1][1] + 20, 2)
    T0, T1 = np.meshgrid(t0, t1)

    j = np.array([computeCost(X, y, np.array([[i], [j]])) for i, j in zip(np.ravel(T0), np.ravel(T1))])
    J = j.reshape(T0.shape)

    fig = plt.figure(figsize=(4,2.75), dpi=DPI)
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(T0, T1, J, cmap='rainbow', alpha=0.85, zorder=0)

    reg0, reg1, reg2 = zip(*regCoeficients)
    ax.plot(reg0, reg1, reg2, color='black', linestyle='-', marker='x', zorder=10, markersize=4)
    ax.title.set_text('Evolución de los coeficientes en el entrenamiento')
    ax.set_xlabel('$w$')
    ax.set_ylabel('$b$')
    ax.set_zlabel('$J(w, b)$')
    ax.zaxis.set_tick_params(labelsize=7)
    fig.tight_layout()

    # Outputing the plot to an image or showning it
    if file:
        plt.savefig('3dGradientEvolution.png', dpi=DPI, bbox_inches='tight')
    else:
        fig.show()
    plt.close(fig)

def main():
    # Reading csv
    df = pd.read_csv('/Users/alvarolopezgarcia/Documents/Documentos Universidad/4o Curso/2o Cuatrimestre/Trabajo de Fin de Grado - Archivos Locales/Ilustraciones/RegresionLineal/Dataset/Coches/CarPrice_Assignment.csv')

    plt.rcParams['text.usetex'] = True
    # Plotting price VS engine size
    plt.figure(figsize=(6,4),dpi=DPI)
    plt.scatter(df['enginesize'], df['price'])
    plt.xlabel('Engine Size')
    plt.ylabel('Price')
    plt.title('Coches en el mercado estadounidense')
    plt.xlim(0, max(df['enginesize']) * 1.1)
    plt.ylim(0, max(df['price']) * 1.1)

    # Outputing the plot to an image or showning it
    if FILE:
        plt.savefig('motorVSprice.png', dpi=DPI)
    else:
        plt.show()

    # Training parameters
    X = np.vstack((df['enginesize'], np.ones(df['enginesize'].shape[0]))).T # Stacking a column of 1s for matmul with theta_0
    y = np.array(([df['price']])).T
    theta_init = np.random.rand(2, 1)
    alpha = 0.000005
    num_iter = 25

    # Plotting regresion and cost evolution
    plotCostEv(X, y, theta_init, alpha, num_iter, file=FILE)
    plotThetaEv(X, y, theta_init, alpha, num_iter, file=FILE)
    plot3dGradientEv(X, y, theta_init, alpha, num_iter, file=FILE)

if __name__ == '__main__':
    main()