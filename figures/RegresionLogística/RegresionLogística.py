import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from matplotlib import gridspec

# Constants
DPI = 300
FILE = True

def sigmoid(z):
    g = 0 * z
    g = 1 / (1 + np.exp(-z))
    return g

def computeCost(X, y, theta):

    m, p = X.shape
    J = 0.0

    h0x = sigmoid(X @ theta)

    J = - y.T @ np.log(h0x) - (1 - y).T @ np.log(1 - sigmoid(h0x))
    J = J[0, 0] / m

    return J

def gradientDescent(X, y, theta, alpha, iterations):

    m, p = X.shape
    for _ in range(iterations):
        theta = theta - (alpha / m) * X.T @ (sigmoid(X @ theta) - y)
    return theta

def plotCostEv(X, y, theta, alpha, iterations, file=False):
    costs = [computeCost(X, y, theta)]

    # Getting the values of the cost for the different iterations
    for _ in range(iterations):
        theta = gradientDescent(X, y, theta, alpha, 1)
        costs.append(computeCost(X, y, theta))

    fig = plt.figure(figsize=(4,2.75), dpi=DPI)
    ax = fig.add_subplot()
    ax.plot(costs)
    ax.set_ylim([0.9 * min(costs), 1.1 * max(costs)])
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

    dataset = np.hstack([X[:,:2], y])

    red, white = dataset[dataset[:,2]==0.], dataset[dataset[:,2]==1.]

    fig = plt.figure(figsize=(6, 4), dpi=DPI)
    ax = fig.add_subplot()
    ax.scatter(white[:, 0], white[:, 1], s=15, marker='^', linewidths=0.5)
    ax.scatter(red[:, 0], red[:, 1], s=15, marker='x', linewidths=0.5)
    for i, coeficients in enumerate(regCoeficients):
        if i % 250 == 0: # Painting 1 in 250 lines
            plot_x = [min(X[:,0]),  max(X[:,0])]
            plot_y = [min(X[:,1]),  max(X[:,1])]
            umbral = [np.array([i, j, 1]).T @ np.array(coeficients) for i , j in zip(plot_x, plot_y)]
            ax.plot(plot_x, umbral, color='r', alpha=np.sqrt(i / (len(regCoeficients)-1)))
    ax.title.set_text('Evolución del umbral de decisión en el entrenamiento')
    ax.set_xlabel('Total Sulfur Dioxide')
    ax.set_ylabel('Alcohol')
    ax.set_xlim([0, max(dataset[:, 0]) * 1.1])
    ax.set_ylim([min(dataset[:, 1]) * 0.9, max(dataset[:, 1]) * 1.1])
    ax.legend(('Vinos Blancos', 'Vinos Tintos'),
               scatterpoints=1,
               loc='upper right',
               fontsize=8)
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

    # Updating the values with the good bias parameter in order to plot a 3d space
    good_bias = theta[2][0]
    not_dropped = []
    for idx, item in enumerate(regCoeficients):
        if idx % 2000 == 0:
            new_theta = np.array([np.hstack([item[0], item[1], good_bias])]).T
            regCoeficients[idx] = [item[0], item[1], computeCost(X, y, new_theta)]
            not_dropped.append(idx)

    regCoeficients = np.array(regCoeficients)[not_dropped]

    t0 = np.arange(regCoeficients[0][0] - 1, regCoeficients[-1][0] + 1, 1)
    t1 = np.arange(regCoeficients[0][1] - 1, regCoeficients[-1][1] + 1, 1)
    T0, T1 = np.meshgrid(t0, t1)

    j = np.array([computeCost(X, y, np.array([[i], [j], [good_bias]])) for i, j in zip(np.ravel(T0), np.ravel(T1))])
    J = j.reshape(T0.shape)

    fig = plt.figure(figsize=(4,2.75), dpi=DPI)
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(T0, T1, J, cmap='rainbow', alpha=0.85, zorder=0)

    reg0, reg1, reg2 = zip(*regCoeficients)
    ax.plot(reg0, reg1, reg2, color='black', linestyle='-', marker='x', zorder=10, markersize=4)
    ax.title.set_text('Evolución de los coeficientes en el entrenamiento')
    ax.set_xlabel('$w_{1}$')
    ax.set_ylabel('$w_{2}$')
    ax.set_zlabel('$J(w, b)$')
    plt.tight_layout()

    # Outputing the plot to an image or showning it
    if file:
        plt.savefig('3dGradientEvolution.png', bbox_inches='tight', dpi=DPI)
    else:
        fig.show()
    plt.close(fig)

def main():
    plt.rcParams['text.usetex'] = True

    # Reading csvs
    df_red = pd.read_csv(r'Dataset/Vinos/winequality-red.csv', sep=';')
    df_white = pd.read_csv(r'Dataset/Vinos/winequality-white.csv', sep=';')

    df_red['type'] = 0
    df_white['type'] = 1

    df = pd.concat([df_red[['total sulfur dioxide', 'alcohol', 'type']], \
                    df_white[['total sulfur dioxide', 'alcohol', 'type']]])

    figure, axes = plt.subplots(figsize=[6, 6], nrows=2, ncols=2, gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [1, 3]})
    axes = axes.flat

    figure.suptitle('Vinos Blancos VS Vinos Tintos')

    # scatterplot
    axes[2].scatter(df_white['total sulfur dioxide'], df_white['alcohol'], s=15, marker='^', linewidths=0.5)
    axes[2].scatter(df_red['total sulfur dioxide'], df_red['alcohol'], s=15, marker='x', linewidths=0.5)
    axes[2].legend(('Vinos Blancos', 'Vinos Tintos'),
              scatterpoints=1,
              loc='upper right',
              fontsize=8)
    axes[2].set_xlim([0, max(df['total sulfur dioxide']) * 1.1])
    axes[2].xaxis.set_ticks(np.arange(0, math.ceil(max(df['total sulfur dioxide']) * 1.1), 25))
    axes[2].set_ylim([min(df['alcohol']) * 0.9, max(df['alcohol']) * 1.1])
    axes[2].yaxis.set_ticks(np.arange(int(min(df['alcohol']) * 0.9), math.ceil(max(df['alcohol']) * 1.1),0.5))
    axes[2].set_xticks(axes[2].get_xticks()[::2])
    axes[2].set_yticks(axes[2].get_yticks()[::2])
    axes[2].set_xlabel('Total Sulfur Dioxide')
    axes[2].set_ylabel('Alcohol')

    # Hide axis
    axes[1].axis('off')

    # histograms
    # horizontal
    axes[0].hist(df_white['total sulfur dioxide'], edgecolor='black', bins=np.arange(0, math.ceil(max(df['total sulfur dioxide']) * 1.1), 25))
    axes[0].hist(df_red['total sulfur dioxide'], edgecolor='black', bins=np.arange(0, math.ceil(max(df['total sulfur dioxide']) * 1.1), 25))
    axes[0].set_xticklabels([])
    axes[0].set_xlim([0, max(df['total sulfur dioxide']) * 1.1])
    axes[0].xaxis.set_ticks(np.arange(0, math.ceil(max(df['total sulfur dioxide']) * 1.1), 25))

    # vertical
    axes[3].hist(df_white['alcohol'], orientation='horizontal', edgecolor='black', bins=np.arange(int(min(df['alcohol']) * 0.9), math.ceil(max(df['alcohol']) * 1.1), 0.5))
    axes[3].hist(df_red['alcohol'], orientation='horizontal', edgecolor='black', bins=np.arange(int(min(df['alcohol']) * 0.9), math.ceil(max(df['alcohol']) * 1.1), 0.5))
    axes[3].set_yticklabels([])
    axes[3].set_ylim([min(df['alcohol']) * 0.9, max(df['alcohol']) * 1.1])
    axes[3].yaxis.set_ticks(np.arange(int(min(df['alcohol']) * 0.9), math.ceil(max(df['alcohol']) * 1.1), 0.5))

    figure.tight_layout(h_pad=0.3, w_pad=0.5)

    # Outputing the plot to an image or showning it
    if FILE:
        plt.savefig('alcoholVStotal sulfur dioxide.png', dpi=600)
    else:
        plt.show()

    # Training parameters
    X = np.vstack(((df['total sulfur dioxide']-df['total sulfur dioxide'].min())/(df['total sulfur dioxide'].max()-df['total sulfur dioxide'].min()), \
                   (df['alcohol']-df['alcohol'].min())/(df['alcohol'].max()-df['alcohol'].min()), \
                   np.ones(df['alcohol'].shape[0]))).T # Stacking a column of 1s for matmul with theta_0
    y = np.array(([df['type']])).T
    theta_init = np.random.rand(X.shape[1], 1)
    alpha = 0.1
    num_iter = 30000

    # Plotting regresion and cost evolution
    plotCostEv(X, y, theta_init, alpha, num_iter, file=FILE)
    plotThetaEv(X, y, theta_init, alpha, num_iter, file=FILE)
    plot3dGradientEv(X, y, theta_init, alpha, num_iter, file=FILE)

    # Plotting sigmoid function
    plt.figure(figsize=(4,2.75), dpi=DPI)
    x = np.arange(-10, 10, 0.05)
    y = 1 / (1 + np.exp(-x))
    plt.plot(x, y)
    plt.xlabel('$x$')
    plt.ylabel('$\sigma(x)$')
    plt.title('Función Sigmoide')
    plt.xlim(-10, 10)
    plt.ylim(min(y) -0.2, max(y) * 1.2)
    plt.axhline(y=0.5, color='r', linestyle='--', linewidth=1)
    plt.text(-9, 0.525, '0.5', rotation=360, color='r')
    plt.axhline(y=0, color='grey', linestyle='--', linewidth=1)
    plt.axhline(y=1, color='grey', linestyle='--', linewidth=1)
    plt.axvline(x=0, color='grey', linestyle='--', linewidth=1)
    plt.grid(axis='both', which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(axis='x', which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.minorticks_on()

    # Outputing the plot to an image or showning it
    if FILE:
        plt.savefig('sigmoid.png', dpi=DPI)
    else:
        plt.show()

if __name__ == '__main__':
    main()