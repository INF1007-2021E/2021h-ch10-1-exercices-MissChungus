#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:
    return np.linspace(-1.3, 2.5, 64)


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    return np.array([(np.sqrt(coord[0]**2 + coord[1]**2), np.arctan2(coord[1], coord[0])) for coord in cartesian_coordinates])


def find_closest_index(values: np.ndarray, number: float) -> int:
    return np.abs(values - number).argmin()


def create_plot():
    x_val = np.linspace(-1, 1, 250)
    y_val = (x_val**2)*np.sin(1/x_val**2) + x_val
    plt.scatter(x_val, y_val)
    plt.title("Exercice 4")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def monte_carlo(iteration: int=5000) -> float:
    x_int = []
    x_ext = []
    y_int = []
    y_ext = []
    x = np.random.random(iteration)
    y = np.random.random(iteration)

    for i in range(len(x)):
        if x[i]**2 + y[i]**2 <= 1:
            x_int.append(x[i])
            y_int.append(y[i])
        else:
            x_ext.append(x[i])
            y_ext.append(y[i])

    plt.scatter(x_int, y_int, s=10)
    plt.scatter(x_ext, y_ext, s=10)
    plt.title("Méthode de Monte Carlo")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    return len(x_int)/len(x) * 4


def integral():
    result = integrate.quad(lambda x: np.exp(-(x**2)), -np.inf, np.inf)
    x = np.linspace(-4, 4, 100)
    y = [integrate.quad(lambda x: np.exp(-(x**2)), 0, x_val)[0] for x_val in x]
    plt.plot(x, y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Intégrale de cette fonction entre [-4, 4]")
    plt.show()
    return result


if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    print(linear_values(), "\n")
    print(coordinate_conversion(cartesian_coordinates=np.array([(15, 30), (-7, 10)])), "\n")
    print(find_closest_index(np.array([10, 15, 20, 12, 13]), 12))
    create_plot()
    print(monte_carlo())
    print(integral())
