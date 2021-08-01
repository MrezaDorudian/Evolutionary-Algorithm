import random

import matplotlib.pyplot as plt

with open('plot data/fitness.txt', 'r') as file:
    current_generation_fitness = []
    all_fitness = []
    data = file.read().split('\n')
    for d in range(len(data) - 1):
        current_generation_fitness.clear()
        for i in data[d].split(', '):
            i = i.replace('[', '')
            i = i.replace(']', '')
            current_generation_fitness.append(int(i))
        all_fitness.append(current_generation_fitness.copy())

    minimum = []
    maximum = []
    mean = []
    for generation in all_fitness:
        minimum.append(generation[-1])
        maximum.append(generation[0])
        mean.append(sum(generation) // len(generation))
    print(maximum)
    print(minimum)
    print(mean)
    plt.plot(maximum, c='green', label='Max')
    plt.plot(mean, c='yellow', label='Avg')
    plt.plot(minimum, c='red', label='Min')
    plt.legend()
    plt.show()
