import random

from player import Player
import numpy as np
import copy
from config import CONFIG
import random


class Evolution():

    def __init__(self, mode):
        self.mode = mode

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutate(self, child):
        for i in range(len(child.nn.w_matrix_layer_1)):
            for j in range(len(child.nn.w_matrix_layer_1[i])):
                chance_1 = random.uniform(0, 1)
                noise_1 = np.random.normal(0, 1)
                if chance_1 > 0.9:
                    child.nn.w_matrix_layer_1[i][j] += noise_1

        for i in range(len(child.nn.w_matrix_layer_2)):
            for j in range(len(child.nn.w_matrix_layer_2[i])):
                chance_2 = random.uniform(0, 1)
                noise_2 = np.random.normal(0, 1)
                if chance_2 > 0.9:
                    child.nn.w_matrix_layer_2[i][j] += noise_2

        for i in range(len(child.nn.b_vector_layer_1)):
            chance_3 = random.uniform(0, 1)
            noise_3 = np.random.normal(0, 1)
            if chance_3 > 0.9:
                child.nn.b_vector_layer_1[i] += noise_3

        for i in range(len(child.nn.b_vector_layer_2)):
            chance_4 = random.uniform(0, 1)
            noise_4 = np.random.normal(0, 1)
            if chance_4 > 0.9:
                child.nn.b_vector_layer_2[i] += noise_4
                child.nn.b_vector_layer_2[i] += noise_4
        return child

    def generate_new_population(self, num_players, prev_players=None):
        # in first generation, we create random players
        if prev_players is None:
            open('plot data/fitness.txt', 'w')
            return [Player(self.mode) for _ in range(num_players)]

        else:
            new_players = []
            sum_of_all_fitness = sum([player.fitness for player in prev_players])
            selection_probability = [player.fitness / sum_of_all_fitness for player in prev_players]
            for i in range(num_players):
                new_players.append(prev_players[np.random.choice(len(prev_players), p=selection_probability, replace=False)])

            mutated_players = []
            for player in new_players:
                mutated_players.append(self.mutate(copy.deepcopy(player)))
            return mutated_players

    def next_population_selection(self, players, num_players):
        players.sort(key=lambda player: player.fitness, reverse=True)
        with open('plot data/fitness.txt', 'a') as file:
            to_write = str([player.fitness for player in players])
            file.write(to_write)
            file.write('\n')
        return players[: num_players]
