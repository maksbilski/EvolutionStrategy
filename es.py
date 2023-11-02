import torch
import random
import math

from typing import Callable


def tournament_selection(
        fitness_function: Callable[[torch.Tensor], float],
        population: torch.Tensor,
        count_of_selected_individuals: int
        ) -> torch.Tensor:
    selected_individuals = torch.empty((count_of_selected_individuals, 2))
    for selected_individual_index in range(count_of_selected_individuals):
        two_random_indices = torch.randint(0, population.shape[0], (2,))
        random_pair_of_individuals = population[two_random_indices]
        better_individual = min(
            random_pair_of_individuals,
            key=lambda x: fitness_function(x))
        selected_individuals[selected_individual_index] = better_individual
    return selected_individuals


def perform_interpolation_mating(
        population: torch.Tensor,
        mate_count: int
        ) -> torch.Tensor:
    children = torch.empty((mate_count, 2))
    for child_index in range(mate_count):
        two_random_indices = torch.randperm(population.shape[0])[:2]
        parents = population[two_random_indices]
        a = random.uniform(0, 1)
        child = a * parents[0] + (1 - a) * parents[1]
        children[child_index] = child
    return children


def eliminate_weak_individuals(
        fitness_function: Callable[[torch.Tensor], float],
        population: torch.Tensor,
        transformed_individuals: torch.Tensor,
        my: int
        ) -> torch.Tensor:
    concatenated = torch.cat((population, transformed_individuals), 0)
    values = torch.empty(concatenated.shape[0])
    for i, row in enumerate(population):
        values[i] = fitness_function(row)
    indices = torch.argsort(values)
    sorted_concatenated = concatenated[indices]
    population = sorted_concatenated[:my]
    return population


def find_minimum_es(
        fitness_function: Callable[[torch.Tensor], float],
        starting_population: torch.Tensor,
        my: int,
        lambd: int,
        standard_deviation: float,
        number_of_generations: int
        ) -> torch.Tensor:

    population = starting_population
    scores = torch.empty(starting_population.shape[0])
    for i, row in enumerate(population):
        scores[i] = fitness_function(row)
    curr_best_individual = min(population, key=lambda x: fitness_function(x))
    curr_best_score = min(scores)

    for _ in range(number_of_generations):
        selected_individuals = tournament_selection(
            fitness_function, population, lambd)

        selected_individuals = perform_interpolation_mating(
            selected_individuals, lambd)

        gaussian_noise = torch.normal(
            mean=0, std=standard_deviation, size=(lambd, 2))
        selected_individuals += gaussian_noise

        batch_scores = torch.empty(selected_individuals.shape[0])
        for i, row in enumerate(selected_individuals):
            batch_scores[i] = fitness_function(row)
        batch_best_score = min(batch_scores)

        if batch_best_score < curr_best_score:
            curr_best_score = batch_best_score
            curr_best_individual = min(
                selected_individuals,
                key=lambda x: fitness_function(x)
            )

        population = eliminate_weak_individuals(
            fitness_function, population, selected_individuals, my)

        scores = torch.empty(population.shape[0])
        for i, row in enumerate(population):
            scores[i] = fitness_function(row)
    return curr_best_individual


def main():
    def fitness_function(arguments: torch.Tensor):
        x = arguments[0]
        y = arguments[1]
        numerator = 10 * x * y
        denominator = math.exp(x ** 2 + 0.5 * x + y ** 2)
        return_value = numerator / denominator
        return return_value

    starting_population = torch.Tensor([[0.01, 0.02],
                                       [0.05, 0.03],
                                       [0.06, 0.07],
                                       [0.05, 0.04],
                                       [0.08, 0.10],
                                       [0.09, 0.13],
                                       [0.20, 0.15],
                                       [0.25, 0.23],
                                       [0.40, -0.40],
                                       [0.25, -0.30],
                                       [0.45, -0.32],
                                       [0.25, -0.42],
                                       [0.13, -0.21],
                                       [0.97, -0.89],
                                       [-0.20, 0.30],
                                       [-0.45, 0.32],
                                       [-0.32, 0.87],
                                       [-0.53, 0.54],
                                       [-0.21, 0.42],
                                       [-0.78, 0.21],
                                       [-0.99, 0.43],
                                       [-0.34, -0.21],
                                       [-0.21, -0.56],
                                       [-0.54, -0.32],
                                       [-0.67, -0.45],
                                       [-0.87, -0.63],
                                       [-0.24, -0.24],
                                       [-0.35, -0.23]])
    print(find_minimum_es(
        fitness_function=fitness_function,
        starting_population=starting_population,
        lambd=20,
        my=140,
        standard_deviation=0.9,
        number_of_generations=50000
            ))


if __name__ == '__main__':
    main()
