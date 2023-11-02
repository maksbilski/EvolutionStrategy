import torch
import random
import math
import time

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
        better_individual = max(
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
        two_random_indices = torch.randint(0, population.shape[0], (2,))
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
    for i, row in enumerate(concatenated):
        values[i] = fitness_function(row)
    indices = torch.argsort(values, descending=True)
    # time.sleep(5)
    # print(values[indices])
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

    curr_best_individual = max(population, key=lambda x: fitness_function(x))
    curr_best_score = fitness_function(curr_best_individual)

    for _ in range(number_of_generations):
        selected_individuals = tournament_selection(
            fitness_function, population, lambd)

        offspring = perform_interpolation_mating(
            selected_individuals, lambd)

        gaussian_noise = torch.normal(
            mean=0, std=standard_deviation, size=(lambd, 2))
        offspring += gaussian_noise

        batch_best_individual = max(
            offspring, key=lambda x: fitness_function(x))
        batch_best_score = fitness_function(batch_best_individual)

        if batch_best_score > curr_best_score:
            curr_best_score = batch_best_score
            curr_best_individual = batch_best_individual

        population = eliminate_weak_individuals(
            fitness_function, population, offspring, my)

    return curr_best_individual


def main():
    def fitness_function(arguments: torch.Tensor):
        x = arguments[0]
        y = arguments[1]
        numerator = 10 * x * y
        denominator = math.exp(x ** 2 + 0.5 * x + y ** 2)
        return_value = numerator / denominator
        return return_value

    starting_population = torch.rand((1000, 2))
    for index, individual in enumerate(starting_population):
        starting_population[index] = individual * (10 - (-10)) + (-10)

    starting_population

    print(starting_population)

    print(find_minimum_es(
        fitness_function=fitness_function,
        starting_population=starting_population,
        lambd=20,
        my=140,
        standard_deviation=0.55,
        number_of_generations=1000
            ))


if __name__ == '__main__':
    main()
