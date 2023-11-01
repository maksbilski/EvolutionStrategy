import torch
import random

from typing import Callable


def tournament_selection(
        fitness_function: Callable[[torch.Tensor], float],
        population: torch.Tensor,
        count_of_selected_individuals: int
        ) -> torch.Tensor:
    selected_individuals = torch.empty_like(population)
    for _ in range(count_of_selected_individuals):
        two_random_indices = torch.randint(0, population.shape[0], (2,))
        random_pair_of_individuals = population[two_random_indices]
        better_individual = min(
            random_pair_of_individuals,
            key=lambda x: fitness_function(x))
        selected_individuals[_] = better_individual
    return selected_individuals


def perform_interpolation_mating(
        population: torch.Tensor,
        mate_count: int
        ) -> torch.Tensor:
    population_after_mating = torch.empty_like(population)
    for _ in range(mate_count):
        two_random_indices = torch.randperm(population.shape[0])[:2]
        parents = population[two_random_indices]
        a = random.uniform(0, 1)
        child = a * parents[0] + (1 - a) * parents[1]
        population_after_mating[_] = child
    return population_after_mating


def find_minimum_es(
        fitness_function: Callable[[torch.Tensor], float],
        starting_population: torch.Tensor,
        my: int,
        lambd: int,
        standard_deviation: float,
        number_of_generations: int
        ) -> torch.Tensor:

    population = starting_population
    scores = population.apply_(fitness_function)
    curr_best_individual = min(population, key=lambda x: fitness_function(x))
    curr_best_score = min(scores)

    for _ in range(number_of_generations):
        selected_individuals = tournament_selection(
            fitness_function, population, lambd)

        individuals_after_mating = perform_interpolation_mating(
            selected_individuals, lambd)

        gaussian_noise = standard_deviation * torch.randn_like(population)
        individuals_after_mating += gaussian_noise

        batch_scores = individuals_after_mating.apply_(fitness_function)
        batch_best_score = min(batch_scores)

        if batch_best_score < curr_best_score:
            curr_best_score = batch_best_score
            curr_best_individual = min(
                individuals_after_mating,
                key=lambda x: fitness_function(x)
            )

        concatenated = torch.cat((population, individuals_after_mating), 0)
        values = concatenated.apply_(fitness_function)
        indices = torch.argsort(values)
        sorted_concatenated = concatenated[indices]
        population = sorted_concatenated[:my]
        scores = population.apply_(fitness_function)
    return curr_best_individual
