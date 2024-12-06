import numpy as np
import scipy.optimize as opt


def initialize_population(bounds, population_size):
    population = []
    for _ in range(population_size):
        individual = [np.random.uniform(low, high) for low, high in bounds]
        population.append(individual)

    return population


def clip_to_bounds(individual, bounds):
    clipped_individual = []
    for val, (low, high) in zip(individual, bounds):
        clipped_val = max(min(val, high), low)
        clipped_individual.append(clipped_val)

    return clipped_individual


def differential_evolution(objective_function, bounds: list, population_size: int, max_generations: int,
                           additional_args_for_obj: dict, fitness1_better_than_fitness2, fitness_initialization,
                           F=0.8, CR=0.7):
    """
    Based on Das2011.

    fitness1_better_than_fitness2 is a function that takes two fitnesses as input and returns if the first one is
    better or at least equally good as the second one.
    """
    population = initialize_population(bounds, population_size)
    best_solution = None
    best_fitness = fitness_initialization

    for generation in range(max_generations):
        new_population = []
        for i, target in enumerate(population):
            indices = np.random.choice(len(population), 3, replace=False)
            a, b, c = [population[idx] for idx in indices]
            mutant = [a[j] + F * (b[j] - c[j]) for j in range(len(bounds))]
            mutant = clip_to_bounds(mutant, bounds)

            crossover_mask = [np.random.uniform() < CR if j != np.random.randint(len(bounds)) else True for j in range(len(bounds))]
            trial = [mutant[j] if crossover_mask[j] else target[j] for j in range(len(bounds))]
            target_fitness = objective_function(target, additional_args_for_obj)
            trial_fitness = objective_function(trial, additional_args_for_obj)

            if fitness1_better_than_fitness2(trial_fitness, target_fitness):
                new_population.append(trial)
                if fitness1_better_than_fitness2(trial_fitness, best_fitness):
                    best_solution = trial
                    best_fitness = trial_fitness
            else:
                new_population.append(target)

        population = new_population

        # print some information after every generation
        print(f"Generation {generation}: best Fitness = {best_fitness}, solution = {best_solution}")

    return best_solution, best_fitness


def improve_solution(solution: np.array, scalar_objective_function, additional_arguments_dict: dict, maxiter: int,
                     bounds: list, print_results: bool):
    # perform Nelder-Mead optimization with the given solution as starting point
    res = opt.minimize(
        fun=scalar_objective_function,
        x0=solution,
        args=additional_arguments_dict,
        method='Nelder-Mead',
        bounds=bounds,
        options={'maxiter': maxiter}
    )

    # print results if necessary
    if print_results:
        print('\nNelder-Mead optimization:')
        print(res)

    # return the improved solution
    return res.x
