import itertools
import os

import numpy as np

from discretization.point_discretization import PointDisc
from phase_eq_construction import PhaseEqAnalysis
from differential_evolution import differential_evolution, improve_solution
from util.helper_functions import (init_binary_NRTL_model_yuan_option,
                                   init_binary_NRTL_model_tau_without_tscaling_option,
                                   init_binary_NRTL_model_oh_option,
                                   evaluate_phase_eq_for_given_feed)


def do_NRTL_parameter_fit(point_disc: PointDisc, names_components: list, T_p_information: dict, feeds: np.array,
                          exp_phases, NRTL_option: str, melting_temperatures: dict or None, melting_heats: dict or None,
                          antoine_models: dict or None, results_path: str):
    # reproducibility
    np.random.seed(42)

    # additional arguments for the objective function
    additional_arguments_dict = {
        'exp_data': exp_phases,
        'feeds': feeds,
        'names_components': names_components,
        'point_disc': point_disc,
        'print_results': False,
        'NRTL_option': NRTL_option,  # referring to the options inside NRTL class
        'melting_heats': melting_heats,
        'melting_temperatures': melting_temperatures,
        'antoine_models': antoine_models,
        'T_p_information': T_p_information
    }

    # check if we optimize over a temperature or pressure range or if we just do it for 1 configuration
    if 'pressures' in T_p_information and 'temperatures' in T_p_information:
        additional_arguments_dict['do_optimization_for_1_configuration'] = False

    else:
        additional_arguments_dict['do_optimization_for_1_configuration'] = True

    # bounds depend on the number of components and the NRTL option
    bounds = None
    if NRTL_option == 'tau_without_temperature_scaling':
        max_factor = 10000.
        if len(names_components) == 2:
            bounds = [(-max_factor, max_factor), (-max_factor, max_factor), (0.1, 1.)]

    elif NRTL_option == 'oh':
        if len(names_components) == 2:
            bounds = [(-10000., 10000.), (-10000., 10000.), (0.1, 1.)]

    elif NRTL_option == 'yuan':
        if len(names_components) == 2:
            bounds = [(-100., 100.), (-100., 100.), (-15000., 15000.), (-15000., 15000.), (0.1, 1.)]

    print('\nPerforming DEA for direct loss on CEM:')

    def cem_fitness_comparison(fitness_1, fitness_2):
        # usable within DEA
        # return if solution 1 is better or equal to solution 2
        if fitness_1['num_feeds_found'] > fitness_2['num_feeds_found']:
            return True

        else:
            if fitness_1['num_feeds_found'] == fitness_2['num_feeds_found']:
                if (fitness_1['avg_abs_error_found_per_found_feed'] <=
                        fitness_2['avg_abs_error_found_per_found_feed']):
                    return True

                else:
                    return False

    population_size = 50
    num_generations = 50
    best_solution, best_fitness = differential_evolution(
        objective_function=cem_objective_function,
        bounds=bounds,
        population_size=population_size,
        max_generations=num_generations,
        additional_args_for_obj=additional_arguments_dict,
        fitness1_better_than_fitness2=cem_fitness_comparison,
        fitness_initialization={
            'num_feeds_found': -1,
            'avg_abs_error_found_per_found_feed': float('inf')
        }
    )

    print('\nBest DEA solution:', best_solution, best_fitness)
    print('Testing it:')
    results_dict = test_NRTL_parameter_solution(
        input_array=best_solution,
        feeds=feeds,
        names_components=names_components,
        point_disc=point_disc,
        exp_phases=exp_phases,
        print_results=True,
        NRTL_option=NRTL_option,
        melting_heats=melting_heats,
        melting_temperatures=melting_temperatures,
        antoine_models=antoine_models,
        T_p_information=additional_arguments_dict['T_p_information'],
        do_optimization_for_1_configuration=additional_arguments_dict['do_optimization_for_1_configuration']
    )

    # store the results as txt file
    file = open(os.path.join(results_path, "results_parameter_fit.txt"), "w+")
    file.write("DEA results:\n\n")
    file.write("num generations: " + str(num_generations) + " , population size: " + str(population_size) + "\n\n")
    file.write("gE-model parameters: " + str(best_solution) + "\n")
    for k, v in results_dict.items():
        if k == 'missing_feeds':
            file.write('\nmissing feeds:\n')
            for feed_dict in v:
                file.write(str(feed_dict) + '\n\n')

            file.write('\n\n')

        else:
            file.write(str(k) + ": " + str(v) + "\n")

    # improve solution by Nelder-Mead
    nelder_mead_solution = improve_solution(
        solution=best_solution,
        scalar_objective_function=cem_objective_wrapper,
        additional_arguments_dict=additional_arguments_dict,
        maxiter=1000,
        bounds=bounds,
        print_results=True
    )

    print('\nFurther improved by Nelder-Mead:')
    print('Testing it:')
    nelder_mead_results_dict = test_NRTL_parameter_solution(
        input_array=nelder_mead_solution,
        feeds=feeds,
        names_components=names_components,
        point_disc=point_disc,
        exp_phases=exp_phases,
        print_results=True,
        NRTL_option=NRTL_option,
        melting_heats=melting_heats,
        melting_temperatures=melting_temperatures,
        antoine_models=antoine_models,
        T_p_information=additional_arguments_dict['T_p_information'],
        do_optimization_for_1_configuration=additional_arguments_dict['do_optimization_for_1_configuration']
    )

    # store the results as txt file
    file.write("\n\nNelder-Mead results:\n\n")
    file.write("gE-model parameters: " + str(nelder_mead_solution) + "\n")
    for k, v in nelder_mead_results_dict.items():
        if k == 'missing_feeds':
            file.write('\nmissing feeds:\n')
            for feed_dict in v:
                file.write(str(feed_dict) + '\n\n')

            file.write('\n\n')

        else:
            file.write(str(k) + ": " + str(v) + "\n")

    file.close()


def wrapper_NRTL_init(input_array: np.array, NRTL_option: str, num_comp: int):
    # init ge_model
    ge_model = None
    if NRTL_option == 'yuan':
        ge_model = init_binary_NRTL_model_yuan_option(input_array=input_array)

    elif NRTL_option == 'tau_without_temperature_scaling':
        if num_comp == 2:
            ge_model = init_binary_NRTL_model_tau_without_tscaling_option(input_array=input_array)

    elif NRTL_option == 'oh':
        if num_comp == 2:
            ge_model = init_binary_NRTL_model_oh_option(input_array=input_array)

    return ge_model


def cem_objective_wrapper(input_array: np.array, additional_arguments_dict: dict):
    results_dict = cem_objective_function(input_array, additional_arguments_dict)

    return results_dict['avg_abs_error_found_per_found_feed'] - results_dict['num_feeds_found']


def cem_objective_function(input_array: np.array, additional_arguments_dict: dict):
    # get num of found feeds
    results_dict = test_NRTL_parameter_solution(
        input_array=input_array,
        feeds=additional_arguments_dict['feeds'],
        names_components=additional_arguments_dict['names_components'],
        point_disc=additional_arguments_dict['point_disc'],
        exp_phases=additional_arguments_dict['exp_data'],
        print_results=additional_arguments_dict['print_results'],
        NRTL_option=additional_arguments_dict['NRTL_option'],
        melting_heats=additional_arguments_dict['melting_heats'],
        melting_temperatures=additional_arguments_dict['melting_temperatures'],
        antoine_models=additional_arguments_dict['antoine_models'],
        T_p_information=additional_arguments_dict['T_p_information'],
        do_optimization_for_1_configuration=additional_arguments_dict['do_optimization_for_1_configuration']
    )

    # this way we can also use ranked objectives (compare first num found feeds for example and
    # afterwards the avg abs error for the found feeds)
    objective = results_dict

    return objective


def test_NRTL_parameter_solution(input_array: np.array, feeds: np.array, names_components: list,
                                 T_p_information: dict, point_disc: PointDisc, exp_phases,
                                 print_results: bool, NRTL_option: str, melting_temperatures: dict or None,
                                 melting_heats: dict or None, antoine_models: dict or None,
                                 do_optimization_for_1_configuration: bool):
    # init ge_model
    ge_model = wrapper_NRTL_init(input_array=input_array, NRTL_option=NRTL_option, num_comp=len(names_components))

    # if we have fixed T and p, we calculate the phase eq one time and predict all feeds there
    if do_optimization_for_1_configuration:
        phase_eq_analysis = PhaseEqAnalysis(
            discretized_system=point_disc,
            temperature_K=T_p_information['temperature'],
            pressure_bar=T_p_information['pressure'],
            num_components=len(names_components),
            gE_model=ge_model,
            antoine_models=antoine_models,
            names_comps_ordered=names_components,
            melting_temperatures=melting_temperatures,
            melting_heats=melting_heats
        )

        # construct phase split simplices
        phase_split_simplices = phase_eq_analysis.construct_phase_equilibrium()

    # loop through feeds and predict splits.
    abs_error = 0.
    counter_found_feeds = 0
    missing_feeds = []
    for j, feed in enumerate(feeds):
        # if we have a range of p and T, we have to calculate a phase eq for every point
        if not do_optimization_for_1_configuration:
            # init phase eq analysis
            phase_eq_analysis = PhaseEqAnalysis(
                discretized_system=point_disc,
                temperature_K=T_p_information['temperatures'][j],
                pressure_bar=T_p_information['pressures'][j],
                num_components=len(names_components),
                gE_model=ge_model,
                antoine_models=antoine_models,
                names_comps_ordered=names_components,
                melting_temperatures=melting_temperatures,
                melting_heats=melting_heats
            )

            # construct phase split simplices
            phase_split_simplices = phase_eq_analysis.construct_phase_equilibrium()

        # predict phase split
        pred_phases_mfr, pred_phases_aggregate_states = evaluate_phase_eq_for_given_feed(
            phase_eq_analysis=phase_eq_analysis,
            molar_flowrates_feed=feed,
            point_disc=point_disc,
            phase_eq_simplices=phase_split_simplices,
            num_comp=len(names_components)
        )

        if print_results:
            print('\nprediction:', pred_phases_mfr, 'exp:', [exp_phases[ind][j] for ind in range(len(exp_phases))])

        # check if the number of phases is matching
        if len(pred_phases_mfr) != len(exp_phases):
            if print_results:
                print('Not found:', [exp_phases[ind][j] for ind in range(len(exp_phases))],
                      'prediction:', pred_phases_mfr, pred_phases_aggregate_states)
                missing_feeds.append(
                    {
                        'feed': feed,
                        'pred_mfr': pred_phases_mfr,
                        'pred_phases': pred_phases_aggregate_states,
                        'exp_mfr': [exp_phases[ind][j] for ind in range(len(exp_phases))]
                    }
                )

        else:
            counter_found_feeds = counter_found_feeds + 1

            # test all combinations for the order and take the minimum
            num_present_phases = len(pred_phases_mfr)
            list_possible_order = list(
                itertools.permutations(list(range(num_present_phases)), num_present_phases)
            )
            for combination_index, _ in enumerate(list_possible_order):
                list_possible_order[combination_index] = list(list_possible_order[combination_index])

            differences_per_order = np.zeros(len(list_possible_order))
            for combination_index, combination in enumerate(list_possible_order):
                for phase_index in range(num_present_phases):
                    differences_per_order[combination_index] = (
                            differences_per_order[combination_index] + np.sum(np.abs(
                        pred_phases_mfr[phase_index] - exp_phases[combination[phase_index]][j]))
                    )

            abs_error = abs_error + np.min(differences_per_order)

    # we evaluate always by dividing per num_present_phases and num_comp
    additional_normalizer = len(exp_phases) * len(names_components)

    # normalize the respective scores also by dividing by the number of samples
    avg_abs_error_per_found_feed = 0.
    if counter_found_feeds > 0:
        avg_abs_error_per_found_feed = abs_error / (counter_found_feeds * additional_normalizer)

    # init results dict and print if specified
    results_dict = {
        'num_feeds_found': counter_found_feeds,
        'num_feeds_not_found': len(feeds) - counter_found_feeds,
        'abs_error_found': abs_error,
        'avg_abs_error_found_per_found_feed': avg_abs_error_per_found_feed
    }
    if print_results:
        for k, v in results_dict.items():
            print(k, v)

        print('\n')
        results_dict['missing_feeds'] = missing_feeds

    return results_dict
