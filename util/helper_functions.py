import numpy as np

import thermodynamic_models as thermo_models


def evaluate_phase_eq_for_given_feed(phase_eq_analysis, molar_flowrates_feed: np.array,
                                     point_disc, phase_eq_simplices: list, num_comp: int):
    # use phase eq simplices to find split
    found_phase_split, resulting_streams = phase_eq_analysis.find_phase_split(
        feed_molar_flowrates=molar_flowrates_feed,
        discretized_system=point_disc,
        phase_split_simplices=phase_eq_simplices,
        num_comp=num_comp
    )

    # convert to molar fractions
    pred_phases_mfr = [el['molar_flowrates'] / sum(el['molar_flowrates']) for el in resulting_streams]
    phases = [el['aggregate_state'] for el in resulting_streams]

    return pred_phases_mfr, phases


def convert_mass_fr_to_molar_fr(mass_fractions, molar_masses):
    molar_fractions = np.zeros(len(mass_fractions))
    denominator = 0
    for i in range(len(mass_fractions)):
        denominator = denominator + (mass_fractions[i] / molar_masses[i])

    for i in range(len(mass_fractions)):
        molar_fractions[i] = (mass_fractions[i] / molar_masses[i]) / denominator

    return molar_fractions


def init_binary_NRTL_model_yuan_option(input_array: np.array):
    # set parameters from input_array
    interaction_parameters = {}
    interaction_parameters['a'] = np.zeros((2, 2))
    interaction_parameters['a'][0][1] = input_array[0]
    interaction_parameters['a'][1][0] = input_array[1]

    interaction_parameters['b'] = np.zeros((2, 2))
    interaction_parameters['b'][0][1] = input_array[2]
    interaction_parameters['b'][1][0] = input_array[3]

    interaction_parameters['alphas'] = np.zeros((2, 2))
    interaction_parameters['alphas'][0][1] = input_array[4]
    interaction_parameters['alphas'][1][0] = input_array[4]

    # init ge-model
    ge_model = thermo_models.NRTLModel(
        data_dict=interaction_parameters,
        option_dict={'ge_model_option': 'yuan'},
        num_comp=2
    )

    return ge_model


def init_binary_NRTL_model_tau_without_tscaling_option(input_array: np.array):
    # set parameters from input_array
    interaction_parameters = {}
    interaction_parameters['tau_without_temperature_scaling'] = np.zeros((2, 2))
    interaction_parameters['tau_without_temperature_scaling'][0][1] = input_array[0]
    interaction_parameters['tau_without_temperature_scaling'][1][0] = input_array[1]

    interaction_parameters['alphas'] = np.zeros((2, 2))
    interaction_parameters['alphas'][0][1] = input_array[2]
    interaction_parameters['alphas'][1][0] = input_array[2]

    # init ge-model
    ge_model = thermo_models.NRTLModel(
        data_dict=interaction_parameters,
        option_dict={'ge_model_option': 'tau_without_temperature_scaling'},
        num_comp=2
    )

    return ge_model


def init_binary_NRTL_model_oh_option(input_array: np.array):
    # set parameters from input_array
    interaction_parameters = {}
    interaction_parameters['A'] = np.zeros((2, 2))
    interaction_parameters['A'][0][1] = input_array[0]
    interaction_parameters['A'][1][0] = input_array[1]

    interaction_parameters['alphas'] = np.zeros((2, 2))
    interaction_parameters['alphas'][0][1] = input_array[2]
    interaction_parameters['alphas'][1][0] = input_array[2]

    # init ge-model
    ge_model = thermo_models.NRTLModel(
        data_dict=interaction_parameters,
        option_dict={'ge_model_option': 'oh'},
        num_comp=2
    )

    return ge_model


def init_ternary_NRTL_model_tau_without_tscaling_option(input_array: np.array):
    # set parameters from input_array
    interaction_parameters = {}
    interaction_parameters['tau_without_temperature_scaling'] = np.zeros((3, 3))
    interaction_parameters['alphas'] = np.zeros((3, 3))

    # 01 binary
    interaction_parameters['tau_without_temperature_scaling'][0][1] = input_array[0]
    interaction_parameters['tau_without_temperature_scaling'][1][0] = input_array[1]
    interaction_parameters['alphas'][0][1] = input_array[2]
    interaction_parameters['alphas'][1][0] = input_array[2]

    # 02 binary
    interaction_parameters['tau_without_temperature_scaling'][0][2] = input_array[3]
    interaction_parameters['tau_without_temperature_scaling'][2][0] = input_array[4]
    interaction_parameters['alphas'][0][2] = input_array[5]
    interaction_parameters['alphas'][2][0] = input_array[5]

    # 12 binary
    interaction_parameters['tau_without_temperature_scaling'][1][2] = input_array[6]
    interaction_parameters['tau_without_temperature_scaling'][2][1] = input_array[7]
    interaction_parameters['alphas'][1][2] = input_array[8]
    interaction_parameters['alphas'][2][1] = input_array[8]

    # init ge-model
    ge_model = thermo_models.NRTLModel(
        data_dict=interaction_parameters,
        option_dict={'ge_model_option': 'tau_without_temperature_scaling'},
        num_comp=3
    )

    return ge_model
