import os
import itertools

import numpy as np

import thermodynamic_models as thermo_models

from discretization.point_discretization import PointDisc
from util.plotter import Plotter
from phase_eq_construction import PhaseEqAnalysis
from data_sources.property_data_init import InitializedGeData, InitializedExperimentalData
from data_sources.property_data_classes import OptionData, AntoineData, SolidData
from ge_model_para_fit import do_NRTL_parameter_fit
from util.helper_functions import evaluate_phase_eq_for_given_feed


class InitClass:
    def __init__(self, general_name: str, names_components: list, point_discretization_rec_steps: int,
                 path_discretization: str, ge_model_name: str, experimental_data_given: bool, task: str,
                 task_specific_information: dict, table_name: str or None):
        print('\n\nDoing Analysis for ', general_name, names_components)

        # define path for that general_name (usually author followed by year of publication)
        general_path = os.path.join(os.getcwd(), "phase_equilibria_results", general_name)
        if not os.path.isdir(general_path):
            os.mkdir(general_path)

        # define sub-directory for task, e.g., VLE or LLE
        general_path = os.path.join(general_path, task)
        if not os.path.isdir(general_path):
            os.mkdir(general_path)

        num_comp = len(names_components)

        # we will need the combined component names as string for some directory names
        names_components_combined = self.combine_names_list_to_string(names_components)

        # load point discretization
        point_disc = PointDisc(
            num_comp=num_comp, recursion_steps=point_discretization_rec_steps, load=True, store=False,
            path=path_discretization
        )

        # init plotter, will usually only be done for N<=3
        plotter_instance = Plotter(num_comp=num_comp)

        # initialize ge model data class
        ge_dataset = InitializedGeData().ge_models_collection.get_model(name=general_name)
        multinary_interactions, pure_component_data = ge_dataset.get_multinary_interactions(names=names_components)

        # option data
        option_data = OptionData()
        option_data.add_option_data(name=general_name)
        option_dict = option_data.get_option_data(name=general_name)

        # initialize experimental data
        experimental_data = None
        if experimental_data_given:
            experimental_data = InitializedExperimentalData().experimental_datasets[general_name]

        # initialize ge model
        ge_model = None
        if ge_model_name == "NRTL":
            ge_model = thermo_models.NRTLModel(
                data_dict=multinary_interactions,
                option_dict=option_dict,
                num_comp=len(names_components)
            )

        melting_temperatures = None
        melting_heats = None
        antoine_models = None

        # VLE task
        if task == "VLE":
            # set up antoine data set
            antoine_data = AntoineData()
            antoine_data.add_model(name=general_name)
            antoine_data_dict = antoine_data.get_model(name=general_name)

            # create a antoine model for every contained component
            antoine_models = {}
            for component_name in antoine_data_dict.keys():
                antoine_models[component_name] = thermo_models.AntoineModel(
                    data_dict=antoine_data_dict[component_name], option_dict=option_dict
                )

            # for a VLE, we need either a pressure or temperature range
            if 'pressure_range' in task_specific_information.keys():
                # similarly as for LLEs we create directories with names that give some of the relevant parameters
                dir_name = (
                    "pxy_" + str(num_comp) + "_" + str(point_discretization_rec_steps) + "__" +
                    names_components_combined + "_T_" +
                    str(round(task_specific_information['temperature_kelvin'], 4)) + "_p1_" +
                    str(round(task_specific_information['pressure_range'][0], 4)) + "_p2_" +
                    str(round(task_specific_information['pressure_range'][1], 4))
                )
                vle_path = os.path.join(general_path, dir_name)
                if not os.path.isdir(vle_path):
                    os.mkdir(vle_path)

                path_to_store_exp_data_comparison = vle_path

                # subfolder for all the phase eqs that are required for the whole VLE
                vle_path_phase_eqs = os.path.join(vle_path, 'phase_eqs')
                if not os.path.isdir(vle_path_phase_eqs):
                    os.mkdir(vle_path_phase_eqs)

                # loop through pressure range and construct VLEs
                list_with_pxy_info = []
                for p_bar in experimental_data.experimental_data_in_tables[table_name]['array_p_bar']:
                    _, phase_eq_simplices, loading_path_phase_eq = self.create_phase_eq(
                        point_disc=point_disc,
                        temperature_K=task_specific_information['temperature_kelvin'],
                        pressure_bar=p_bar,
                        num_comp=num_comp,
                        ge_model=ge_model,
                        antoine_models=antoine_models,
                        names_components=names_components,
                        plotter_instance=plotter_instance,
                        path_for_storage=vle_path_phase_eqs,
                        names_components_combined=names_components_combined,
                        melting_heats=melting_heats,
                        melting_temperatures=melting_temperatures
                    )

                    # collect pxy information
                    info_dict = {
                        'p_bar': p_bar,
                        'loading_path': loading_path_phase_eq
                    }
                    if num_comp == 2:
                        info_dict['list_phase_eq_simplices'] = phase_eq_simplices

                    list_with_pxy_info.append(info_dict)

                # for N=2, we can plot a pxy diagram
                if num_comp == 2:
                    # init plotter
                    plotter_instance.plot_pxy(
                        list_with_pxy_info=list_with_pxy_info,
                        pressure_range=task_specific_information['pressure_range'],
                        name_comp1=names_components[0],
                        plot_lines=True,
                        location=None
                    )

                    # save plot
                    plotter_instance.save_plot(path=os.path.join(vle_path, "pxy_plot.png"))

            # for a VLE, we need either a pressure or temperature range
            if 'temperature_range' in task_specific_information.keys():
                # similarly as for LLEs we create directories with names that give some of the relevant parameters
                dir_name = (
                        "Txy_" + str(num_comp) + "_" + str(point_discretization_rec_steps) + "__" +
                        names_components_combined + "_T1_" +
                        str(round(task_specific_information['temperature_range'][0], 4)) + "_T2_" +
                        str(round(task_specific_information['temperature_range'][1], 4)) + "_p_" +
                        str(round(task_specific_information['pressure_bar'], 4))
                )
                vle_path = os.path.join(general_path, dir_name)
                if not os.path.isdir(vle_path):
                    os.mkdir(vle_path)

                path_to_store_exp_data_comparison = vle_path

                # subfolder for all the phase eqs that are required for the whole VLE
                vle_path_phase_eqs = os.path.join(vle_path, 'phase_eqs')
                if not os.path.isdir(vle_path_phase_eqs):
                    os.mkdir(vle_path_phase_eqs)

                # loop through t range and construct VLEs
                list_with_Txy_info = []
                for T_K in experimental_data.experimental_data_in_tables[table_name]['array_T_K']:
                    _, phase_eq_simplices, loading_path_phase_eq = self.create_phase_eq(
                        point_disc=point_disc,
                        temperature_K=T_K,
                        pressure_bar=task_specific_information['pressure_bar'],
                        num_comp=num_comp,
                        ge_model=ge_model,
                        antoine_models=antoine_models,
                        names_components=names_components,
                        plotter_instance=plotter_instance,
                        path_for_storage=vle_path_phase_eqs,
                        names_components_combined=names_components_combined,
                        melting_heats=melting_heats,
                        melting_temperatures=melting_temperatures
                    )

                    # collect pxy information
                    info_dict = {
                        't_K': T_K,
                        'loading_path': loading_path_phase_eq
                    }
                    if num_comp == 2:
                        info_dict['list_phase_eq_simplices'] = phase_eq_simplices

                    list_with_Txy_info.append(info_dict)

                # for N=2, we can plot a pxy diagram
                if num_comp == 2:
                    plotter_instance.plot_txy(
                        list_with_txy_info=list_with_Txy_info,
                        t_range=task_specific_information['temperature_range'],
                        name_comp1=names_components[0],
                        plot_lines=True,
                        location=None
                    )

                    # save plot
                    plotter_instance.save_plot(path=os.path.join(vle_path, "Txy_plot.png"))

        # LLE task
        if task == "LLE":
            _, _, path_to_phase_eq = self.create_phase_eq(
                point_disc=point_disc,
                temperature_K=task_specific_information['temperature_kelvin'],
                pressure_bar=task_specific_information['pressure_bar'],
                num_comp=num_comp,
                ge_model=ge_model,
                antoine_models=antoine_models,
                names_components=names_components,
                plotter_instance=plotter_instance,
                path_for_storage=general_path,
                names_components_combined=names_components_combined,
                melting_heats=melting_heats,
                melting_temperatures=melting_temperatures
            )
            path_to_store_exp_data_comparison = path_to_phase_eq

        # SLE task
        if task == "SLE":
            # get the data on the solid states
            solid_data = SolidData()
            solid_data.add_model(name=general_name)
            solid_data_data_dict = solid_data.get_model(name=general_name)

            # convert format
            melting_temperatures = {}
            melting_heats = {}
            for name in names_components:
                melting_heats[name] = solid_data_data_dict[name]['h_melt']
                melting_temperatures[name] = solid_data_data_dict[name]['T_melt']

            # for a SLE, we need most likely a temperature range
            if 'temperature_range' in task_specific_information.keys():
                # similarly as for LLEs we create directories with names that give some of the relevant parameters
                dir_name = (
                        "Txy_" + str(num_comp) + "_" + str(point_discretization_rec_steps) + "__" +
                        names_components_combined + "_T1_" +
                        str(round(task_specific_information['temperature_range'][0], 4)) + "_T2_" +
                        str(round(task_specific_information['temperature_range'][1], 4)) + "_p_" +
                        str(round(task_specific_information['pressure_bar'], 4))
                )
                sle_path = os.path.join(general_path, dir_name)
                if not os.path.isdir(sle_path):
                    os.mkdir(sle_path)

                path_to_store_exp_data_comparison = sle_path

                # subfolder for all the phase eqs that are required for the whole SLE
                sle_path_phase_eqs = os.path.join(sle_path, 'phase_eqs')
                if not os.path.isdir(sle_path_phase_eqs):
                    os.mkdir(sle_path_phase_eqs)

                # loop through pressure range and construct SLEs
                list_with_Txy_info = []
                for T_K in experimental_data.experimental_data_in_tables[table_name]['array_T_K']:
                    _, phase_eq_simplices, loading_path_phase_eq = self.create_phase_eq(
                        point_disc=point_disc,
                        temperature_K=T_K,
                        pressure_bar=task_specific_information['pressure_bar'],
                        num_comp=num_comp,
                        ge_model=ge_model,
                        antoine_models=antoine_models,
                        names_components=names_components,
                        plotter_instance=plotter_instance,
                        path_for_storage=sle_path_phase_eqs,
                        names_components_combined=names_components_combined,
                        melting_heats=melting_heats,
                        melting_temperatures=melting_temperatures
                    )

                    # collect pxy information
                    info_dict = {
                        't_K': T_K,
                        'loading_path': loading_path_phase_eq
                    }
                    if num_comp == 2:
                        info_dict['list_phase_eq_simplices'] = phase_eq_simplices

                    list_with_Txy_info.append(info_dict)

                # for N=2, we can plot a txy diagram
                if num_comp == 2:
                    plotter_instance.plot_txy(
                        list_with_txy_info=list_with_Txy_info,
                        t_range=task_specific_information['temperature_range'],
                        name_comp1=names_components[0],
                        plot_lines=True,
                        location=None
                    )

                    # save plot
                    plotter_instance.save_plot(path=os.path.join(sle_path, "Txy_plot.png"))

            # if we do it for a specific T
            elif 'temperature_kelvin' in task_specific_information.keys():
                _, _, path_to_phase_eq = self.create_phase_eq(
                    point_disc=point_disc,
                    temperature_K=task_specific_information['temperature_kelvin'],
                    pressure_bar=task_specific_information['pressure_bar'],
                    num_comp=num_comp,
                    ge_model=ge_model,
                    antoine_models=antoine_models,
                    names_components=names_components,
                    plotter_instance=plotter_instance,
                    path_for_storage=general_path,
                    names_components_combined=names_components_combined,
                    melting_heats=melting_heats,
                    melting_temperatures=melting_temperatures
                )
                path_to_store_exp_data_comparison = path_to_phase_eq

        if experimental_data is not None:
            # loop over given feeds and compare to predicted results
            sum_abs_differences = 0
            missing_feeds_indices = []
            missing_feeds = []
            for j, feed in enumerate(experimental_data.experimental_data_in_tables[table_name]['feeds']):
                loaded_phase_eq = None
                loaded_phase_eq_simplices = None
                if task == 'LLE' or (task == 'SLE' and 'temperature_kelvin' in task_specific_information.keys() and
                    'pressure_bar' in task_specific_information.keys()):
                    loaded_phase_eq, loaded_phase_eq_simplices = self.load_phase_eq(
                        point_disc=point_disc,
                        temperature_K=task_specific_information['temperature_kelvin'],
                        pressure_bar=task_specific_information['pressure_bar'],
                        num_comp=num_comp,
                        ge_model=ge_model,
                        antoine_models=antoine_models,
                        names_components=names_components,
                        loading_path_phase_eq=path_to_phase_eq,
                        melting_heats=melting_heats,
                        melting_temperatures=melting_temperatures
                    )

                elif 'pressure_range' in task_specific_information.keys():
                    loaded_phase_eq, loaded_phase_eq_simplices = self.load_phase_eq(
                        point_disc=point_disc,
                        temperature_K=task_specific_information['temperature_kelvin'],
                        pressure_bar=list_with_pxy_info[j]['p_bar'],
                        num_comp=num_comp,
                        ge_model=ge_model,
                        antoine_models=antoine_models,
                        names_components=names_components,
                        loading_path_phase_eq=list_with_pxy_info[j]['loading_path'],
                        melting_heats=melting_heats,
                        melting_temperatures=melting_temperatures
                    )

                elif 'temperature_range' in task_specific_information.keys():
                    loaded_phase_eq, loaded_phase_eq_simplices = self.load_phase_eq(
                        point_disc=point_disc,
                        temperature_K=list_with_Txy_info[j]['t_K'],
                        pressure_bar=task_specific_information['pressure_bar'],
                        num_comp=num_comp,
                        ge_model=ge_model,
                        antoine_models=antoine_models,
                        names_components=names_components,
                        loading_path_phase_eq=list_with_Txy_info[j]['loading_path'],
                        melting_heats=melting_heats,
                        melting_temperatures=melting_temperatures
                    )

                pred_phases_mfr, pred_phases = evaluate_phase_eq_for_given_feed(
                        phase_eq_analysis=loaded_phase_eq,
                        molar_flowrates_feed=feed,
                        point_disc=point_disc,
                        phase_eq_simplices=loaded_phase_eq_simplices,
                        num_comp=num_comp
                    )

                if len(pred_phases_mfr) != experimental_data.experimental_data_in_tables[table_name]['num_phases']:
                    missing_feeds_indices.append(
                        [j, len(pred_phases_mfr), experimental_data.experimental_data_in_tables[table_name]['num_phases']]
                    )
                    missing_feeds.append(
                        {
                            'feed': feed,
                            'experimental_phases': [experimental_data.experimental_data_in_tables[
                                table_name]['corresponding_phases'][ind][j] for ind in range(
                                experimental_data.experimental_data_in_tables[table_name]['num_phases'])],
                            'simulated_phases': pred_phases_mfr
                        }
                    )
                    print("missing feed")
                    print(loaded_phase_eq.temperature_K, loaded_phase_eq.pressure_bar, pred_phases)
                    print(feed)
                    print("simu", len(pred_phases_mfr), pred_phases_mfr)
                    print("exp", experimental_data.experimental_data_in_tables[table_name]['num_phases'],
                          [experimental_data.experimental_data_in_tables[
                           table_name]['corresponding_phases'][ind][j] for ind in range(
                              experimental_data.experimental_data_in_tables[table_name]['num_phases'])])
                    print("\n")

                else:
                    # we have to find, which phases from the paper belong to which phases from our computation
                    # the easy way is just to test all combinations and take the lowest resulting error per phase
                    num_present_phases = experimental_data.experimental_data_in_tables[table_name]['num_phases']
                    list_possible_order = list(
                        itertools.permutations(list(range(num_present_phases)), num_present_phases)
                    )
                    for combination_index, _ in enumerate(list_possible_order):
                        list_possible_order[combination_index] = list(list_possible_order[combination_index])

                    squared_differences_per_order = np.zeros(len(list_possible_order))
                    abs_differences_per_order = np.zeros(len(list_possible_order))
                    for combination_index, combination in enumerate(list_possible_order):
                        for phase_index in range(num_present_phases):
                            squared_differences_per_order[combination_index] = (
                                    squared_differences_per_order[combination_index] + np.sum(np.square(
                                pred_phases_mfr[phase_index] - experimental_data.experimental_data_in_tables[
                                    table_name]['corresponding_phases'][combination[phase_index]][j]
                            )))

                            abs_differences_per_order[combination_index] = (
                                    abs_differences_per_order[combination_index] + np.sum(np.abs(
                                pred_phases_mfr[phase_index] - experimental_data.experimental_data_in_tables[
                                    table_name]['corresponding_phases'][combination[phase_index]][j]
                            )))

                    # take the minimum
                    sum_abs_differences = sum_abs_differences + np.min(abs_differences_per_order)

            # can only add up when at least one feed was predicted with the correct num of phases
            if len(experimental_data.experimental_data_in_tables[table_name]['feeds']) > len(missing_feeds_indices):
                num_present_phases = experimental_data.experimental_data_in_tables[table_name]['num_phases']
                mean_deviation = sum_abs_differences / (
                        num_comp * num_present_phases * (len(experimental_data.experimental_data_in_tables[table_name]['feeds']) -
                                        len(missing_feeds_indices)))

            print("final mean deviation (absolute error normalized):", mean_deviation)
            print("missing feeds: ", len(missing_feeds_indices))

            # store the results as txt file
            file = open(os.path.join(path_to_store_exp_data_comparison, "comparison_simu_exp.txt"), "w+")
            file.write("mean  deviation: " + str(mean_deviation) + "\n")
            file.write("number of feeds: " + str(len(experimental_data.experimental_data_in_tables[table_name]['feeds']
                                                     )) + "\n")
            file.write("missing feeds: " + str(len(missing_feeds_indices)) + "\n\n")
            for m_f_index, m_f in enumerate(missing_feeds_indices):
                file.write("index " + str(m_f[0]) + " simu " + str(m_f[1]) + " exp " + str(m_f[2]) + "\n")
                file.write(str(missing_feeds[m_f_index]) + "\n\n")

            file.close()

            # if experimental data is given, we want to ge-model parameter fit (for binary systems due to speed)
            if num_comp == 2:
                do_NRTL_parameter_fit(
                    point_disc=point_disc,
                    names_components=names_components,
                    T_p_information={
                        'temperatures': experimental_data.experimental_data_in_tables[table_name]['array_T_K'],
                        'pressures': experimental_data.experimental_data_in_tables[table_name]['array_p_bar']
                    },
                    feeds=experimental_data.experimental_data_in_tables[table_name]['feeds'],
                    exp_phases=experimental_data.experimental_data_in_tables[table_name]['corresponding_phases'],
                    NRTL_option=option_dict['ge_model_option'],
                    melting_heats=melting_heats,
                    melting_temperatures=melting_temperatures,
                    antoine_models=antoine_models,
                    results_path=path_to_store_exp_data_comparison
                )

    @staticmethod
    def combine_names_list_to_string(names_components: list):
        names_components_combined = ""
        for i in range(len(names_components) - 1):
            names_components_combined = names_components_combined + names_components[i] + "--"

        names_components_combined = names_components_combined + names_components[-1]

        return names_components_combined

    @staticmethod
    def load_phase_eq(point_disc: PointDisc, temperature_K: float, pressure_bar: float, num_comp: int, ge_model,
                      antoine_models: dict or None, names_components: list, loading_path_phase_eq: str,
                      melting_heats: dict or None, melting_temperatures: dict or None):
        # init phase eq analysis
        phase_eq_analysis = PhaseEqAnalysis(
            discretized_system=point_disc,
            temperature_K=temperature_K,
            pressure_bar=pressure_bar,
            num_components=num_comp,
            gE_model=ge_model,
            antoine_models=antoine_models,
            names_comps_ordered=names_components,
            melting_heats=melting_heats,
            melting_temperatures=melting_temperatures
        )

        # just done to ensure that load works...
        loaded_phase_eq_simplices = phase_eq_analysis.load_phase_eq_liquid(loading_path_phase_eq)

        return phase_eq_analysis, loaded_phase_eq_simplices

    @staticmethod
    def create_phase_eq(point_disc: PointDisc, temperature_K: float, pressure_bar: float, num_comp: int, ge_model,
                        antoine_models: dict or None, names_components: list, plotter_instance: Plotter,
                        path_for_storage: str, names_components_combined: str, melting_heats: dict or None,
                        melting_temperatures: dict or None):
        # init phase eq analysis
        phase_eq_analysis = PhaseEqAnalysis(
            discretized_system=point_disc,
            temperature_K=temperature_K,
            pressure_bar=pressure_bar,
            num_components=num_comp,
            gE_model=ge_model,
            antoine_models=antoine_models,
            names_comps_ordered=names_components,
            melting_temperatures=melting_temperatures,
            melting_heats=melting_heats
        )

        # construct phase split simplices
        phase_split_simplices = phase_eq_analysis.construct_phase_equilibrium()

        # store the phase eq
        loading_path_phase_eq = phase_eq_analysis.store_phase_eq_liquid(
            names_components_combined=names_components_combined,
            phase_split_simplices=phase_split_simplices,
            standard_path=path_for_storage
        )

        # plot phase eq, if possible (in VLE case, this will not be a very useful plot for N=2, for example...)
        if num_comp <= 4:
            plotter_instance.plot_outer_simplex(
                "black",
                color_dict={'l': 'blue', 'g': 'magenta', 's': 'lime'}
            )

            for simplex in phase_split_simplices:
                plotter_instance.plot_phase_split_simplex(
                    simplex, color_dict={
                        'l': 'blue', 'g': 'magenta', 's': 'lime'
                    }
                )

            plotter_instance.save_plot(os.path.join(loading_path_phase_eq, "phase_eq.png"))

            # plot dgmix for binary systems
            if num_comp == 2:
                plotter_instance.plot_dgmix(
                    phase_eq_analysis.dict_dgmix_values_per_phase,
                    color_dict={
                        'l': 'blue', 'g': 'magenta', 's': 'lime'
                    },
                    do_scatter=False
                )
                plotter_instance.save_plot(path=os.path.join(loading_path_phase_eq, "dgmix.png"))

                # plot conv hull
                plotter_instance.plot_conv_hull_dgmix(
                    phase_eq_analysis.dict_dgmix_values_per_phase,
                    color_dict={
                        'l': 'blue', 'g': 'magenta', 's': 'lime'
                    },
                    phase_split_simplices=phase_split_simplices
                )
                plotter_instance.save_plot(path=os.path.join(loading_path_phase_eq, "convhull_dgmix.png"))

        return phase_eq_analysis, phase_split_simplices, loading_path_phase_eq
