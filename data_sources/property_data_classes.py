import os
import itertools

import pandas as pd
import numpy as np

from util.helper_functions import convert_mass_fr_to_molar_fr


class OptionData:
    def __init__(self):
        self.data = {}
        self.standard_path = os.path.join(os.getcwd(), "data_sources", "data", "options.xlsx")

    def add_option_data(self, name: str):
        # no doubled names, name should be in lowercase (name of first author, followed by year)
        if name in self.data.keys():
            raise Exception('Try to add options with existing author name!')

        data = pd.read_excel(self.standard_path, sheet_name=name)

        # get attributes to store
        list_attribute_names = [i for i in data.keys()]

        # everything is stored in dict format
        self.data[name] = {}
        for attribute_name in list_attribute_names:
            self.data[name][attribute_name] = data[attribute_name][0]

    def get_option_data(self, name: str):
        if name not in self.data.keys():
            raise Exception('Try to get options for non-existing author name!')

        return self.data[name]


class AntoineData:
    def __init__(self):
        self.data = {}
        self.standard_path = os.path.join(os.getcwd(), "data_sources", "data", "antoine_data.xlsx")

    def add_model(self, name: str):
        # no doubled names, name should be in lowercase (name of first author, followed by year)
        if name in self.data.keys():
            raise Exception('Try to add Antoine model with existing author name!')

        data = pd.read_excel(self.standard_path, sheet_name=name, index_col=0)

        # get component names and attributes to store
        list_component_names = [i for i in data.keys()]
        list_attribute_names = [i for i in data.index]

        # we add a dict with author name as key, the value is a dict that contains
        # the attributes for every component in the given dataset
        self.data[name] = {}
        for i, name_component in enumerate(list_component_names):
            self.data[name][name_component] = {}
            for j, name_attribute in enumerate(list_attribute_names):
                self.data[name][name_component][name_attribute] = data[name_component][name_attribute]

    def get_model(self, name: str):
        if name not in self.data.keys():
            raise Exception('Try to get Antoine model for non-existing author name!')

        return self.data[name]


class SolidData:
    def __init__(self):
        self.data = {}
        self.standard_path = os.path.join(os.getcwd(), "data_sources", "data", "solid_data.xlsx")

    def add_model(self, name: str):
        # no doubled names, name should be in lowercase (name of first author, followed by year)
        if name in self.data.keys():
            raise Exception('Try to add Antoine model with existing author name!')

        data = pd.read_excel(self.standard_path, sheet_name=name, index_col=0)

        # get component names and attributes to store
        list_component_names = [i for i in data.keys()]
        list_attribute_names = [i for i in data.index]

        # we add a dict with author name as key, the value is a dict that contains
        # the attributes for every component in the given dataset
        self.data[name] = {}
        for i, name_component in enumerate(list_component_names):
            self.data[name][name_component] = {}
            for j, name_attribute in enumerate(list_attribute_names):
                self.data[name][name_component][name_attribute] = data[name_component][name_attribute]

    def get_model(self, name: str):
        if name not in self.data.keys():
            raise Exception('Try to get solid data for non-existing author name!')

        return self.data[name]


class GEDataset:
    def __init__(self):
        # this class stores binary interactions of a given paper
        # the types of the binary interactions can vary depending on
        # author and model
        self.data_dict = {}  # keys will be name pairs
        self.data_dict['pure_component_data'] = {}  # as placeholder, if we need to store some pure component data

    def get_binary_interaction_para(self, name_1: str, name_2: str):
        relevant_name = str(name_1 + "--" + name_2)
        invert = False
        if relevant_name not in self.data_dict.keys():
            # invert name pair
            relevant_name = str(name_2 + "--" + name_1)
            invert = True
            if relevant_name not in self.data_dict.keys():
                raise Exception('Name pair not found!')

        # create the dictionary with the binary interactions
        if invert:
            # invert all binary interactions
            dict_to_return = {}
            for key in self.data_dict[relevant_name].keys():
                dict_to_return[key] = np.flip(self.data_dict[relevant_name][key])

        else:
            dict_to_return = self.data_dict[relevant_name]

        return dict_to_return

    def get_multinary_interactions(self, names):
        # goal is not to get the binary interaction vectors, but to create
        # matrices that describe higher order systems through binaries
        matrices_dict = {}

        # get all possible binaries
        todo = itertools.combinations(list(range(len(names))), 2)

        # inside a dataset from one author the data is assumed to be homogeneous.
        # therefore, we can expect that any binary here, contains all relevant keys.
        # we just take the first binary, copy the keys, and insert empty matrices.
        first_binary = self.get_binary_interaction_para(names[0], names[1])
        for key in first_binary.keys():
            matrices_dict[key] = np.zeros((len(names), len(names)))

        # now that we have the empty matrices, we can fill them with the binary information
        for combination in todo:
            binary_dict = self.get_binary_interaction_para(names[combination[0]], names[combination[1]])
            for key in matrices_dict.keys():
                matrices_dict[key][combination[0]][combination[1]] = binary_dict[key][0]
                matrices_dict[key][combination[1]][combination[0]] = binary_dict[key][1]

        return matrices_dict, self.data_dict['pure_component_data']

    def add_pair(self, name_1: str, name_2: str, dict_binary_interactions: dict):
        name_pair = str(name_1 + "--" + name_2)
        # we fill some zero values, if they were not set specifically
        for key_to_add in ['c', 'd', 'e']:
            if key_to_add not in dict_binary_interactions.keys():
                dict_binary_interactions[key_to_add] = np.zeros(2)

        self.data_dict[name_pair] = dict_binary_interactions  # keys of this dict are already attributes of ge-model

    def add_pure_component_data(self, name: str, data: dict):
        if name in self.data_dict['pure_component_data'].keys():
            raise Exception('Trying to add doubled pure component data!')

        self.data_dict['pure_component_data'][name] = data


class GEDatasets:
    def __init__(self):
        self.data = {}

    def add_model(self, name: str, dataset: GEDataset):
        # no doubled names
        if name in self.data.keys():
            raise Exception('Try to add ge-model model with existing author name!')

        self.data[name] = dataset

    def get_model(self, name: str):
        if name not in self.data.keys():
            raise Exception('Try to get ge-model for non-existing author name!')

        return self.data[name]


class ExperimentalDataset:
    def __init__(self, name_excel_file: str, sheet_names: list, list_temperature: list, list_pressure: list,
                 molar_fractions_given: bool, feed_given: bool, ordered_component_names_per_sheet: list,
                 list_molar_masses: list):
        # a class that stores all experimental data from one paper
        self.experimental_data_in_tables = {}  # keys are the sheet names in the respective excel
        self.standard_path = os.path.join(os.getcwd(), "data_sources", "data", name_excel_file + '.xlsx')

        # iterate over sheet names and store the data from the excel file.
        for i, sheet_name in enumerate(sheet_names):
            dataframe = pd.read_excel(self.standard_path, sheet_name=sheet_name, header=None, index_col=None)

            # for each sheet name, t and p are given. inside one paper, only molar or mass fractions are given.
            # if there are for example two and three phase regions, please make separated tables for that...
            self.experimental_data_in_tables[sheet_name] = self.add_table(
                dataframe=dataframe, temperature=list_temperature[i], pressure=list_pressure[i],
                list_names_components=ordered_component_names_per_sheet[i], molar_fractions_given=molar_fractions_given,
                molar_masses=list_molar_masses[i], feeds_given=feed_given
            )

            # if we construct VLEs, we have usually a range of p, T given, this part we capture in a separate sheet
            loaded_excel_file = pd.ExcelFile(self.standard_path)
            present_sheet_names = loaded_excel_file.sheet_names
            if sheet_name + '_pT' in present_sheet_names:
                pT_dataframe = pd.read_excel(
                    self.standard_path, sheet_name=sheet_name + '_pT', header=None, index_col=None
                )
                pT_dataframe = pT_dataframe.to_numpy().transpose()
                self.experimental_data_in_tables[sheet_name]['array_p_bar'] = pT_dataframe[0]
                self.experimental_data_in_tables[sheet_name]['array_T_K'] = pT_dataframe[1]

    @staticmethod
    def add_table(dataframe, temperature: float or None, pressure: float or None, list_names_components: list,
                  molar_fractions_given: bool, molar_masses: np.array or None, feeds_given: bool):
        table_dict = {
            'temperature': temperature,
            'pressure': pressure,
            'list_names_components': list_names_components
        }

        # transform dataframe to np array
        matrix_to_read = dataframe.to_numpy()
        num_comp = len(list_names_components)

        # if the feed is given in the dataframe, we assume it to be in the first num_comp entries
        # of the matrix_to_read. otherwise, we just have the phases given. in both cases, we can
        # compute the number of phases by division by the num of components
        _, num_cols = matrix_to_read.shape
        if feeds_given:
            num_phases = int((num_cols - num_comp) / num_comp)

        else:
            num_phases = int(num_cols / num_comp)

        # set the phases from the dataset
        phases = [[] for _ in range(num_phases)]
        feeds = []
        for i in range(len(matrix_to_read)):
            for j in range(num_phases):
                if feeds_given:
                    phases[j].append(matrix_to_read[i][(j * num_comp) + num_comp:((j + 1) * num_comp) + num_comp])

                else:
                    phases[j].append(matrix_to_read[i][j * num_comp:(j + 1) * num_comp])

            # if the feed is given, just copy from the dataset
            if feeds_given:
                feeds.append(matrix_to_read[i][:num_comp])

            # if not, take the middle of the phases
            else:
                sum_phases = np.zeros(num_comp)
                for j in range(num_phases):
                    sum_phases = sum_phases + phases[j][i]

                feeds.append(sum_phases / sum(sum_phases))

        if not molar_fractions_given:
            for i in range(len(matrix_to_read)):
                feeds[i] = convert_mass_fr_to_molar_fr(feeds[i], molar_masses)
                for j in range(num_phases):
                    phases[j][i] = convert_mass_fr_to_molar_fr(phases[j][i], molar_masses)

        table_dict['feeds'] = feeds
        table_dict['corresponding_phases'] = phases
        table_dict['num_phases'] = num_phases

        return table_dict
