import numpy as np


class NRTLModel:
    def __init__(self, data_dict: dict, option_dict: dict, num_comp: int):
        """
        Class stores the binary parameters of all components.

        Computes the activity coefficient of component index as described in Molecular
        Thermodynamics of Fluid-Phase Equilibria 1999. There, we have two parameters tau_ij and
        tau_ji (non symmetric) and alpha (symmetric).

        The interaction parameters are most likely not provided in a common format. Therefore, we will have several
        options to choose from.

        Option 1:
            tau_option == 'tau_without_temperature_scaling':
                An unscaled tau matrix is given which is rescaled by division by the temperature in Kelvin:
                tau_ij = tau_without_temperature_scaling_ij / T

        Option 2:
            tau_option == 'oh' (from paper oh2003)
                Aij, Aji, and alpha (symmetric is provided)
                Aij = gij - gjj
                tau_ij = Aij / RT

        Option 3:
            tau_option == 'yuan' (from paper yuan2019)
                tau_ij = a_ij + (b_ij / T)
                alpha is also provided (seems to be symmetric)
        """
        # contains options for calculation method
        self.options = option_dict

        # contains property data
        self.data_dict = data_dict

        self.num_comp = num_comp

    def get_binary_parameters(self, temperature):
        """
        Binary interactions depend on the temperature (provided in Kelvin) and are calculated here.
        """
        tau_matrix = None
        upper_G_matrix = None

        # check the different options
        if self.options['ge_model_option'] == 'tau_without_temperature_scaling':
            tau_matrix = np.empty((self.num_comp, self.num_comp))
            upper_G_matrix = np.empty((self.num_comp, self.num_comp))
            for i in range(self.num_comp):
                for j in range(self.num_comp):
                    tau_matrix[i][j] = self.data_dict['tau_without_temperature_scaling'][i][j] / temperature
                    upper_G_matrix[i][j] = np.exp(-1 * self.data_dict['alphas'][i][j] * tau_matrix[i][j])

        elif self.options['ge_model_option'] == 'oh':
            tau_matrix = np.empty((self.num_comp, self.num_comp))
            upper_G_matrix = np.empty((self.num_comp, self.num_comp))
            for i in range(self.num_comp):
                for j in range(self.num_comp):
                    tau_matrix[i][j] = self.data_dict['A'][i][j] / (8.314 * temperature)
                    upper_G_matrix[i][j] = np.exp(-1 * self.data_dict['alphas'][i][j] * tau_matrix[i][j])

        elif self.options['ge_model_option'] == 'yuan':
            tau_matrix = np.empty((self.num_comp, self.num_comp))
            upper_G_matrix = np.empty((self.num_comp, self.num_comp))
            for i in range(self.num_comp):
                for j in range(self.num_comp):
                    tau_matrix[i][j] = self.data_dict['a'][i][j] + (self.data_dict['b'][i][j] / temperature)
                    upper_G_matrix[i][j] = np.exp(-1 * self.data_dict['alphas'][i][j] * tau_matrix[i][j])

        return tau_matrix, upper_G_matrix

    def compute_activity_coefficient(self, molar_fractions, index, temperature):
        """
        Index refers to the component, where the activity coefficient should be calculated.
        """
        # get interaction parameters to given temperature (unit K)
        tau_matrix, upper_G_matrix = self.get_binary_parameters(temperature)

        # we use the same formula as in the reference from the description (6-169)
        # it is split in several terms for the sake of readability
        numerator_1 = 0
        denominator_1 = 0
        for j in range(self.num_comp):
            numerator_1 = numerator_1 + (tau_matrix[j][index] * upper_G_matrix[j][index] *
                                         molar_fractions[j])

            denominator_1 = denominator_1 + (upper_G_matrix[j][index] * molar_fractions[j])

        summand_1 = numerator_1 / denominator_1

        summand_2 = 0
        for j in range(self.num_comp):
            denom_first_part_sum_2 = 0
            for l in range(self.num_comp):
                denom_first_part_sum_2 = denom_first_part_sum_2 + (
                        upper_G_matrix[l][j] * molar_fractions[l])

            first_part_summand_2 = molar_fractions[j] * upper_G_matrix[index][j] / denom_first_part_sum_2

            numerator_2 = 0
            for r in range(self.num_comp):
                numerator_2 = numerator_2 + (molar_fractions[r] * tau_matrix[r][j] *
                                             upper_G_matrix[r][j])

            second_part_summand_2 = tau_matrix[index][j] - (numerator_2 / denom_first_part_sum_2)
            summand_2 = summand_2 + (first_part_summand_2 * second_part_summand_2)

        activity_coefficient = np.exp(summand_1 + summand_2)

        return np.clip(activity_coefficient, a_min=0.0000001, a_max=None)


class AntoineModel:
    def __init__(self, data_dict: dict, option_dict: dict):
        self.A = data_dict["A"]
        self.B = data_dict["B"]
        self.C = data_dict["C"]
        self.Tmin = data_dict["Tmin"]  # assumed in K
        self.Tmax = data_dict["Tmax"]  # assumed in K

        # options (necessary if different versions of Antoine models are used)
        self.options = option_dict

    def compute_vapor_pressure_bar(self, temperature_K):
        """
        log_10 p/Pa = ln(p/Pa) / ln(10) = A - (B / (C + T/K))

        -> ln(p/Pa) = ln(10) * (A - (B / (C + T/K)))

        -> p/Pa = exp(ln(10) * (A - (B / (C + T/K))))
        """
        if temperature_K > self.Tmax or temperature_K < self.Tmin:
            raise Exception('Trying to get vapor pressure outside temperature limits!',
                            temperature_K, self.Tmax, self.Tmin)

        p_bar = None
        if self.options['antoine_option'] == 'oh':
            # kPa are computed from Celsius
            temperature_C = temperature_K - 273.15
            p_kPa = np.exp(np.log(10) * (self.A - (self.B / (self.C + temperature_C))))
            p_bar = p_kPa / 100

        elif self.options['antoine_option'] == 'xiao':
            # kPa are computed from Kelvin
            p_kPa = np.exp(np.log(10) * (self.A - (self.B / (self.C + temperature_K))))
            p_bar = p_kPa / 100

        elif self.options['antoine_option'] == 'nist':
            p_bar = np.exp(np.log(10) * (self.A - (self.B / (self.C + temperature_K))))

        return p_bar
