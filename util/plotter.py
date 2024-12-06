import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from discretization.point_discretization import PointDisc


class Plotter:
    def __init__(self, num_comp: int):
        self.num_comp = num_comp

        # to be able to plot the simplex for the whole component system, we just construct it
        # in the same way as in the simplex discretization
        self.vertices_outer_simplex = []
        for i in range(self.num_comp - 1):
            basis_vector = np.zeros(self.num_comp - 1)
            basis_vector[i] = 1 / np.sqrt(2)
            self.vertices_outer_simplex.append(basis_vector)

        # last point
        self.vertices_outer_simplex.append(np.ones(self.num_comp - 1) * (1 + np.sqrt(self.num_comp)) / (
                (self.num_comp - 1) * np.sqrt(2)))

        if self.num_comp == 3:
            # rotation with psi
            psi = 2 * np.pi * 285 / 360
            rotation_matrix = np.array([[np.cos(psi), -1 * np.sin(psi)], [np.sin(psi), np.cos(psi)]])
            for i in range(len(self.vertices_outer_simplex)):
                self.vertices_outer_simplex[i] = np.matmul(rotation_matrix, self.vertices_outer_simplex[i])

        self.matrix_mfr_to_cart, self.matrix_cart_to_mfr = PointDisc.get_basis_change(
            self.vertices_outer_simplex)

        if self.num_comp == 4:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')

    def plot_conv_hull_dgmix(self, dict_dgmix_per_phases: dict, color_dict: dict, phase_split_simplices: list):
        self.plot_dgmix(dict_dgmix_per_phases, color_dict, do_scatter=True)

        lines_to_plot = []
        for simplex in phase_split_simplices:
            if len(simplex.points_molar_fractions) == 2:
                mfrs = [simplex.points_molar_fractions[0][0], simplex.points_molar_fractions[1][0]]
                lines_to_plot.append(
                    [mfrs, simplex.dgmix_values]
                )

        for lines in lines_to_plot:
            plt.plot(lines[0], lines[1], color='black', zorder=5, linestyle='dashed', linewidth=3.0)

    def plot_dgmix(self, dict_dgmix_per_phases: dict, color_dict: dict, do_scatter: bool):
        # plot liquid phase
        self.plot_dgmix_for_one_phase(
            dgmix_values=dict_dgmix_per_phases['l'],
            molar_fractions=dict_dgmix_per_phases['corresponding_mfr'],
            color=color_dict['l'],
            label='liquid',
            marker='s',
            do_scatter=do_scatter
        )

        # plot solid and vapor, if existing
        if dict_dgmix_per_phases['g'] is not None:
            self.plot_dgmix_for_one_phase(
                dgmix_values=dict_dgmix_per_phases['g'],
                molar_fractions=dict_dgmix_per_phases['corresponding_mfr'],
                color=color_dict['g'],
                label='gas',
                marker='o',
                do_scatter=do_scatter
            )

        if dict_dgmix_per_phases['s'] is not None:
            self.plot_dgmix_for_one_phase(
                dgmix_values=dict_dgmix_per_phases['s'],
                molar_fractions=dict_dgmix_per_phases['corresponding_mfr'],
                color=color_dict['s'],
                label='solid',
                marker='^',
                do_scatter=do_scatter
            )

        # format plot
        plt.xlim(0, 1)
        plt.ylabel('$\Delta g^{\mathrm{mix}}$')
        plt.xticks([0, 1])
        plt.yticks([0])
        plt.xlabel('$x_1$ / mol$\cdot$mol$^{-1}$')
        plt.legend()

    def plot_dgmix_for_one_phase(self, dgmix_values: np.array, molar_fractions: list, color: str, label: str,
                                 marker: str, do_scatter: bool):
        if self.num_comp == 2:
            # we have to order the molar fractions and dgmix accordingly
            mol_fractions = np.array([mfr[0] for mfr in molar_fractions])
            ordered_indices = np.argsort(mol_fractions)

            # plot the ordered arrays
            plt.plot(mol_fractions[ordered_indices], dgmix_values[ordered_indices], color=color, label=label, zorder=2)
            if do_scatter:
                plt.scatter(mol_fractions[ordered_indices], dgmix_values[ordered_indices], color=color, marker=marker, zorder=2)

    def plot_txy(self, list_with_txy_info: list, t_range: list, name_comp1: str, plot_lines: bool,
                 location: str or None):
        if self.num_comp == 2:
            # a list where every element is a dict
            # {'t_K': t, 'list_phase_eq_simplices': list}
            t_liquid_points = []
            mfr_liquid_points = []
            t_gas_points = []
            mfr_gas_points = []
            t_solid_points = []
            mfr_solid_points = []
            lines_to_plot = []
            for list_el in list_with_txy_info:
                t = list_el['t_K']
                for simplex in list_el['list_phase_eq_simplices']:
                    for mfr_ind, mfr in enumerate(simplex.points_molar_fractions):
                        relevant_mfr = mfr[0]
                        if simplex.points_aggregate_state[mfr_ind] == 'g':
                            t_gas_points.append(t)
                            mfr_gas_points.append(relevant_mfr)

                        if simplex.points_aggregate_state[mfr_ind] == 'l':
                            t_liquid_points.append(t)
                            mfr_liquid_points.append(relevant_mfr)

                        if simplex.points_aggregate_state[mfr_ind] == 's':
                            t_solid_points.append(t)
                            mfr_solid_points.append(relevant_mfr)

                    if len(simplex.points_molar_fractions) == 2:
                        mfrs = [simplex.points_molar_fractions[0][0], simplex.points_molar_fractions[1][0]]
                        ts = [t, t]
                        lines_to_plot.append(
                            [mfrs, ts]
                        )

            if plot_lines:
                for lines in lines_to_plot:
                    plt.plot(lines[0], lines[1], color='black', zorder=1)

            if len(mfr_liquid_points) > 0:
                plt.scatter(mfr_liquid_points, t_liquid_points, color='blue', label='liquid', zorder=2)

            if len(mfr_gas_points) > 0:
                plt.scatter(mfr_gas_points, t_gas_points, color='magenta', label='gas', zorder=2)

            if len(mfr_solid_points) > 0:
                plt.scatter(mfr_solid_points, t_solid_points, color='lime', label='solid', zorder=2)

            # final plot formats
            plt.xlim(0, 1)
            plt.ylim(t_range[0], t_range[1])
            plt.ylabel('$T$ / K')
            plt.xlabel(r'$x_{\mathrm{' + name_comp1 + '}}$')
            plt.legend(loc=location)

    def plot_pxy(self, list_with_pxy_info: list, pressure_range: list, name_comp1: str, plot_lines: bool,
                 location: str or None):
        if self.num_comp == 2:
            # a list whre every element is a dict
            # {'p_bar': p, 'list_phase_eq_simplices': list}
            p_liquid_points = []
            mfr_liquid_points = []
            p_gas_points = []
            mfr_gas_points = []
            lines_to_plot = []
            for list_el in list_with_pxy_info:
                p = list_el['p_bar']
                for simplex in list_el['list_phase_eq_simplices']:
                    for mfr_ind, mfr in enumerate(simplex.points_molar_fractions):
                        relevant_mfr = mfr[0]
                        if simplex.points_aggregate_state[mfr_ind] == 'g':
                            p_gas_points.append(p)
                            mfr_gas_points.append(relevant_mfr)

                        if simplex.points_aggregate_state[mfr_ind] == 'l':
                            p_liquid_points.append(p)
                            mfr_liquid_points.append(relevant_mfr)

                    if len(simplex.points_molar_fractions) == 2:
                        mfrs = [simplex.points_molar_fractions[0][0], simplex.points_molar_fractions[1][0]]
                        ps = [p, p]
                        lines_to_plot.append(
                            [mfrs, ps]
                        )

            if plot_lines:
                for lines in lines_to_plot:
                    plt.plot(lines[0], lines[1], color='black', zorder=1)

            plt.scatter(mfr_liquid_points, p_liquid_points, color='blue', label='liquid', zorder=2)
            plt.scatter(mfr_gas_points, p_gas_points, color='magenta', label='gas', zorder=2)

            # final plot formats
            plt.xlim(0, 1)
            plt.ylim(pressure_range[0], pressure_range[1])
            plt.ylabel('$p$ / bar')
            plt.xlabel(r'$x_{\mathrm{' + name_comp1 + '}}$')
            plt.legend(loc=location)

    def plot_outer_simplex(self, color, color_dict: dict):
        if self.num_comp == 3:
            plt.gca().set_aspect('equal', adjustable='box')
            points = [self.vertices_outer_simplex[j] for j in range(self.num_comp)]
            for j in range(self.num_comp):
                plt.plot([points[j - 1][0], points[j][0]], [points[j - 1][1], points[j][1]], color=color,
                         linewidth=1)

            plt.xticks([])
            plt.yticks([])
            plt.axis('off')

            # legend
            custom_legend = [
                mpatches.Patch(color=color_dict['l'], label='liquid'),
                mpatches.Patch(color=color_dict['g'], label='gas'),
                mpatches.Patch(color=color_dict['s'], label='solid')
            ]

            # Add the custom legend
            plt.legend(handles=custom_legend, loc='upper right')

        elif self.num_comp == 4:
            points = [self.vertices_outer_simplex[j] for j in range(self.num_comp)]
            for i in range(self.num_comp):
                for j in range(i+1, self.num_comp):
                    self.ax.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]],
                                 zs=[points[i][2], points[j][2]], color=color)

    def plot_phase_split_simplex(self, simplex, color_dict: dict):
        if self.num_comp == 2:
            # plot points
            for i in range(self.num_comp):
                plt.plot(simplex.points_molar_fractions[i][0], 0, marker="o",
                         color=color_dict[simplex.points_aggregate_state[i]])

        elif self.num_comp == 3:
            # for 3 phase simplices, plot transparent area in the simplex
            if np.sum(simplex.edge_classification) == 6:
                t1 = plt.Polygon(simplex.points_coordinates_cart, color='orange', alpha=0.5)
                plt.gca().add_patch(t1)

            else:
                for i in range(self.num_comp):
                    for j in range(i+1, self.num_comp):
                        plt.plot([simplex.points_coordinates_cart[i][0], simplex.points_coordinates_cart[j][0]],
                                 [simplex.points_coordinates_cart[i][1], simplex.points_coordinates_cart[j][1]],
                                 color='black', linewidth=1)

            # plot points
            for i in range(self.num_comp):
                plt.plot(simplex.points_coordinates_cart[i][0], simplex.points_coordinates_cart[i][1], marker="o",
                         color=color_dict[simplex.points_aggregate_state[i]], markersize=3)

        elif self.num_comp == 4:
            for i in range(self.num_comp):
                for j in range(i+1, self.num_comp):
                    self.ax.plot([simplex.points_coordinates_cart[i][0], simplex.points_coordinates_cart[j][0]],
                                 [simplex.points_coordinates_cart[i][1], simplex.points_coordinates_cart[j][1]],
                                 zs=[simplex.points_coordinates_cart[i][2], simplex.points_coordinates_cart[j][2]],
                                 color='black')

    @staticmethod
    def save_plot(path):
        plt.savefig(path, dpi=400)
        plt.close()

    @staticmethod
    def show_plot():
        plt.show()
        plt.close()
