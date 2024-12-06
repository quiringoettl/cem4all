import itertools
import copy
import os

import numpy as np
import scipy.spatial as spat

from discretization.point_discretization import PointDisc
from thermodynamic_models import NRTLModel


class PhaseEqSimplex:
    def __init__(self, points_coords_cart, points_mfr, points_ind,
                 points_aggregate_state: list, matrix, matrix_inv, dgmix_values=None):
        """
        Class to store a simplex, which contains points which split into multiple phases that are in
        equilibrium.

        All important attributes are stored in those simplices to be able to define a unit operation only
        with those.

        These simplices always have dimension n (N=n+1 vertices) and define some phase split.
        For this, we classify the edges into homogeneous (between neighboring points) and
        heterogeneous. Neighboring points are two points, which are contained in one subsimplex
        from the discretization.
        """
        self.index = None
        self.points_coordinates_cart = points_coords_cart
        self.points_molar_fractions = points_mfr
        self.points_indices = points_ind  # point indices from the simplex discretization
        # list where element with index i refers to the aggregate state of point i
        # 'l': liquid, 'g': gas, 's': solid
        self.points_aggregate_state = points_aggregate_state
        self.matrix = matrix  # matrix * lambda = (1, p), lambda barycentric coordinates
        self.matrix_inv = matrix_inv

        # symmetric N x N matrix, entry[i][j] == 1 means that the edge from point[i] to point [j]
        # is heterogeneous, matrix is filled afterwards (default is a homogeneous simplex without
        # an occurring phase split).
        self.edge_classification = np.zeros((len(self.points_indices), len(self.points_indices)))

        # for each phase, we append an block = [indices], where indices refers to the point indices
        # which form the basis of the respective phase. the order here is the fixed order of the phases
        self.phase_blocks = []  # the indices in here range from 0 to N-1

        # store the minimum dgmix values at the respective points
        self.dgmix_values = dgmix_values

    def get_middle(self):
        middle = np.zeros(len(self.points_molar_fractions[0]))
        for mfr in self.points_molar_fractions:
            middle = middle + mfr

        return middle / len(self.points_molar_fractions)


class PhaseEqAnalysis:
    def __init__(self, discretized_system: PointDisc, temperature_K: float, pressure_bar: float,
                 num_components: int, gE_model: NRTLModel or None, antoine_models: dict or None,
                 names_comps_ordered: list, melting_temperatures: dict or None, melting_heats: dict or None,
                 hanna_model=None):
        """
        Construct the respective phase equilibrium using the CEM.
        This is a generalization of Ryll2009, Ryll2012, Göttl2023.
        """
        self.discretized_system = discretized_system
        self.temperature_K = temperature_K
        self.pressure_bar = pressure_bar
        self.num_comp = num_components
        self.gE_model = gE_model  # e.g. NRTL, UNIQUAC
        # in this specific implementation the HANNA model is provided already with the respective SMILES as input
        self.hanna_model = hanna_model
        self.antoine_models = antoine_models
        self.names_comps_ordered = names_comps_ordered

        # set solid properties, if existing
        self.melting_heats = None
        self.melting_temperatures = None
        if melting_heats is not None:
            self.melting_temperatures = np.zeros(num_components)
            self.melting_heats = np.zeros(num_components)
            for i, name in enumerate(names_comps_ordered):
                self.melting_heats[i] = melting_heats[name]
                self.melting_temperatures[i] = melting_temperatures[name]

        self.epsilon = 0.0001  # for comparisons
        self.phases_names = ['l', 'g', 's']  # to have a fixed order for identification

        # to access later on the dgmix values
        self.dict_dgmix_values_per_phase = {
            'l': None,
            'g': None,
            's': None,
            'corresponding_mfr': None
        }

    def find_phase_split(self, feed_molar_flowrates: np.array, discretized_system: PointDisc,
                         phase_split_simplices: list, num_comp: int):
        """
        Returned is a list with dicts for the resulting phases and a bool indicating if there is a phase split.
        """
        found_phase_split = False
        resulting_streams = []
        # ensure non negative flowrates
        feed_molar_flowrates = np.clip(feed_molar_flowrates, 0, None)
        if sum(feed_molar_flowrates) > 0:
            feed_molar_fractions = feed_molar_flowrates / sum(feed_molar_flowrates)
            # check if it is contained in a simplex with a phase split
            feed_cartesian = discretized_system.transform_molar_fr_to_cartesian(feed_molar_fractions)

            # search all phase split simplices
            in_gap = False
            for simplex_ind, simplex in enumerate(phase_split_simplices):
                if PhaseEqAnalysis.point_in_simplex_via_bary(simplex, feed_cartesian):
                    in_gap = True
                    relevant_simplex = phase_split_simplices[simplex_ind]
                    break

            if not in_gap:
                resulting_streams.append({'molar_flowrates': feed_molar_flowrates, 'aggregate_state': None})

            else:
                found_phase_split = True

                # we get the barycentric coordinates of our feed with respect to the relevant simplex
                ext_feed_cartesian = np.ones(num_comp)
                ext_feed_cartesian[1:] = feed_cartesian
                bary_feed_rel_simplex = np.matmul(relevant_simplex.matrix_inv, ext_feed_cartesian)

                # the split ratios are the sum of the barycentric coordinates of the points, which
                # belong to the respective phase block
                num_phases = len(relevant_simplex.phase_blocks)
                split_ratios = np.zeros(num_phases)

                # this way we get the split ratios and with those, we can compute the mfrs of the
                # phases. if we have those, we can get the flowrates.
                mfr_phases = []
                for i, block in enumerate(relevant_simplex.phase_blocks):
                    split_ratios[i] = sum([bary_feed_rel_simplex[j] for j in block])
                    phase_cart = np.zeros(num_comp - 1)
                    for j in range(len(block)):
                        phase_cart = phase_cart + ((bary_feed_rel_simplex[block[j]] / split_ratios[i]) *
                                                   relevant_simplex.points_coordinates_cart[block[j]])

                    mfr_phases.append(discretized_system.transform_cartesian_to_molar_fr(phase_cart))

                splits_flowrates = PhaseEqAnalysis.get_split_flowrates(feed_molar_flowrates, mfr_phases,
                                                                       split_ratios)

                for flowr in splits_flowrates:
                    resulting_streams.append({'molar_flowrates': flowr, 'aggregate_state': None})

            # get aggregate states
            for i, stream in enumerate(resulting_streams):
                mfr = stream['molar_flowrates'] / sum(stream['molar_flowrates'])
                resulting_streams[i]['aggregate_state'] = self.get_aggregate_state(molar_fractions=mfr)

            return found_phase_split, resulting_streams

        else:
            raise Exception('Negative flowrates inserted:', feed_molar_flowrates)

    def get_aggregate_state(self, molar_fractions: np.array):
        dgmix_l = self.compute_delta_g_mix_liquid(
            molar_fractions, temperature=self.temperature_K
        )

        dgmix_g = float('inf')
        if self.antoine_models is not None:
            vap_p = np.zeros(self.num_comp)
            # compute vapor pressures in order
            for name_ind, name in enumerate(self.names_comps_ordered):
                vap_p[name_ind] = self.antoine_models[name].compute_vapor_pressure_bar(
                    temperature_K=self.temperature_K
                )

            dgmix_g = self.compute_delta_g_mix_gas(
                molar_fractions=molar_fractions, vapor_pressures=vap_p,
                temperature=self.temperature_K, pressure=self.pressure_bar
            )

        dgmix_s = float('inf')
        if self.melting_heats is not None:
            dgmix_s = self.compute_delta_g_mix_solid(
                molar_fractions=molar_fractions, temperature=self.temperature_K,
                melting_temperatures=self.melting_temperatures, melting_heats=self.melting_heats
            )

        phases = ['l', 'g', 's']

        return phases[np.argmin([dgmix_l, dgmix_g, dgmix_s])]

    @staticmethod
    def get_split_flowrates(molar_flowrates_feed, phases_mfr, split_ratio):
        """
        phases_mfr contains the molar fractions of the phases, split ratio is a vector with nonnegative
        entries summing up to 1, function returns the flowrates of the ordered phases
        """
        epsilon = 0.0001
        split_flowrates = []
        for i in range(len(split_ratio) - 1):
            flowrates_phase = np.zeros(len(molar_flowrates_feed))
            total_flowrate_phase = sum(molar_flowrates_feed) * split_ratio[i]
            for j in range(len(molar_flowrates_feed)):
                flowrates_phase[j] = total_flowrate_phase * phases_mfr[i][j]

            split_flowrates.append(flowrates_phase)

        # set the last phase
        last_phase_flowrates = molar_flowrates_feed
        for fr in split_flowrates:
            last_phase_flowrates = last_phase_flowrates - fr

        # check for safety
        if np.min(last_phase_flowrates) < -1 * epsilon * sum(molar_flowrates_feed):
            print(molar_flowrates_feed, split_flowrates, last_phase_flowrates)
            return None

        split_flowrates.append(last_phase_flowrates)

        return split_flowrates

    def construct_phase_equilibrium(self):
        relevant_dg_mix_values, present_phases = self.construct_dgmix_graph()
        hull_simplices, graph_points_real_indices, graph = self.construct_convex_hull(
            relevant_dg_mix_values=relevant_dg_mix_values
        )
        phase_split_simplices = self.get_phase_split_simplices(
            hull_simplices=hull_simplices, graph_points_real_indices=graph_points_real_indices, graph=graph,
            present_phases=present_phases
        )

        return phase_split_simplices

    def store_phase_eq_liquid(self, names_components_combined: str, phase_split_simplices: list, standard_path: str):
        # create paths
        if not os.path.isdir(standard_path):
            os.mkdir(standard_path)

        # name is usually a combination of the components names
        standard_path = os.path.join(standard_path, str(self.num_comp) + "_" + str(
                                     self.discretized_system.recursion_steps) + "___" + names_components_combined +
                                     "_T_" + str(round(self.temperature_K, 4)) + "_p_" +
                                     str(round(self.pressure_bar, 4)))

        if not os.path.isdir(standard_path):
            os.mkdir(standard_path)

        # store PhaseEqSimplex sets with all important attributes
        phase_eq_simpl_points_coords_cart = np.empty((len(phase_split_simplices), self.num_comp, self.num_comp - 1))
        phase_eq_simpl_points_mfrs = np.empty((len(phase_split_simplices), self.num_comp, self.num_comp))
        phase_eq_simpl_points_ind = np.empty((len(phase_split_simplices), self.num_comp))
        phase_eq_simpl_points_agg_states = np.empty((len(phase_split_simplices), self.num_comp))
        phase_eq_simpl_mat = np.empty((len(phase_split_simplices),
                                       len(self.discretized_system.matrix_mfr_to_cart),
                                       len(self.discretized_system.matrix_mfr_to_cart)))
        phase_eq_simpl_mat_inv = np.empty((len(phase_split_simplices),
                                           len(self.discretized_system.matrix_mfr_to_cart),
                                           len(self.discretized_system.matrix_mfr_to_cart)))
        edge_classifications = np.empty((len(phase_split_simplices), self.num_comp, self.num_comp))
        phase_blocks = np.empty((len(phase_split_simplices), (self.num_comp * 2) + 1))

        for i, simplex in enumerate(phase_split_simplices):
            for j in range(self.num_comp):
                phase_eq_simpl_points_coords_cart[i][j] = simplex.points_coordinates_cart[j]
                phase_eq_simpl_points_mfrs[i][j] = simplex.points_molar_fractions[j]

            phase_eq_simpl_points_ind[i] = simplex.points_indices
            phase_eq_simpl_mat[i] = simplex.matrix
            phase_eq_simpl_mat_inv[i] = simplex.matrix_inv
            edge_classifications[i] = simplex.edge_classification

            # get aggregate states
            for j in range(len(simplex.points_aggregate_state)):
                phase_eq_simpl_points_agg_states[i][j] = self.phases_names.index(simplex.points_aggregate_state[j])

            # we store each cluster and add -1 mark(s) its end
            current_ind = 0
            for cluster in simplex.phase_blocks:
                for j in cluster:
                    phase_blocks[i][current_ind] = j
                    current_ind = current_ind + 1

                phase_blocks[i][current_ind] = -1
                current_ind = current_ind + 1

            phase_blocks[i][current_ind:] = -1

        np.save(os.path.join(standard_path, "simpl_points_coords_cart.npy"), phase_eq_simpl_points_coords_cart)
        np.save(os.path.join(standard_path, "simpl_points_mfrs.npy"), phase_eq_simpl_points_mfrs)
        np.save(os.path.join(standard_path, "simpl_points_ind.npy"), phase_eq_simpl_points_ind)
        np.save(os.path.join(standard_path, "simpl_points_agg_states.npy"), phase_eq_simpl_points_agg_states)
        np.save(os.path.join(standard_path, "simpl_mat.npy"), phase_eq_simpl_mat)
        np.save(os.path.join(standard_path, "simpl_mat_inv.npy"), phase_eq_simpl_mat_inv)
        np.save(os.path.join(standard_path, "edge_classifications.npy"), edge_classifications)
        np.save(os.path.join(standard_path, "phase_blocks.npy"), phase_blocks)

        return standard_path

    def load_phase_eq_liquid(self, path):
        phase_split_simplices = []

        # basically reverse the storing procedure
        phase_eq_simpl_points_coords_cart = np.load(os.path.join(path, "simpl_points_coords_cart.npy"))
        phase_eq_simpl_points_mfrs = np.load(os.path.join(path, "simpl_points_mfrs.npy"))

        # for the index arrays and aggregate states, we need integers
        phase_eq_simpl_points_ind = np.load(os.path.join(path, "simpl_points_ind.npy")).astype(int)
        phase_eq_simpl_points_agg_states = np.load(os.path.join(path, "simpl_points_agg_states.npy")).astype(int)

        phase_eq_simpl_mat = np.load(os.path.join(path, "simpl_mat.npy"))
        phase_eq_simpl_mat_inv = np.load(os.path.join(path, "simpl_mat_inv.npy"))
        edge_classifications = np.load(os.path.join(path, "edge_classifications.npy"))

        # this one has to be transformed into clusters with ints as indices
        phase_blocks_source = np.load(os.path.join(path, "phase_blocks.npy"))

        for i in range(len(phase_eq_simpl_points_ind)):
            agg_states = []
            for agg_state_index in phase_eq_simpl_points_agg_states[i]:
                agg_states.append(self.phases_names[agg_state_index])

            simplex = PhaseEqSimplex(
                points_coords_cart=phase_eq_simpl_points_coords_cart[i],
                points_mfr=phase_eq_simpl_points_mfrs[i],
                points_ind=phase_eq_simpl_points_ind[i],
                points_aggregate_state=agg_states,
                matrix=phase_eq_simpl_mat[i],
                matrix_inv=phase_eq_simpl_mat_inv[i]
            )

            simplex.edge_classification = edge_classifications[i]

            phase_blocks = []
            block = []
            for j in range((self.num_comp * 2) + 1):
                # end of a cluster is marked
                if phase_blocks_source[i][j] < 0:
                    phase_blocks.append(block)
                    # check if we are at the end
                    if phase_blocks_source[i][j+1] < 0:
                        break

                    else:
                        block = []

                else:
                    block.append(int(phase_blocks_source[i][j]))

            simplex.phase_blocks = phase_blocks
            simplex.index = i

            phase_split_simplices.append(simplex)

        return phase_split_simplices

    def construct_dgmix_graph(self):
        values_dgmix_liquid = np.zeros(len(self.discretized_system.points_mfr))

        # gas and solid will sometimes not be constructed, therefore, we init the values as inf so that
        # they will for sure not be the minimum dg value later on
        values_dgmix_gas = np.full(len(self.discretized_system.points_mfr), float('inf'))
        values_dgmix_solid = np.full(len(self.discretized_system.points_mfr), float('inf'))

        # should be the minimum of the curves (pointwise)
        relevant_dg_mix_values = []
        present_phases = []
        for i, point_mfr in enumerate(self.discretized_system.points_mfr):
            values_dgmix_liquid[i] = self.compute_delta_g_mix_liquid(
                point_mfr, temperature=self.temperature_K
            )

            if self.antoine_models is not None:
                vap_p = np.zeros(self.num_comp)
                # compute vapor pressures in order
                for name_ind, name in enumerate(self.names_comps_ordered):
                    vap_p[name_ind] = self.antoine_models[name].compute_vapor_pressure_bar(
                        temperature_K=self.temperature_K
                    )

                values_dgmix_gas[i] = self.compute_delta_g_mix_gas(
                    molar_fractions=point_mfr, vapor_pressures=vap_p,
                    temperature=self.temperature_K, pressure=self.pressure_bar
                )

            if self.melting_heats is not None:
                values_dgmix_solid[i] = self.compute_delta_g_mix_solid(
                    molar_fractions=point_mfr, temperature=self.temperature_K,
                    melting_temperatures=self.melting_temperatures, melting_heats=self.melting_heats
                )

            to_compare = [values_dgmix_liquid[i], values_dgmix_gas[i], values_dgmix_solid[i]]
            min_index = np.argmin(to_compare)

            # store the final dg_mix value and the present phase
            relevant_dg_mix_values.append(np.min(to_compare))
            present_phases.append(self.phases_names[min_index])

        # store values for plots later (only relevant for binary systems)
        if self.num_comp == 2:
            self.dict_dgmix_values_per_phase['corresponding_mfr'] = self.discretized_system.points_mfr
            self.dict_dgmix_values_per_phase['l'] = values_dgmix_liquid
            if self.antoine_models is not None:
                self.dict_dgmix_values_per_phase['g'] = values_dgmix_gas

            if self.melting_heats is not None:
                self.dict_dgmix_values_per_phase['s'] = values_dgmix_solid

        return relevant_dg_mix_values, present_phases

    def construct_convex_hull(self, relevant_dg_mix_values):
        # to ensure a proper roof (horizontal line from 0 to 1 at dgmix=0 for the binary case), we add pseudo points.
        # if those are added anyway (e.g., when in a vle case the vapor phase at 0 mfr has a positive value for dmix)
        # we will track that and sort out later on the pseudo points that were added artificially.
        pseudo_point_already_present = [False] * self.num_comp

        # we transform the graph values a bit before we construct the convex hull
        graph = []
        graph_points_real_indices = []
        dg_mix_graph = []
        indices_pure_components = []
        for i, value in enumerate(relevant_dg_mix_values):
            # only negative values matter for this method as positive values for delta_g_mix never
            # lead to a stable state
            corrected_value = np.min([0, relevant_dg_mix_values[i]])

            # we only care for negative values for dgmix or for pure components (to get a
            # proper roof from the convex hull construction)
            if corrected_value < -1 * self.epsilon or np.max(self.discretized_system.points_mfr[i]) > 1 - self.epsilon:
                graphvalue = np.zeros(self.num_comp)
                graphvalue[:-1] = self.discretized_system.points_cart[i]
                graphvalue[-1] = corrected_value
                graph.append(graphvalue)
                graph_points_real_indices.append(i)
                dg_mix_graph.append(corrected_value)

                # track if this would replace a pseudo point s.t. we don't have to add it artificially
                if np.max(self.discretized_system.points_mfr[i]) > 1 - self.epsilon:
                    indices_pure_components.append(i)
                    if corrected_value > -1 * self.epsilon:
                        pseudo_point_already_present[np.argmax(self.discretized_system.points_mfr[i])] = True

        # to ensure a proper roof, we add pure components with dgmix equal to 0 and sort them out later
        for i in range(self.num_comp):
            if not pseudo_point_already_present[i]:
                # get mfr of pure comp
                mfr = np.zeros(self.num_comp)
                mfr[i] = 1

                # convert to cartesian
                cart = self.discretized_system.transform_molar_fr_to_cartesian(molar_fractions=mfr)

                # get pseudopoint for graph
                graphvalue = np.zeros(self.num_comp)
                graphvalue[:-1] = cart
                graphvalue[-1] = 0
                graph.append(graphvalue)

        # basic check for safety
        if len(indices_pure_components) != self.num_comp:
            raise Exception('Found', len(indices_pure_components), 'comps in a', self.num_comp,  'system!')

        # the dgmix values are floored to a max of 0. If their min is also 0, it means that there is no stable state
        # and thus, the miscibility gap is between all pure components. In this case, there is no need to construct a
        # convex hull (and it is also not possible due to dimensionality)
        if np.min(dg_mix_graph) > -1 * self.epsilon:
            hull_simplices = [indices_pure_components]

        else:
            # construct hull
            hull_simplices = spat.ConvexHull(graph).simplices

        return hull_simplices, graph_points_real_indices, graph

    def get_phase_split_simplices(self, hull_simplices, graph_points_real_indices, graph, present_phases):
        """
        At this stage we already have a convex hull of our graph which consists of the cartesian coordinates
        and the last entry corresponds to delta_g_mix.

        Now we check for all simplices in the convex hull, if those connect neighboring points in the original
        setting (without delta_g_mix values). If a simplex does not connect neighboring points, it models a
        phase split.
        """
        phase_split_simplices = []

        # expected length depends on discretization
        expected_length = np.sqrt(2) * self.discretized_system.stepsize
        expected_simplex_volume = self.discretized_system.volume_simplex(
            self.discretized_system.vertices_outer_simplex) / (
                int(2 ** self.discretized_system.recursion_steps) ** (self.num_comp - 1))

        for index, simplex_points_indices in enumerate(hull_simplices):
            # sort out the artificially added pseudo points
            pseudopoint_included = False
            for i in simplex_points_indices:
                if i >= len(graph_points_real_indices):
                    pseudopoint_included = True

            if not pseudopoint_included:
                values_dgmix_simplex = [graph[i][-1] for i in simplex_points_indices]
                point_indices_simplex = [graph_points_real_indices[i] for i in simplex_points_indices]
                point_aggregate_states = [present_phases[i] for i in point_indices_simplex]
                vertices_mfr = [self.discretized_system.points_mfr[i] for i in point_indices_simplex]
                vertices_cartesian = [self.discretized_system.points_cart[i] for i in point_indices_simplex]
                simplex_volume = self.discretized_system.volume_simplex(vertices_cartesian)

                resulting_legal_simplex = self.check_if_phase_split_simplex(
                    values_dgmix_simplex=values_dgmix_simplex,
                    volume_simplex=simplex_volume,
                    expected_volume_simplex=expected_simplex_volume,
                    vertices_mfr_simplex=vertices_mfr,
                    expected_length=expected_length,
                    vertices_cartesian_simplex=vertices_cartesian,
                    point_indices_simplex=point_indices_simplex,
                    point_aggregate_states=point_aggregate_states
                )

                if resulting_legal_simplex is not None:
                    phase_split_simplices.append(resulting_legal_simplex)

        return phase_split_simplices

    def check_if_phase_split_simplex(self, values_dgmix_simplex: list, volume_simplex: float,
                                     expected_volume_simplex: float, vertices_mfr_simplex: list,
                                     expected_length: float, vertices_cartesian_simplex: list,
                                     point_indices_simplex: list, point_aggregate_states: list):
        """
        Collection of small tests that check if the simplex shows a phase split or not.
        """
        # as described in Ryll2009, we don't have to care about subsimplices, where deltag_g_mix is 0
        # for all vertices. this way we automatically filter the "roof" (consisting of the pure components
        # of our convex hull).
        if np.min(values_dgmix_simplex) > -1 * self.epsilon:
            return None

        # if the simplex has a volume close to 0, it has no phase split (the convex hull procedure
        # sometimes generates simplices with volume close to 0 that are not relevant...).
        if np.abs(volume_simplex) <= self.epsilon * expected_volume_simplex:
            return None

        # if we have more than 1 aggregate state in the simplex, it is always a phase split simplex
        only_one_aggregate_state_inside_simplex = True
        if point_aggregate_states.count(point_aggregate_states[0]) != len(point_aggregate_states):
            only_one_aggregate_state_inside_simplex = False

        # now we check, if there is a deviation from the expected volume (the check for 0 volume before is
        # still necessary since 0 volume is also a deviation)
        if np.abs(volume_simplex - expected_volume_simplex) / expected_volume_simplex <= self.epsilon:
            # if we have the expected volume and only one aggregate state, it is no phase split
            if only_one_aggregate_state_inside_simplex:
                return None

        # now, we compute the distances between the vertices of the simplex
        # afterwards, we want to check for deviations to the expected length
        distance_matrix = np.zeros((self.num_comp, self.num_comp))
        for i in range(self.num_comp):
            for j in range(i + 1, self.num_comp):
                distance_matrix[i][j] = self.discretized_system.euclidean_distance(
                    vertices_mfr_simplex[i], vertices_mfr_simplex[j]
                )

                distance_matrix[j][i] = distance_matrix[i][j]

        # check for deviations in the lengths
        if np.abs(np.max(distance_matrix) - expected_length) / expected_length < self.epsilon:
            # if we have the expected length and only one aggregate state, it is no phase split
            if only_one_aggregate_state_inside_simplex:
                return None

        # init a candidate simplex, which we will check for phase splits
        matrix, matrx_inv = self.discretized_system.get_basis_change(vertices_cartesian_simplex)
        candidate_simplex = PhaseEqSimplex(
            points_coords_cart=vertices_cartesian_simplex, points_mfr=vertices_mfr_simplex,
            points_ind=point_indices_simplex, points_aggregate_state=point_aggregate_states,
            matrix=matrix, matrix_inv=matrx_inv, dgmix_values=values_dgmix_simplex
        )

        # we want to check if we can model the phase split in this simplex, for this, we have
        # to determine for each edge if it is homo- or heterogeneous
        for i in range(self.num_comp):
            for j in range(i + 1, self.num_comp):
                # if the edge is too long, it is heterogeneous
                if np.abs(distance_matrix[i][j] - expected_length) / expected_length > self.epsilon:
                    # symmetric matrix
                    candidate_simplex.edge_classification[i][j] = 1
                    candidate_simplex.edge_classification[j][i] = 1

                # another criterion for a heterogeneous edge is that the
                # aggregate states of the respective points differ
                if point_aggregate_states[i] != point_aggregate_states[j]:
                    candidate_simplex.edge_classification[i][j] = 1
                    candidate_simplex.edge_classification[j][i] = 1

        # if all edges are homogeneous, it is no phase split simplex
        if candidate_simplex.edge_classification.sum() < self.epsilon:
            return None

        # we check for all vertices, if they are only connected to heterogeneous
        # edges (which means they represent a phase) or if they are also connected
        # to homogeneous edges (and collect those to check if they form a lower dim simplex)
        homogeneous_edges = []
        for i in range(self.num_comp):
            # if all connections are heterogeneous, we have a phase
            if sum(candidate_simplex.edge_classification[i]) > 0.99 * (self.num_comp - 1):
                candidate_simplex.phase_blocks.append([i])

            else:
                for j in range(i + 1, self.num_comp):
                    if candidate_simplex.edge_classification[i][j] < 0.01:
                        homogeneous_edges.append([i, j])

        # now we cluster all homogeneous edges (two edges are in the same cluster,
        # if they have one index in common).
        clusters = []
        current_cluster = []
        # if we add new edges to this cluster, we store a copy here, so that we check also for
        # those, if there are connections left in homogeneous edges
        todo = []
        while len(homogeneous_edges) > 0 or len(todo) > 0:
            # we always compare the remaining homogeneous edges with a current edge
            if len(todo) == 0:
                # if empty, a new cluster is started
                current_edge = homogeneous_edges[0]
                homogeneous_edges.remove(homogeneous_edges[0])
                current_cluster.append(current_edge)

            else:
                current_edge = todo[0]
                todo.remove(todo[0])

            # here we store the found connections
            to_remove = []
            for i, edge in enumerate(homogeneous_edges):
                # each edge occurs only once in homogeneous edges (due to the construction)
                if current_edge[0] in edge or current_edge[1] in edge:
                    to_remove.append(i)

            for i in to_remove:
                current_cluster.append(homogeneous_edges[i])
                todo.append(homogeneous_edges[i])

            # remove from homogeneous edges
            for i in reversed(to_remove):
                homogeneous_edges.remove(homogeneous_edges[i])

            # if we did not find any new edges for the cluster and do not have anything
            # to do, we need a new cluster
            if len(to_remove) == 0 and len(todo) == 0:
                clusters.append(current_cluster)
                current_cluster = []

        # for each cluster, we check now, if it is exactly a low-dimensional simplex (not less or
        # more, as then we cannot model the phase split linearly and we omit this simplex)
        omit_candidate_simplex = False
        for i, cluster in enumerate(clusters):
            # it is enough to check if the cluster is a k simplex, where k+1 is the number of points
            # in the cluster. attention: up to now we just stored the edges of the cluster, so we have
            # to get the unique points first
            point_ind_list = []
            for edge in cluster:
                if edge[0] not in point_ind_list:
                    point_ind_list.append(edge[0])

                if edge[1] not in point_ind_list:
                    point_ind_list.append(edge[1])

            # now we just check for every point index, if there are exactly k edges containing this
            # index in the cluster (which means we would have a low dim simplex)
            for point_index in point_ind_list:
                edge_count = 0
                for edge in cluster:
                    if point_index in edge:
                        edge_count = edge_count + 1

                # if this condition is not fulfilled only once, we can stop
                if edge_count != len(point_ind_list) - 1:
                    omit_candidate_simplex = True
                    break

            # we add the point list, which specifies the phase
            candidate_simplex.phase_blocks.append(point_ind_list)

        # sometimes we encounter simplices that we cannot model (e.g. close to the
        # critical point). This means simplices, which we cannot model as for example
        # the phase blocks are not isolated. Here we try to reduce them, which means
        # we check, if some of the heterogeneous edges could be homogeneous (similar as
        # discussed in Ryll2009) and if this leads to a legal misc gap simplex.
        # Note that this procedure is carried out always (even a legal candidate simplex
        # from the procedure above can be reduced sometimes).
        reduced_simplex = self.reduce_simplex(candidate_simplex=candidate_simplex)

        # returned is the reduced simplex, if it exists, otherwise candidate_simplex,
        # but only if it is a legal simplex.
        simplex_to_return = None
        if reduced_simplex is None:
            if not omit_candidate_simplex:
                simplex_to_return = candidate_simplex

        else:
            simplex_to_return = reduced_simplex

        return simplex_to_return

    def reduce_simplex(self, candidate_simplex: PhaseEqSimplex):
        """
        It is checked if the candidate_simplex can be reduced to some other legal PhaseEqSimplex.
        The resulting legal PhaseEqSimplex is returned (this can also be the original candidate_simplex).
        """
        # get all heterogeneous edges and the euclidean lengths
        # the edges that are between different aggregate states are forced edges
        hetero_index_pairs = []
        hetero_lengths = []
        unforced_hetero_lengths = []
        flags_forced_heteros = []
        for i in range(self.num_comp):
            for j in range(i + 1, self.num_comp):
                if candidate_simplex.edge_classification[i][j] == 1:
                    hetero_index_pairs.append([i, j])
                    length = self.discretized_system.euclidean_distance(
                        candidate_simplex.points_molar_fractions[i],
                        candidate_simplex.points_molar_fractions[j]
                    )
                    hetero_lengths.append(length)
                    if candidate_simplex.points_aggregate_state[i] != candidate_simplex.points_aggregate_state[j]:
                        flags_forced_heteros.append(True)

                    else:
                        flags_forced_heteros.append(False)
                        unforced_hetero_lengths.append(length)

        # if there are heterogeneous edges, which are quite short, we will check if it makes sense
        # to classify them as homogeneous (e.g., it could be that due to too sparse discretization
        # some edges that in reality should be homogeneous, are classified as heterogeneous).
        candidate_indices = []
        small_edge_present = False
        forced_indices = []
        for i, pair in enumerate(hetero_index_pairs):
            if flags_forced_heteros[i]:
                forced_indices.append(i)

            else:
                # the value of 0.6 * max_len was found to provide good results for this procedure
                if hetero_lengths[i] < 0.6 * np.max(hetero_lengths):
                    small_edge_present = True

                candidate_indices.append(i)

        # the current simplex has all indices in candidate_indices as heterogeneous edges. now, we look at
        # simplices that only have a subset of candidate_indices as heterogeneous edges.
        index_subsets = []
        for i in candidate_indices:
            index_subsets.append([i])

        # if all indices are candidate indices, we just look at real subsets, as it just
        # seems quite unlikely that it is not at least some phase split there, but if there are
        # less candidate indices than hetero edges, we just look at all possible (sub)sets, as it
        # may be possible to reduce them all
        relevant_length = np.min([len(hetero_index_pairs) - 1, len(candidate_indices)])
        for i in range(2, relevant_length + 1):
            for el in itertools.combinations(candidate_indices, i):
                index_subsets.append(list(el))

        # ensure that the forced indices are not in the subsets (so that they are always kept)
        temp = []
        for i in range(len(index_subsets)):
            no_forced_index_in_subset = True
            for forced_ind in forced_indices:
                if forced_ind in index_subsets[i]:
                    no_forced_index_in_subset = False
                    break

            # we only take index subsets without forced indices
            if no_forced_index_in_subset:
                temp.append(index_subsets[i])

        # replace index subsets by the 'cleaned' list
        index_subsets = temp

        # for each combination of candidate indices, check if this would be a legal PhaseEqSimplex
        # if the edges from the combination are set to homogeneous (so basically he same procedure as before
        # for the original simplex)
        new_simplices = []
        for comb_ind, combination in enumerate(index_subsets):
            new_simplex = copy.deepcopy(candidate_simplex)
            # reset phase blocks
            new_simplex.phase_blocks = []

            # set the edges from the combination to homogeneous
            for index in combination:
                pair = hetero_index_pairs[index]
                new_simplex.edge_classification[pair[0]][pair[1]] = 0
                new_simplex.edge_classification[pair[1]][pair[0]] = 0

            # we check for all vertices, if they are only connected to heterogeneous
            # edges (which means they represent a phase) or if there are also homogeneous
            # edges (and collect those to check if they form a lower dim simplex)
            homogeneous_edges = []
            for i in range(self.num_comp):
                # if all connections are heterogeneous, we have a phase
                if sum(new_simplex.edge_classification[i]) > 0.99 * (self.num_comp - 1):
                    new_simplex.phase_blocks.append([i])

                else:
                    for j in range(i + 1, self.num_comp):
                        if new_simplex.edge_classification[i][j] < 0.01:
                            homogeneous_edges.append([i, j])

            # now we cluster all homogeneous edges (two edges are in the same cluster,
            # if they have one index in common).
            clusters = []
            cluster = []
            # if we add new edges to this cluster, we store a copy here, so that we check also for
            # those, if there are connections left in homogeneous edges
            todo = []
            while len(homogeneous_edges) > 0 or len(todo) > 0:
                # we always compare the remaining homogeneous edges with a current edge
                if len(todo) == 0:
                    # if to_do is empty, a new cluster was started
                    current_edge = homogeneous_edges[0]
                    homogeneous_edges.remove(homogeneous_edges[0])
                    cluster.append(current_edge)

                else:
                    current_edge = todo[0]
                    todo.remove(todo[0])

                # here we store the found connections
                to_remove = []
                for i, edge in enumerate(homogeneous_edges):
                    # each edge occurs only once in homogeneous edges (due to the construction)
                    if current_edge[0] in edge or current_edge[1] in edge:
                        to_remove.append(i)

                # add to cluster and to to_do
                for i in to_remove:
                    cluster.append(homogeneous_edges[i])
                    todo.append(homogeneous_edges[i])

                # remove from homogeneous edges
                for i in reversed(to_remove):
                    homogeneous_edges.remove(homogeneous_edges[i])

                # if we did not find any new edges for the cluster and do not have anything in
                # the to_do list left, we need a new cluster
                if len(to_remove) == 0 and len(todo) == 0:
                    clusters.append(cluster)
                    cluster = []

            # for each cluster, we check now, if it is exactly a low-dimensional simplex (not less and
            # more, as then we cannot model the phase split linearly and omit this simplex)
            omit_new_simplex = False
            for i, cluster in enumerate(clusters):
                # it is enough to check if the cluster is a k simplex, where k+1 is the number of points
                # in the cluster. attention: up to now we just stored the edges of the cluster, so we have
                # to get the unique points first
                point_ind_list = []
                for edge in cluster:
                    if edge[0] not in point_ind_list:
                        point_ind_list.append(edge[0])

                    if edge[1] not in point_ind_list:
                        point_ind_list.append(edge[1])

                # now we just check for every point index, if there are exactly k edges containing this
                # index in the cluster (which means we would have a simplex that we cannot model).
                for point_index in point_ind_list:
                    edge_count = 0
                    for edge in cluster:
                        if point_index in edge:
                            edge_count = edge_count + 1

                    # if this condition is not fulfilled only once, we can stop
                    if edge_count != len(point_ind_list) - 1:
                        omit_new_simplex = True
                        break

                # we add the point list, which specifies the phase
                new_simplex.phase_blocks.append(point_ind_list)

            # if this is a split, which we can model, we add the simplex
            if not omit_new_simplex:
                new_simplices.append([comb_ind, combination, new_simplex])

        # if we found new, reduced, and legal simplices and if we found a small edge before, it is very likely
        # that our original simplex should be reduced. Note that from the procedure above, it is also possible
        # that the original candidate_simplex is contained in new_simplices and returned after the following checks.
        simplex_to_return = None
        if len(new_simplices) > 0 and small_edge_present:
            # if there are more candidates, we first search for the simplex with the least phase blocks
            # if this is not unique, we return the simplex, with the largest minimal hetero edge length.
            # if this is still not unique, we return the simplex with the highest average len in the hetero edges

            # in a first step, those stats are obtained for each simplex:
            cand_num_phases = np.zeros(len(new_simplices))
            min_hetero_edge_len = np.zeros(len(new_simplices))
            for ind, lis in enumerate(new_simplices):
                cand_num_phases[ind] = len(lis[-1].phase_blocks)
                current_min = float('inf')
                for i in range(self.num_comp):
                    for j in range(i + 1, self.num_comp):
                        if lis[-1].edge_classification[i][j] == 1:
                            edge_len = self.discretized_system.euclidean_distance(
                                lis[-1].points_molar_fractions[i],
                                lis[-1].points_molar_fractions[j]
                            )

                            if edge_len < current_min:
                                current_min = edge_len

                min_hetero_edge_len[ind] = current_min

            # get the minimum of phases among new_simplices and check if more than one simplex
            # have this number of phases
            min_phases = min(cand_num_phases)
            counter = 0
            cands_with_min_phases = []
            min_phases_min_hetero_lens = []
            for ind, c in enumerate(new_simplices):
                if np.abs(min_phases - cand_num_phases[ind]) < 0.1:
                    counter = counter + 1
                    cands_with_min_phases.append(c)
                    min_phases_min_hetero_lens.append(min_hetero_edge_len[ind])

            # if there are more simplices with the minimum number of phases,
            # we check for the largest minimal hetero edge length
            second_reduced_cands = []
            if len(cands_with_min_phases) > 1:
                max_min_hetero_len = np.max(min_phases_min_hetero_lens)
                for ind, c in enumerate(cands_with_min_phases):
                    if np.abs(min_phases_min_hetero_lens[ind] - max_min_hetero_len) < 0.0001:
                        second_reduced_cands.append(c)

                # check if this leads to a unique choice
                if len(second_reduced_cands) == 1:
                    simplex_to_return = second_reduced_cands[0][-1]

                else:
                    # if this still is not enough, we check for the highest average len
                    # in the hetero edges
                    averages = np.zeros(len(second_reduced_cands))
                    for c_ind, c in enumerate(second_reduced_cands):
                        counter = 0
                        for iw in range(len(c[-1].edge_classification)):
                            for jw in range(iw + 1, len(c[-1].edge_classification)):
                                if c[-1].edge_classification[iw][jw] == 1:
                                    counter = counter + 1
                                    averages[c_ind] = averages[c_ind] + self.discretized_system.euclidean_distance(
                                        c[-1].points_molar_fractions[iw], c[-1].points_molar_fractions[jw]
                                    )

                        averages[c_ind] = averages[c_ind] / counter

                    simplex_to_return = second_reduced_cands[np.argmax(averages)][-1]

            else:
                simplex_to_return = cands_with_min_phases[0][-1]

        return simplex_to_return

    def compute_delta_g_mix_solid(self, molar_fractions, temperature, melting_heats: np.array,
                                  melting_temperatures: np.array):
        """
        For solid phases, we compute as in Kim2011:
        R * T * -1 * sum_j x_j * ((Delta h_j^f / (R * T_j^f)) * ((T_j^f / T) - 1))

        Delta h_j^f is in J/mol.

        Note that in Reyes2001, Delta C_p is also used, we neglect that for simplicity.
        """
        delta_g_mix = 0
        for j in range(len(molar_fractions)):
            # in this case, it is ok to have a non-zero value at a pure component (molar fraction equal to 1).
            # this is since we normalize to liquid state (there, dgmix is 0 for pure components)
            if molar_fractions[j] > self.epsilon:
                delta_g_mix = delta_g_mix - (
                        molar_fractions[j] * ((melting_heats[j] / (8.314 * melting_temperatures[j])) * (
                            ((melting_temperatures[j] / temperature) - 1))))

        delta_g_mix = delta_g_mix * temperature * 8.314

        return delta_g_mix

    def compute_delta_g_mix_gas(self, molar_fractions, vapor_pressures: np.array, temperature, pressure):
        """
        For gaseous phases, we compute as in Olaya2010:
        Delta g_mix = R * T * sum_j (x_j * ln(x_j)) + (x_j * ln(p / p*_j)

        Note that similar as in liquid phase (and also stated by Olaya2010), this is the sum of the
        ideal contribution xln(x) and the nonideal contribution.
        """
        delta_g_mix = 0
        for j in range(len(molar_fractions)):
            # in this case, it is ok to have a non-zero value at a pure component (molar fraction equal to 1).
            # this is since we normalize to liquid state (there, dgmix is 0 for pure components)
            if molar_fractions[j] > self.epsilon:
                delta_g_mix = delta_g_mix + (
                        (molar_fractions[j] * np.log(molar_fractions[j])) + (
                         molar_fractions[j] * np.log(pressure / vapor_pressures[j]))
                )

        # in Olaya2010 this is not scaled, but also dgmix for liquid then hss to be provided unscaled
        # we just do it here, to have also the correct units
        delta_g_mix = delta_g_mix * temperature * 8.314

        return delta_g_mix

    def compute_delta_g_mix_liquid(self, molar_fractions, temperature):
        """
        For liquid phases, we compute as in Göttl2023 (see there for details, sources):
        Delta g_mix = R * T * sum_j x_j * ln(x_j * gamma_j), where R = 8.314 J/(K mol).
        """
        if self.gE_model is None:
            gE_model = self.hanna_model

        else:
            gE_model = self.gE_model

        delta_g_mix = 0
        for j in range(len(molar_fractions)):
            # exclude pure components and also prevent log(0)
            if 1 - self.epsilon > molar_fractions[j] > self.epsilon:
                activity_coefficient = gE_model.compute_activity_coefficient(molar_fractions, j, temperature)
                delta_g_mix = delta_g_mix + (molar_fractions[j] * np.log(molar_fractions[j] * activity_coefficient))

        delta_g_mix = delta_g_mix * temperature * 8.314

        return delta_g_mix

    @staticmethod
    def point_in_simplex_via_bary(simplex, point_cartesian):
        """
        For a simplex class and cartesian coordinates of a point.

        We just get the barycentric coordinates of the point (we assume
        that the required matrices exist already in the simplex).
        """
        epsilon = 0.0001
        bary_coords = np.matmul(simplex.matrix_inv, np.array([1] + list(point_cartesian)))

        # this should never happen
        if np.abs(sum(bary_coords) - 1) > epsilon:
            return None

        # check if all coords are in the interval [0, 1]
        if np.min(bary_coords) < -1 * epsilon or np.max(bary_coords) > 1 + epsilon:
            return False

        else:
            return True

    @staticmethod
    def check_if_list_consists_of_equal_elements(list_to_check: list):
        all_elements_equal = True
        for el_1 in list_to_check:
            for el_2 in list_to_check:
                if el_1 != el_2:
                    all_elements_equal = False
                    break

        return all_elements_equal
