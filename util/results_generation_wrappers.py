import os

from init_class import InitClass
from phase_eq_construction import PhaseEqAnalysis
from discretization.point_discretization import PointDisc
from hanna.hanna_wrapper import HANNAWrapper
from thermodynamic_models import AntoineModel
from util.plotter import Plotter


def generate_chen2000_LLEs(discretization_path: str):
    # chen2000

    # 3 comp systems
    names_list = [["n_hexane", "benzene", "sulfolane"], ["n_hexane", "toluene", "sulfolane"],
                  ["n_hexane", "xylene", "sulfolane"], ["n_octane", "benzene", "sulfolane"],
                  ["n_octane", "toluene", "sulfolane"], ["n_octane", "xylene", "sulfolane"]]
    temperature_kelvin = 298.15
    pressure = 1.013
    general_name = "chen2000"
    ge_model_name = "NRTL"
    experimental_data_given = True
    table_names = [
        "Table_1_1", "Table_1_2", "Table_1_3", "Table_2_1", "Table_2_2", "Table_2_3"
    ]

    for j in range(len(names_list)):
        InitClass(
            general_name=general_name,
            names_components=names_list[j],
            point_discretization_rec_steps=7,
            path_discretization=discretization_path,
            ge_model_name=ge_model_name,
            experimental_data_given=experimental_data_given,
            task="LLE",
            task_specific_information={
                'temperature_kelvin': temperature_kelvin,
                'pressure_bar': pressure,
            },
            table_name=table_names[j]
        )

    # 4 comp systems
    names_list = [["n_hexane", "n_octane", "benzene", "sulfolane"],
                  ["n_hexane", "benzene", "xylene", "sulfolane"],
                  ["n_octane", "toluene", "xylene", "sulfolane"]]
    temperature_kelvin = 298.15
    pressure = 1.013
    general_name = "chen2000"
    ge_model_name = "NRTL"
    experimental_data_given = True
    table_names = [
        "Table_5_1", "Table_5_2", "Table_5_3"
    ]

    for j in range(len(names_list)):
        InitClass(
            general_name=general_name,
            names_components=names_list[j],
            point_discretization_rec_steps=6,
            path_discretization=discretization_path,
            ge_model_name=ge_model_name,
            experimental_data_given=experimental_data_given,
            task="LLE",
            task_specific_information={
                'temperature_kelvin': temperature_kelvin,
                'pressure_bar': pressure,
            },
            table_name=table_names[j]
        )

    # 5 comp system
    names_list = ["n_hexane", "n_octane", "benzene", "toluene", "sulfolane"]
    temperature_kelvin = 298.15
    pressure = 1.013
    general_name = "chen2000"
    ge_model_name = "NRTL"
    experimental_data_given = True

    InitClass(
        general_name=general_name,
        names_components=names_list,
        point_discretization_rec_steps=5,
        path_discretization=discretization_path,
        ge_model_name=ge_model_name,
        experimental_data_given=experimental_data_given,
        task="LLE",
        task_specific_information={
            'temperature_kelvin': temperature_kelvin,
            'pressure_bar': pressure,
        },
        table_name="Table_7"
    )


def generate_chen2001_LLEs(discretization_path: str):
    # chen2001

    # 4 comp systems
    names_list = [
        ["hexane", "heptane", "toluene", "sulfolane"],
        ["heptane", "octane", "m_xylene", "sulfolane"],
        ["heptane", "benzene", "toluene", "sulfolane"]
    ]
    temperature_kelvin = 298.15
    pressure = 1.013
    general_name = "chen2001"
    ge_model_name = "NRTL"
    experimental_data_given = True
    table_names = ["Table_1_1", "Table_1_2", "Table_1_3"]

    for j in range(len(names_list)):
        InitClass(
            general_name=general_name,
            names_components=names_list[j],
            point_discretization_rec_steps=6,
            path_discretization=discretization_path,
            ge_model_name=ge_model_name,
            experimental_data_given=experimental_data_given,
            task="LLE",
            task_specific_information={
                'temperature_kelvin': temperature_kelvin,
                'pressure_bar': pressure,
            },
            table_name=table_names[j]
        )

    # 5 comp systems
    names_list = [
        ["hexane", "heptane", "toluene", "m_xylene", "sulfolane"],
        ["heptane", "octane", "benzene", "m_xylene", "sulfolane"]
    ]
    temperature_kelvin = 298.15
    pressure = 1.013
    general_name = "chen2001"
    ge_model_name = "NRTL"
    experimental_data_given = True
    table_names = ["Table_2_1", "Table_2_2"]

    for j in range(len(names_list)):
        InitClass(
            general_name=general_name,
            names_components=names_list[j],
            point_discretization_rec_steps=5,
            path_discretization=discretization_path,
            ge_model_name=ge_model_name,
            experimental_data_given=experimental_data_given,
            task="LLE",
            task_specific_information={
                'temperature_kelvin': temperature_kelvin,
                'pressure_bar': pressure,
            },
            table_name=table_names[j]
        )


def generate_yuan2019_LLEs(discretization_path: str):
    # yuan2019

    # 4 comp systems
    names_list = [
        ["ethanol", "octanol", "dodecane", "water"],
        ["ethanol", "octanol", "dodecane", "water"],
        ["ethanol", "octanol", "dodecane", "water"],
        ["ethanol", "octanol", "tridecane", "water"],
        ["ethanol", "octanol", "tridecane", "water"],
        ["ethanol", "octanol", "tridecane", "water"],
        ["ethanol", "nonanol", "dodecane", "water"],
        ["ethanol", "nonanol", "dodecane", "water"],
        ["ethanol", "nonanol", "dodecane", "water"],
        ["ethanol", "nonanol", "tridecane", "water"],
        ["ethanol", "nonanol", "tridecane", "water"],
        ["ethanol", "nonanol", "tridecane", "water"]
    ]
    temperature_kelvin = [
        293.15, 298.15, 303.15,
        293.15, 298.15, 303.15,
        293.15, 298.15, 303.15,
        293.15, 298.15, 303.15
    ]
    pressure = 1.013
    general_name = "yuan2019"
    ge_model_name = "NRTL"
    experimental_data_given = True
    table_names = [
        "Table_2_1", "Table_2_2", "Table_2_3",
        "Table_3_1", "Table_3_2", "Table_3_3",
        "Table_4_1", "Table_4_2", "Table_4_3",
        "Table_5_1", "Table_5_2", "Table_5_3"
    ]

    for j in range(len(names_list)):
        InitClass(
            general_name=general_name,
            names_components=names_list[j],
            point_discretization_rec_steps=6,
            path_discretization=discretization_path,
            ge_model_name=ge_model_name,
            experimental_data_given=experimental_data_given,
            task="LLE",
            task_specific_information={
                'temperature_kelvin': temperature_kelvin[j],
                'pressure_bar': pressure,
            },
            table_name=table_names[j]
        )


def generate_oh2003_VLEs(discretization_path: str):
    # oh2003

    # 2 comp systems
    names_list = [
        ["MTBE", "methanol"],
        ["MTBE", "benzene"],
        ["MTBE", "toluene"],
        ["methanol", "benzene"],
        ["methanol", "toluene"]
    ]
    temperature_kelvin = [
        313.15, 313.15, 313.15, 313.15, 313.15
    ]
    pressure_ranges_bar = [
        [0.3, 0.75], [0.2, 0.7], [0.05, 0.7], [0.2, 0.6], [0.05, 0.6]
    ]
    general_name = "oh2003"
    ge_model_name = "NRTL"
    experimental_data_given = True
    table_names = ["Table_2_1", "Table_2_2", "Table_2_3", "Table_2_4", "Table_2_5"]

    for rec_steps in [5, 8]:
        for j in range(len(names_list)):
            InitClass(
                general_name=general_name,
                names_components=names_list[j],
                point_discretization_rec_steps=rec_steps,
                path_discretization=discretization_path,
                ge_model_name=ge_model_name,
                experimental_data_given=experimental_data_given,
                task="VLE",
                task_specific_information={
                    'temperature_kelvin': temperature_kelvin[j],
                    'pressure_range': pressure_ranges_bar[j]
                },
                table_name=table_names[j]
            )

    # 3 comp systems
    names_list = [
        ["MTBE", "methanol", "benzene"],
        ["MTBE", "methanol", "toluene"]
    ]
    temperature_kelvin = [
        313.5, 313.5
    ]
    pressure_ranges_bar = [
        [0.3, 0.75], [0.2, 0.75]
    ]
    general_name = "oh2003"
    ge_model_name = "NRTL"
    experimental_data_given = True
    table_names = ["Table_4", "Table_5"]

    for j in range(len(names_list)):
        InitClass(
            general_name=general_name,
            names_components=names_list[j],
            point_discretization_rec_steps=7,
            path_discretization=discretization_path,
            ge_model_name=ge_model_name,
            experimental_data_given=experimental_data_given,
            task="VLE",
            task_specific_information={
                'temperature_kelvin': temperature_kelvin[j],
                'pressure_range': pressure_ranges_bar[j]
            },
            table_name=table_names[j]
        )


def generate_xiao2013_VLEs(discretization_path: str):
    # 2 comp systems
    names_list = [
        ["methyl-acetate", "isopropyl-acetate"]
    ]
    pressure_bar = [1.013]
    temperature_range = [
        [320., 370.]
    ]
    general_name = "xiao2013"
    ge_model_name = "NRTL"
    experimental_data_given = True
    table_names = ["Table_3"]

    for rec_steps in [5, 8]:
        for j in range(len(names_list)):
            InitClass(
                general_name=general_name,
                names_components=names_list[j],
                point_discretization_rec_steps=rec_steps,
                path_discretization=discretization_path,
                ge_model_name=ge_model_name,
                experimental_data_given=experimental_data_given,
                task="VLE",
                task_specific_information={
                    'temperature_range': temperature_range[j],
                    'pressure_bar': pressure_bar[j]
                },
                table_name=table_names[j]
            )

    # 4 comp systems
    names_list = [
        ["methyl-acetate", "methanol", "isopropanol", "isopropyl-acetate"]
    ]
    pressure_bar = [1.013]
    temperature_range = [
        [320., 360.]
    ]
    general_name = "xiao2013"
    ge_model_name = "NRTL"
    experimental_data_given = True
    table_names = ["Table_4"]

    for j in range(len(names_list)):
        InitClass(
            general_name=general_name,
            names_components=names_list[j],
            point_discretization_rec_steps=6,
            path_discretization=discretization_path,
            ge_model_name=ge_model_name,
            experimental_data_given=experimental_data_given,
            task="VLE",
            task_specific_information={
                'temperature_range': temperature_range[j],
                'pressure_bar': pressure_bar[j]
            },
            table_name=table_names[j]
        )


def generate_kim2011_SLE(discretization_path: str):
    # 2 comp system
    names_list = ["chloroform", "acetylacetone"]
    pressure_bar = 1.0133
    temperature_range = [180., 260.]
    general_name = "kim2011"
    ge_model_name = "NRTL"
    experimental_data_given = True
    table_name = "Table_6"

    # only 1 system
    for rec_steps in [5, 8]:
        InitClass(
            general_name=general_name,
            names_components=names_list,
            point_discretization_rec_steps=rec_steps,
            path_discretization=discretization_path,
            ge_model_name=ge_model_name,
            experimental_data_given=experimental_data_given,
            task="SLE",
            task_specific_information={
                'temperature_range': temperature_range,
                'pressure_bar': pressure_bar
            },
            table_name=table_name
        )


def generate_gao2018_SLEs(discretization_path: str):
    # 2 comp systems
    names_list = [["2_naphthaldehyde", "ethyl_acetate"],
                  ["4_methylphthalic_anhydride", "ethyl_acetate"]]
    pressure_bar = 1.0133
    temperature_range = [[283.15, 313.15], [283.15, 313.15]]
    general_name = "gao2018"
    ge_model_name = "NRTL"
    experimental_data_given = True
    table_names = ["Table_3_1", "Table_3_2"]

    for rec_steps in [5, 8]:
        for i in range(len(table_names)):
            InitClass(
                general_name=general_name,
                names_components=names_list[i],
                point_discretization_rec_steps=rec_steps,
                path_discretization=discretization_path,
                ge_model_name=ge_model_name,
                experimental_data_given=experimental_data_given,
                task="SLE",
                task_specific_information={
                    'temperature_range': temperature_range[i],
                    'pressure_bar': pressure_bar
                },
                table_name=table_names[i]
            )

    # ternary systems
    names_list = [["2_naphthaldehyde", "4_methylphthalic_anhydride", "ethyl_acetate"],
                  ["2_naphthaldehyde", "4_methylphthalic_anhydride", "ethyl_acetate"],
                  ["2_naphthaldehyde", "4_methylphthalic_anhydride", "ethyl_acetate"]]
    pressure_bar = 1.0133
    temperature = [288.15, 298.15, 308.15]
    general_name = "gao2018"
    ge_model_name = "NRTL"
    experimental_data_given = False

    for i in range(len(names_list)):
        InitClass(
            general_name=general_name,
            names_components=names_list[i],
            point_discretization_rec_steps=7,
            path_discretization=discretization_path,
            ge_model_name=ge_model_name,
            experimental_data_given=experimental_data_given,
            task="SLE",
            task_specific_information={
                'temperature_kelvin': temperature[i],
                'pressure_bar': pressure_bar
            },
            table_name=None
        )


def generate_li2016_SLEs(discretization_path: str):
    # ternary systems
    names_list = [
        ["3_4_dichloronitrobenzene", "2_3_dichloronitrobenzene", "ethanol"],
        ["3_4_dichloronitrobenzene", "2_3_dichloronitrobenzene", "ethanol"],
        ["3_4_dichloronitrobenzene", "2_3_dichloronitrobenzene", "ethanol"],
        ["3_4_dichloronitrobenzene", "2_3_dichloronitrobenzene", "n_propanol"],
        ["3_4_dichloronitrobenzene", "2_3_dichloronitrobenzene", "n_propanol"],
        ["3_4_dichloronitrobenzene", "2_3_dichloronitrobenzene", "n_propanol"]
    ]
    pressure_bar = 1.0133
    temperature = [283.15, 293.15, 303.15,
                   283.15, 293.15, 303.15]
    general_name = "li2016"
    ge_model_name = "NRTL"
    experimental_data_given = True
    table_names = ["Table_2_1", "Table_2_2", "Table_2_3",
                   "Table_3_1", "Table_3_2", "Table_3_3"]

    for i in range(len(names_list)):
        InitClass(
            general_name=general_name,
            names_components=names_list[i],
            point_discretization_rec_steps=7,
            path_discretization=discretization_path,
            ge_model_name=ge_model_name,
            experimental_data_given=experimental_data_given,
            task="SLE",
            task_specific_information={
                'temperature_kelvin': temperature[i],
                'pressure_bar': pressure_bar
            },
            table_name=table_names[i]
        )


def generate_li2017_SLEs(discretization_path: str):
    # 4 comp systems
    names_lists = [
        ["adipic_acid", "glutaric_acid", "succinic_acid", "ethanol"],
        ["adipic_acid", "glutaric_acid", "succinic_acid", "ethanol"],
        ["adipic_acid", "glutaric_acid", "succinic_acid", "ethanol"]
    ]
    pressure_bar = 1.012
    temperatures_kelvin = [283.15, 303.15, 313.15]
    general_name = "li2017"
    ge_model_name = "NRTL"
    experimental_data_given = True
    table_names = ["Table_2", "Table_3", "Table_4"]

    for i in range(len(names_lists)):
        InitClass(
            general_name=general_name,
            names_components=names_lists[i],
            point_discretization_rec_steps=6,
            path_discretization=discretization_path,
            ge_model_name=ge_model_name,
            experimental_data_given=experimental_data_given,
            task="SLE",
            task_specific_information={
                'temperature_kelvin': temperatures_kelvin[i],
                'pressure_bar': pressure_bar
            },
            table_name=table_names[i]
        )


def HANNA_example(discretization_path: str):
    # load hanna
    hanna_wrapper = HANNAWrapper()

    # 2 comp system
    names_list = ["butanol", "water"]
    pressure_bar = 1.012
    temperature_range_kelvin = [260., 400.]

    antoine_models = {
        "butanol": AntoineModel(
            data_dict={
                "A": 4.54607,
                "B": 1351.555,
                "C": -93.34,
                "Tmin": temperature_range_kelvin[0] - 1.,
                "Tmax": temperature_range_kelvin[1] + 1.,
            },
            option_dict={
                'antoine_option': 'nist'
            }
        ),
        "water": AntoineModel(
            data_dict={
                "A": 4.6543,
                "B": 1435.264,
                "C": -64.848,
                "Tmin": temperature_range_kelvin[0] - 1.,
                "Tmax": temperature_range_kelvin[1] + 1.
            },
            option_dict={
                'antoine_option': 'nist'
            }
        )
    }
    melting_temps = {
        "water": 273.15,  # nist or any other source
        "butanol": 184.5  # nist
    }
    melting_enth = {
        "water": 6020.,  # chemteam, heat of fusion
        "butanol": 9372.  # nist
    }

    # init plotter, makes only sense for N<=4 and will usually only be done for N<=3
    plotter_instance = Plotter(num_comp=2)

    # load point discretization
    for rec_steps in [5, 8]:
        point_disc = PointDisc(
            num_comp=len(names_list), recursion_steps=rec_steps, load=True, store=False,
            path=discretization_path
        )

        # define path for that general_name
        general_path = os.path.join(os.getcwd(), "phase_equilibria_results", "HANNA_example")
        if not os.path.isdir(general_path):
            os.makedirs(general_path, exist_ok=True)

        general_path = os.path.join(general_path, "PhaseEq")
        if not os.path.isdir(general_path):
            os.makedirs(general_path, exist_ok=True)

        dir_name = (
                "Txy_" +
                "butanol_water" + "_T1_" +
                str(round(temperature_range_kelvin[0], 4)) + "_T2_" +
                str(round(temperature_range_kelvin[1], 4)) + "_p_" +
                str(round(pressure_bar, 4))
        )
        vle_path = os.path.join(general_path, dir_name)
        if not os.path.isdir(vle_path):
            os.mkdir(vle_path)

        temps = [260., 265., 270., 275., 280., 285., 290., 295., 300., 305., 310., 315., 320.,
                 325., 330., 335., 340., 345., 350., 355., 360., 361., 362., 363., 364., 365., 366.,
                 366.2, 366.4, 366.6, 366.8, 367., 368., 369., 370., 371.,
                 372., 373., 374., 375., 376., 377., 378., 380., 385., 390., 395., 400.]
        list_with_Txy_info = []
        for temperature_K in temps:
            print('Doing T:', temperature_K)
            phase_eq_analysis = PhaseEqAnalysis(
                discretized_system=point_disc,
                temperature_K=temperature_K,
                pressure_bar=pressure_bar,
                num_components=len(names_list),
                gE_model=None,
                antoine_models=antoine_models,
                names_comps_ordered=names_list,
                melting_temperatures=melting_temps,
                melting_heats=melting_enth,
                hanna_model=hanna_wrapper
            )
            phase_split_simplices = phase_eq_analysis.construct_phase_equilibrium()
            loading_path_phase_eq = phase_eq_analysis.store_phase_eq_liquid(
                names_components_combined="butanol_water", phase_split_simplices=phase_split_simplices,
                standard_path=vle_path
            )

            # collect pxy information
            info_dict = {
                't_K': temperature_K,
                'loading_path': loading_path_phase_eq,
                'list_phase_eq_simplices': phase_split_simplices
            }
            list_with_Txy_info.append(info_dict)

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

        plotter_instance.plot_txy(
            list_with_txy_info=list_with_Txy_info,
            t_range=[temps[0] - 5, temps[-1] + 5],
            name_comp1=names_list[0],
            plot_lines=True,
            location='lower right'
        )

        # save plot
        plotter_instance.save_plot(path=os.path.join(vle_path, "Txy_" + str(rec_steps) + "_plot.png"))
