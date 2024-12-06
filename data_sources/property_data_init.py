import numpy as np

from data_sources.property_data_classes import GEDataset, GEDatasets, ExperimentalDataset


class InitializedGeData:
    def __init__(self):
        self.ge_models_collection = GEDatasets()

        # Chen2000
        chen2000_ge_model_data = GEDataset()
        chen2000_ge_model_data.add_pair("n_hexane", "benzene",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([1.523, 90.89])
                                        })
        chen2000_ge_model_data.add_pair("n_hexane", "sulfolane",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([2045, 870.6])
                                        })
        chen2000_ge_model_data.add_pair("benzene", "sulfolane",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([403.2, -103.8])
                                        })
        chen2000_ge_model_data.add_pair("n_hexane", "toluene",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([347, -267.6])
                                        })
        chen2000_ge_model_data.add_pair("toluene", "sulfolane",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([562.3, -53.59])
                                        })
        chen2000_ge_model_data.add_pair("n_hexane", "xylene",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([115.5, -98.56])
                                        })
        chen2000_ge_model_data.add_pair("xylene", "sulfolane",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([413.7, 145.1])
                                        })
        chen2000_ge_model_data.add_pair("n_octane", "benzene",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([-163.5, 283.7])
                                        })
        chen2000_ge_model_data.add_pair("n_octane", "sulfolane",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([2235, 1231])
                                        })
        chen2000_ge_model_data.add_pair("n_octane", "toluene",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([115.5, -84.22])
                                        })
        chen2000_ge_model_data.add_pair("n_octane", "xylene",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([-331.8, 364.5])
                                        })
        chen2000_ge_model_data.add_pair("n_octane", "n_hexane",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.zeros(2)
                                        })
        chen2000_ge_model_data.add_pair("benzene", "toluene",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.zeros(2)
                                        })
        chen2000_ge_model_data.add_pair("benzene", "xylene",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.zeros(2)
                                        })
        chen2000_ge_model_data.add_pair("toluene", "xylene",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.zeros(2)
                                        })

        self.ge_models_collection.add_model(name="chen2000", dataset=chen2000_ge_model_data)

        # Chen2001
        chen2001_ge_model_data = GEDataset()
        chen2001_ge_model_data.add_pair("hexane", "heptane",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([0., 0.])
                                        })
        chen2001_ge_model_data.add_pair("hexane", "octane",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([0., 0.])
                                        })
        chen2001_ge_model_data.add_pair("hexane", "benzene",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([1.523, 90.89])
                                        })
        chen2001_ge_model_data.add_pair("hexane", "toluene",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([347, -267.6])
                                        })
        chen2001_ge_model_data.add_pair("hexane", "m_xylene",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([115.5, -98.56])
                                        })
        chen2001_ge_model_data.add_pair("hexane", "sulfolane",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([2045, 870.6])
                                        })
        chen2001_ge_model_data.add_pair("heptane", "octane",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([0., 0.])
                                        })
        chen2001_ge_model_data.add_pair("heptane", "benzene",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([-18.13, 98.93])
                                        })
        chen2001_ge_model_data.add_pair("heptane", "toluene",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([359.8, -279.8])
                                        })
        chen2001_ge_model_data.add_pair("heptane", "m_xylene",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([139.8, -58.49])
                                        })
        chen2001_ge_model_data.add_pair("heptane", "sulfolane",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([1960, 870.6])
                                        })
        chen2001_ge_model_data.add_pair("octane", "benzene",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([-163.5, 283.7])
                                        })
        chen2001_ge_model_data.add_pair("octane", "toluene",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([115.5, -84.22])
                                        })
        chen2001_ge_model_data.add_pair("octane", "m_xylene",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([-331.8, 364.5])
                                        })
        chen2001_ge_model_data.add_pair("octane", "sulfolane",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([2235, 1231])
                                        })
        chen2001_ge_model_data.add_pair("benzene", "toluene",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([0., 0.])
                                        })
        chen2001_ge_model_data.add_pair("benzene", "m_xylene",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([0., 0.])
                                        })
        chen2001_ge_model_data.add_pair("benzene", "sulfolane",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([403.2, -103.8])
                                        })
        chen2001_ge_model_data.add_pair("toluene", "m_xylene",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([0., 0.])
                                        })
        chen2001_ge_model_data.add_pair("toluene", "sulfolane",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([562.3, -53.59])
                                        })
        chen2001_ge_model_data.add_pair("m_xylene", "sulfolane",
                                        dict_binary_interactions={
                                            "alphas": np.array([0.2, 0.2]),
                                            "tau_without_temperature_scaling": np.array([413.7, 145.1])
                                        })

        self.ge_models_collection.add_model(name="chen2001", dataset=chen2001_ge_model_data)

        # Yuan2019
        yuan2019_ge_model_data = GEDataset()
        yuan2019_ge_model_data.add_pair("octanol", "nonanol",
                                        dict_binary_interactions={
                                            "a": np.array([5.46, -0.66]),
                                            "b": np.array([-1329.55, -158.88]),
                                            "alphas": np.array([0.3, 0.3])
                                        })
        yuan2019_ge_model_data.add_pair("octanol", "water",
                                        dict_binary_interactions={
                                            "a": np.array([-1.45, 5.92]),
                                            "b": np.array([-3011.98, -1581.72]),
                                            "alphas": np.array([0.26, 0.26])
                                        })
        yuan2019_ge_model_data.add_pair("nonanol", "water",
                                        dict_binary_interactions={
                                            "a": np.array([-0.27, 10.30]),
                                            "b": np.array([-2326.25, -1670.40]),
                                            "alphas": np.array([0.2, 0.2])
                                        })
        yuan2019_ge_model_data.add_pair("dodecane", "water",
                                        dict_binary_interactions={
                                            "a": np.array([-6.09, 23.43]),
                                            "b": np.array([-248.17, -4478.56]),
                                            "alphas": np.array([0.2, 0.2])
                                        })
        yuan2019_ge_model_data.add_pair("tridecane", "water",
                                        dict_binary_interactions={
                                            "a": np.array([-6.91, 24.64]),
                                            "b": np.array([346.42, -4906.39]),
                                            "alphas": np.array([0.2, 0.2])
                                        })
        yuan2019_ge_model_data.add_pair("ethanol", "water",
                                        dict_binary_interactions={
                                            "a": np.array([-0.80, 3.46]),
                                            "b": np.array([-3837.76, -884.89]),
                                            "alphas": np.array([0.3, 0.3])
                                        })
        yuan2019_ge_model_data.add_pair("octanol", "dodecane",
                                        dict_binary_interactions={
                                            "a": np.array([0., 0.]),
                                            "b": np.array([-306.76, -428.98]),
                                            "alphas": np.array([0.3, 0.3])
                                        })
        yuan2019_ge_model_data.add_pair("octanol", "ethanol",
                                        dict_binary_interactions={
                                            "a": np.array([0., 0.]),
                                            "b": np.array([-804.11, -342.30]),
                                            "alphas": np.array([0.3, 0.3])
                                        })
        yuan2019_ge_model_data.add_pair("dodecane", "ethanol",
                                        dict_binary_interactions={
                                            "a": np.array([0., 0.]),
                                            "b": np.array([439.59, 1047.59]),
                                            "alphas": np.array([0.3, 0.3])
                                        })
        yuan2019_ge_model_data.add_pair("octanol", "tridecane",
                                        dict_binary_interactions={
                                            "a": np.array([0., 0.]),
                                            "b": np.array([-202.10, -614.853]),
                                            "alphas": np.array([0.3, 0.3])
                                        })
        yuan2019_ge_model_data.add_pair("tridecane", "ethanol",
                                        dict_binary_interactions={
                                            "a": np.array([0., 0.]),
                                            "b": np.array([478.69, 1133.72]),
                                            "alphas": np.array([0.3, 0.3])
                                        })
        yuan2019_ge_model_data.add_pair("nonanol", "dodecane",
                                        dict_binary_interactions={
                                            "a": np.array([0., 0.]),
                                            "b": np.array([-444.49, -408.667]),
                                            "alphas": np.array([0.3, 0.3])
                                        })
        yuan2019_ge_model_data.add_pair("nonanol", "ethanol",
                                        dict_binary_interactions={
                                            "a": np.array([0., 0.]),
                                            "b": np.array([-685.72, -707.19]),
                                            "alphas": np.array([0.3, 0.3])
                                        })
        yuan2019_ge_model_data.add_pair("nonanol", "tridecane",
                                        dict_binary_interactions={
                                            "a": np.array([0., 0.]),
                                            "b": np.array([-410.03, -561.45]),
                                            "alphas": np.array([0.3, 0.3])
                                        })
        yuan2019_ge_model_data.add_pair("dodecane", "tridecane",
                                        dict_binary_interactions={
                                            "a": np.array([0., 0.]),
                                            "b": np.array([-48.69, 50.33]),
                                            "alphas": np.array([0.3, 0.3])
                                        })

        self.ge_models_collection.add_model(name="yuan2019", dataset=yuan2019_ge_model_data)

        # Oh2003
        oh2003_ge_model_data = GEDataset()
        # below are the parameter from the paper
        oh2003_ge_model_data.add_pair(
            "MTBE", "methanol",
            dict_binary_interactions={
                "alphas": np.array([0.3, 0.3]),
                "A": np.array([1799.0230, 1218.5638])
            })
        oh2003_ge_model_data.add_pair(
            "MTBE", "benzene",
            dict_binary_interactions={
                "alphas": np.array([0.3, 0.3]),
                "A": np.array([-1605.0272, 2572.2002])
            })
        oh2003_ge_model_data.add_pair(
            "MTBE", "toluene",
            dict_binary_interactions={
                "alphas": np.array([0.3, 0.3]),
                "A": np.array([4959.0840, -2382.6299])
            })
        oh2003_ge_model_data.add_pair(
            "methanol", "benzene",
            dict_binary_interactions={
                "alphas": np.array([0.5152, 0.5152]),
                "A": np.array([2532.0484, 5326.2385])
            })
        oh2003_ge_model_data.add_pair(
            "methanol", "toluene",
            dict_binary_interactions={
                "alphas": np.array([0.4749, 0.4749]),
                "A": np.array([3380.8926, 4666.5010])
            })

        # outcommented below with parameters from 50 50 opt followed by Nelder-Mead, seed 42
        '''oh2003_ge_model_data.add_pair(
            "MTBE", "methanol",
            dict_binary_interactions={
                "alphas": np.array([1.53892093e-01, 1.53892093e-01]),
                "A": np.array([1.95602456e+03, 9.17845815e+02])
            })
        oh2003_ge_model_data.add_pair(
            "MTBE", "benzene",
            dict_binary_interactions={
                "alphas": np.array([1.42275721e-01, 1.42275721e-01]),
                "A": np.array([-5.38025596e+03, 8.24213600e+03])
            })
        oh2003_ge_model_data.add_pair(
            "MTBE", "toluene",
            dict_binary_interactions={
                "alphas": np.array([8.04997536e-01, 8.04997536e-01]),
                "A": np.array([2.73340289e+03, -7.16391438e+02])
            })
        oh2003_ge_model_data.add_pair(
            "methanol", "benzene",
            dict_binary_interactions={
                "alphas": np.array([3.26962134e-01, 3.26962134e-01]),
                "A": np.array([1.67756524e+03, 4.28549569e+03])
            })
        oh2003_ge_model_data.add_pair(
            "methanol", "toluene",
            dict_binary_interactions={
                "alphas": np.array([4.86146280e-01, 4.86146280e-01]),
                "A": np.array([3.34868107e+03, 6.34177686e+03])
            })'''

        self.ge_models_collection.add_model(name="oh2003", dataset=oh2003_ge_model_data)

        # Xiao2013
        xiao2013_ge_model_data = GEDataset()
        xiao2013_ge_model_data.add_pair(
            "methyl-acetate", "isopropyl-acetate",
            dict_binary_interactions={
                "alphas": np.array([0.3, 0.3]),
                "A": np.array([1616.8, -1192.3])
            })
        xiao2013_ge_model_data.add_pair(
            name_1="methanol", name_2="isopropanol",
            dict_binary_interactions={
                "alphas": np.array([0.3, 0.3]),
                "A": np.array([1507.9, -1706.1])
            })
        xiao2013_ge_model_data.add_pair(
            name_1="methyl-acetate", name_2="isopropanol",
            dict_binary_interactions={
                "alphas": np.array([0.3, 0.3]),
                "A": np.array([194.9, 1926.4])
            })
        xiao2013_ge_model_data.add_pair(
            name_1="methyl-acetate", name_2="methanol",
            dict_binary_interactions={
                "alphas": np.array([0.3, 0.3]),
                "A": np.array([1466.6, 1617.8])
            })
        xiao2013_ge_model_data.add_pair(
            name_1="methanol", name_2="isopropyl-acetate",
            dict_binary_interactions={
                "alphas": np.array([1.03, 1.03]),
                "A": np.array([2525.75, 1767.46])
            })
        xiao2013_ge_model_data.add_pair(
            name_1="isopropanol", name_2="isopropyl-acetate",
            dict_binary_interactions={
                "alphas": np.array([0.2, 0.2]),
                "A": np.array([1160.468, 632.2797])
            })

        self.ge_models_collection.add_model(name="xiao2013", dataset=xiao2013_ge_model_data)

        # Kim2011
        kim2011_ge_model_data = GEDataset()
        kim2011_ge_model_data.add_pair(
            name_1="chloroform", name_2="acetylacetone",
            dict_binary_interactions={
                "alphas": np.array([0.31, 0.31]),
                "A": np.array([288.02, -2300.25])
            })

        self.ge_models_collection.add_model(name="kim2011", dataset=kim2011_ge_model_data)

        # Gao2018
        gao2018_ge_model_data = GEDataset()
        gao2018_ge_model_data.add_pair(
            name_1="2_naphthaldehyde", name_2="ethyl_acetate",
            dict_binary_interactions={
                "a": np.array([4.328, -3.107]),
                "b": np.array([-1151, 1196]),
                "alphas": np.array([0.3, 0.3])
            })
        gao2018_ge_model_data.add_pair(
            name_1="4_methylphthalic_anhydride", name_2="ethyl_acetate",
            dict_binary_interactions={
                "a": np.array([6.685, -2.832]),
                "b": np.array([-1152, 593.738]),
                "alphas": np.array([0.3, 0.3])
            })
        gao2018_ge_model_data.add_pair(
            name_1="2_naphthaldehyde", name_2="4_methylphthalic_anhydride",
            dict_binary_interactions={
                "a": np.array([40.571, -6.581]),
                "b": np.array([-11050, 1722]),
                "alphas": np.array([0.3, 0.3])
            })

        self.ge_models_collection.add_model(name="gao2018", dataset=gao2018_ge_model_data)

        # Li2016
        li2016_ge_model_data = GEDataset()
        li2016_ge_model_data.add_pair(
            name_1="3_4_dichloronitrobenzene", name_2="2_3_dichloronitrobenzene",
            dict_binary_interactions={
                "a": np.array([-4.849, -7.518]),
                "b": np.array([1985, 2331]),
                "alphas": np.array([0.3, 0.3])
            })
        li2016_ge_model_data.add_pair(
            name_1="3_4_dichloronitrobenzene", name_2="ethanol",
            dict_binary_interactions={
                "a": np.array([-99.582, -19.644]),
                "b": np.array([31960, 6339]),
                "alphas": np.array([0.3, 0.3])
            })
        li2016_ge_model_data.add_pair(
            name_1="2_3_dichloronitrobenzene", name_2="ethanol",
            dict_binary_interactions={
                "a": np.array([-44.829, -3.706]),
                "b": np.array([16950, 1791]),
                "alphas": np.array([0.3, 0.3])
            })
        li2016_ge_model_data.add_pair(
            name_1="3_4_dichloronitrobenzene", name_2="n_propanol",
            dict_binary_interactions={
                "a": np.array([-22.842, 4.643]),
                "b": np.array([7316, -995.741]),
                "alphas": np.array([0.3, 0.3])
            })
        li2016_ge_model_data.add_pair(
            name_1="2_3_dichloronitrobenzene", name_2="n_propanol",
            dict_binary_interactions={
                "a": np.array([2.299, 2.169]),
                "b": np.array([-624.969, 37.681]),
                "alphas": np.array([0.3, 0.3])
            })

        self.ge_models_collection.add_model(name="li2016", dataset=li2016_ge_model_data)

        # Li2017
        li2017_ge_model_data = GEDataset()
        li2017_ge_model_data.add_pair(
            name_1="adipic_acid", name_2="glutaric_acid",
            dict_binary_interactions={
                "a": np.array([17.58, -79.74]),
                "b": np.array([-2896, 24460]),
                "alphas": np.array([0.2, 0.2])
            })
        li2017_ge_model_data.add_pair(
            name_1="adipic_acid", name_2="ethanol",
            dict_binary_interactions={
                "a": np.array([1.49, 0.97]),
                "b": np.array([40.19, -959.25]),
                "alphas": np.array([0.2, 0.2])
            })
        li2017_ge_model_data.add_pair(
            name_1="glutaric_acid", name_2="ethanol",
            dict_binary_interactions={
                "a": np.array([-2.64, 10.67]),
                "b": np.array([401.35, -2279]),
                "alphas": np.array([0.3, 0.3])
            })
        li2017_ge_model_data.add_pair(
            name_1="glutaric_acid", name_2="succinic_acid",
            dict_binary_interactions={
                "a": np.array([1.68 * 1000000, 33.27]),
                "b": np.array([4.14 * 10000000, -6.28 * 1000]),
                "alphas": np.array([0.2, 0.2])
            })
        li2017_ge_model_data.add_pair(
            name_1="succinic_acid", name_2="ethanol",
            dict_binary_interactions={
                "a": np.array([44.97, 1.49]),
                "b": np.array([93300, -873.31]),
                "alphas": np.array([0.2, 0.2])
            })
        li2017_ge_model_data.add_pair(
            name_1="adipic_acid", name_2="succinic_acid",
            dict_binary_interactions={
                "a": np.array([7.24, 7.89]),
                "b": np.array([-1106, 2265]),
                "alphas": np.array([0.2, 0.2])
            })

        self.ge_models_collection.add_model(name="li2017", dataset=li2017_ge_model_data)


class InitializedExperimentalData:
    def __init__(self):
        self.experimental_datasets = {}

        # chen2000
        self.experimental_datasets['chen2000'] = ExperimentalDataset(
            name_excel_file='chen2000',
            sheet_names=["Table_1_1", "Table_1_2", "Table_1_3", "Table_2_1", "Table_2_2", "Table_2_3",
                         "Table_5_1", "Table_5_2", "Table_5_3", "Table_7"],
            list_temperature=[298.15] * 10,
            list_pressure=[1.013] * 10,
            molar_fractions_given=True,
            feed_given=False,
            ordered_component_names_per_sheet=[
                ["n_hexane", "benzene", "sulfolane"], ["n_hexane", "toluene", "sulfolane"],
                ["n_hexane", "xylene", "sulfolane"], ["n_octane", "benzene", "sulfolane"],
                ["n_octane", "toluene", "sulfolane"], ["n_octane", "xylene", "sulfolane"],
                ["n_hexane", "n_octane", "benzene", "sulfolane"],
                ["n_hexane", "benzene", "xylene", "sulfolane"],
                ["n_octane", "toluene", "xylene", "sulfolane"],
                ["n_hexane", "n_octane", "benzene", "toluene", "sulfolane"]],
            list_molar_masses=[None] * 10
        )

        # chen2001
        self.experimental_datasets['chen2001'] = ExperimentalDataset(
            name_excel_file='chen2001',
            sheet_names=["Table_1_1", "Table_1_2", "Table_1_3", "Table_2_1", "Table_2_2"],
            list_temperature=[298.15] * 5,
            list_pressure=[1.013] * 5,
            molar_fractions_given=True,
            feed_given=False,
            ordered_component_names_per_sheet=[
                ["hexane", "heptane", "toluene", "sulfolane"],
                ["heptane", "octane", "m_xylene", "sulfolane"],
                ["heptane", "benzene", "toluene", "sulfolane"],
                ["hexane", "heptane", "toluene", "m_xylene", "sulfolane"],
                ["heptane", "octane", "benzene", "m_xylene", "sulfolane"]
            ],
            list_molar_masses=[None] * 5
        )

        # yuan2019
        self.experimental_datasets['yuan2019'] = ExperimentalDataset(
            name_excel_file='yuan2019',
            sheet_names=[
                "Table_2_1", "Table_2_2", "Table_2_3",
                "Table_3_1", "Table_3_2", "Table_3_3",
                "Table_4_1", "Table_4_2", "Table_4_3",
                "Table_5_1", "Table_5_2", "Table_5_3",
                "Table_7_1", "Table_7_2", "Table_7_3"
            ],
            list_temperature=[
                293.15, 298.15, 303.15,
                293.15, 298.15, 303.15,
                293.15, 298.15, 303.15,
                293.15, 298.15, 303.15,
                293.15, 298.15, 303.15
            ],
            list_pressure=[1.013] * 15,
            molar_fractions_given=False,
            feed_given=False,
            ordered_component_names_per_sheet=[
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
                ["ethanol", "nonanol", "tridecane", "water"],
                ["ethanol", "octanol", "nonanol", "dodecane", "tridecane", "water"],
                ["ethanol", "octanol", "nonanol", "dodecane", "tridecane", "water"],
                ["ethanol", "octanol", "nonanol", "dodecane", "tridecane", "water"]
            ],
            list_molar_masses=[
                np.array([46.07, 130.23, 170.33, 18.01528]),
                np.array([46.07, 130.23, 170.33, 18.01528]),
                np.array([46.07, 130.23, 170.33, 18.01528]),
                np.array([46.07, 130.23, 184.37, 18.01528]),
                np.array([46.07, 130.23, 184.37, 18.01528]),
                np.array([46.07, 130.23, 184.37, 18.01528]),
                np.array([46.07, 144.26, 170.33, 18.01528]),
                np.array([46.07, 144.26, 170.33, 18.01528]),
                np.array([46.07, 144.26, 170.33, 18.01528]),
                np.array([46.07, 144.26, 184.37, 18.01528]),
                np.array([46.07, 144.26, 184.37, 18.01528]),
                np.array([46.07, 144.26, 184.37, 18.01528]),
                np.array([46.07, 130.23, 144.26, 170.33, 184.37, 18.01528]),
                np.array([46.07, 130.23, 144.26, 170.33, 184.37, 18.01528]),
                np.array([46.07, 130.23, 144.26, 170.33, 184.37, 18.01528])
            ]
        )

        # oh2003
        self.experimental_datasets['oh2003'] = ExperimentalDataset(
            name_excel_file='oh2003',
            sheet_names=["Table_2_1", "Table_2_2", "Table_2_3", "Table_2_4", "Table_2_5", "Table_4", "Table_5"],
            list_temperature=[None] * 7,
            list_pressure=[None] * 7,
            molar_fractions_given=True,
            feed_given=False,
            ordered_component_names_per_sheet=[
                ["MTBE", "methanol"],
                ["MTBE", "benzene"],
                ["MTBE", "toluene"],
                ["methanol", "benzene"],
                ["methanol", "toluene"],
                ["MTBE", "methanol", "benzene"],
                ["MTBE", "methanol", "toluene"]
            ],
            list_molar_masses=[None] * 7
        )

        # xiao2013
        self.experimental_datasets['xiao2013'] = ExperimentalDataset(
            name_excel_file='xiao2013',
            sheet_names=["Table_3", "Table_4"],
            list_temperature=[None] * 2,
            list_pressure=[None] * 2,
            molar_fractions_given=True,
            feed_given=False,
            ordered_component_names_per_sheet=[
                ["methyl-acetate", "isopropyl-acetate"],
                ["methyl-acetate", "methanol", "isopropanol", "isopropyl-acetate"]
            ],
            list_molar_masses=[None] * 2
        )

        # kim2011
        self.experimental_datasets['kim2011'] = ExperimentalDataset(
            name_excel_file='kim2011',
            sheet_names=["Table_6"],
            list_temperature=[None],
            list_pressure=[None],
            molar_fractions_given=True,
            feed_given=False,
            ordered_component_names_per_sheet=[
                ["chloroform", "acetylacetone"]
            ],
            list_molar_masses=[None]
        )

        # gao2018
        self.experimental_datasets['gao2018'] = ExperimentalDataset(
            name_excel_file='gao2018',
            sheet_names=["Table_3_1", "Table_3_2"],
            list_temperature=[None, None],
            list_pressure=[None, None],
            molar_fractions_given=True,
            feed_given=False,
            ordered_component_names_per_sheet=[
                ["2_naphthaldehyde", "ethyl_acetate"],
                ["4_methylphthalic_anhydride", "ethyl_acetate"]
            ],
            list_molar_masses=[None, None]
        )

        # li2016
        self.experimental_datasets['li2016'] = ExperimentalDataset(
            name_excel_file='li2016',
            sheet_names=["Table_2_1", "Table_2_2", "Table_2_3",
                         "Table_3_1", "Table_3_2", "Table_3_3"],
            list_temperature=[None] * 6,
            list_pressure=[None] * 6,
            molar_fractions_given=False,
            feed_given=False,
            ordered_component_names_per_sheet=[
                ["3_4_dichloronitrobenzene", "2_3_dichloronitrobenzene", "ethanol"],
                ["3_4_dichloronitrobenzene", "2_3_dichloronitrobenzene", "ethanol"],
                ["3_4_dichloronitrobenzene", "2_3_dichloronitrobenzene", "ethanol"],
                ["3_4_dichloronitrobenzene", "2_3_dichloronitrobenzene", "n_propanol"],
                ["3_4_dichloronitrobenzene", "2_3_dichloronitrobenzene", "n_propanol"],
                ["3_4_dichloronitrobenzene", "2_3_dichloronitrobenzene", "n_propanol"]
            ],
            list_molar_masses=[
                np.array([192, 192, 32]),
                np.array([192, 192, 32]),
                np.array([192, 192, 32]),
                np.array([192, 192, 60.1]),
                np.array([192, 192, 60.1]),
                np.array([192, 192, 60.1])
            ]
        )

        # li2017
        self.experimental_datasets['li2017'] = ExperimentalDataset(
            name_excel_file='li2017',
            sheet_names=["Table_2", "Table_3", "Table_4"],
            list_temperature=[None] * 3,
            list_pressure=[None] * 3,
            molar_fractions_given=False,
            feed_given=False,
            ordered_component_names_per_sheet=[
                ["adipic_acid", "glutaric_acid", "succinic_acid", "ethanol"],
                ["adipic_acid", "glutaric_acid", "succinic_acid", "ethanol"],
                ["adipic_acid", "glutaric_acid", "succinic_acid", "ethanol"]
            ],
            list_molar_masses=[
                np.array([146.14, 132.12, 118.09, 32]),
                np.array([146.14, 132.12, 118.09, 32]),
                np.array([146.14, 132.12, 118.09, 32])
            ]
        )
