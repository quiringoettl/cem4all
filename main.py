import os

from discretization.create_point_disc import create_discs
from util import results_generation_wrappers as results_gen


if __name__ == "__main__":
    # ensure discs are there
    disc_path = create_discs()

    # ensure results path
    results_path = os.path.join(os.getcwd(), "phase_equilibria_results")
    if not os.path.isdir(results_path):
        os.mkdir(results_path)

    # VLEs
    results_gen.generate_oh2003_VLEs(discretization_path=disc_path)
    results_gen.generate_xiao2013_VLEs(discretization_path=disc_path)

    # SLEs
    results_gen.generate_kim2011_SLE(discretization_path=disc_path)
    results_gen.generate_gao2018_SLEs(discretization_path=disc_path)
    results_gen.generate_li2016_SLEs(discretization_path=disc_path)
    results_gen.generate_li2017_SLEs(discretization_path=disc_path)

    # HANNA example
    results_gen.HANNA_example(discretization_path=disc_path)

    # LLEs (a subset of those done in Göttl et al. 2023, just to check if LLEs still work)
    results_gen.generate_chen2000_LLEs(discretization_path=disc_path)
    results_gen.generate_chen2001_LLEs(discretization_path=disc_path)
    results_gen.generate_yuan2019_LLEs(discretization_path=disc_path)
