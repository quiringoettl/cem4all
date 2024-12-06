import os

from discretization.point_discretization import PointDisc


def create_discs():
    # create point discretizations as specified below [N_comp, N_disc]
    discretizations_todo = [
        [2, 5], [2, 8],
        [3, 7],
        [4, 6],
        [5, 5]
    ]

    discretization_path = os.path.join(os.getcwd(), "discretization", "discs")
    os.makedirs(discretization_path, exist_ok=True)

    for todo_el in discretizations_todo:
        filename = os.path.join(discretization_path, str(todo_el[0]) + "_" + str(todo_el[1]))
        if not os.path.isdir(filename):
            PointDisc(
                num_comp=todo_el[0], recursion_steps=todo_el[1],
                load=False, store=True, path=discretization_path
            )

    return discretization_path


if __name__ == "__main__":
    create_discs()
