# cem4all
This repository contains the implementation for the convex envelope method (CEM) for arbitrary chemical systems (in terms of aggregate state, number of components, and number of phases) from the paper [1]. The present version of the CEM is based on our previous publication [2] that describes the CEM for arbitrary liquid phase equilibria. The repository also includes code from the implementation of the HANNA-model [3].

# How to cite

To cite the CEM or when you are using code from this repository, please cite our works on that topic [1, 2]. When you are using the HANNA-model, please cite [3].

# Installation

We were using Python 3.11.7. To reproduce the results from [1], just install all packages from requirements.txt and run the script main.py.

# References

[1] Q. Göttl, N. Rosen, J. Burger, 2025. Convex envelope method for T, p flash calculations for chemical systems with an arbitrary number of components and arbitrary aggregate states. arXiv:2502.09402.

[2] Q. Göttl, J. Pirnay, D. G. Grimm, J. Burger, 2023. Convex Envelope Method for determining liquid multi-phase equilibria in systems with arbitrary number of components. Computers & Chemical Engineering. 177, 108321. Preprint available: arXiv:2304.12025.

[3] T. Specht, M. Nagda, S. Fellenz, A. Mandt, H. Hasse, F. Jirasek, 2024. HANNA: hard-constraint neural network for consistent activity coefficient prediction. Chemical Science. 15, 19777.

# License
This project is licensed under the MIT License. See the LICENSE file for details.
