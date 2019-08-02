# SECSI
Demo implementation for the **SE**mi-algebraic framework for the approximate **C**P decompositions via **SI**multaneous Matrix Diagonalizations (SECSI) [1].


SECSI is a framework for the semi-algebraic CP-decomposition of N-th order multidimensional arrays (tensors).

 * **SECSI** provides an enhanced performance compared to other solvers in difficult scenarios, e.g., correlated factors 
 * **SECSI** offers complexity-accuracy tradeoff by selecting appropriate heuristics
 * **SECSI** flexibly adapts to the data it is given: it can handle real- and complex-valued data with an arbitrary dimension. It automatically detects certain special cases where closed-form solutions are available for improved speed and performance.
 

# Dependencies
The sources for SECSI are self-contained. SECSI operates on conventional multi-dimensional arrays as provided by MATLAB.

# Demo
We have included a demo that showcases certain features of SECSI, including:

 * Exact recovery of randomly generated three-way low-rank tensor
 * Approximate recovery in the noisy case
 * Performance/Speed comparison of various heuristics
 * How to set up a custom heuristic
 * Recovery of a 4-way tensor
 * Special cases rank-2 and 2-slab CPD

# SECSI-GU (Upcoming Release)
Please note that for the CP-decomposition of tensors of order N>3 we recommend the use of SECSI-GU, where optimum performance is achieved by making use of generalized unfoldings (GU) [2]. The sources are currently being prepared for an upcoming release.


# References
[1] F. Roemer and M. Haardt, “A Semi-Algebraic Framework for Approximate CP Decompositions via Simultaneous Matrix Diagonalizations (SECSI),” Signal Processing, vol. 93, no. 9, pp. 2722–2738, 2013.

[2] F. Roemer, C. Schroeter, and M. Haardt, “A Semi-Algebraic Framework for Approximate CP Decompositions via Joint Matrix Diagonalization and Generalized Unfoldings,” in Conference Record of the Forty Sixth Asilomar Conference on Signals, Systems and Computers (ASILOMAR), 2012, 2012, pp. 2023–2027.

# Citation
If you use this code as part of any published research, please acknowledge the following paper.

```
@article{roemer2013a,
author = {Roemer, Florian and Haardt, Martin},
doi = {10.1016/j.sigpro.2013.02.016},
journal = {Signal Processing},
number = {9},
pages = {2722--2738},
title = {A Semi-Algebraic Framework for Approximate CP Decompositions via Simultaneous Matrix Diagonalizations (SECSI)},
volume = {93},
year = {2013}
}
```