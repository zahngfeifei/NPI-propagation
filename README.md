## Neural Perturbational Inference

Participant-specific effective connectivity was estimated using the
Neural Perturbational Inference (NPI) framework described by Luo et al.

The original NPI implementation is publicly available at:
https://github.com/ncclab-sustech/NPI/

This repository does not provide a new implementation of NPI.
Instead, it contains study-specific analysis code for processing
NPI-derived effective-connectivity matrices and reproducing the
downstream analyses reported in the manuscript, including cortical
gradient analysis, sensory-seeded propagation, gradient–propagation
coupling, statistical analyses, and transcriptomic analyses.

## Data

This study used publicly available ABIDE I and ABIDE II neuroimaging
and phenotypic data. Participant-level ABIDE data are not redistributed
in this repository.

Data can be obtained from:
http://fcon_1000.projects.nitrc.org/indi/abide/

## System requirements

Python 3.11.9

Main dependencies:
- NumPy 1.26.4
- SciPy 1.10.1
- pandas 1.5.3
- scikit-learn 1.7.1
- statsmodels 0.14.5
- nibabel 5.3.2
- Nilearn 0.12.0
- BrainSpace 0.1.22
- BrainSMASH 0.11.0
- abagen 0.1.3
- PyTorch 2.5.1
- CUDA 12.1
