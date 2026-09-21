# NPI-propagation

Code accompanying the study:

**Neural Perturbational Inference Reveals Reduced Unimodal–Transmodal Gradient Differentiation and Altered Sensory-Seeded Propagation, with Weakened Gradient–Propagation Coupling in Autism**

This repository contains study-specific analysis code for investigating
effective-connectivity gradients, sensory-seeded propagation,
gradient–propagation coupling, brain–behavior associations, and
transcriptomic associations in autism spectrum disorder (ASD) using
ABIDE I and ABIDE II.

---

## Overview

The main analysis workflow of this study was:

```text
ABIDE I / ABIDE II
        |
        v
Resting-state fMRI preprocessing
        |
        v
Schaefer400 ROI time-series extraction
        |
        v
Neural Perturbational Inference (NPI)
        |
        v
Participant-specific effective connectivity
        |
        +-----------------------------+
        |                             |
        v                             v
Effective-connectivity          Sensory-seeded
gradient analysis               propagation analysis
        |                             |
        +--------------+--------------+
                       |
                       v
             Gradient–propagation
                  coupling
                       |
             +---------+---------+
             |                   |
             v                   v
       Group analysis       Brain–behavior
                              analysis
             |
             v
    Regional spatial maps
             |
             v
Transcriptomic and ASD-risk-gene analysis
```

ABIDE I was used as the discovery cohort and ABIDE II as an independent
replication cohort.

The two cohorts were analyzed independently. Cohort-specific
harmonization parameters and gradient templates were estimated
separately.

---

## Neural Perturbational Inference

Participant-specific effective connectivity was estimated using the
Neural Perturbational Inference (NPI) framework described in the
original NPI study.

The original NPI implementation is publicly available at:

https://github.com/ncclab-sustech/NPI/

This repository does not introduce a new implementation of NPI.

Instead, it contains study-specific code used to process NPI-derived
effective-connectivity matrices and perform the downstream analyses
reported in the manuscript, including:

- effective-connectivity analysis;
- NPI validation analyses;
- cortical gradient analysis;
- multisite harmonization;
- sensory-seeded propagation analysis;
- gradient–propagation coupling analysis;
- ASD–control statistical comparisons;
- brain–behavior association analysis;
- transcriptomic and ASD-risk-gene analysis;
- spatial-null and sensitivity analyses;
- discovery–replication analyses; and
- manuscript and supplementary figure generation.

---

## Data

### ABIDE

Resting-state fMRI and phenotypic data were obtained from the publicly
available Autism Brain Imaging Data Exchange (ABIDE I and ABIDE II).

Participant-level ABIDE data are not redistributed in this repository.

ABIDE data can be obtained from:

http://fcon_1000.projects.nitrc.org/indi/abide/

Users wishing to reproduce the participant-level analyses should obtain
the data directly from the original ABIDE repository.

### Allen Human Brain Atlas

Human cortical gene-expression data were obtained from the Allen Human
Brain Atlas (AHBA):

https://human.brain-map.org/

### SFARI Gene

ASD-associated gene annotations were obtained from the SFARI Gene
database:

https://gene.sfari.org/

The SFARI gene annotations used for the reported analyses correspond to
the dataset version specified in the manuscript.

No new primary neuroimaging or transcriptomic datasets were generated
in this study.

---

# Computational environments

The analyses were performed using two computing environments.

Environment 1 was used for NPI estimation and the main neuroimaging
analyses.

Environment 2 was used only for the transcriptomic and gene-related
analyses.

---

## Environment 1 — NPI and neuroimaging analyses

This system was used for participant-specific NPI estimation and the
main neuroimaging analyses, including effective-connectivity,
gradient, propagation, coupling, statistical, behavioral, sensitivity,
replication, and figure-generation analyses.

### Operating system

- Microsoft Windows
- Windows version/build: 10.0.17763.737
- 64-bit AMD64 architecture

### CPU

- 2 × Intel Xeon Gold 6226R @ 2.90 GHz
- 16 physical cores per processor
- 32 physical CPU cores in total
- 32 logical processors per processor
- 64 logical processors in total

### System memory

- 255.66 GB RAM
- 8 × 32 GB Samsung memory modules

### GPU

- 2 × NVIDIA Tesla T4
- 15,360 MiB VRAM per GPU
- NVIDIA driver 539.41

### CUDA environment

System CUDA Toolkit:

```text
CUDA Toolkit 12.2
nvcc 12.2.91
```

PyTorch environment:

```text
PyTorch 2.5.1+cu121
PyTorch CUDA build 12.1
CUDA available: True
```

The installed CUDA Toolkit version and the CUDA runtime version used by
PyTorch are therefore not identical. PyTorch 2.5.1+cu121 was used for
the NPI analyses.

### Python

```text
Python 3.11.9
```

### Main Python packages

```text
NumPy          1.26.4
SciPy          1.10.1
pandas         1.5.3
scikit-learn   1.7.1
statsmodels    0.14.5
nibabel        5.3.2
Nilearn        0.12.0
BrainSpace     0.1.22
PyTorch        2.5.1+cu121
```

Resting-state fMRI preprocessing used:

```text
fMRIPrep 24.1.1
```

GPU acceleration was used for NPI estimation.

The downstream gradient, propagation, statistical, behavioral, and
figure-generation analyses do not require the same dual-GPU hardware
configuration.

---

## Environment 2 — transcriptomic and gene analyses

A separate workstation was used specifically for the transcriptomic,
spatial-null, and ASD-risk-gene analyses.

### Operating system

```text
Microsoft Windows 10 Pro
Version 10.0.19045
Build 10.0.19045.6466
64-bit
```

### CPU

```text
Intel Core i9-10900K @ 3.70 GHz
10 physical cores
20 logical processors
```

### System memory

```text
31.86 GB RAM
2 × 16 GB Corsair memory modules
```

### GPU

The workstation contained:

```text
NVIDIA GeForce RTX 3070
8192 MiB VRAM
NVIDIA driver 610.62
```

However, the transcriptomic and gene analyses performed on this
workstation did not require GPU acceleration.

The Python environment used a CPU-only PyTorch installation.

### Python

```text
Python 3.10.8
```

### Key packages used for transcriptomic and gene analyses

```text
abagen          0.1.3
BrainSMASH      0.11.0
NumPy           2.2.6
SciPy           1.15.3
pandas          2.3.3
scikit-learn    1.7.2
statsmodels     0.14.6
gseapy          1.3.0
nibabel         5.4.2
Nilearn         0.14.0
```

The complete environment also contained:

```text
PyTorch         2.4.1+cpu
BrainSpace      0.2.1
neuroCombat     0.2.12
neuromaps       0.0.7
netneurotools   0.3.0
bctpy           0.6.1
```

BrainSpace 0.2.1 installed on this workstation was not used for the
principal cortical-gradient analyses. The principal gradient analyses
used BrainSpace 0.1.22 in Environment 1.

No CUDA-enabled PyTorch installation was required for the
transcriptomic analyses.

---

# Installation

Clone this repository:

```bash
git clone https://github.com/zahngfeifei/NPI-propagation.git
cd NPI-propagation
```

The original NPI implementation should be obtained separately from:

```text
https://github.com/ncclab-sustech/NPI/
```

For the principal neuroimaging analyses, a Python environment matching
Environment 1 should be used.

For the transcriptomic and gene analyses, a Python environment matching
Environment 2 should be used.

Because different analysis stages were performed in different
environments, users do not need to install all packages listed above
into a single Python environment.

---

## Example Environment 1 installation

A Python 3.11 environment can be created and the principal dependencies
installed using an environment manager of the user's choice.

The versions used in this study were:

```text
numpy==1.26.4
scipy==1.10.1
pandas==1.5.3
scikit-learn==1.7.1
statsmodels==0.14.5
nibabel==5.3.2
nilearn==0.12.0
brainspace==0.1.22
torch==2.5.1+cu121
```

The appropriate PyTorch installation command depends on the operating
system and CUDA configuration. Users should follow the official PyTorch
installation instructions for the required CUDA build.

---

## Example Environment 2 installation

For the transcriptomic analyses, the principal packages used were:

```text
abagen==0.1.3
brainsmash==0.11.0
numpy==2.2.6
scipy==1.15.3
pandas==2.3.3
scikit-learn==1.7.2
statsmodels==0.14.6
gseapy==1.3.0
nibabel==5.4.2
nilearn==0.14.0
```

For compatibility with `abagen==0.1.3`, users should ensure that its
required Python dependencies, including a compatible installation
providing `pkg_resources`, are available.

---

# Repository structure

The repository contains scripts corresponding to different stages of
the analyses reported in the manuscript.

Because the scripts were developed for the study-specific analysis
workflow, filenames retain their original analysis-stage labels.

---

## Effective-connectivity estimation

Example:

```text
1.EC矩阵计算_固定_LOSS_保存EC_参数一致.py
```

This stage generates participant-specific NPI-derived directed
effective-connectivity matrices.

---

## NPI validation

Examples:

```text
2.NPI方法验证.py
2.NPI方法验证_组平均计算.py
```

These analyses evaluate the ability of NPI-derived models to reproduce
empirical functional-connectivity structure.

---

## Functional connectivity and cortical gradients

Examples:

```text
2.因果梯度分析_0.功能链接.py
2.因果梯度分析_1.梯度计算.py
2.因果梯度分析_2.群体模板.py
```

These scripts construct connectivity matrices, estimate cortical
gradients, generate group-level gradient templates, and align
participant-level gradients.

---

## Site harmonization

Scripts containing `combat` implement multisite harmonization for
relevant participant-level imaging measures.

Examples:

```text
2.因果梯度分析_3.梯度combat.py
2.因果梯度分析_3.有效连接combat.py
2.因果梯度分析_3.功能连接梯度combat.py
2.因果梯度分析_3.负向梯度combat.py
```

Discovery and replication cohorts should be harmonized independently.

---

## Group-level statistical analyses

The repository contains scripts for ASD–control comparisons at the
participant, network, hierarchical-level, and cortical-parcel levels.

These analyses include covariate-adjusted regression models,
multiple-comparison correction, spatial-null inference, and sensitivity
analyses as specified in the manuscript.

---

## Sensory-seeded propagation analysis

The propagation analyses use positive-output effective-connectivity
matrices to construct directed transition matrices and quantify
sensory-seeded propagation across cortical hierarchical levels.

Visual and somatomotor systems are used as sensory starting systems as
specified in the manuscript.

The derived measures include early propagation profiles and summary
indices used in subsequent group comparisons.

---

## Gradient–propagation coupling

These analyses quantify the relationship between cortical hierarchical
gradient position and sensory-seeded propagation within participants.

Participant-level coupling estimates are subsequently used for
ASD–control comparisons and behavioral association analyses.

---

## Brain–behavior analyses

The behavioral analyses evaluate associations between
gradient–propagation coupling and available ADOS measures.

Both linear and nonlinear models used in the manuscript are implemented
in the corresponding analysis scripts.

---

# Transcriptomic and gene analyses

The transcriptomic analyses integrate regional imaging-effect maps with
Allen Human Brain Atlas expression data and ASD-related gene sets.

They include:

- AHBA expression processing;
- regional gene-expression mapping;
- ASD-risk-gene analyses;
- spatial-null testing;
- matched-gene-set null analyses;
- donor and hemisphere sensitivity analyses;
- ranked-gene analyses; and
- replication/sensitivity analyses.

Relevant scripts include:

```text
补充分析_1.公共配置与函数.py
补充分析_2.匹配基因集零模型.py
补充分析_3.供体与半球敏感性.py
补充分析_4.汇总统计.py
补充分析_5.排名基因GSEA.py
补充分析_6.结果验证.py
补充分析_7.生成分析摘要.py
补充分析_8.运行补充分析.py
```

These analyses were performed using Environment 2.

---

# Figure generation

Scripts beginning with:

```text
plot-
```

are used to generate main-text and supplementary figures from the saved
analysis outputs.

Figure-generation scripts operate on previously generated numerical
results rather than rerunning the complete NPI pipeline.

---

# Instructions for use

The scripts were developed for the directory structure used in this
study.

Before executing a script, users should configure the input and output
paths defined in the corresponding Python file according to their local
directory structure.

A complete analysis generally follows the sequence below.

### 1. Obtain public datasets

Obtain ABIDE I and/or ABIDE II directly from the ABIDE repository.

For transcriptomic analyses, obtain the required AHBA and SFARI
resources from their original repositories.

### 2. Preprocess resting-state fMRI

Preprocess the imaging data according to the preprocessing procedure
specified in the manuscript.

### 3. Extract cortical ROI time series

Extract resting-state time series using the Schaefer 400-region cortical
parcellation.

### 4. Estimate participant-specific effective connectivity

Use the original NPI framework to derive participant-specific directed
effective-connectivity matrices.

### 5. Perform gradient analyses

Compute functional- and effective-connectivity gradients and align
participant-level gradients to the appropriate cohort-specific
reference.

### 6. Perform multisite harmonization

Apply the cohort-specific harmonization procedures described in the
manuscript.

ABIDE I and ABIDE II should be processed independently.

### 7. Run sensory-seeded propagation analyses

Construct the directed propagation model and estimate propagation from
the predefined sensory systems.

### 8. Calculate gradient–propagation coupling

Estimate participant-level relationships between cortical gradient
position and sensory-seeded propagation.

### 9. Perform group and behavioral analyses

Run ASD–control statistical comparisons and the corresponding
brain–behavior association analyses.

### 10. Perform transcriptomic analyses

Use the regional imaging-effect maps together with AHBA expression
profiles and SFARI ASD-risk-gene annotations.

These analyses should be performed using the transcriptomic environment
described under Environment 2.

### 11. Generate figures

Use the corresponding `plot-*.py` scripts to generate the main-text and
supplementary figures.

---

# Reproducing the study analyses

The Methods section of the associated manuscript provides the definitive
specification of analysis parameters.

In particular, users should follow the manuscript for:

- participant inclusion and exclusion criteria;
- image preprocessing;
- motion-control procedures;
- cortical parcellation;
- NPI model configuration;
- effective-connectivity construction;
- gradient estimation and alignment;
- multisite harmonization;
- sensory-seeded propagation parameters;
- gradient–propagation coupling;
- statistical covariates;
- multiple-comparison correction;
- behavioral models;
- spatial-null procedures;
- transcriptomic analyses; and
- discovery and replication procedures.

ABIDE I and ABIDE II should not be pooled when reproducing the
discovery–replication analyses.

Cohort-specific templates, harmonization parameters, and statistical
models should be estimated separately as described in the manuscript.

---

# Input data

Participant-level input data are not included in this repository.

Users are responsible for downloading the original publicly available
datasets and configuring their local paths.

Depending on the analysis stage, inputs include:

```text
Schaefer400 ROI time series
Participant phenotypic information
Participant-specific effective-connectivity matrices
Gradient estimates
Propagation estimates
ADOS measures
Regional imaging-effect maps
AHBA gene-expression data
SFARI gene annotations
```

---

# Outputs

Depending on the analysis stage, the code generates:

```text
Participant-specific effective-connectivity matrices
NPI validation results
Functional-connectivity matrices
Effective-connectivity gradients
Cohort-level gradient templates
Harmonized participant-level measures
Sensory-seeded propagation profiles
Hierarchical propagation measures
Gradient–propagation coupling estimates
ASD–control statistical results
Brain–behavior association results
Regional cortical statistical maps
Transcriptomic association results
Spatial-null distributions
Gene-set null distributions
Sensitivity-analysis results
Main-text figures
Supplementary figures
```

Output paths are configured within the corresponding analysis scripts.

---

# Reproducibility notes

The following principles should be retained when reproducing the
analyses:

1. ABIDE I and ABIDE II are treated as independent discovery and
   replication cohorts.

2. Site harmonization is performed independently within each cohort.

3. Gradient templates are estimated independently within the relevant
   cohort rather than being fitted to pooled ABIDE I and ABIDE II data.

4. The same principal NPI, gradient, propagation, coupling, and
   statistical procedures are applied to both cohorts.

5. Transcriptomic analyses use regional imaging-effect maps generated
   by the preceding neuroimaging analyses.

6. Participant-level ABIDE data are obtained directly from ABIDE and
   are not redistributed through this repository.

---

# Software availability

The original Neural Perturbational Inference implementation is available
from:

https://github.com/ncclab-sustech/NPI/

The study-specific downstream analysis code is available from:

https://github.com/zahngfeifei/NPI-propagation

---

# Data availability

Resting-state fMRI and phenotypic data:

https://fcon_1000.projects.nitrc.org/indi/abide/

Allen Human Brain Atlas:

https://human.brain-map.org/

SFARI Gene:

https://gene.sfari.org/

No new primary neuroimaging or transcriptomic datasets were generated
in this study.

---

# Citation

If using the effective-connectivity methodology, please cite the
original Neural Perturbational Inference study.

If using the study-specific analysis code contained in this repository,
please also cite the associated manuscript.
