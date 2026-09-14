from __future__ import annotations

import csv
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


sourceDocument = Path(r"F:\NPI-4-code\方法\npi-manuscript-v27-citation-placeholders-revised.docx")
outputDocument = Path(r"F:\NPI-4-code\方法\npi-manuscript-v27-citation-placeholders-rn-replaced.docx")
outputCsv = Path(r"F:\NPI-4-code\方法\citation-placeholder-doi-map.csv")

# Replace multi-key citations first so the resulting citation lists remain concise and semantically scoped.
citationReplacements = [
    ("RN75,RN76,RN77", "REF_connectome_spreading_dynamics,REF_network_diffusion"),
    ("RN76,RN77", "REF_network_diffusion"),
    ("RN79,RN75", "REF_connectome_spreading_dynamics"),
    ("RN34,RN79", "REF_Yeo_networks"),
    ("RN220,RN221", "REF_B_splines,REF_generalized_additive_models"),
    ("RN66", "REF_ABIDE_I_dataset"),
    ("RN69", "REF_ABIDE_II_dataset"),
    ("RN70", "REF_fMRIPrep"),
    ("RN72", "REF_ABIDE_I_dataset,REF_ABIDE_II_dataset"),
    ("RN176", "REF_NPI_architecture"),
    ("RN38", "REF_diffusion_maps"),
    ("RN37", "REF_BrainSpace"),
    ("RN79", "REF_Yeo_networks"),
    ("RN82", "REF_BH_FDR"),
    ("RN75", "REF_connectome_spreading_dynamics"),
    ("RN76", "REF_network_diffusion"),
    ("RN77", "REF_network_diffusion"),
    ("RN78", "REF_brain_connectivity_toolbox"),
    ("RN81", "REF_HC3_robust_covariance"),
    ("RN219", "REF_ADOS"),
    ("RN179", "REF_AIC"),
    ("RN223", "REF_spline_basis_residualization"),
    ("RN208", "REF_robust_Wald_test"),
    ("RN89", "REF_abagen"),
]

doiMap = {
    "REF_ABIDE_I_dataset": "10.1038/mp.2013.78",
    "REF_ABIDE_II_dataset": "10.1038/sdata.2017.10",
    "REF_ABIDE_site_specific_TR_parameters": "10.1038/mp.2013.78; 10.1038/sdata.2017.10",
    "REF_AHBA": "10.1038/nature11405",
    "REF_ADOS": "10.1002/1097-4687(200012)37:3<205::AID-JCLP1>3.0.CO;2-9",
    "REF_AIC": "10.1109/TAC.1974.1100705",
    "REF_Adam_optimizer": "arXiv:1412.6980 (no DOI)",
    "REF_BH_FDR": "10.1111/j.2517-6161.1995.tb02031.x",
    "REF_BIC": "10.1214/aos/1176344136",
    "REF_BrainSMASH": "10.1016/j.neuroimage.2020.117038",
    "REF_BrainSpace": "10.1038/s42003-020-0794-7",
    "REF_B_splines": "10.1214/ss/1038425655",
    "REF_ComBat": "10.1093/biostatistics/kxj037",
    "REF_CompCor": "10.1016/j.neuroimage.2007.04.042",
    "REF_Conte69_surface": "10.1093/cercor/bhr291",
    "REF_FD_DVARS_outlier_thresholds_precedent": "10.1038/s41592-018-0235-4",
    "REF_FSL_MCFLIRT": "10.1006/nimg.2002.1132",
    "REF_Fisher_z_transform": "10.1093/biomet/10.4.507",
    "REF_Friedman_test": "10.1080/01621459.1937.10503522",
    "REF_GSEA": "10.1073/pnas.0506580102",
    "REF_HC1_robust_covariance": "10.1016/0304-4076(85)90158-7",
    "REF_HC3_robust_covariance": "10.1016/0304-4076(85)90158-7",
    "REF_MNI152NLin2009cAsym_template": "10.1016/S1053-8119(09)70884-5",
    "REF_Morans_I": "10.1093/biomet/37.1-2.17",
    "REF_NPI_architecture": "10.1038/s41592-025-02654-x",
    "REF_Nilearn": "10.3389/fninf.2014.00014",
    "REF_SFARI_database": "10.1242/dmm.005439",
    "REF_Schaefer_parcellation": "10.1093/cercor/bhx179",
    "REF_Wilcoxon_Pratt": "10.1080/01621459.1959.10501526",
    "REF_Yeo_networks": "10.1152/jn.00338.2011",
    "REF_abagen": "10.7554/eLife.72129",
    "REF_boundary_based_registration": "10.1016/j.neuroimage.2009.06.060",
    "REF_brain_connectivity_toolbox": "10.1016/j.neuroimage.2009.10.003",
    "REF_cluster_robust_standard_errors": "10.1111/j.1468-0084.1987.mp49004006.x",
    "REF_connectome_spreading_dynamics": "10.1016/j.neuron.2015.05.035",
    "REF_derivative_quadratic_confounds_expansion": "10.1016/j.neuroimage.2012.08.052",
    "REF_differential_stability": "10.1038/nn.4171",
    "REF_diffusion_maps": "10.1007/s10208-006-0225-0",
    "REF_empirical_permutation_p_value": "10.2202/1544-6115.1585",
    "REF_expression_matched_gene_set_null_precedent": "10.1002/hbm.25711",
    "REF_fMRIPrep": "10.1038/s41592-018-0235-4",
    "REF_generalized_additive_models": "10.1201/9781315370279",
    "REF_global_signal_regression": "10.1016/j.neuroimage.2016.11.052",
    "REF_intersection_union_test": "10.1214/ss/1032280304",
    "REF_mean_FD_0_3_exclusion_threshold_precedent": "10.1038/s41467-019-08944-1",
    "REF_motion_scrubbing": "10.1016/j.neuroimage.2011.10.018",
    "REF_network_diffusion": "10.1016/j.neuroimage.2013.10.069",
    "REF_neuroimaging_ComBat": "10.1016/j.neuroimage.2017.11.024",
    "REF_nonparametric_bootstrap": "10.1214/aos/1176344552",
    "REF_orthogonal_Procrustes": "10.1007/BF02289451",
    "REF_resting_state_temporal_filtering": "PMID:11415917 (no DOI)",
    "REF_robust_Wald_test": "10.2307/1912934",
    "REF_spline_basis_residualization": "10.1007/978-3-319-19425-7",
    "REF_spin_permutation": "10.1016/j.neuroimage.2018.05.070",
    "REF_standardized_mean_difference_balance_diagnostic": "10.1002/sim.3697",
}

with ZipFile(sourceDocument, "r") as inputZip, ZipFile(outputDocument, "w", ZIP_DEFLATED) as outputZip:
    for zipItem in inputZip.infolist():
        content = inputZip.read(zipItem.filename)
        if zipItem.filename == "word/document.xml":
            documentText = content.decode("utf-8")
            for oldCitation, newCitation in citationReplacements:
                documentText = documentText.replace(oldCitation, newCitation)
            content = documentText.encode("utf-8")
        outputZip.writestr(zipItem, content)

with outputCsv.open("w", newline="", encoding="utf-8-sig") as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(["citationPlaceholder", "doiOrIdentifier"])
    for placeholder in sorted(doiMap):
        writer.writerow([placeholder, doiMap[placeholder]])

print(outputDocument)
print(outputCsv)
