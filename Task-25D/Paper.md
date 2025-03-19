The study utilizes a convolutional neural network (CNN) to model cropped 2.5D data at the slice level, where each slice is treated as an independent input for feature extraction. Through the application of a multi-instance learning (MIL) framework, features from multiple slices are aggregated. These aggregated features are subsequently used in survival analysis to predict patient outcomes. The approach emphasizes the integration of intra- and inter-slice variability, enhancing the model's ability to reflect the complex tumor landscape. This methodology shows promise for advancing predictive models in oncological prognostics. Figure 1 illustrates the workflow of this study.

![image-20241124224607644](doc\pipeline.png)

Fig1. Workflow of this work.

## Methods

### Patients

> 这里填一下你的入组出组标准

The study cohort was divided into two groups: a training cohort comprising 70% of the dataset and an internal validation cohort accounting for the remaining 30%. Furthermore, the model's generalizability was assessed using an independent external test dataset to ensure robust performance across different populations.

### Image Acquisition

> 数据获取对应的参数，参考一下下面这个描述

**ROI Segmentation**: In this study, the Regions of Interest (ROI) in the training dataset were manually delineated using ITK-SNAP. Two experienced radiologists independently annotated the ROIs, with any inconsistencies between their annotations being resolved by a senior radiologist with over 20 years of clinical experience. 

**Data Preprocessing**: Rigorous preprocessing was applied to enhance the consistency of medical images across diverse sources. Voxel spacing was standardized to $1mm \times 1mm \times 1mm$ to ensure spatial uniformity. 

### 2.5D Multi-Instance Learning

In deep learning, traditional models often focus on the largest cross-section of the ROI, potentially missing crucial contextual information. While 3D CNNs provide more comprehensive feature extraction, they introduce increased complexity and a higher risk of overfitting. To address this, we developed a 2.5D deep learning model that incorporates multiple adjacent slices surrounding the central slice, integrating data from different perspectives to achieve a more detailed representation of the ROI. 

For survival analysis, our workflow was divided into two steps. First, we grouped patients based on their 3-year survival status and trained a classification model, selecting the best-performing model on the internal validation set. This model was then used within the multi-instance learning framework to aggregate features, which were subsequently utilized to build OS and DFS models using Cox regression.

#### Data Generation

**2.5D Image Construction:** We constructed the 2.5D dataset by selecting the largest cross-sectional slice of the ROI, supplemented with adjacent slices to capture more contextual information. Specifically, slices positioned at ±1, ±2, and ±4 relative to the central slice were included, resulting in a total of seven slices per patient. Using the OKT-crop_max_roi tool from the OnekeyAI Platform, we ensured precise cropping that preserved key structural details of the ROI, creating a comprehensive 2.5D representation.

#### Slice-Level Model Training

For model training, the generated 2.5D data was used to evaluate performance across various deep learning architectures. We tested three models—DenseNet121, DenseNet201, and ResNet18—and selected the one that yielded the best results on the validation set. Detailed model configurations and training setups are provided in Supplementary Material 1A.

#### Multi-Instance Learning Fusion

To integrate the slice-level predictions, we employed two multi-instance learning fusion techniques. First, we generated Predict Likelihood Histograms (PLH) from the 2.5D models, which captured predictive probabilities and labels for each slice, providing an overall probabilistic view. We also applied a Bag of Words (BoW) method, where each sample’s seven slices were treated as individual instances, and their predictive outputs were processed using the Term Frequency-Inverse Document Frequency (TF-IDF) approach. Finally, we enhanced the model by combining PLH and BoW features with radiomic data, leveraging multiple data sources to improve classification accuracy. Detailed fusion procedures can be found in Supplementary Material 1B.

### Signature Building

**MIL Signature (MIL):** For the multi-instance learning features, we first applied a correlation-based filter to reduce redundancy by retaining only one feature from each highly correlated pair (Pearson correlation coefficient > 0.9). Following this, we used univariate Cox regression to select features with a p-value < 0.05 for inclusion in models predicting overall survival (OS) and disease-free survival (DFS).

**Clinical Signature:** Similar to the process for MIL features, we selected relevant clinical features using univariate Cox regression, retaining those with a p-value < 0.05 for constructing separate OS and DFS models.

**Combined Model:** To evaluate the combined predictive power of MIL and clinical data, we integrated the selected clinical features (p-value < 0.05) with the MIL Signature using a Cox proportional hazards model to predict both OS and DFS outcomes.

**Survival Analysis:** X-tile software was used to determine optimal cut-off points, allowing for patient stratification into high and low-risk groups. Kaplan-Meier survival curves were then used to compare these groups, with statistical significance assessed through the log-rank test.

### Statistical Analysis

We evaluated the normality of clinical features using the Shapiro-Wilk test. Depending on distribution, continuous variables were analyzed using either the t-test or the Mann-Whitney U test, while categorical variables were assessed using Chi-square (χ²) tests. Baseline characteristics for all cohorts are detailed in Table 1, with p-values exceeding 0.05 between groups, indicating no significant differences and confirming that the cohort divisions were unbiased.

All data analyses were performed on the OnekeyAI platform (version 3.9.1) with Python (version 3.7.12). Statistical analyses were conducted using Statsmodels (version 0.13.2), and radiomics feature extraction was completed with PyRadiomics (version 3.0.1). Deep learning models were developed with PyTorch (version 1.11.0) and optimized with CUDA (version 11.3.1) and cuDNN (version 8.2.1) for enhanced performance.

|            feature_name |        train |          val | test         | pvalue |
| ----------------------: | -----------: | -----------: | ------------ | -----: |
|          age_at_surgery |  60.72±11.43 |  61.89±10.56 | 63.91±9.28   |  0.472 |
|                  height |    1.67±0.08 |    1.69±0.07 | 1.67±0.08    |  0.247 |
|                  weight |  66.37±10.61 |  67.84±11.36 | 65.77±9.59   |  0.285 |
|                     BMI |   23.65±3.24 |   23.83±3.35 | 23.50±3.10   |  0.787 |
|                     SMA | 127.65±29.92 | 132.47±31.46 | 129.71±25.60 |  0.187 |
|                     SMI |   45.21±8.78 |   46.28±9.13 | 46.16±8.18   |  0.279 |
|                     sex |              |              |              |  0.374 |
|                  female |    91(33.96) |    33(28.70) | 19(25.33)    |        |
|                    male |   177(66.04) |    82(71.30) | 56(74.67)    |        |
|              sarcopenia |              |              |              |  0.435 |
|                       0 |   198(73.88) |    90(78.26) | 53(70.67)    |        |
|                       1 |    70(26.12) |    25(21.74) | 22(29.33)    |        |
|          clinical_stage |              |              |              |  0.901 |
|                       Ⅱ |   120(44.78) |    53(46.09) | 38(50.67)    |        |
|                       Ⅲ |   148(55.22) |    62(53.91) | 37(49.33)    |        |
|        clinical_T_stage |              |              |              |  0.381 |
|                       2 |     16(5.97) |      3(2.61) | 6(8.00)      |        |
|                       3 |   230(85.82) |   102(88.70) | 64(85.33)    |        |
|                       4 |     22(8.21) |     10(8.70) | 5(6.67)      |        |
|       Lymph_node_status |              |              |              |  0.728 |
|                       0 |   119(44.40) |    54(46.96) | 38(50.67)    |        |
|                       1 |   149(55.60) |    61(53.04) | 37(49.33)    |        |
|                location |              |              |              |  0.125 |
|                  5-10cm |   122(45.52) |    64(55.65) | 34(45.33)    |        |
|                    <5cm |   105(39.18) |    33(28.70) | 31(41.33)    |        |
|                   >10cm |    41(15.30) |    18(15.65) | 10(13.33)    |        |
|  Preoperative_serum_CEA |              |              |              |  0.143 |
|                      <5 |   181(67.54) |    87(75.65) | 49(65.33)    |        |
|                      ≥5 |    87(32.46) |    28(24.35) | 26(34.67)    |        |
|       Surgical_approach |              |              |              |    1.0 |
|              laraoscope |   236(88.06) |   101(87.83) | 65(86.67)    |        |
|                    open |    32(11.94) |    14(12.17) | 10(13.33)    |        |
|     Neoadjuvant_therapy |              |              |              |  0.007 |
|                       0 |   184(68.66) |    95(82.61) | 62(82.67)    |        |
|                       1 |    84(31.34) |    20(17.39) | 13(17.33)    |        |
|   Adjuvant_chemotherapy |              |              |              |  0.249 |
|                       0 |    54(20.15) |    30(26.09) | 33(44.00)    |        |
|                       1 |   214(79.85) |    85(73.91) | 42(56.00)    |        |
|   Adjuvant_radiotherapy |              |              |              |  0.303 |
|                       0 |   242(90.30) |    99(86.09) | 40(53.33)    |        |
|                       1 |     26(9.70) |    16(13.91) | 35(46.67)    |        |
|                pN_stage |              |              |              |  0.704 |
|                       0 |   134(50.00) |    60(52.17) | 40(53.33)    |        |
|                       1 |   111(41.42) |    43(37.39) | 26(34.67)    |        |
|                       2 |     23(8.58) |    12(10.43) | 9(12.00)     |        |
|                pT_stage |              |              |              |  0.266 |
|                       0 |      8(2.99) |      2(1.74) | null         |        |
|                       1 |      3(1.12) |      2(1.74) | null         |        |
|                       2 |    41(15.30) |      9(7.83) | 5(6.67)      |        |
|                       3 |   205(76.49) |    95(82.61) | 66(88.00)    |        |
|                       4 |     11(4.10) |      7(6.09) | 4(5.33)      |        |
| lymphovascular_invasion |              |              |              |  0.576 |
|                       0 |   243(90.67) |   107(93.04) | 54(72.00)    |        |
|                       1 |     25(9.33) |      8(6.96) | 21(28.00)    |        |

Table1. baseline characters of our cohorts.

## Results

### Clinical Features

**Univariable Analysis:** We performed a comprehensive univariate analysis of all clinical features, calculating hazard ratios (HR) and corresponding p-values for each variable. Age was specifically incorporated into the final fusion model. Clinical features with a multivariate p-value < 0.05 were selected for inclusion in the clinical model.

|           features_name |    HR | lower 95%CI | upper 95%CI | pvalue |
| ----------------------: | ----: | ----------: | ----------: | -----: |
|          age_at_surgery | 1.013 |       0.993 |       1.033 |  0.202 |
|                     sex | 1.386 |       0.860 |       2.234 |   0.18 |
|                  height | 2.642 |       0.144 |      48.368 |  0.512 |
|                  weight | 1.000 |       0.979 |       1.021 |  0.978 |
|                     BMI | 0.985 |       0.919 |       1.055 |  0.667 |
|                     SMA | 1.001 |       0.994 |       1.008 |  0.816 |
|                     SMI | 1.000 |       0.975 |       1.025 |  0.985 |
|              sarcopenia | 1.786 |       1.117 |       2.857 |  <0.05 |
|          clinical_stage | 1.487 |       0.950 |       2.327 |  0.083 |
|        clinical_T_stage | 1.244 |       0.695 |       2.227 |  0.462 |
|       Lymph_node_status | 1.323 |       0.848 |       2.064 |  0.217 |
|                location | 0.874 |       0.642 |       1.189 |  0.389 |
|  Preoperative_serum_CEA | 1.035 |       0.651 |       1.647 |  0.884 |
|       Surgical_approach | 1.320 |       0.705 |       2.470 |  0.385 |
|     Neoadjuvant_therapy | 1.172 |       0.737 |       1.865 |  0.503 |
|   Adjuvant_chemotherapy | 0.840 |       0.496 |       1.424 |  0.518 |
|   Adjuvant_radiotherapy | 1.113 |       0.544 |       2.278 |   0.77 |
|                pN_stage | 1.561 |       1.127 |       2.162 |  <0.05 |
|                pT_stage | 1.229 |       0.862 |       1.753 |  0.255 |
| lymphovascular_invasion | 1.004 |       0.474 |       2.130 |  0.991 |

Table 2.A Univariable Analysis of clinical features in OS

|           features_name |    HR | lower 95%CI | upper 95%CI | pvalue |
| ----------------------: | ----: | ----------: | ----------: | -----: |
|          age_at_surgery | 1.008 |       0.991 |       1.025 |  0.365 |
|                     sex | 1.236 |       0.825 |       1.852 |  0.304 |
|                  height | 1.469 |       0.124 |      17.456 |  0.761 |
|                  weight | 0.994 |       0.977 |       1.012 |  0.542 |
|                     BMI | 0.975 |       0.918 |       1.035 |  0.406 |
|                     SMA | 1.000 |       0.994 |       1.007 |  0.884 |
|                     SMI | 1.000 |       0.979 |       1.022 |  0.979 |
|              sarcopenia | 1.184 |       0.778 |       1.801 |  0.432 |
|          clinical_stage | 1.718 |       1.163 |       2.537 |  <0.05 |
|        clinical_T_stage | 1.131 |       0.680 |       1.878 |  0.636 |
|       Lymph_node_status | 1.475 |       1.003 |       2.169 |  <0.05 |
|                location | 0.860 |       0.661 |       1.118 |  0.259 |
|  Preoperative_serum_CEA | 1.128 |       0.758 |       1.677 |  0.553 |
|       Surgical_approach | 1.118 |       0.637 |       1.964 |  0.697 |
|     Neoadjuvant_therapy | 1.051 |       0.701 |       1.576 |  0.809 |
|   Adjuvant_chemotherapy | 0.981 |       0.616 |       1.563 |  0.937 |
|   Adjuvant_radiotherapy | 0.919 |       0.477 |       1.772 |  0.802 |
|                pN_stage | 1.673 |       1.268 |       2.209 |  <0.05 |
|                pT_stage | 1.337 |       0.973 |       1.836 |  0.073 |
| lymphovascular_invasion | 1.369 |       0.748 |       2.507 |  0.308 |

Table 2.B Univariable Analysis of clinical features in DFS

### Results of MIL Signature

#### Slice Level Results

For the AUC analysis, ResNet18 demonstrated the best performance across all cohorts. In the training cohort, ResNet18 achieved an AUC of 0.702 (95% CI: 0.6672-0.7367), indicating relatively strong discrimination. Its performance remained consistent in the validation cohort with an AUC of 0.691 (95% CI: 0.6232-0.7578), and improved in the test cohort, reaching an AUC of 0.716 (95% CI: 0.6451-0.7874). In contrast, DenseNet models exhibited lower AUCs, particularly DenseNet201, which achieved 0.627 (95% CI: 0.5640-0.6892) on the validation set and 0.601 (95% CI: 0.5224-0.6804) on the test set. DenseNet121 also showed underperformance with AUC values of 0.636 (95% CI: 0.5709-0.7021) and 0.639 (95% CI: 0.5651-0.7136) on the validation and test sets, respectively.

In conclusion, ResNet18 outperformed both DenseNet121 and DenseNet201 in terms of AUC across all cohorts. The better convergence of ResNet18, a smaller model, suggests that simpler architectures may achieve superior performance when the sample size is limited, as they are less prone to overfitting compared to more complex models like DenseNet.

|   ModelName |   Acc |   AUC |        95% CI | Sensitivity | Specificity |   PPV |   NPV | Cohort |
| ----------: | ----: | ----: | ------------: | ----------: | ----------: | ----: | ----: | ------ |
|    resnet18 | 0.684 | 0.702 | 0.6672-0.7367 |       0.606 |       0.699 | 0.265 | 0.908 | train  |
|    resnet18 | 0.792 | 0.691 | 0.6232-0.7578 |       0.519 |       0.822 | 0.245 | 0.939 | val    |
|    resnet18 | 0.747 | 0.716 | 0.6451-0.7874 |       0.613 |       0.767 | 0.286 | 0.929 | test   |
| densenet201 | 0.726 | 0.728 | 0.6959-0.7604 |       0.588 |       0.751 | 0.297 | 0.910 | train  |
| densenet201 | 0.592 | 0.627 | 0.5640-0.6892 |       0.597 |       0.591 | 0.140 | 0.930 | val    |
| densenet201 | 0.585 | 0.601 | 0.5224-0.6804 |       0.597 |       0.583 | 0.179 | 0.905 | test   |
| densenet121 | 0.535 | 0.674 | 0.6388-0.7085 |       0.724 |       0.501 | 0.207 | 0.910 | train  |
| densenet121 | 0.692 | 0.636 | 0.5709-0.7021 |       0.506 |       0.712 | 0.164 | 0.928 | val    |
| densenet121 | 0.743 | 0.639 | 0.5651-0.7136 |       0.419 |       0.792 | 0.234 | 0.900 | test   |

Table3. Slice level results of different CNN models.

<img src="img/DTL_2.5D_train_roc.svg" style="zoom: 33%;" /><img src="img/DTL_2.5D_val_roc.svg" style="zoom: 33%;" /><img src="img/DTL_2.5D_test_roc.svg" style="zoom: 33%;" />

Fig2. ROC of different CNN models in slice level prediction.

#### Grad-CAM

We used Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize the activations in the final convolution layer related to class predictions. Figure 3 shows how Grad-CAM highlights key image regions influencing the model's decisions, improving interpretability.

![primary-135.nii_-04](doc\primary-135.nii_-04.png)

![primary-298.nii_-01](doc\primary-298.nii_-01.png)

Figure3 displays Grad-CAM visualizations for two representative samples, demonstrating how the model selectively focuses on different regions of the images to make its predictions. This visualization is crucial for understanding the model's attention mechanism in practical applications.

### Survival Analysis

In terms of c-index, the MIL model outperformed the clinical model for both overall survival (OS) and disease-free survival (DFS) across all cohorts. For OS, the MIL model achieved a c-index of 0.757 in the training cohort, 0.754 in the validation cohort, and 0.741 in the test cohort, compared to 0.666, 0.772, and 0.756 for the clinical model, respectively. Similarly, for DFS, the MIL model demonstrated c-indices of 0.735 (train), 0.732 (validation), and 0.728 (test), surpassing the clinical model’s performance, which ranged from 0.624 to 0.637 across cohorts. The combined model, which integrates MIL features with clinical features, further improved predictive performance, with the highest c-index values observed for both OS (0.819, 0.822, 0.759) and DFS (0.768, 0.750, 0.721) across all cohorts.

In conclusion, the MIL model significantly outperformed the clinical feature-based model for both OS and DFS prediction, highlighting its superior predictive power. Moreover, the combined model, which fuses MIL and clinical features, further enhanced the model’s predictive performance, suggesting that integrating these diverse data sources offers a more comprehensive and accurate prediction of patient outcomes.

|        Clinical-OS |             MIL-OS |        Combined-OS |       Clinical-DFS |            MIL-DFS |       Combined-DFS | Cohort |
| -----------------: | -----------------: | -----------------: | -----------------: | -----------------: | -----------------: | ------ |
| 0.666(0.610-0.723) | 0.757(0.705-0.808) | 0.819(0.773-0.865) | 0.627(0.570-0.685) | 0.735(0.682-0.788) | 0.769(0.719-0.819) | train  |
| 0.772(0.696-0.849) | 0.754(0.676-0.833) | 0.822(0.752-0.892) | 0.624(0.535-0.712) | 0.732(0.651-0.813) | 0.747(0.668-0.827) | val    |
| 0.756(0.658-0.853) | 0.741(0.642-0.840) | 0.759(0.663-0.856) | 0.637(0.528-0.745) | 0.728(0.627-0.829) | 0.721(0.620-0.823) | test   |

Table4. C-index of different signature in prediction OS, DFS

<img src="img/Clinical-OS_KM_train.svg" style="zoom: 33%;" /><img src="img/Clinical-OS_KM_val.svg" style="zoom: 33%;" /><img src="img/Clinical-OS_KM_test.svg" style="zoom: 33%;" />

<img src="img/MIL-OS_KM_train.svg" style="zoom: 33%;" /><img src="img/MIL-OS_KM_val.svg" style="zoom: 33%;" /><img src="img/MIL-OS_KM_test.svg" style="zoom: 33%;" />

<img src="img/Combined-OS_KM_train.svg" style="zoom: 33%;" /><img src="img/Combined-OS_KM_val.svg" style="zoom: 33%;" /><img src="img/Combined-OS_KM_test.svg" style="zoom: 33%;" />

Fig. 4A: Kaplan-Meier (KM) analysis for overall survival (OS) predictions, displayed from left to right for the training, validation, and test cohorts. From top to bottom, the rows represent the clinical model, MIL model, and combined model, respectively.

<img src="img/Clinical-DFS_KM_train.svg" style="zoom: 33%;" /><img src="img/Clinical-DFS_KM_val.svg" style="zoom: 33%;" /><img src="img/Clinical-DFS_KM_test.svg" style="zoom: 33%;" />

<img src="img/MIL-DFS_KM_train.svg" style="zoom: 33%;" /><img src="img/MIL-DFS_KM_val.svg" style="zoom: 33%;" /><img src="img/MIL-DFS_KM_test.svg" style="zoom: 33%;" />

<img src="img/Combined-DFS_KM_train.svg" style="zoom: 33%;" /><img src="img/Combined-DFS_KM_val.svg" style="zoom: 33%;" /><img src="img/Combined-DFS_KM_test.svg" style="zoom: 33%;" />

Fig. 4B: Kaplan-Meier (KM) analysis for disease free survival (DFS) predictions, displayed from left to right for the training, validation, and test cohorts. From top to bottom, the rows represent the clinical model, MIL model, and combined model, respectively.

**Interpretation**: Fig. 5 illustrates the application of the combined model in a clinical context, where both clinical features and MIL signature are visualized for enhanced interoperability.

![](img/OS_nomogram.png)

![](img/DFS_nomogram.png)

Fig. 5: Nomogram for clinical application, illustrating predictions for overall survival (OS) and disease-free survival (DFS).

### Time-Dependent Analysis

In the 3-year time-dependent ROC analysis for overall survival (OS), the MIL-OS model consistently outperformed the clinical model across all cohorts. In the training cohort, MIL-OS achieved an AUC of 0.818 (95% CI: 0.7437–0.8917), compared to 0.684 (95% CI: 0.5993–0.7689) for the clinical model. In the validation cohort, MIL-OS reached an AUC of 0.812 (95% CI: 0.7090–0.9154), again surpassing the clinical model’s 0.750 (95% CI: 0.6059–0.8936). Similarly, in the test cohort, MIL-OS had an AUC of 0.784 (95% CI: 0.6402–0.9279), while the clinical model obtained an AUC of 0.800 (95% CI: 0.6327–0.9677). The combined model further improved the AUC across all cohorts, with values of 0.886, 0.834, and 0.815 for the training, validation, and test cohorts, respectively.

For the 5-year time-dependent ROC analysis, MIL-OS continued to outperform the clinical model in most cases. In the training cohort, MIL-OS achieved an AUC of 0.781 (95% CI: 0.6980–0.8642), compared to 0.685 (95% CI: 0.5891–0.7817) for the clinical model. In the validation cohort, MIL-OS underperformed slightly with an AUC of 0.647 (95% CI: 0.4878–0.8063), compared to the clinical model’s 0.817 (95% CI: 0.6826–0.9517). In the test cohort, MIL-OS showed strong performance with an AUC of 0.844 (95% CI: 0.6486–1.0000), surpassing the clinical model's AUC of 0.781 (95% CI: 0.5774–0.9851). The combined model again demonstrated improved performance, with AUCs of 0.842, 0.818, and 0.818 for the training, validation, and test cohorts, respectively.

|   Signature | Accuracy |   AUC |          95% CI | Sensitivity | Specificity |   PPV |   NPV | Survival | Cohort |
| ----------: | -------: | ----: | --------------: | ----------: | ----------: | ----: | ----: | -------: | ------ |
| Clinical-OS |    0.464 | 0.667 | 0.5794 - 0.7549 |       0.402 |       0.842 | 0.939 | 0.189 |   3Years | Train  |
|      MIL-OS |    0.843 | 0.836 | 0.7669 - 0.9045 |       0.869 |       0.684 | 0.943 | 0.464 |   3Years | Train  |
| Combined-OS |    0.809 | 0.882 | 0.8275 - 0.9368 |       0.803 |       0.842 | 0.968 | 0.416 |   3Years | Train  |
| Clinical-OS |    0.461 | 0.754 | 0.6117 - 0.8970 |       0.413 |       0.909 | 0.977 | 0.141 |   3Years | Val    |
|      MIL-OS |    0.470 | 0.792 | 0.6838 - 0.9001 |       0.413 |       1.000 | 1.000 | 0.153 |   3Years | Val    |
| Combined-OS |    0.739 | 0.825 | 0.6874 - 0.9629 |       0.731 |       0.818 | 0.974 | 0.243 |   3Years | Val    |
| Clinical-OS |    0.453 | 0.806 | 0.6390 - 0.9721 |       0.394 |       0.889 | 0.963 | 0.167 |   3Years | Test   |
|      MIL-OS |    0.440 | 0.747 | 0.5994 - 0.8938 |       0.379 |       0.889 | 0.962 | 0.163 |   3Years | Test   |
| Combined-OS |    0.733 | 0.778 | 0.6002 - 0.9554 |       0.712 |       0.889 | 0.979 | 0.296 |   3Years | Test   |

Table 5: OS Time-dependent ROC AUC for different signatures at 3 and 5 years.

<img src="img\3Years_OS_train_auc.svg" alt="" style="zoom: 33%;" /><img src="img\3Years_OS_val_auc.svg" alt="" style="zoom: 33%;" /><img src="img\3Years_OS_test_auc.svg" alt="" style="zoom: 33%;" />


Fig. 6: OS ROC Curves for Time-Dependent Analysis in Training , validation and Testing Cohorts

In the 3-year time-dependent ROC analysis for disease-free survival (DFS), the MIL-DFS model consistently outperformed the clinical model across all cohorts. In the training cohort, the MIL-DFS achieved an AUC of 0.773 (95% CI: 0.7042–0.8416), compared to 0.661 (95% CI: 0.5908–0.7310) for the clinical model. In the validation cohort, MIL-DFS reached an AUC of 0.792 (95% CI: 0.6730–0.9103), surpassing the clinical model's AUC of 0.631 (95% CI: 0.4862–0.7749). Similarly, in the test cohort, the MIL-DFS model achieved an AUC of 0.756 (95% CI: 0.6413–0.8706), while the clinical model obtained an AUC of 0.641 (95% CI: 0.4764–0.8057). The combined model further enhanced predictive performance, with AUCs of 0.823, 0.796, and 0.742 for the training, validation, and test cohorts, respectively.

For the 5-year time-dependent ROC analysis, MIL-DFS also showed superior performance. In the training cohort, MIL-DFS had an AUC of 0.780 (95% CI: 0.7029–0.8576), compared to the clinical model's 0.638 (95% CI: 0.5470–0.7289). In the validation cohort, MIL-DFS achieved an AUC of 0.757 (95% CI: 0.6184–0.8952), outperforming the clinical model's 0.677 (95% CI: 0.5335–0.8205). In the test cohort, MIL-DFS also demonstrated better performance with an AUC of 0.773 (95% CI: 0.5484–0.9978), compared to the clinical model's 0.588 (95% CI: 0.3443–0.8321). The combined model further improved AUC scores, achieving 0.814, 0.795, and 0.748 for the training, validation, and test cohorts, respectively.

In conclusion, the MIL model consistently demonstrated stronger performance in predicting DFS for both 3-year and 5-year survival, with the combined model offering the highest AUC values across all cohorts. This indicates that the integration of MIL features with clinical data significantly improves predictive accuracy for DFS outcomes.

|    Signature | Accuracy |   AUC |          95% CI | Sensitivity | Specificity |   PPV |   NPV | Survival | Cohort |
| -----------: | -------: | ----: | --------------: | ----------: | ----------: | ----: | ----: | -------: | ------ |
| Clinical-DFS |    0.624 | 0.661 | 0.5908 - 0.7310 |       0.595 |       0.699 | 0.837 | 0.398 |   3Years | Train  |
|      MIL-DFS |    0.768 | 0.773 | 0.7042 - 0.8416 |       0.821 |       0.630 | 0.852 | 0.575 |   3Years | Train  |
| Combined-DFS |    0.757 | 0.828 | 0.7698 - 0.8855 |       0.763 |       0.740 | 0.884 | 0.545 |   3Years | Train  |
| Clinical-DFS |    0.591 | 0.631 | 0.4862 - 0.7749 |       0.589 |       0.600 | 0.869 | 0.245 |   3Years | Val    |
|      MIL-DFS |    0.864 | 0.792 | 0.6730 - 0.9103 |       0.911 |       0.650 | 0.921 | 0.619 |   3Years | Val    |
| Combined-DFS |    0.864 | 0.793 | 0.6713 - 0.9153 |       0.922 |       0.600 | 0.912 | 0.632 |   3Years | Val    |
| Clinical-DFS |    0.563 | 0.641 | 0.4764 - 0.8057 |       0.554 |       0.600 | 0.838 | 0.265 |   3Years | Test   |
|      MIL-DFS |    0.634 | 0.756 | 0.6413 - 0.8706 |       0.554 |       0.933 | 0.969 | 0.359 |   3Years | Test   |
| Combined-DFS |    0.775 | 0.745 | 0.6034 - 0.8870 |       0.821 |       0.600 | 0.885 | 0.474 |   3Years | Test   |

Table 6: DFS Time-dependent ROC AUC for different signatures at 3 and 5 years.

<img src="img\3Years_DFS_train_auc.svg" alt="" style="zoom: 33%;" /><img src="img\3Years_DFS_val_auc.svg" alt="" style="zoom: 33%;" /><img src="img\3Years_DFS_test_auc.svg" alt="" style="zoom: 33%;" />


Fig. 6: ROC Curves for Time-Dependent Analysis in Training , validation and Testing Cohorts

## Supplementary

### 1A. 2.5D Deep Learning Training Process Details

**Transfer Learning**: We evaluated advanced architectures, including ResNet18, DenseNet201, and DenseNet121, to enhance the performance of traditional CNN models. Comparative analyses focused on key performance metrics to select the architecture best suited to our research objectives.

**Data Preparation**: We standardized the intensity distribution across RGB channels using Z-score normalization. For training, we applied real-time data augmentation techniques such as random cropping, and horizontal and vertical flipping, while test images underwent only normalization. Additionally, grayscale values were standardized using a min-max transformation, and images were resized to 224 × 224 pixels using nearest neighbor interpolation.

**Training Parameters**: To improve model generalization, we employed a cosine decay strategy to adjust the learning rate, given by the equation:
$$
\eta_t = \eta_{min}^i + \frac{1}{2}(\eta_{max}^i - \eta_{min}^i)(1 + \cos(\frac{T_{cur}}{T_i}\pi))
$$
where $\eta_{min}^i = 0$ and $\eta_{max}^i = 0.01$. Stochastic Gradient Descent (SGD) was used as the optimizer, and softmax cross-entropy was applied as the loss function.

### 1B. Multi-Instance Learning-Based Feature Fusion

Our multi-instance learning approach aimed to improve predictive accuracy by integrating data from multiple slices of each sample into a unified feature set, consisting of:

1. **Slice Prediction**: Each slice was processed by the deep learning model to generate prediction probabilities ($Slice_{prob}$) and predicted labels ($Slice_{pred}$), both rounded to two decimal places.

2. **Multi-Instance Learning Feature Aggregation**:

   - **Histogram Feature Aggregation**:
     - Prediction values were categorized into "bins" to count occurrences of each type.
     - Frequencies of $Slice_{prob}$ and $Slice_{pred}$ were tallied and normalized using min-max scaling, producing $Histo_{prob}$ and $Histo_{pred}$.

   - **Bag of Words (BoW) Feature Aggregation**:
     - A dictionary of unique elements in $Slice_{prob}$ and $Slice_{pred}$ was created.
     - Each slice was represented as a vector indicating the frequency of dictionary elements, with a TF-IDF transformation applied to emphasize key features.
     - This resulted in a BoW feature representation for each slice, capturing both the presence and significance of features.

3. **Feature Early Fusion**: We combined $Histo_{prob}$, $Histo_{pred}$, $Bow_{prob}$, and $Bow_{pred}$ using feature concatenation ($\oplus$), producing a unified feature vector:
   $$
   feature_{fusion} = Histo_{prob} \oplus Histo_{pred} \oplus Bow_{prob} \oplus Bow_{pred}
   $$
