# AHP-G: Analytic Hierarchy Process Gaussian

This repository contains the implementation of the **Gaussian AHP (AHP-G)** method, a Multi-Criteria Decision Making (MCDM) tool designed to rank alternatives based on multiple criteria.

## Description
The algorithm automates the selection of alternatives by using data variation to determine weights, reducing subjectivity and the cognitive effort required for exhaustive pairwise comparisons. The methodology is based on the study by **Santos, Costa, and Gomes (2021)** on the selection of warships for the Brazilian Navy.

## The Gaussian Factor
The main innovation of this approach is the use of the standard deviation ($\sigma$) and the mean ($\mu$) of the alternatives to calculate the **Gaussian Factor** ($GF$) for each criterion:

$$GF = \frac{\sigma}{\mu}$$

The higher the value of this factor, the greater the discriminatory power of the criterion in the final ranking.

## Features
* **Dual Compatibility**: Native support for **Pandas** and **PySpark** DataFrames.
* **Inverse Criteria Handling**: Management of variables where "higher is worse".
* **Automatic Normalization**: Data scaling to handle different units and negative values.
* **Sorted Output**: Generates a consolidated ranking of the best options.

## Requirements
* Python 3.x
* Pandas or PySpark
* Scikit-learn (`MinMaxScaler`)

## Usage Example
This example uses the data presented in section **5. Application of the AHP method** of the paper.

```python
import pandas as pd

# Define data based on the article's decision matrix
article_data = {
    'Model': ['Model 1', 'Model 2', 'Model 3'],
    'Action Radius (C1)': [4000, 9330, 10660],
    'Fuel Endurance (C2)': [11, 26, 30],
    'Autonomy (C3)': [30, 25, 35],
    'Primary Cannon (C4)': [25, 25, 120],
    'Secondary Cannon (C5)': [1, 2, 2],
    'AAW Missiles (C6)': [0, 1, 1],
    'Initial Cost (C7)': [290000000, 310000000, 310000000],
    'Life Cycle Cost (C8)': [592000000, 633000000, 633000000],
    'Construction Time (C9)': [6, 8, 8]
}

# Create dataframe
vessels_df = pd.DataFrame(article_data)

# Separate parameter lists for the function
all_criteria = vessels_df.columns[1:].tolist()
lower_is_better_criteria = ['Initial Cost (C7)', 'Life Cycle Cost (C8)', 'Construction Time (C9)']

# Call function from gaussian_ahp.py
result_df = gaussian_ahp(vessels_df, all_criteria, lower_is_better_criteria, sort=True)

print(result_df)
```

### Result

The generated ranking reflects the precision of the sensitivity analysis described in the study:

| Model | Action Radius (C1) | ... | Construction Time (C9) | **score_AHP_G** |
| :--- | :--- | :--- | :--- | :--- |
| **Model 3** | 10660 | ... | 8 | **0.514318** |
| **Model 2** | 9330 | ... | 8 | **0.339228** |
| **Model 1** | 4000 | ... | 6 | **0.146454** |

## Reference

SANTOS, M.; COSTA, I. P. A.; GOMES, C. F. S. Multicriteria decision-making in the selection of warships: a new approach to the AHP method. **International Journal of the Analytic Hierarchy Process**, v. 13, n. 1, 2021. DOI: [10.13033/ijahp.v13i1.833](https://doi.org/10.13033/ijahp.v13i1.833).

---


**Author**: Erik Yuri Dutzig
