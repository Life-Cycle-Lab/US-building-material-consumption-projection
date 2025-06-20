# Material Consumption Projections for the US Building Materials (2020–2050)

This repository contains two complementary methodologies, Jupyter Notebook files, and input/output data for modeling and projecting material consumption in U.S. construction, supporting climate scenario analysis from 2020 to 2050.

---

## Project Context

This modeling effort was developed as part of the Wedge Project, led by the Carbon Leadership Forum and in collaboration with Rocky Mountain Institute (RMI) and the University of Washington's Life Cycle Lab. The goal of the primary project is to estimate and compare future demand for construction materials and their associated carbon impacts under a range of mitigation strategies. The work incorporates both historical consumption analysis (Approach 1) and a bottom-up, floor-area-based estimations (Approach 2) of building material consumption.

---

## Methodological Overview

### Approach 1 – SARIMA-Based Time Series Forecasting

Forecasts construction material use based on historical U.S. consumption data (1900–2020) from the US Geological Survey (USGS) and National Asphalt Pavement Association (NAPA) using SARIMA models.

**Outputs:**
- Projected material quantities (total and construction-only)

![approach1](https://github.com/user-attachments/assets/9eba7491-dddd-4d8f-90ff-edd9813ef755)

### Approach 2 – Material Intensity × Floor Area

Combines:
- Future building floor area projections (developed by RMI)
- CLF-derived (as part of the WBLCA Benchmark Study) material use intensity (kg/m²)

**Outputs:**
- Material stock by building typology, material, and year

![approach2](https://github.com/user-attachments/assets/33d905e6-2cec-44fe-bc8f-4bf015b2a9cc)

### Combined Approach – Reconciliation

Final forecasts reconcile overlaps between the two approaches. Combined outputs emphasize consistency and conservativeness by defaulting to the average estimate across methods.

![approach1and2](https://github.com/user-attachments/assets/311f49f9-0d36-4d62-b5f1-f825b66f3d1f)
---

## Repository Layout

```plaintext
.
├── Approach 1/
│   ├── Input Data/
│   ├── Output Data/
│   └── material_intensity_wedge_final - Approach 1.ipynb
├── Approach 2/
│   ├── Input Data/
│   ├── Output Data/
│   └── material_intensity_wedge_final - Approach 2.ipynb
├── material_intensity_wedge_final - Approach 1 and 2.ipynb
├── combined_approach_1_and_2.xlsx
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Execute notebooks

Run the following notebooks in this order:

1. **Approach 1:** SARIMA-based forecasting
2. **Approach 2:** Bottom-up MUI × area
3. **Combined:** Merge and harmonize results

---

## Major Outputs

| File                                            | Description                              |
|-------------------------------------------------|------------------------------------------|
| `material_projections_best_fit.csv`            | Approach 1: material forecast             |
| `material_stock_projection_by_wedge_categories.csv` | Approach 2: material demand           |
| `combined_approach_1_and_2.xlsx`               | Final harmonized building material consumption projections       |


---

## References and Acknowledgments

This repository builds directly on the data processing and analysis developed in the following foundational repository:

> **Life-Cycle-Lab/wblca-benchmark-v2-material-use-embodied-carbon-intensity**  
> GitHub: [https://github.com/Life-Cycle-Lab/wblca-benchmark-v2-material-use-embodied-carbon-intensity](https://github.com/Life-Cycle-Lab/wblca-benchmark-v2-material-use-embodied-carbon-intensity)

That repository contributed significantly to this project by providing:

- Cleaned and standardized WBLCA datasets (Benchmark V2)
- Material use intensity values (kg/m²) across building typologies
- Harmonized project metadata and typology classification
- Embodied carbon intensity benchmarks by scope and category

We gratefully acknowledge the original authors and contributors for making this dataset and methodology openly available.

---

## Contact

Milad Zokaei Ashtiani  
Research Scientist  
University of Washington  
[ashtiani@uw.edu](mailto:ashtiani@uw.edu)

---

## Citation

> AUTHORS (2025). *TITLE*. Carbon Leadership Forum.
