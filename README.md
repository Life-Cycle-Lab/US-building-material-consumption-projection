# Material Consumption and Embodied Carbon Forecasting for the US Building Materials (2020–2050)

This repository contains two complementary methodologies, Jupyter Notebook files, and input/output data for modeling and projecting material consumption and embodied carbon emissions in U.S. construction, supporting climate scenario analysis from 2020 to 2050.

---

## Project Context

This modeling effort was developed as part of the Wedge Project, led by the Carbon Leadership Forum and in collaboration with Rocky Mountain Institute (RMI) and the University of Washington's Life Cycle Lab. The goal is to estimate and compare future demand for construction materials and their associated carbon impacts under a range of mitigation strategies. The work incorporates both historical consumption analysis (Approach 1) and a bottom-up, floor-area-based estimations (Approach 2) of building material consumption.

---

## Methodological Overview

### Approach 1 – SARIMA-Based Time Series Forecasting

Forecasts construction material use based on historical U.S. consumption data (1900–2020) from the US Geological Survey (USGS) and National Asphalt Pavement Association (NAPA) using SARIMA models.

**Outputs:**
- Projected material quantities (total and construction-only)
- A1–A3 embodied carbon emissions based on national GHG inventories

### Approach 2 – Material Intensity × Floor Area

Combines:
- Future building floor area projections (developed by RMI)
- CLF-derived (as part of the WBLCA Benchmark Study) material use intensity (kg/m²)
- Embodied carbon intensity data (kgCO2/m²)

**Outputs:**
- Material stock by building typology, material, and year
- GWP projections by material and strategy scenario

### Combined Approach – Reconciliation

Final forecasts reconcile overlaps between the two approaches. Combined outputs emphasize consistency and conservativeness by defaulting to the average estimate across methods.

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
| `approach1_construction_co2.xlsx`              | Approach 1: GHG emissions (A1–A3)         |
| `material_stock_projection_by_wedge_categories.csv` | Approach 2: material demand           |
| `carbon_stock_projection_by_wedge_categories.csv`   | Approach 2: GWP emissions              |
| `combined_approach_1_and_2.xlsx`               | Final harmonized wedge projections       |

---

## Contact

Milad Zokaei Ashtiani  
Research Scientist  
University of Washington  
[ashtiani@uw.edu](mailto:ashtiani@uw.edu)

---

## Citation

> AUTHORS (2025). *TITLE*. Carbon Leadership Forum.
