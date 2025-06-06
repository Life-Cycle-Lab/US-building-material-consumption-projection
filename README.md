# Material Intensity and Embodied Carbon Forecasting (2020–2050)

This repository contains two complementary methodologies for modeling and projecting material consumption and embodied carbon emissions in U.S. construction, supporting climate scenario analysis from 2020 to 2050.

---

## Project Context

This modeling effort was developed as part of the Wedge Project, led by the Carbon Leadership Forum. The goal is to estimate and compare future demand for construction materials and their associated carbon impacts under a range of mitigation strategies. The work incorporates both historical consumption analysis and forward-looking, floor-area-based estimations of material demand.

---

## Methodological Overview

### 📈 Approach 1 – SARIMA-Based Time Series Forecasting

Forecasts construction material use based on historical U.S. consumption data (1900–2020) from the USGS using SARIMA models. Exogenous drivers include:
- GDP per capita
- Urbanization rates
- Population

**Outputs:**
- Projected material quantities (total and construction-only)
- A1–A3 embodied carbon emissions based on national GHG inventories

### 🏗 Approach 2 – Material Intensity × Floor Area

Combines:
- Future building floor area projections (RMI)
- CLF-derived material use intensity (kg/m²)
- GWP emissions factors

**Outputs:**
- Material stock by typology, material, and year
- GWP projections by material and strategy scenario

### 🔗 Combined Approach – Reconciliation

Final forecasts reconcile overlaps between the two approaches. Combined outputs emphasize consistency and conservativeness by defaulting to the median estimate across methods.

---

## Strategic Scenarios

Each scenario modifies model parameters to reflect varying mitigation pathways:

| ID  | Name              | Focus                                    |
|-----|-------------------|-------------------------------------------|
| S1  | Balanced          | Moderate and proportional mitigation      |
| S2  | Slow Start        | Delayed action with rapid later reduction |
| S3  | Best Case         | Aggressive mitigation across all sectors  |
| S4  | Energy Transition | Clean energy and electrification focus    |
| S5  | R&D               | Advanced material innovation              |
| S6  | Circularity       | High rates of reuse, recycling            |
| S7  | Design Innovation | Lightweight and alternative design        |

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

## Outputs

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
University of Washington – Carbon Leadership Forum  
📧 [ashtiani@uw.edu](mailto:ashtiani@uw.edu)

---

## Citation

> Ashtiani, M.Z., et al. (2025). *Material Use and Embodied Carbon Intensity of New Construction Buildings in North America*. University of Washington.
