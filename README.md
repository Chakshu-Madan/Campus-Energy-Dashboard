# Campus Energy-Use Dashboard

A Python-based data analysis project for monitoring and visualizing building-wise energy consumption across a university campus.

---

## 📌 Project Overview

The Campus Energy-Use Dashboard analyzes electricity consumption data collected from multiple buildings and generates:

- Consumption trends over time  
- Building-level summaries  
- Daily & weekly aggregates  
- Automated dashboard visualizations  
- Executive summary reports  

The goal is to support energy optimization and sustainability decisions on campus.

---

## 🏛️ Buildings Included

- Building A  
- Building B  
- Building C  

Each building contains a CSV file with timestamped kWh usage data.

---

## ⚙️ Features

### ✔ 1. Data Ingestion & Cleaning
- Loads multiple CSV files automatically  
- Validates required columns (`timestamp`, `kwh`)  
- Converts timestamps to datetime format  
- Removes invalid or missing rows  
- Merges all building data into a single DataFrame  

### ✔ 2. Aggregations
- Daily total energy usage  
- Weekly energy aggregates  
- Building-wise summary statistics (mean, min, max, total kWh)  

### ✔ 3. OOP-Based Design
- `MeterReading` dataclass for kWh readings  
- `Building` class to hold readings and create building-level reports  
- `BuildingManager` for campus-wide statistics  
- Computes:
  - Campus total energy consumption  
  - Highest-consuming building  
  - Peak load timestamp and value
  
### ✔ 4. Visual Dashboard
Generated using Matplotlib:

- Daily energy trend lines (per building)  
- Weekly average consumption bar chart  
- Timestamp vs kWh scatter plot 


---

## 📊 Sample Output Files

- **dashboard.png** — all visualizations  
- **cleaned_energy_data.csv** — processed dataset  
- **building_summary.csv** — per-building statistics  
- **summary.txt** — executive summary  

---
