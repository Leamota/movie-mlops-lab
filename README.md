#  Movie MLOps Lab: Data Drift Monitoring with Evidently AI

## Overview
This repository demonstrates how to apply modern MLOps tools to a movie-streaming scenario.  
We build a simple recommendation prototype, simulate user behavior, and use **Evidently AI** to monitor data drift in production-like conditions.

---


## Strengths, Limitations, and Engineering Decisions

###  Strengths
- Synthetic data ensures reproducibility and avoids privacy concerns.  
- Evidently AI provides clear, interpretable drift reports.  
- Modular pipeline design makes it easy to extend into retraining or streaming workflows.  

###  Limitations
- Synthetic data may not fully capture the complexity of real‑world streaming platforms.  
- Baseline model is intentionally simple, so performance metrics are limited.  
- Current setup focuses on batch monitoring rather than real‑time drift detection.  

###  Engineering Decisions
- **Evidently AI** chosen for its strong visualization and reporting capabilities.  
- **Python scripts** used for simplicity and accessibility across environments.  
- **GitHub Pages** selected for publishing the blog to make results publicly accessible.  
- **Virtual environment setup** ensures reproducibility across different operating systems.  

---


## Goals
- Explore the ecosystem of tools for production ML systems.
- Show how data drift impacts recommendation quality.
- Demonstrate Evidently AI for monitoring and reporting.

## Repository Structure
movie_drift_demo/
├─ data_simulation.py
├─ train_baseline.py
├─ monitor_drift.py
├─ images/
│  └─ evidently_report.png
├─ utils.py
└─ requirements.txt
└─ README.md


## Setup Guide

Follow these steps to reproduce the experiment:

### 1. Clone the repository
```bash 
git clone https://github.com/Leamota/movie-mlops-lab.git
cd movie-mlops-lab 

```



### 2. create a virtual environment
```bash 
python3 -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows

```


### 3. Install dependencies
```bash
pip install -r requirements.txt

```


### 4. Generate synthetic data
```bash
python data_simulation.py

```

### 5. Train the baseline model
```bash
python train_baseline.py

```

### 6. Monitor drift
```bash
python monitor_drift.py

```

### 7. View the report
```bash
data_drift_report.html

```


### 8. (Optional) Quick drift alert
```bash
python quick_check.py

```
