#  Movie MLOps Lab: Data Drift Monitoring with Evidently AI

## Overview
This repository demonstrates how to apply modern MLOps tools to a movie-streaming scenario.  
We build a simple recommendation prototype, simulate user behavior, and use **Evidently AI** to monitor data drift in production-like conditions.

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
git clone https://github.com/your-username/movie-mlops-lab.git
cd movie-mlops-lab

### 2. create a virtual environment
'''bash
python3 -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows

### 3. Install dependencies
pip install -r requirements.txt

### 4. Generate synthetic data
python data_simulation.py


### 5. Train the baseline model
python train_baseline.py


### 6. Monitor drift
python monitor_drift.py

### 7. View the report
data_drift_report.html

### 8. (Optional) Quick drift alert
python quick_check.py
