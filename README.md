# PunchRecognition

Repo of the Sport Tech course project.

## Overview
This workspace contains:
- Data processing and labeling scripts in [Data/Analyzer](Data/Analyzer).
- Model training code in [Data/Model](Data/Model).
- A web dashboard in [dashboard/Code/web_app](dashboard/Code/web_app).

## Python scripts
### Analyzer
- [Data/Analyzer/bvhToAcc.py](Data/Analyzer/bvhToAcc.py): Parse BVH motion files and export accelerations to CSV.
- [Data/Analyzer/calculate_height.py](Data/Analyzer/calculate_height.py): Estimate subject height from skeletal data (just for testing purposes).
- [Data/Analyzer/downsample.py](Data/Analyzer/downsample.py): Downsample high-frequency data to a lower rate.
- [Data/Analyzer/labellingScript.py](Data/Analyzer/labellingScript.py): Assist in labeling punch segments in CSVs.
- [Data/Analyzer/visualizer.py](Data/Analyzer/visualizer.py): Visualize and inspect skeleton/acceleration time series.

### Model
- [Data/Model/ModelTraining.py](Data/Model/ModelTraining.py): Train deep learning models (including TCN) on labeled data.
- [Data/Model/ModelStatistics.py](Data/Model/ModelStatistics.py): Compute evaluation metrics and summarize results (just for testing purposes).
- [Data/Model/RandomForest.py](Data/Model/RandomForest.py): Train and evaluate multiple types of models (just for testing purposes).

## Install & Start
### Install dependencies
From [dashboard/Code/web_app](dashboard/Code/web_app):

```sh
cd dashboard/Code/web_app

# Install Python dependencies
pip install -r python/requirements.txt

# Install Node dependencies
npm install
```

### Start Web dashboard
From [dashboard/Code/web_app](dashboard/Code/web_app):

```sh
node server.js
```

Then open `http://localhost:3000`.

## Notes
- The web backend uses [`WebPunchAnalyzer`](dashboard/Code/web_app/python/analizer.py) for punch analysis (model.keras is used).
- Model training scripts are in [Data/Model/ModelTraining.py](Data/Model/ModelTraining.py) and [Data/Model/RandomForest.py](Data/Model/RandomForest.py).
