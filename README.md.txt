# Apple Music Preference Predictor

## Project Overview
An end-to-end machine learning pipeline built in Python to predict whether a user will "Like" a song based on historical Apple Music listening habits. The project ingests raw JSON and CSV exports from Apple Music, processes the data, engineers temporal features, and trains an XGBoost classification model to predict user preferences.

## Data Source
The data for this project comes from personal Apple Music data exports, including:
* `Apple Music - Play History Daily Tracks.csv`
* `Apple Music - Top Content.csv`
* `Apple Music Library Tracks.json`

*Note: For privacy reasons, the raw data files and intermediate cleaned CSVs are not included in this repository.*

## Pipeline Structure

### 1. Data Cleaning (`Proj_Data_cleaning_V1.1.py`)
* Merges daily track history with library metadata using inner joins on Track Identifiers.
* Filters out duplicate track entries and handles missing values (e.g., filling null media types with 'AUDIO').
* Converts play durations from milliseconds to seconds and imputes anomalous/zero-duration plays with the column mean.

### 2. Feature Engineering (`Proj_Feat_engineering_V1.py`)
* **Multi-hot Encoding for Time of Day:** Parses the `Hours` column to engineer categorical features representing when a song was played (`Morning`, `Afternoon`, `Night`).
* **Target Variable Mapping:** Maps categorical string ratings ('Like', 'Dislike', 'Neutral') into a binary `Is_like` target variable.
* **Standardization:** Applies `StandardScaler` to continuous variables including `Track Year`, `Play Duration Seconds`, and temporal features.
* **Dummy Variables:** Uses `pd.get_dummies()` to handle categorical variables like `Genre`.

### 3. Modeling & Evaluation (`Proj_modeling_XGBoost_V1.py`)
* Splits the data into a stratified Train (80%), Test (10%), and Validation (10%) set to maintain class balance.
* Trains an XGBoost classifier to predict the `Is_like` target.
* Evaluates the model comprehensively using a confusion matrix and outputs detailed metrics including:
  * Accuracy
  * Sensitivity (True Positive Rate)
  * Specificity (True Negative Rate)
  * Precision
  * Negative Predictive Value
  * False Discovery Rate
  * ROC AUC Score
* Generates visualizations including an ROC curve and appends run statistics to a continuous `ProjectStats.csv` file.

## Setup & Installation

1. Clone this repository to your local machine.
2. Ensure you have Python installed.
3. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt