# Medical Insurance Cost Prediction — Full ML Project

## Dataset
Public CSV: https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv
Expected features: age, sex, bmi, children, smoker, region. Target: charges.

## Ubuntu setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Train and evaluate
```bash
python train.py
```
This cleans the data, splits it 75:25, applies median/mode imputation, standard scaling and one-hot encoding inside a pipeline, trains four regressors, compares MAE/MSE/RMSE/R2, and saves the lowest-RMSE model.

## Launch app
```bash
streamlit run app.py
```
## Files generated
- data/medical_insurance_raw.csv
- data/medical_insurance_cleaned.csv
- outputs/model_comparison.csv
- models/best_insurance_model.joblib
- models/model_metadata.json

Do not invent model scores. Report the values printed by your own run. This is a coursework demonstration, not a real insurance quote or underwriting tool.
