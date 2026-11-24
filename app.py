import joblib, pandas as pd, streamlit as st
from pathlib import Path
st.set_page_config(page_title="Insurance Cost Predictor",page_icon="🏥")
st.title("Medical Insurance Cost Predictor")
st.write("Enter the details to estimate medical insurance charges.")
path=Path("models/best_insurance_model.joblib")
if not path.exists():
 st.warning("Model not found. Run `python train.py` first."); st.stop()
model=joblib.load(path)
with st.form("input"):
 age=st.number_input("Age",18,100,30)
 sex=st.selectbox("Sex",["male","female"])
 bmi=st.number_input("BMI",10.0,60.0,25.0)
 children=st.number_input("Children",0,10,0)
 smoker=st.selectbox("Smoker",["no","yes"])
 region=st.selectbox("Region",["northeast","northwest","southeast","southwest"])
 go=st.form_submit_button("Predict")
if go:
 x=pd.DataFrame([{"age":age,"sex":sex,"bmi":bmi,"children":children,"smoker":smoker,"region":region}])
 st.success(f"Estimated insurance charges: ${model.predict(x)[0]:,.2f}")
 st.caption("Educational estimate only; not an official insurance quote.")
