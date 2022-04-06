import pandas as pd
import pickle

def encode_data(race, diabetic):
	race_encoder     = pickle.load(open("utils/race_encoder.pkl", "rb"))
	diabetic_encoder = pickle.load(open("utils/diabetic_encoder.pkl", "rb"))

	encoded_race     = race_encoder.transform([race])[0]
	encoded_diabetic = diabetic_encoder.transform([diabetic])[0]

	return encoded_race, encoded_diabetic

def min_max_scaling(bmi, physical_health, mental_health, sleep_time):
	bmi_scaler             = pickle.load(open("utils/bmi_scaler.pkl", "rb"))
	physical_health_scaler = pickle.load(open("utils/physical_health_scaler.pkl", "rb")) 
	mental_health_scaler   = pickle.load(open("utils/mental_health_scaler.pkl", "rb")) 
	sleep_time_scaler      = pickle.load(open("utils/sleep_time_scaler.pkl", "rb"))  

	scaled_bmi             = bmi_scaler.transform([[bmi]])[0][0]
	scaled_physical_health = physical_health_scaler.transform([[physical_health]])[0][0]
	scaled_mental_health   = mental_health_scaler.transform([[mental_health]])[0][0]
	scaled_sleep_time      = sleep_time_scaler.transform([[sleep_time]])[0][0]

	return scaled_bmi, scaled_physical_health, scaled_mental_health, scaled_sleep_time

def load_model():
	file = open("utils/saved_model_lgbm.h5", "rb")
	model = pickle.load(file) 
	return model

def predict(sex, age, race, height, weight, smoking, alcohol, general_health, sleep_time, mental_health, physical_health, physical_activity, diff_walking, stroke, diabetic, asthma, skin_cancer, kidney_disease):
	
	gen_health_dict = {"Poor":0, "Fair":1, "Good":2, "Very good":3, "Excellent":4}
	age_category_dict = {"18-24":0, "25-29":1, "30-34":2, "35-39":3, "40-44":4, "45-49":5, "50-54":6, "55-59":7, "60-64":8, "65-69":9, "70-74":10, "75-79":11, "80 or older": 12}
	bmi               = int(weight) / ((int(height)/100) **2)
	print(">>>>> bmi", bmi)

	smoking           = 1 if smoking == "Yes"  else 0
	alcohol           = 1 if alcohol == "Yes"  else 0 
	stroke            = 1 if stroke == "Yes"  else 0
	diff_walking      = 1 if diff_walking == "Yes"  else 0
	physical_activity = 1 if physical_activity == "Yes"  else 0 
	asthma            = 1 if asthma == "Yes"  else 0
	kidney_disease    = 1 if kidney_disease == "Yes"  else 0
	skin_cancer       = 1 if skin_cancer == "Yes"  else 0
	sex               = 1 if sex == "Female"  else 0
	general_health    = gen_health_dict[general_health]
	age               = age_category_dict[age]
	race, diabetic    = encode_data(race, diabetic)
	bmi, physical_health, mental_health, sleep_time = min_max_scaling(bmi, physical_health, mental_health, sleep_time)

	df_dict = {"BMI": bmi, "Smoking":smoking, "AlcoholDrinking":alcohol, "Stroke": stroke, "PhysicalHealth": physical_health, "MentalHealth":mental_health, "DiffWalking":diff_walking, "Sex":sex, "AgeCategory":age, "Race":race, "Diabetic":diabetic, "PhysicalActiviy":physical_activity, "GenHealth":general_health, "SleepTime":sleep_time, "Asthma":asthma, "KidneyDisease":kidney_disease, "SkinCancer":skin_cancer}
	temp_df = pd.DataFrame(df_dict, index=[0])
	temp_df = temp_df.iloc[0]
	model = load_model()
	predict_proba = model.predict_proba([temp_df])[0][1] * 100

	return round(predict_proba, 2)