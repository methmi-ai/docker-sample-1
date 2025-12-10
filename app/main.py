from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI(title="Student Grade Predictor")

model = joblib.load("grade_model.pkl")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class StudyHours(BaseModel):
    hours: float

@app.get("/")
def home():
    return {"message": "Student Grade Predictor API"}

@app.post("/predict")
def predict(data: StudyHours):
    prediction = model.predict([[data.hours]])[0]
    return {"hours_studied": data.hours, "predicted_grade": round(prediction, 2)}
