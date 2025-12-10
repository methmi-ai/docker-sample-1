import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def train_model():
    data = pd.read_csv("grades.csv")
    X = data[['hours_studied']]
    y = data['final_grade']

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, "grade_model.pkl")
    print(" Model trained and saved successfully!")

if __name__ == "__main__":
    train_model()
