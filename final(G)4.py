# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE

# Load dataset
file_path = "Depression Professional Dataset.csv"
data = pd.read_csv(file_path)

# Handle missing values
data = data.copy()  # Fix FutureWarning by working on a copy
data["Financial Stress"] = data["Financial Stress"].fillna(data["Financial Stress"].median())

# Encode categorical variables
label_encoders = {}
for col in data.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str).str.lower())  # Convert to lowercase for consistency
    label_encoders[col] = le  # Store encoders for later use

# Normalize numerical features
feature_columns = data.drop(columns=["Depression"]).columns
scaler = MinMaxScaler()
data[feature_columns] = scaler.fit_transform(data[feature_columns])

# Split dataset
X = data.drop(columns=["Depression"])
y = data["Depression"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Logistic Regression model
model = LogisticRegression(max_iter=500, C=1.0, solver='liblinear')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC curve

# Accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Define activity points (Gamification Feature)
activity_points = {
    "meditation": 10,
    "walking": 15,
    "exercise": 20,
    "healthy diet": 5
}

# Function to log user activities and calculate points
def log_user_activities():
    print("\n💡 Complete activities to earn points and improve your mental health!\n")
    
    total_points = 0
    activities_done = []

    for activity, points in activity_points.items():
        response = input(f"Did you do {activity}? (yes/no): ").strip().lower()
        if response == "yes":
            total_points += points
            activities_done.append(activity)
    
    # Save progress to a file
    save_progress(total_points, activities_done)
    
    print(f"\n🎉 You earned {total_points} points today! 🎉")
    if total_points >= 30:
        print("✅ Great job! Keep it up. 😊")
    else:
        print("⚠️ Try to complete more activities tomorrow for better well-being!")

# Function to save user progress
def save_progress(points, activities):
    try:
        df = pd.read_csv("user_progress.csv")  # Load existing data if available
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Day", "Points", "Activities"])
    
    new_entry = pd.DataFrame({"Day": [len(df) + 1], "Points": [points], "Activities": [", ".join(activities)]})
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv("user_progress.csv", index=False)

# Function to take user input and predict depression with probability percentage
def predict_depression():
    print("\n🔹 Enter the following details (case insensitive for text inputs):")
    user_data = []
    
    for col in feature_columns:
        value = input(f"{col}: ").strip()
        
        if col in label_encoders:  # Encode categorical input
            try:
                value = label_encoders[col].transform([value.lower()])[0]
            except ValueError:
                print(f"⚠️ Invalid input for {col}. Please enter a valid option.")
                return
        else:
            try:
                value = float(value)
            except ValueError:
                print(f"⚠️ Invalid numeric input for {col}. Please enter a valid number.")
                return
        
        user_data.append(value)
    
    # Scale input data
    user_data = np.array(user_data).reshape(1, -1)
    user_data = scaler.transform(user_data)
    
    # Predict probability
    probability = model.predict_proba(user_data)[0][1]  # Probability of depression
    percentage = round(probability * 100, 2)  # Convert to percentage

    # Display result
    if probability >= 0.5:
        print(f"\n🧠 Prediction: Depressed ({percentage}% likelihood) 🧠")
        risk_level = "⚠️ High Risk ⚠️" if percentage >= 80 else "🔸 Moderate Risk 🔸"
        print(f"\n{risk_level}: You have a {percentage}% chance of being depressed. Consider seeking support.")
    else:
        print(f"\n✅ Prediction: Not Depressed ({percentage}% likelihood) ✅")
        risk_level = "🟢 Low Risk 🟢" if percentage < 30 else "🟡 Mild Risk 🟡"
        print(f"\n{risk_level}: You have a {percentage}% chance of depression. Maintain a healthy lifestyle!")

    # Only suggest gamification if risk is Moderate (30-80%) or High (>80%)
    if percentage >= 30:
        print("\n💡 Since your risk level is moderate or high, let's try some positive activities.")
        log_user_activities()
    else:
        print("\n✅ You are at low risk! Keep maintaining a healthy lifestyle. 😊")

# Run the prediction function
predict_depression()
