from sklearn.preprocessing import StandardScaler
import joblib

# Assuming X_train is your training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save the scaler
joblib.dump(scaler, 'models/scaler.pkl')

