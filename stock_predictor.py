import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# --- Step 1: Define tickers ---
tickers = ["AAPL"]

# --- Step 2: Download data ---
data = yf.download(tickers, start="2020-01-01", end="2024-12-31")['Close']

# pick AAPL as an example
stock = data['AAPL'].dropna()

# --- Step 3: Feature Engineering ---
df = pd.DataFrame(stock)
df['Return'] = df['AAPL'].pct_change()
df['MA5'] = df['AAPL'].rolling(window=5).mean()
df['MA10'] = df['AAPL'].rolling(window=10).mean()
df['Volatility'] = df['Return'].rolling(window=10).std()

# Drop NaN values
df = df.dropna()

# Target: 1 if tomorrow’s close > today’s, else 0
df['Target'] = (df['AAPL'].shift(-1) > df['AAPL']).astype(int)

# --- Step 4: Prepare dataset ---
X = df[['Return', 'MA5', 'MA10', 'Volatility']]
y = df['Target']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 5: Train model ---
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# --- Step 6: Evaluate ---
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# --- Step 7: Predict tomorrow ---
latest_features = X.iloc[-1:]
latest_scaled = scaler.transform(latest_features)
prediction = model.predict(latest_scaled)

print("Tomorrow’s prediction for AAPL:", "Up" if prediction[0] == 1 else "Down")


