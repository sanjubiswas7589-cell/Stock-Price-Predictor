import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Download Apple stock data
stock = yf.download("AAPL", start="2020-01-01", end="2025-01-01")

# Keep only Close price
stock = stock[['Close']]

# Create prediction column
stock['Prediction'] = stock[['Close']].shift(-1)

# Remove last row
stock = stock[:-1]

X = stock[['Close']]
y = stock['Prediction']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = model.score(X_test, y_test)

print("Model Accuracy:", accuracy)

# Plot
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual")
plt.plot(predictions, label="Predicted")
plt.title("Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()

plt.savefig("prediction.png")
plt.show()

# Predict next day's price
last_price = stock[['Close']].tail(1)
next_day = model.predict(last_price)

print("Next Day Predicted Price:", next_day[0])
