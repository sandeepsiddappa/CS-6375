import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Data Handling Class
class DataHandler:
    def __init__(self, file_path):
        self.dataframe = pd.read_csv(file_path)

    def clean_data(self):
        if self.dataframe.isnull().values.any():
            self.dataframe.dropna(inplace=True)

        if self.dataframe.duplicated().any():
            self.dataframe.drop_duplicates(inplace=True)

data_handler = DataHandler('Apple.csv')  # Change to your file path
data_handler.clean_data()
processed_data = data_handler.dataframe

closing_prices = processed_data['Close'].values.reshape(-1, 1)

# Normalizing Data
normalizer = MinMaxScaler(feature_range=(0, 1))
normalized_prices = normalizer.fit_transform(closing_prices)

# Function to Generate Sequences
def prepare_time_series(data, window_size):
    # Initialize empty lists to store subsequences and corresponding targets
    subsequences, next_values = [], []
    
    # Generate subsequences of the specified window size
    for idx in range(len(data) - window_size):
        # Slice the window of data
        window = data[idx: idx + window_size]
        # Get the next value in the sequence as the target
        next_val = data[idx + window_size]
        
        # Append the current window and next value to their respective lists
        subsequences.append(window)
        next_values.append(next_val)
    
    # Convert the lists to NumPy arrays for subsequent processing
    return np.array(subsequences), np.array(next_values)

# Configure the length of the time series window
time_series_length = 100

X, y = prepare_time_series(normalized_prices, time_series_length)


# Allocate portions of the data for training and testing purposes
train_ratio = 0.8  # 80% of the data will be used for training
split_index = int(len(X) * train_ratio)  # Determine the split index

# Splitting the Dataset
train_split = int(len(X) * 0.8)
training_features, testing_features = X[:train_split], X[train_split:]
training_targets, testing_targets = y[:train_split], y[train_split:]

# Activation Functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.square(z)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

# RNN Model
class RecurrentNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, activation_fn='tanh'):
        self.Wih = np.random.randn(hidden_dim, input_dim) * 0.01
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.Who = np.random.randn(output_dim, hidden_dim) * 0.01
        self.bh = np.zeros((hidden_dim, 1))
        self.bo = np.zeros((output_dim, 1))

        if activation_fn == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation_fn == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        else:
            self.activation = tanh
            self.activation_derivative = tanh_derivative

    def forward(self, inputs):
        hidden_states = {}
        hidden_states[-1] = np.zeros((self.Whh.shape[0], 1))
        for t in range(len(inputs)):
            xt = np.reshape(inputs[t], (self.Wih.shape[1], 1))
            hidden_states[t] = self.activation(np.dot(self.Wih, xt) + np.dot(self.Whh, hidden_states[t - 1]) + self.bh)
        output = np.dot(self.Who, hidden_states[len(inputs) - 1]) + self.bo
        return output, hidden_states

    def backward(self, inputs, hidden_states, output, target):
        dWih, dWhh, dWho = np.zeros_like(self.Wih), np.zeros_like(self.Whh), np.zeros_like(self.Who)
        dbh, dbo = np.zeros_like(self.bh), np.zeros_like(self.bo)

        doutput = output - target
        dWho += np.dot(doutput, hidden_states[len(inputs) - 1].T)
        dbo += doutput

        dhidden_next = np.zeros_like(hidden_states[0])
        for t in reversed(range(len(inputs))):
            dhidden = np.dot(self.Who.T, doutput) + dhidden_next
            dhidden_raw = self.activation_derivative(hidden_states[t]) * dhidden
            dbh += dhidden_raw
            dWih += np.dot(dhidden_raw, np.reshape(inputs[t], (1, -1)))
            dWhh += np.dot(dhidden_raw, hidden_states[t - 1].T)
            dhidden_next = np.dot(self.Whh.T, dhidden_raw)

        return dWih, dWhh, dWho, dbh, dbo

    def train(self, X, y, epochs, lr):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                input_seq, target = X[i], y[i]
                output, hidden_states = self.forward(input_seq)
                total_loss += np.sum((target - output) ** 2)

                gradients = self.backward(input_seq, hidden_states, output, target)
                self.Wih -= lr * gradients[0]
                self.Whh -= lr * gradients[1]
                self.Who -= lr * gradients[2]
                self.bh -= lr * gradients[3]
                self.bo -= lr * gradients[4]

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / len(X)}")

    def predict(self, X):
        predictions = []
        for x in X:
            output, _ = self.forward(x)
            predictions.append(output)
        return np.array(predictions)

# Training the Model
def evaluate_model(model, X, y, scaler):
    pred = model.predict(X).squeeze()
    pred_rescaled = scaler.inverse_transform(pred.reshape(-1, 1))
    y_rescaled = scaler.inverse_transform(y.reshape(-1, 1))
    mse = mean_squared_error(y_rescaled, pred_rescaled)
    r2 = r2_score(y_rescaled, pred_rescaled)
    rmse = np.sqrt(mse)
    return mse, r2, rmse, pred_rescaled

# Data Normalization and Sequencing
# Existing code for data normalization and sequence generation...

# Training and Hyperparameter Tuning
activation_funcs = ['tanh', 'sigmoid', 'relu']
hidden_sizes = [4, 8]
learning_rates = [0.001, 0.005]
epochs_choices = [30, 50]

performance_metrics = {}

for activation in activation_funcs:
    for hidden_size in hidden_sizes:
        for lr in learning_rates:
            for epochs in epochs_choices:
                print(f"\nTraining: Activation={activation}, Hidden Size={hidden_size}, Learning Rate={lr}, Epochs={epochs}")
                rnn = RecurrentNeuralNetwork(1, hidden_size, 1, activation)
                rnn.train(training_features, training_targets, epochs, lr)

               
                output = rnn.predict(testing_features)
                y_pred_reshaped = output.reshape(-1, 1)
                y_pred_rescaled = normalizer.inverse_transform(y_pred_reshaped)

                # Calculate metrics
                y_test_rescaled = normalizer.inverse_transform(testing_targets)
                mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
                r2 = r2_score(y_test_rescaled, y_pred_rescaled)
                rmse = np.sqrt(mse)
                performance_metrics[(activation, hidden_size, lr, epochs)] = (mse, r2, rmse, y_test_rescaled)

# Finding the Best Configuration
best_config = min(performance_metrics, key=lambda k: performance_metrics[k][0])
best_mse, best_r2, best_rmse, best_predictions = performance_metrics[best_config]

# Re-Training the Best Model
best_rnn = RecurrentNeuralNetwork(1, best_config[1], 1, best_config[0])
best_rnn.train(training_features, training_targets, best_config[3], best_config[2])

# Evaluation on Train and Test Sets
train_mse, train_r2, train_rmse, train_pred = evaluate_model(best_rnn, training_features, training_targets, normalizer)
test_mse, test_r2, test_rmse, test_pred = evaluate_model(best_rnn, testing_features, testing_targets, normalizer)

# Displaying Evaluation Metrics
print(f"Train MSE: {train_mse}, R-squared: {train_r2}, RMSE: {train_rmse}")
print(f"Test MSE: {test_mse}, R-squared: {test_r2}, RMSE: {test_rmse}")

# Plotting Results
plt.figure(figsize=(12, 6))

# Training Data Plot
plt.subplot(1, 2, 1)
plt.plot(normalizer.inverse_transform(training_targets), label='Actual Train Prices')
plt.plot(train_pred, label='Predicted Train Prices', alpha=0.7)
plt.title('Training Data: Actual vs Predicted')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()

# Test Data Plot
plt.subplot(1, 2, 2)
plt.plot(normalizer.inverse_transform(testing_targets), label='Actual Test Prices')
plt.plot(test_pred, label='Predicted Test Prices', alpha=0.7)
plt.title('Testing Data: Actual vs Predicted')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()

plt.tight_layout()
plt.show()