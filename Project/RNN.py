import pandas as pnd
import numpy as nmp
import matplotlib.pyplot as graph
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Data Handling Class
class DataHandler:
    def __init__(self, file_path):
        self.dataframe = pnd.read_csv(file_path)

    def clean_data(self):
        if self.dataframe.isnull().values.any():
            self.dataframe.dropna(inplace=True)

        if self.dataframe.duplicated().any():
            self.dataframe.drop_duplicates(inplace=True)

data_handler = DataHandler('https://raw.githubusercontent.com/sandeepsiddappa/CS-6375/main/Project/Apple.csv?token=GHSAT0AAAAAACK7I45DHM5M7Z4KP6B6ELW6ZLKU3LA')  
data_handler.clean_data()
processed_data = data_handler.dataframe

closing_prices = processed_data['Close'].values.reshape(-1, 1)

# Normalizing Data
data_normalize = MinMaxScaler(feature_range=(0, 1))
normalized_prices = data_normalize.fit_transform(closing_prices)

# Function to Generate Sequences
def prepare_time_series(data, window_size):
    subsequences, next_values = [], []
    
   
    for idx in range(len(data) - window_size):
        window = data[idx: idx + window_size]
        next_val = data[idx + window_size]
        subsequences.append(window)
        next_values.append(next_val)

    return nmp.array(subsequences), nmp.array(next_values)

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
    return 1 / (1 + nmp.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

def tanh(z):
    return nmp.tanh(z)

def tanh_derivative(z):
    return 1 - nmp.square(z)

def relu(z):
    return nmp.maximum(0, z)

def relu_derivative(z):
    return nmp.where(z > 0, 1, 0)

# RNN Model
class RecurrentNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, activation_fn='tanh'):
        self.Wih = nmp.random.randn(hidden_dim, input_dim) * 0.01
        self.Whhid = nmp.random.randn(hidden_dim, hidden_dim) * 0.01
        self.Who = nmp.random.randn(output_dim, hidden_dim) * 0.01
        self.bhid = nmp.zeros((hidden_dim, 1))
        self.bo = nmp.zeros((output_dim, 1))

#selecting the activation function
        if activation_fn == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation_fn == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        else:
            self.activation = tanh
            self.activation_derivative = tanh_derivative

#forward calculation function
    def forward(self, inputs):
        hidden_states = {}
        hidden_states[-1] = nmp.zeros((self.Whhid.shape[0], 1))
        for t in range(len(inputs)):
            xt = nmp.reshape(inputs[t], (self.Wih.shape[1], 1))
            hidden_states[t] = self.activation(nmp.dot(self.Wih, xt) + nmp.dot(self.Whhid, hidden_states[t - 1]) + self.bhid)
        output = nmp.dot(self.Who, hidden_states[len(inputs) - 1]) + self.bo
        return output, hidden_states

#backward calculations
    def backward(self, inputs, hidden_states, output, target):
        dWih, dWhhid, dWho = nmp.zeros_like(self.Wih), nmp.zeros_like(self.Whhid), nmp.zeros_like(self.Who)
        dbhid, dbo = nmp.zeros_like(self.bhid), nmp.zeros_like(self.bo)

        doutput = output - target
        dWho += nmp.dot(doutput, hidden_states[len(inputs) - 1].T)
        dbo += doutput

        dhidden_next = nmp.zeros_like(hidden_states[0])
        for t in reversed(range(len(inputs))):
            dhidden = nmp.dot(self.Who.T, doutput) + dhidden_next
            dhidden_raw = self.activation_derivative(hidden_states[t]) * dhidden
            dbhid += dhidden_raw
            dWih += nmp.dot(dhidden_raw, nmp.reshape(inputs[t], (1, -1)))
            dWhhid += nmp.dot(dhidden_raw, hidden_states[t - 1].T)
            dhidden_next = nmp.dot(self.Whhid.T, dhidden_raw)

        return dWih, dWhhid, dWho, dbhid, dbo

#train method
    def train(self, X, y, epochs, lr):
        loss_history = []
        for iter in range(epochs):
            total_loss = 0
            for k in range(len(X)):
                input_seq, target = X[k], y[k]
                output, hidden_states = self.forward(input_seq)
                total_loss += nmp.sum((target - output) ** 2)

                gradients = self.backward(input_seq, hidden_states, output, target)
                self.Wih -= lr * gradients[0]
                self.Whhid -= lr * gradients[1]
                self.Who -= lr * gradients[2]
                self.bhid -= lr * gradients[3]
                self.bo -= lr * gradients[4]

            epoch_loss = total_loss / len(X)
            loss_history.append(epoch_loss) 

            if iter % 10 == 0:
                print(f"Epoch {iter}, Loss: {epoch_loss}")

        return loss_history

#predict method
    def predict(self, X):
        return nmp.array(list(map(lambda x: self.forward(x)[0], X)))

# Training the Model
def evaluate_model(model, X, y, scaler):
    pred = model.predict(X).squeeze()
    pred_rescaled = scaler.inverse_transform(pred.reshape(-1, 1))
    y_rescaled = scaler.inverse_transform(y.reshape(-1, 1))
    mse = mean_squared_error(y_rescaled, pred_rescaled)
    r2 = r2_score(y_rescaled, pred_rescaled)
    rmse = nmp.sqrt(mse)
    return mse, r2, rmse, pred_rescaled


# Training and Hyperparameter Tuning
activation_funcs = ['tanh', 'sigmoid', 'relu']
hidden_sizes = [4, 8]
learning_rates = [0.0001, 0.005]
epochs_choices = [30, 80]

performance_metrics = {}
logs=[]
iteration=0
for activation in activation_funcs:
    for hidden_size in hidden_sizes:
        for lr in learning_rates:
            for epochs in epochs_choices:
                #Training the model with different parameters
                print(f"\nTraining: Activation={activation}, Hidden Size={hidden_size}, Learning Rate={lr}, Epochs={epochs}")
                rnn = RecurrentNeuralNetwork(1, hidden_size, 1, activation)
                rnn.train(training_features, training_targets, epochs, lr)

               
                output = rnn.predict(testing_features)
                y_pred_reshaped = output.reshape(-1, 1)
                y_pred_rescaled = data_normalize.inverse_transform(y_pred_reshaped)

                # Calculate metrics
                y_test_rescaled = data_normalize.inverse_transform(testing_targets)
                mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
                r2 = r2_score(y_test_rescaled, y_pred_rescaled)
                rmse = nmp.sqrt(mse)
                performance_metrics[(activation, hidden_size, lr, epochs)] = (mse, r2, rmse, y_test_rescaled)
                iteration=iteration+1
                log_entry = {
                    'Experiment': iteration,
                    'Activation': activation,
                    'Learning_Rate': lr,
                    'Max_Iterations': epochs,
                    'Hidden_Neurons': hidden_size,
                    'Train_Test Split': f"{int(train_ratio * 100)}:{int(100 - train_ratio * 100)}",
                    'MSE_Test': mse,
                    'RMSE_Test': rmse,
                    'R2_Test': r2
                }
                logs.append(log_entry)
 
log_df = pnd.DataFrame(logs)
log_df.to_csv('training_log.csv', index=False)

# Finding the Best Configuration
best_config = min(performance_metrics, key=lambda k: performance_metrics[k][0])
best_mse, best_r2, best_rmse, best_predictions = performance_metrics[best_config]

# Re-Training the Best Model
print("Retraining the best Model!!!!")
best_rnn = RecurrentNeuralNetwork(1, best_config[1], 1, best_config[0])
loss_history = best_rnn.train(training_features, training_targets, best_config[3], best_config[2])

# Evaluation on Train and Test Sets
train_mse, train_r2, train_rmse, train_pred = evaluate_model(best_rnn, training_features, training_targets, data_normalize)
test_mse, test_r2, test_rmse, test_pred = evaluate_model(best_rnn, testing_features, testing_targets, data_normalize)

# Displaying Evaluation Metrics
print(f"{'Metric':<15}{'Training':>10}{'Testing':>10}")
print(f"{'-'*35}")
print(f"{'MSE':<15}{train_mse:>10.4f}{test_mse:>10.4f}")
print(f"{'R-squared':<15}{train_r2:>10.4f}{test_r2:>10.4f}")
print(f"{'Root MSE':<15}{train_rmse:>10.4f}{test_rmse:>10.4f}")


# Plotting Results
graph.figure(figsize=(12, 6))

# Training Data Plot
graph.subplot(1, 2, 1)
graph.plot(data_normalize.inverse_transform(training_targets), label='Actual Train Prices')
graph.plot(train_pred, label='Predicted Train Prices', alpha=0.7)
graph.title('Training Data: Actual vs Predicted')
graph.xlabel('Days')
graph.ylabel('Stock Price')
graph.legend()

# Test Data Plot
graph.subplot(1, 2, 2)
graph.plot(data_normalize.inverse_transform(testing_targets), label='Actual Test Prices')
graph.plot(test_pred, label='Predicted Test Prices', alpha=0.7)
graph.title('Testing Data: Actual vs Predicted')
graph.xlabel('Days')
graph.ylabel('Stock Price')
graph.legend()

graph.tight_layout()
graph.show()

#Learning curve 
graph.figure(figsize=(10, 5))
graph.plot(loss_history, label='Training Loss')
graph.title('Learning Curve')
graph.xlabel('Epochs')
graph.ylabel('Loss')
graph.legend()
graph.show()