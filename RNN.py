import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import joblib
from sklearn.preprocessing import OneHotEncoder

with open("text_data.txt", "r") as file:
    data = file.read()


chars = list(set(data))
chars.append('<END>')
num_chars = len(chars)




hidden_size = 100  # Количество скрытых единиц в RNN
seq_length = 40  # Количество предыдущих символов для прогнозирования
epochs = 100
idx = 0


class RNNLayer:
    def __init__(self, hidden_size, seq_length, activation='tanh', learning_rate=0.001, type_layer="Many-to-Many", optimization=None):
        self.grad_input = None
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.activation = activation
        self.type_layer = type_layer
        self.optimization = optimization

        self.hs = np.zeros((self.seq_length, hidden_size))
        self.weights_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bias_h = np.zeros(hidden_size)
        self.weights_xh = None
        self.weights_hy = None
        self.bias_y = None

        self.output = []
        self.grad_weights_hh = np.zeros_like(self.weights_hh)
        self.grad_bias_h = np.zeros_like(self.bias_h)
        self.grad_h_next = np.zeros_like(self.hs[0])
        # self.beta1 = 0.9  # Adam parameter - exponential decay rate for the first moment estimates
        # self.beta2 = 0.999  # Adam parameter - exponential decay rate for the second moment estimates
        # self.epsilon = 1e-8  # Adam parameter - small value to avoid division by zero

        # Momentum variables
        # self.m_weights_xh = np.zeros((hidden_size, hidden_size))
        # self.v_weights_xh = np.zeros((hidden_size, hidden_size))
        # self.m_weights_hh = np.zeros((hidden_size, hidden_size))
        # self.v_weights_hh = np.zeros((hidden_size, hidden_size))
        # self.m_weights_hy = np.zeros((1, hidden_size))
        # self.v_weights_hy = np.zeros((1, hidden_size))
        # self.m_bias_h = np.zeros(hidden_size)
        # self.v_bias_h = np.zeros(hidden_size)
        # self.m_bias_y = np.zeros(1)
        # self.v_bias_y = np.zeros(1)

    def set_prev_layer(self, prev_layer):
        self.prev_layer = prev_layer

    def forward(self, inputs):
        if self.prev_layer is not None:
            self.weights_xh = self.prev_layer.weights_xh.copy()
            self.weights_hh = self.prev_layer.weights_hh.copy()
            self.weights_hy = self.prev_layer.weights_hy.copy()
            self.bias_h = self.prev_layer.bias_h.copy()
            self.bias_y = self.prev_layer.bias_y.copy()

        if self.type_layer == "Many-to-Many":
            # print('Many-to-Many')
            if self.weights_xh is None:
                # print('self.weights_xh is None')
                self.weights_xh = np.random.randn(self.hidden_size, inputs.shape[1]) * 0.01
                self.weights_hy = np.random.randn(inputs.shape[1], self.hidden_size) * 0.01
                self.bias_h = np.zeros(self.hidden_size)
                self.bias_y = np.zeros(inputs.shape[1])

            for t in range(self.seq_length):
                self.hs[t] = self.activation_func(np.dot(self.weights_xh, inputs[t]) + self.bias_h)
                y = np.dot(self.weights_hy, self.hs[t]) + self.bias_y
                self.output.append(y)
        else:
            # print('Many-to-One')
            if self.weights_xh is None:
                # print('self.weights_xh is None')
                self.weights_xh = np.random.randn(self.hidden_size, inputs.shape[0]) * 0.01
                self.weights_hh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
                self.weights_hy = np.random.randn(1, self.hidden_size) * 0.01
                self.bias_h = np.zeros(self.hidden_size)
                self.bias_y = np.zeros(1)
            for t in range(self.seq_length):
                self.hs[t] = self.activation_func(
                    np.dot(self.weights_xh, inputs[t]) + np.dot(self.weights_hh, self.hs[t - 1]) + self.bias_h)
                y = np.dot(self.weights_hy, self.hs[t]) + self.bias_y
                self.output = self.softmax(y)
        self.inputs = np.array(self.output)
        return np.array(self.output)

    def backward(self, grad_output):
        if self.activation:
            grad_output = self.activation_func_deriv(grad_output)

        grad_weights_xh = np.zeros_like(self.weights_xh)
        grad_weights_hh = np.zeros_like(self.weights_hh)
        grad_weights_hy = np.zeros_like(self.weights_hy)
        grad_bias_h = np.zeros_like(self.bias_h)
        grad_bias_y = np.zeros_like(self.bias_y)

        if self.type_layer == "Many-to-Many":
            grad_h = np.dot(grad_output, self.weights_hy)
            grad_h_next = np.zeros_like(grad_h[0])

            for t in reversed(range(self.seq_length)):
                grad_weights_hy += np.outer(grad_output[t], self.hs[t])
                grad_bias_y += grad_output[t]
                grad_h_total = grad_h[t] + grad_h_next
                grad_h_act = self.activation_func_deriv(self.hs[t]) * grad_h_total
                grad_weights_xh += np.outer(grad_h_act, self.inputs[t])
                grad_bias_h += grad_h_act
                grad_h_next = np.dot(self.weights_hh.T, grad_h_act)
            self.grad_input = grad_h_next

        else:  # Many-to-One
            self.output = self.softmax_deriv(self.output)
            grad_y = np.array(self.output)
            grad_weights_hy += np.outer(grad_y.T, self.hs[self.seq_length - 1])
            grad_bias_y += grad_y.sum(axis=0)
            grad_h = np.dot(grad_y, self.weights_hy.T)
            grad_h_next = np.zeros_like(grad_h[0])

            for t in reversed(range(self.seq_length)):
                grad_h_total = grad_h[t] + grad_h_next
                grad_h_act = self.activation_func_deriv(self.hs[t]) * grad_h_total
                grad_weights_xh += np.outer(grad_h_act, self.inputs[t])
                grad_bias_h += grad_h_act
                grad_h_next = np.dot(self.weights_hh.T, grad_h_act)
                # grad_h_next = self.softmax_deriv(grad_h_next)

        self.grad_weights_xh = grad_weights_xh
        self.grad_weights_hh = grad_weights_hh
        self.grad_weights_hy = grad_weights_hy
        self.grad_bias_h = grad_bias_h
        self.grad_bias_y = grad_bias_y
        self.clip_gradients()
        self.update_parameters()
        self.grad_input = grad_h_next

    def clip_gradients(self):
        # Clip gradients to mitigate the exploding gradients problem
        for grad_param in [self.grad_weights_xh, self.grad_weights_hh, self.grad_weights_hy,
                           self.grad_bias_h, self.grad_bias_y]:
            np.clip(grad_param, -3, 3, out=grad_param)

    def update_parameters(self):
        # if self.optimization == 'Adam':
        #     self.m_weights_xh = self.beta1 * self.m_weights_xh + (1 - self.beta1) * self.grad_weights_xh
        #     self.v_weights_xh = self.beta2 * self.v_weights_xh + (1 - self.beta2) * (self.grad_weights_xh ** 2)
        #     self.m_weights_hh = self.beta1 * self.m_weights_hh + (1 - self.beta1) * self.grad_weights_hh
        #     self.v_weights_hh = self.beta2 * self.v_weights_hh + (1 - self.beta2) * (self.grad_weights_hh ** 2)
        #     self.m_weights_hy = self.beta1 * self.m_weights_hy + (1 - self.beta1) * self.grad_weights_hy
        #     self.v_weights_hy = self.beta2 * self.v_weights_hy + (1 - self.beta2) * (self.grad_weights_hy ** 2)
        #     self.m_bias_h = self.beta1 * self.m_bias_h + (1 - self.beta1) * self.grad_bias_h
        #     self.v_bias_h = self.beta2 * self.v_bias_h + (1 - self.beta2) * (self.grad_bias_h ** 2)
        #     self.m_bias_y = self.beta1 * self.m_bias_y + (1 - self.beta1) * self.grad_bias_y
        #     self.v_bias_y = self.beta2 * self.v_bias_y + (1 - self.beta2) * (self.grad_bias_y ** 2)
        #
        ##     Bias correction
            # m_weights_xh_corrected = self.m_weights_xh / (1 - self.beta1)
            # v_weights_xh_corrected = self.v_weights_xh / (1 - self.beta2)
            # m_weights_hh_corrected = self.m_weights_hh / (1 - self.beta1)
            # v_weights_hh_corrected = self.v_weights_hh / (1 - self.beta2)
            # m_weights_hy_corrected = self.m_weights_hy / (1 - self.beta1)
            # v_weights_hy_corrected = self.v_weights_hy / (1 - self.beta2)
            # m_bias_h_corrected = self.m_bias_h / (1 - self.beta1)
            # v_bias_h_corrected = self.v_bias_h / (1 - self.beta2)
            # m_bias_y_corrected = self.m_bias_y / (1 - self.beta1)
            # v_bias_y_corrected = self.v_bias_y / (1 - self.beta2)
            #
            # self.weights_xh -= self.learning_rate * m_weights_xh_corrected / (
            #     np.sqrt(v_weights_xh_corrected) + self.epsilon)
            # self.weights_hh -= self.learning_rate * m_weights_hh_corrected / (
            #     np.sqrt(v_weights_hh_corrected) + self.epsilon)
            # self.weights_hy -= self.learning_rate * m_weights_hy_corrected / (
            #     np.sqrt(v_weights_hy_corrected) + self.epsilon)
            # self.bias_h -= self.learning_rate * m_bias_h_corrected / (np.sqrt(v_bias_h_corrected) + self.epsilon)
            # self.bias_y -= self.learning_rate * m_bias_y_corrected / (np.sqrt(v_bias_y_corrected) + self.epsilon)
        # else:
        self.weights_xh -= self.learning_rate * self.grad_weights_xh
        self.weights_hh -= self.learning_rate * self.grad_weights_hh
        self.weights_hy -= self.learning_rate * self.grad_weights_hy
        self.bias_h -= self.learning_rate * self.grad_bias_h
        self.bias_y -= self.learning_rate * self.grad_bias_y

    def activation_func(self, x):
        if self.activation == 'relu':
            return np.maximum(0.0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            return x

    def activation_func_deriv(self, x):
        if self.activation == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation == 'sigmoid':
            return x * (1 - x)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        else:
            return 1

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def softmax_deriv(self, x):
        softmax_output = self.softmax(x)
        grad = softmax_output * (1 - softmax_output)
        return grad

class RNNNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        if self.layers:
            prev_layer = self.layers[-1]
            layer.set_prev_layer(prev_layer)
        self.layers.append(layer)

    def forward(self, inputs, reset=True):
        if reset:
            self.layers[0].i = 0
        for layer in self.layers:
            inputs = np.array(inputs)
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            layer.backward(grad_output)
            grad_output = layer.grad_input  # передаем grad_input следующему слою

    def fit(self, X_train, y_train, data, chars, num_epochs, batch_size):
        num_batches = len(X_train) // batch_size
        for epoch in range(num_epochs):
            batch_losses = []
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                X_batch = np.array(X_batch)
                y_pred = self.forward(X_batch)
                grad_output = y_pred - y_batch
                self.backward(grad_output)  # передаем только grad_output
                batch_loss = -np.mean(y_batch * np.log(y_pred + 1e-8))
                batch_losses.append(batch_loss)
            average_loss = np.mean(batch_losses)
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}')
            if epoch % 3 == 0:
                self.evaluate(data, chars)

    def evaluate(self, data, chars):
        num_chars = len(chars)
        seq_length = self.layers[0].seq_length

        start_idx = np.random.randint(0, len(data) - seq_length)
        input_seq = data[start_idx:start_idx + seq_length]
        print(f'Input Sequence: {input_seq}')

        input_seq = np.array([np.eye(num_chars)[chars.index(c)] for c in input_seq])
        predicted_text = ""

        for _ in range(25):
            output = self.forward(input_seq)
            predicted_char = np.random.choice(len(chars), p=output)
            predicted_text += chars[predicted_char]
            if predicted_text[-1] == '.':
                break

            input_seq[:-1] = input_seq[1:]
            input_seq[-1] = np.eye(num_chars)[predicted_char]
            # print(f'Text: {"".join([chars[np.argmax(vec)] for vec in input_seq])}')

        print(f'Predicted Text: {predicted_text}')

    def loss(self, X_test, y_test):
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        total_loss = 0.0
        num_blocks = len(X_test) // self.layers[0].seq_length
        for block in range(num_blocks):
            start_idx = block * self.layers[0].seq_length
            end_idx = start_idx + self.layers[0].seq_length
            X_block = X_test[start_idx:end_idx]
            y_block = y_test[start_idx:end_idx]
            y_pred_block = self.forward(X_block)
            block_loss = -np.mean(y_block * np.log(y_pred_block + 1e-8))  # Cross-entropy loss
            total_loss += block_loss
        loss = total_loss / num_blocks
        return loss

    def save_model(self, file_path):
        model_params = {
            'layers': self.layers
        }
        with open(file_path, 'wb') as f:
            pickle.dump(model_params, f)

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as f:
            model_params = pickle.load(f)
        model = RNNNetwork()
        model.layers = model_params['layers']
        return model




# Создание экземпляра OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoder.fit(np.array(chars).reshape(-1, 1))

# Кодирование символов в формате one-hot
encoded_data = encoder.transform(np.array(list(data)).reshape(-1, 1))
print(encoded_data, encoded_data.shape)

# Добавление символа конца текста к закодированным данным
end_symbol = np.zeros((1, num_chars))
end_symbol[0, -1] = 1
encoded_data = np.concatenate([encoded_data, end_symbol.repeat(len(data), axis=0)], axis=0)


encoded_data_train = encoded_data[len(encoded_data)//5:]
encoded_data_test = encoded_data[:len(encoded_data)//5]

X_train = np.array(encoded_data_train[:-1])
y_train = np.array(encoded_data_train[1:])

X_test = np.array(encoded_data_test[:-1])
y_test = np.array(encoded_data_test[1:])



network = RNNNetwork()

layer1 = RNNLayer(hidden_size=72, seq_length=25, num_chars=num_chars, activation='tanh')
network.add_layer(layer1)
layer1.set_prev_layer(None)  # Set prev_layer attribute for the first layer

layer2 = RNNLayer(hidden_size=72, seq_length=25, num_chars=64, activation='tanh')
network.add_layer(layer2)
layer2.set_prev_layer(layer1)

layer3 = RNNLayer(hidden_size=72, seq_length=25, num_chars=num_chars, activation='tanh', type_layer="Many-to-One")
network.add_layer(layer3)
layer3.set_prev_layer(layer2)


print('Обучение модели')
network.fit(X_train, y_train, data, chars, num_epochs=epochs, batch_size=seq_length)

print('Оценка потерь модели на тестовых данных')
loss = network.loss(X_test, y_test)
print('Loss:', loss)

print('Преобразование обратно в исходный формат данных')
data = ''.join(encoder.inverse_transform(encoded_data).flatten())

char_to_idx = {char: i for i, char in enumerate(chars)}
print('char_to_idx', char_to_idx)
idx_to_char = {i: char for i, char in enumerate(chars)}
print('idx_to_char', idx_to_char)
data_size = len(data)
print(data_size)
