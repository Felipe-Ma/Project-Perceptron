import numpy as np
import pandas as pd
import logging
import json

# Log to the console
logging.basicConfig(level=logging.INFO)


def unit_step_func(x):
    return np.where(x > 0, 1, 0)


class SingleLayerNN:
    def __init__(self, input_size, output_size, learning_rate=0.01, n_iters=10000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = np.zeros((input_size, output_size))
        self.bias = np.zeros(output_size)

    def fit(self, X, y):
        n_samples, _ = X.shape

        y_encoded = np.zeros((n_samples, self.bias.size))
        y_encoded[np.arange(n_samples), y] = 1

        for _ in range(self.n_iters):
            linear_output = np.dot(X, self.weights) + self.bias
            y_predicted = self.activation_func(linear_output)

            # Update rule
            self.weights += self.lr * np.dot(X.T, (y_encoded - y_predicted))
            self.bias += self.lr * np.sum(y_encoded - y_predicted, axis=0)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return np.argmax(y_predicted, axis=1)

    def activation_func(self, x):
        return np.where(x > 0, 1, 0)

    def save_weights(self, filename):
        model_params = {
            "weights": self.weights.tolist(),  # Converting numpy array to list
            "bias": self.bias.tolist()
        }
        with open(filename, 'w') as f:
            json.dump(model_params, f)

    def load_weights(self, filename):
        with open(filename, 'r') as f:
            model_params = json.load(f)
        self.weights = np.array(model_params["weights"])
        self.bias = np.array(model_params["bias"])

    def evaluate(self, X_test, y_test):
        linear_output = np.dot(X_test, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)

        # Convert y_test to one-hot encoding
        n_samples = y_test.size
        y_test_encoded = np.zeros((n_samples, self.bias.size))
        y_test_encoded[np.arange(n_samples), y_test] = 1

        # Calculate statistics
        correct_classification = (np.sum(y_predicted == y_test_encoded, axis=1) == self.bias.size)
        multiple_neurons_fired = np.sum(y_predicted, axis=1) > 1
        zero_neurons_fired = np.sum(y_predicted, axis=1) == 0

        stats = {
            "perfect_classification": np.mean(correct_classification) * 100,
            "multiple_neurons_fired": np.mean(multiple_neurons_fired) * 100,
            "zero_neurons_fired": np.mean(zero_neurons_fired) * 100
        }

        # Individual neuron statistics
        neuron_stats = []
        for i in range(self.bias.size):
            neuron_fired = y_predicted[:, i] == 1
            correct_fire = (neuron_fired & (y_test == i))
            incorrect_fire = (neuron_fired & (y_test != i))
            missed_fire = (~neuron_fired & (y_test == i))

            neuron_stats.append({
                "neuron": i,
                "correct_fire": np.mean(correct_fire) * 100,
                "incorrect_fire": np.mean(incorrect_fire) * 100,
                "missed_fire": np.mean(missed_fire) * 100
            })

        return stats, neuron_stats


def test_network(model, data, label_map):
    X_test = data[['latitude', 'longitude']].values
    y_test = data['label'].values
    predictions = model.predict(X_test)

    # Initialize counters
    correct_predictions = 0
    multiple_firings = 0
    no_firings = 0
    class_stats = {label: {'correct': 0, 'false_positive': 0, 'false_negative': 0} for label in label_map}

    for i, (pred, actual) in enumerate(zip(predictions, y_test)):
        if pred == actual:
            correct_predictions += 1
            class_stats[label_map.inverse[actual]]['correct'] += 1
        else:
            class_stats[label_map.inverse[actual]]['false_negative'] += 1
            class_stats[label_map.inverse[pred]]['false_positive'] += 1

        if sum(pred) > 1:
            multiple_firings += 1
        elif sum(pred) == 0:
            no_firings += 1

    total = len(y_test)
    print(f"Percentage of perfectly classified examples: {correct_predictions / total * 100:.2f}%")
    print(f"Percentage of examples causing multiple neurons to fire: {multiple_firings / total * 100:.2f}%")
    print(f"Percentage of examples causing zero neurons to fire: {no_firings / total * 100:.2f}%")
    for label, stats in class_stats.items():
        print(f"Neuron: {label}")
        for stat, count in stats.items():
            print(f"   {stat.title()}: {count / total * 100:.2f}%")



def load_data(filename):
    data = pd.read_csv(filename, sep='\t', header=None, names=['latitude', 'longitude', 'label'])
    labels = data['label'].unique()
    print(labels)
    print(data['label'].value_counts())

    label_map = {label: i for i, label in enumerate(labels)}

    data['label'] = data['label'].map(label_map)
    return data, label_map

def normalize_data(data):
    data['latitude'] = (data['latitude'] - data['latitude'].mean()) / data['latitude'].std()
    data['longitude'] = (data['longitude'] - data['longitude'].mean()) / data['longitude'].std()
    return data


def main_menu():
    logging.info("Welcome to the Single Layer Neural Network Classifier!")
    model = None
    label_map = None

    while True:
        print("\n1. Train model\n2. Save weights\n3. Load weights\n4. Test model\n5. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            filename = input("Enter the training data filename: ")
            data, label_map = load_data(filename)
            data = normalize_data(data)
            X = data[['latitude', 'longitude']].values
            y = data['label'].values
            model = SingleLayerNN(input_size=2, output_size=len(label_map))
            model.fit(X, y)
            logging.info("Model trained successfully!")
        elif choice == "2":
            if model:
                save_filename = input("Enter the filename to save the model: ")
                model.save_weights(save_filename)
                logging.info("Model saved successfully!")
            else:
                logging.info("No model trained!")
        elif choice == "3":
            filename = input("Enter the filename to load weights: ")
            if not model:
                logging.info("Initialize the model first by training it.")
            else:
                model.load_weights(filename)
                logging.info(f"Weights loaded from {filename}")

        elif choice == "4":
            if not model:
                print("Please train or load a model first.")
            else:
                test_filename = input("Enter the testing data filename: ")
                test_data, label_map = load_data(test_filename)
                #print(label_map)
                #print(test_data)
                test_data = normalize_data(test_data)
                X_test = test_data[['latitude', 'longitude']].values
                y_test = test_data['label'].values

                stats, neuron_stats = model.evaluate(X_test, y_test)

                print("\nModel Evaluation Statistics:")
                print(f"Perfect Classification: {stats['perfect_classification']:.2f}%")
                print(f"Multiple Neurons Fired: {stats['multiple_neurons_fired']:.2f}%")
                print(f"Zero Neurons Fired: {stats['zero_neurons_fired']:.2f}%")

                print("\nNeuron-wise Statistics:")
                for neuron_stat in neuron_stats:
                    print(f"Neuron {neuron_stat['neuron']}:")
                    print(f"  Correctly Fired: {neuron_stat['correct_fire']:.2f}%")
                    print(f"  Fired Incorrectly: {neuron_stat['incorrect_fire']:.2f}%")
                    print(f"  Missed Fire: {neuron_stat['missed_fire']:.2f}%")

        elif choice == "5":
            break




if __name__ == "__main__":
    main_menu()