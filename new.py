import numpy as np

# Good
#LEARNING_RATE = 0.01
#EPOCHS = 10
#THRESHOLD = 0.5

# Bad
LEARNING_RATE = .01
EPOCHS = 1000
THRESHOLD = 0.1

def load_data(filename):
    with open(filename, 'r') as f:
        data = f.readlines()

    features = []
    labels = []
    for line in data:
        lat, lon, label = line.strip().split('\t')
        features.append([float(lat), float(lon)])
        labels.append(label)

    unique_labels = list(set(labels))
    label_encoding = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = [label_encoding[label] for label in labels]

    features = np.array(features)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    normalized_features = (features - mean) / std

    return normalized_features, np.array(encoded_labels), unique_labels


def initialize_weights(num_features, num_classes):
    return np.random.rand(num_features, num_classes)


def perceptron_step(inputs, weights, threshold):
    activation = np.dot(inputs, weights)
    return activation > threshold


def train_perceptron(features, labels, epochs, threshold, learning_rate):
    num_features = features.shape[1]
    num_classes = len(set(labels))
    weights = initialize_weights(num_features, num_classes)

    for epoch in range(epochs):
        for x, y in zip(features, labels):
            prediction = perceptron_step(x, weights, threshold)
            for class_idx in range(num_classes):
                if y == class_idx and not prediction[class_idx]:
                    weights[:, class_idx] += learning_rate * x
                elif y != class_idx and prediction[class_idx]:
                    weights[:, class_idx] -= learning_rate * x
    return weights


def evaluate_model(weights, features, labels, unique_labels, threshold):
    predictions = np.array([perceptron_step(x, weights, threshold) for x in features])
    num_classes = len(unique_labels)
    stats = {}

    for i, class_name in enumerate(unique_labels):
        tp = np.sum((labels == i) & (predictions[:, i] == True))
        tn = np.sum((labels != i) & (predictions[:, i] == False))
        fp = np.sum((labels != i) & (predictions[:, i] == True))
        fn = np.sum((labels == i) & (predictions[:, i] == False))

        stats[class_name] = {
            "Correct": (tp + tn) / len(labels) * 100,
            "True Positives": tp / len(labels) * 100,
            "True Negatives": tn / len(labels) * 100,
            "False Positives": fp / len(labels) * 100,
            "False Negatives": fn / len(labels) * 100
        }
    return stats


def save_weights(weights, filename):
    with open(filename, 'w') as file:
        for neuron_weights in weights.T:
            weights_str = ','.join(map(str, neuron_weights))
            file.write(weights_str + '\n')


def load_weights(filename, num_features, num_classes):
    with open(filename, 'r') as file:
        lines = file.readlines()
        weights = [list(map(float, line.strip().split(','))) for line in lines]
        return np.array(weights).T


def main():
    trained_weights = None
    num_classes = 10
    unique_labels = None

    while True:
        print("\nPerceptron Neural Network")
        print("1. Load training data and train the model")
        print("2. Save weights to a file")
        print("3. Load initial weights from a file or initialize randomly")
        print("4. Open testing data and evaluate the model")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            filename = input("Enter the filename for the training data: ")
            features, labels, unique_labels = load_data(filename)
            print(features)
            print(labels)
            print(unique_labels)

            trained_weights = train_perceptron(features, labels, EPOCHS, THRESHOLD, LEARNING_RATE)
            print("Model trained successfully.")
        elif choice == '2':
            if trained_weights is not None:
                save_filename = input("Enter the filename to save the weights: ")
                save_weights(trained_weights, save_filename)
                print("Weights saved successfully.")
            else:
                print("No trained model found. Please train the model first.")
        elif choice == '3':
            sub_choice = input("Load weights from a file (y/n)? ")
            if sub_choice.lower() == 'y':
                weights_file = input("Enter the filename for the weights: ")
                trained_weights = load_weights(weights_file, 2, num_classes)  # Assuming 2 features (lat, lon)
                print("Weights loaded successfully.")
            else:
                trained_weights = initialize_weights(2, num_classes)  # Assuming 2 features (lat, lon)
                print("Weights initialized randomly.")

        elif choice == '4':
            if trained_weights is not None and unique_labels is not None:
                test_filename = input("Enter the filename for the testing data: ")
                test_features, test_labels, _ = load_data(test_filename)
                stats = evaluate_model(trained_weights, test_features, test_labels, unique_labels, THRESHOLD)

                for neuron, neuron_stats in stats.items():
                    print(f"Neuron: {neuron}")
                    for stat_name, stat_value in neuron_stats.items():
                        print(f"   {stat_name}: {stat_value:.2f}%")
            else:
                print("No trained model found. Please train the model first or load weights.")

        elif choice == '5':
            break
        else:
            print("Invalid choice. Please try again.")
    """
    # Load and train on training data
    train_features, train_labels, unique_labels = load_data("nnTrainData.txt")
    epochs = 10
    threshold = 0.50
    learning_rate = 0.01
    weights = train_perceptron(train_features, train_labels, epochs, threshold, learning_rate)

    # Load and evaluate on test data
    test_features, test_labels, _ = load_data("nnTestData.txt")
    stats = evaluate_model(weights, test_features, test_labels, unique_labels, threshold)

    # Print the results
    for neuron, neuron_stats in stats.items():
        print(f"Neuron: {neuron}")
        for stat_name, stat_value in neuron_stats.items():
            print(f"   {stat_name}: {stat_value:.2f}%")
    """


if __name__ == "__main__":
    main()
