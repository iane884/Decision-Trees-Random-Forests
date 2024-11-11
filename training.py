import ID3
import parse
import random
import matplotlib.pyplot as plt

# Function to run 100 random runs for given training size, with or without pruning
def run_experiment(data, train_sizes, with_pruning=True):
    average_accuracies = []

    for train_size in train_sizes:
        accuracies = []

        for _ in range(100):
            # Shuffle data and split into training and testing
            random.shuffle(data)
            train = data[:train_size]
            test = data[train_size:]

            # Train the decision tree
            tree = ID3.ID3(train, 'democrat')

            # Prune the tree if pruning is enabled
            if with_pruning:
                validation = data[train_size//2:3*train_size//4]
                ID3.prune(tree, validation)

            # Test the tree on the remaining examples
            accuracy = ID3.test(tree, test)
            accuracies.append(accuracy)

        # Calculate average accuracy for this training size
        avg_accuracy = sum(accuracies) / len(accuracies)
        average_accuracies.append(avg_accuracy)
        print(f"Train size {train_size}, Average accuracy: {avg_accuracy:.4f}")

    return average_accuracies

# Plot the learning curves
def plot_learning_curves(train_sizes, acc_with_pruning, acc_without_pruning):
    plt.plot(train_sizes, acc_with_pruning, label="With Pruning")
    plt.plot(train_sizes, acc_without_pruning, label="Without Pruning")
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Accuracy on Test Data")
    plt.title("Accuracy vs. Training Examples")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function to run the experiment
def main():
    # Load the dataset
    data = parse.parse('house_votes_84.data')

    # Training sizes to experiment with
    train_sizes = list(range(10, 301))

    # Run the experiment with pruning
    print("Running experiment with pruning...")
    acc_with_pruning = run_experiment(data, train_sizes, with_pruning=True)

    # Run the experiment without pruning
    print("Running experiment without pruning...")
    acc_without_pruning = run_experiment(data, train_sizes, with_pruning=False)

    # Plot the learning curves
    plot_learning_curves(train_sizes, acc_with_pruning, acc_without_pruning)

if __name__ == "__main__":
    main()
