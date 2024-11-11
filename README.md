# Modeling with Decision Trees and Random Forests

## Project Overview
This project focuses on implementing the ID3 decision tree algorithm and constructing a random forest classifier. Using various datasets, the goal was to understand the performance of decision trees and how pruning and ensemble methods like random forests impact accuracy and overfitting.

## Steps Completed

### Node Data Structure Modification
I modified the Node data structure to add a `make_label` function for assigning labels and an `add_child` function for linking parent and child nodes. These changes improved code readability and ensured that the nodes could represent decision tree structures effectively.

### Handling Missing Attributes
I chose to ignore instances with missing attributes during training. While alternatives like replacing missing values with the most common attribute value were considered, this approach minimized bias and retained the dataset's integrity.

### Pruning Implementation
The pruning function recursively simplifies nodes by replacing them with a majority class label if it improves accuracy. This method helps reduce overfitting by removing unnecessary branches, which is especially beneficial when the training size is large enough to introduce noise.

### Learning Curves and Overfitting Analysis
Using the `house_votes_84.data` dataset, I plotted learning curves for pruned and unpruned decision trees. As the training size increased, both pruned and unpruned trees improved in accuracy until they began overfitting, where pruning provided better generalization by reducing noise from excess data.

### Decision Tree Performance on Cars Dataset
I trained the decision tree on `cars_train.data` and observed overfitting, achieving 100% accuracy on training but lower accuracy on validation and test sets. Pruning improved performance on validation and test sets by reducing overfitting.

### Random Forest Construction
I extended the ID3 algorithm to create a random forest by developing a `random_ID3` function to build trees with random attribute subsets. The `random_forest` function created an ensemble of decision trees, which were then evaluated on different datasets. Testing on `house_votes_84` and `candy.data`, the random forest outperformed single trees by reducing variance and improving generalizability. This validated the ensemble’s ability to reduce overfitting and provided a robust alternative to a single decision tree.

---
