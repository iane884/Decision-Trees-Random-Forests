from node import Node
import math
from parse import parse
import random

def calculate_entropy(examples):
    '''
    Find Entropy of Examples
    '''
    # This will hold KV pairs of Class:Count
    class_sum = {}
    for example in examples:
    # Sum the instances of each class
        class_sum[example['Class']] = class_sum.get(example['Class'], 0) + 1
  
    sum_entropy = 0
    for count in class_sum.values():
    # Determine probability of each class
        prob = count / len(examples)

    # Find total entropy of the examples
    sum_entropy += -1 * prob * math.log2(prob)

    return sum_entropy

def info_gain(examples, attribute):
    '''
    Determine information gained from an attribute
    '''
    original_entropy = calculate_entropy(examples)

  # Determine all possible versions of the attribute (blue, green, red...)
    attribute_possibilities = set(example[attribute] for example in examples)
    attribute_entropy = 0
  
    for possibility in attribute_possibilities:
    # Find all instances of when the specific attribute is the specific possibility
    # Like color == blue
        attribute_subset = [example for example in examples if example[attribute] == possibility]
 
    # weighted entropy
        proportion = len(attribute_subset) / len(examples)
        attribute_entropy += calculate_entropy(attribute_subset) * proportion

    gain = original_entropy - attribute_entropy

    return gain
    
def pick_best_attr(examples, attributes):
    # Find largest infromation gain
    max_gain = float('-inf')
    best_attrs = []
    for attribute in sorted(attributes):
        gain = info_gain(examples, attribute)
        if gain > max_gain:
            max_gain = gain
            best_attrs = [attribute]
        # pick random on tie? idk
        elif gain == max_gain:
           best_attrs.append(attribute)
    return random.choice(best_attrs)

# good default
def find_most_frequent_class(examples):
    #edge case handling
    if not examples:
        return 0

    # This will hold KV pairs of Class:Count
    class_sum = {}
    for example in examples:
    # Sum the instances of each class
        class_sum[example['Class']] = class_sum.get(example['Class'], 0) + 1

    max_count = max(class_sum, key=class_sum.get)
    return max_count


def ID3(examples, default):
    '''
    Takes in an array of examples, and returns a tree (an instance of Node) 
    trained on the examples.  Each example is a dictionary of attribute:value pairs,
    and the target class variable is a special attribute with the name "Class".
    Any missing attributes are denoted with a value of "?"
    '''

    root = Node()

    # Edge case- empty dataset
    if len(examples) == 0:
        root.label = default
        return root
    
    # Edge Case- all classes are the same
    if all(example['Class'] == examples[0]['Class'] for example in examples):
        root.label = examples[0]['Class']
        return root

    # Find the best attribute based on information gain
    attributes = [attribute for attribute in examples[0].keys() if attribute != 'Class']

    # Edge case - no attributes
    if not attributes:
        root.label = find_most_frequent_class(examples)
        return root
    
    # this iteratuion's node will be the best attribute at this iteration
    best_attribute = pick_best_attr(examples, attributes)
    root.label = best_attribute

    # What are the values this attribute can take? Example: y, n, ?
    attr_vals = set(example[best_attribute] for example in examples if example[best_attribute] != '?')

    for val in attr_vals:
        # Find the examples in which the best attribute takes on this value
        subset = [example for example in examples if example[best_attribute] == val]
        
        if not subset:
            child = Node()
            child.label = find_most_frequent_class(examples)
        else:
            # take out the curr best example
            new_examples = []
            for example in subset:
                new_example = {}
                for k, v in example.items():
                    if k != best_attribute:
                        new_example[k] = v
                new_examples.append(new_example)

            child = ID3(new_examples, find_most_frequent_class(subset))

        root.add_child(val, child)

    return root

def prune(node, examples):
    '''
    Takes in a trained tree and a validation set of examples.  Prunes nodes in order
    to improve accuracy on the validation data; the precise pruning strategy is up to you.
    '''

  # edge cases
    if not node.children:
        return node
  
    if not examples:
        return node

    for value in node.children.keys():
        child = node.children[value]
        reduced_examples = [example for example in examples if example[node.label]==value]
        node.children[value] = prune(child, reduced_examples)

    acc = test(node, examples)
    pruned = Node()
    pruned.label = find_most_frequent_class(examples)
    pruned_acc = test(pruned, examples)

    if pruned_acc >= acc:
        return pruned
    else:
        return node
   

def test(node, examples):
    '''
    Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
    of examples the tree classifies correctly).
    '''

    if not examples:
        return 0
  
    score = 0
  
    for example in examples:
        pred = evaluate(node, example)
        if pred == example['Class']:
            score = score + 1

    return score / len(examples)


def evaluate(node, example):
    '''
    Takes in a tree and one example. Returns the Class value that the tree
    assigns to the example.
    '''
    current = node
    
    while current.children:
        attr = current.label
        val = example.get(attr)
        
        # Handle ???
        if val == '?' or val not in current.children:
            child_classes = [child.label for child in current.children.values() if not child.children]
            if child_classes:
                return max(set(child_classes), key=child_classes.count)
            else:
                return current.label
        
        current = current.children[val]
    
    return current.label

def prune_tree(examples, default):
    root = ID3(examples, default)
    pruned_root = prune(root, examples)
    return pruned_root

def random_ID3(examples, default, n_attr):

    root = Node()

    # Edge case- empty dataset
    if len(examples) == 0:
        root.label = default
        return root
    
    # Edge Case- all classes are the same
    if all(example['Class'] == examples[0]['Class'] for example in examples):
        root.label = examples[0]['Class']
        return root

    # Find the best attribute based on information gain
    attributes = [attribute for attribute in examples[0].keys() if attribute != 'Class']

    # Edge case - no attributes
    if not attributes:
        root.label = find_most_frequent_class(examples)
        return root
    
    n_attr = min(n_attr, len(attributes))
    random_attributes = random.sample(attributes, n_attr)
    
    best_attribute = pick_best_attr(examples, random_attributes)
    root.label = best_attribute

    # What are the values this attribute can take? Example: y, n, ?
    attr_vals = set(example[best_attribute] for example in examples if example[best_attribute] != '?')

    for val in attr_vals:
        # Find the examples in which the best attribute takes on this value
        subset = [example for example in examples if example[best_attribute] == val]
        
        if not subset:
            child = Node()
            child.label = find_most_frequent_class(examples)
        else:
            # Create a new examples set without the best attribute
            new_examples = [{k: v for k, v in example.items() if k != best_attribute} for example in subset]
            child = random_ID3(new_examples, find_most_frequent_class(subset), n_attr)

        root.add_child(val, child)

    return root

def random_forest(examples, n_tree, n_attr):
    forest = []
    for i in range(n_tree):
        sample = random.choices(examples, k=len(examples))
        tree = random_ID3(sample, find_most_frequent_class(examples), n_attr)
        forest.append(tree)
    return forest

def evaluate_forest(forest, example):
  # take vote
    predictions = [evaluate(tree, example) for tree in forest]
    return max(set(predictions), key=predictions.count)

def test_forest(forest, examples):
  # get accuracy
    if not examples:
        return 0
  
    score = 0
  
    for example in examples:
        pred = evaluate_forest(forest, example)
        if pred == example['Class']:
            score = score + 1

    return score/len(examples)

house_file = 'house_votes_84.data'
tennis_file = 'tennis.data'
cars_file = 'cars_train.data'
candy_file = 'candy.data'

root_house = ID3(parse(house_file), 0)
root_cars = ID3(parse(cars_file), find_most_frequent_class(parse(cars_file)))
root_candy = ID3(parse(candy_file), 0)
root_tennis = ID3(parse(tennis_file), 0)
#print(test(root_tennis, parse(tennis_file)))
#print(test(prune_tree(parse(tennis_file), 0), parse(tennis_file)))

#print(test(root_house, parse(house_file)))
#print(test(prune_tree(parse(house_file), 0), parse(house_file)))

#print(test(root_candy, parse(candy_file)))
#print(test(prune_tree(parse(candy_file), 0), parse(candy_file)))

#print_decision_tree(root_cars)

#print(test(root_cars, parse('cars_test.data')))
#print(test(root_cars, parse('cars_valid.data')))
#print(test(root_cars, parse('cars_train.data')))

#print(test(prune_tree(parse(cars_file), '0'), parse('cars_valid.data')))
#print(test(prune_tree(parse(cars_file), 'unacc'), parse('cars_test.data')))
data = parse(house_file)
train_data = data[0:360]
test_data = data[360:]
##train_data = parse('cars_train.data')#data[:360]
#test_data = parse('cars_test.data')#data[360:]
#valid_data = parse('cars_valid.data')
    
    # tree
single_tree = ID3(train_data, find_most_frequent_class(train_data))
single_tree_accuracy = test(single_tree, test_data)
print(single_tree_accuracy)
pruned_tree = prune_tree(train_data, find_most_frequent_class(train_data))
pruned_tree_accuracy = test(pruned_tree, test_data)
    # Rf
n_trees = 100
k = max(1, int(math.log2(len(data[0]) - 1)))
forest = random_forest(train_data, n_trees, k)
forest_accuracy = test_forest(forest, test_data)

single_tree_correct = 0
forest_correct = 0
pruned_tree_correct = 0
for i, example in enumerate(test_data):
    single_tree_pred = evaluate(single_tree, example)
    pruned_tree_pred = evaluate(pruned_tree, example)
    forest_pred = evaluate_forest(forest, example)
    if single_tree_pred == example['Class']:
        single_tree_correct += 1
    if forest_pred == example['Class']:
        forest_correct += 1
    if pruned_tree_pred == example['Class']:
        pruned_tree_correct += 1
    
print(f"\nSingle Tree Correct: {single_tree_correct}/{len(test_data)}")
print(f"\nRandom Forest Correct: {forest_correct}/{len(test_data)}")
print(f"\nPruned Tree Correct: {pruned_tree_correct}/{len(test_data)}")


#print(f"\nCars Train Accuracy: {test(single_tree, train_data)}")
##print(f"\nCars Test Accuracy: {test(single_tree, test_data)}")
#print(f"\nCars Test Pruned Accuracy: {test(pruned_tree, test_data)}")
#print(f"\nCars Valid Accuracy: {test(single_tree, valid_data)}")
#print(f"\nCars Valid Pruned Accuracy: {test(pruned_tree, valid_data)}")
