import sys
import csv
import statistics
import numpy as np
import math

def read_data(csv_path):
    """Read in the input data from csv.
    
    Args:
        csv_path: the path to the input file.  The data file is assumed to have a header row, and
                the class value is found in the last column.
        standardize: flag to indicate whether the attributes (not including the class label) should 
                be transformed into z-scores before returning

    Returns: a list of lists containing the labeled training examples, with each row representing a 
        single example of the format [ attr1, attr2, ... , class_label ]
    """
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # header row
        features = []
        labels = []
        for row in reader:
            features.append([ float(r) for r in row[:-1] ])            
            labels.append(row[-1])
        examples = [ row + [labels[i]] for i, row in enumerate(features) ]
        return examples


def standardize(examples):
    """Transform data to use z-scores instead of raw values.  
        
    Args:
        examples: a list of lists containing the training examples, with each row representing a 
            single example of the format [ attr1, attr2, ... , class_label ]
    
    Returns: a list of lists containing the transformed training examples, with each row 
        representing a single example of the format [ zscore1, zscore2, ... , class_label ]

    See: https://en.wikipedia.org/wiki/Standard_score for more detail on z-scores.  N.B.: the last
        field each row is assumed to contain the class label and is not transformed!    
    """
    meanArr = []
    stdevArr = []
    exArr = np.array(examples).transpose()
    for row in exArr[:-1]:
        row = [float(x) for x in row.tolist()]
        stdevArr.append(statistics.stdev(row))
        meanArr.append(sum(row) / len(examples))
    for i in range(len(examples)):
        for j in range(13):
            examples[i][j] = (examples[i][j] - meanArr[j]) / stdevArr[j]  # Fscore calculation
    return examples  # Fix this line!

def learn_weights(examples):
    """Learn attribute weights for a multiclass perceptron.

    Args:
        examples: a list of lists containing the training examples, with each row representing a 
            single example of the format [ attr1, attr2, ... , class_label ]
                  
    Returns: a dictionary containing the weights for each attribute, for each class, that correctly
            classify the training data.  The keys of the dictionary should be the class values, and
            the values should be lists attribute weights in the order they appear in the data file.
            For example, if there are four attributes and three classes (1-3), the output might take
            the form:
                { 1 => [0.1, 0.8, 0.5, 0.01],
                  2 => [0.9, 0.01, 0.05, 0.4],
                  3 => [0.01, 0.2, 0.3, 0.85] }
    """

    weights = {'1':[[0]*13], '2':[[0]*13], '3':[[0]*13]}  # one set of weights for each class
    # print(len(weights['1']))
    # print(len(examples[0]))
    # weights = [[2] * 13]
    # weights[0][0] = 54
    # 
    # Fill this in!
    #
    one = np.array(weights['1'], dtype=float)
    two = np.array(weights['2'], dtype=float)
    three = np.array(weights['3'], dtype=float)
    minErrors = math.inf
    minErrorIt = 0
    for i in range(1000):  # If more than 1000 iters, return
        successCount = 0
        errorCount = 0
        for row in examples:
            npRow = np.array(row[0:13])
            # print(npRow)
            # break
            maxDP = np.dot(one, npRow)  # Set to first dot product
            maxClass = 1
            wineClass = 1
            for vector in [one, two, three]:
                dotProd = np.dot(vector, npRow)
                if dotProd > maxDP:
                    maxDP = dotProd
                    maxClass = wineClass
                wineClass+=1
            if row[13] == str(maxClass):
            #  Correct Classification
                successCount += 1
            else:
            #  Incorrect Classification
                errorCount += 1
                if maxClass == 1:   # Subtract from incorrect weight vector
                    one = np.subtract(one, npRow)
                elif maxClass == 2:
                    two = np.subtract(two, npRow)
                else:
                    three = np.subtract(three, npRow)
                if row[13] == '1':  # Add to correct weight vector
                    one = np.add(one, npRow)
                elif row[13] == '2':
                    two = np.add(two, npRow)
                else:
                    three = np.add(three, npRow)
            # print("Class 1: ", one)
            # print("Class 2: ", two)
            # print("Class 3: ", three)
        # print(errorCount)
        if errorCount == 0:
            # print("perfect")
            break   # Perfect class of all wines
        # print(npRow)
        # print(maxClass)
        # print(one)
        # break
        # print(errorCount)
        if errorCount < minErrors:
            minErrors = errorCount
            minErrorIt = i

    # print("Minimum Wines Classified Wrong:",minErrors,"(Iteration {})".format(minErrorIt))  # For questions
    # print("Final Wines Classified Wrong:",errorCount,"(Iteration {})".format(i))
    weights['1'] = one[0]
    weights['2'] = two[0]
    weights['3'] = three[0]

    return weights


def print_weights(class__weights):
    for c, wts in sorted(class__weights.items()):
        print("class {}: {}".format(c, ",".join([str(w) for w in wts])))




#############################################

if __name__ == '__main__':

    path_to_csv = "wine.csv"
    training_data = read_data(path_to_csv)

    class__weights = learn_weights(training_data)
    print_weights(class__weights)

    training_data = standardize(training_data)
    class__weights = learn_weights(training_data)
    print_weights(class__weights)







