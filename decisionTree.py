import csv
import numpy as np
import copy # based on python documentation on how to
# copy lists
import sys
import pickle
from scipy.io import arff



class Node:
    def __init__(self, children, splitFeature,
                 splitChildren, output, label = None, depth = 0):
        self.children = children
        self.splitFeature = splitFeature
        self.splitChildren = splitChildren
        self.output = output
        self.label = label
        self.depth = depth

# getCategoryProportions takes in a column of data and
# returns a dictionary of lists of indices paired with the
# proportion that the category shows up in the data
def getCategoryProportions(colData):
    categoryDict = dict()
    for i in range(len(colData)):
        if (categoryDict.get(colData[i]) is None):
            categoryDict[colData[i]] = [[i],1]
        else:
            indices = categoryDict[colData[i]][0]
            numPerClass = categoryDict[colData[i]][1]
            categoryDict[colData[i]] = [indices+[i],numPerClass+1]
    # classDict is a dictionary containing categories as keys
    # and the values are two-element lists where the first element is
    # the list of indices where that category exists and the second
    # element is the proportion of the category in the column of data
    outputClasses = []
    for key in categoryDict:
        outputClasses.append(key)
        categoryDict[key] = [categoryDict[key][0],
                     float(categoryDict[key][1])/len(colData)]
    return categoryDict, outputClasses

class DecisionTree(object):
    def __init__(self, trainFile, testFile, maxDepth, trainOutputLabels,
                 testOutputLabels, metricsOutput):
        (self.trainData,
        self.attributes,
        self.outputAttribute) = self.getData(trainFile)
        self.outputClasses = list(getCategoryProportions(\
                                self.trainData[self.outputAttribute])[0].keys())

        (self.testData,_,_) = self.getData(testFile) # testData presumed to
        # have same attributes as trainData.
        self.maxDepth = maxDepth
        self.trainOutputLabels = trainOutputLabels
        self.testOutputLabels = testOutputLabels
        self.metricsOutput = metricsOutput

    # getData takes in filename to turn the csv into a dictionary
    # containing column names containing the corresponding
    # column lists with the data.
    def getData(self,filename):
        #dataRows = csv.reader(open(filename), delimiter = "\t")
        dataframe,metadata = arff.loadarff(filename)
        data = dict()
        rowIndex = 0
        attributes = metadata._attrnames
        for att in attributes:
            data[att] = []
        for row in dataframe:
            for i in range(len(row)):
                data[attributes[i]].append(row[i])
        return data,attributes[:-1],attributes[-1]
    # majorityVote behaves similarly to majorityVote in inspection.py
    # It creates a dictionary and gets the category in the data
    # with the greatest count.
    def majorityVote(self,data):
        countDict = dict()
        for val in data:
            if (countDict.get(val) is None):
                countDict[val] = 1
            else:
                countDict[val] += 1
        maxCount = -1
        maxClass = None
        for key,count in countDict.items():
            if (count > maxCount):
                maxCount = count
                maxClass = key
            elif (count == maxCount):
                if (maxClass is not None and key > maxClass):
                    maxClass = key
        return maxClass
    # getEntropy behaves in the same way as the getEntropy function in
    # inspection.py but takes in a colData argument because the data
    # will gradually dwindle down as the tree is further split.
    # getEntropy is akin to H(Y) in the calculation of mutual information.
    # getEntropy uses numpy to take the log of base 2 of probabilities
    # - numpy np.log2 based on documentation
    def getEntropy(self,colData):
        classDict,_ = getCategoryProportions(colData)
        entropy = 0
        for key in classDict:
            prob = classDict[key][1] # proportion of class with name key
            if (prob != 0):
                entropy += prob*np.log2(prob)
        return -1*entropy
    # getCondEntropy takes in an attribute "X" and data dictionary
    # that is a dictionary of features (columns) associated with
    # lists that represent column data aand returns the conditional
    # entropy. It's the sum of the product between the proportion of
    # a category of attribute "X" and the entropy of output "Y" conditioned
    # on those associated with "X" values of only that category.
    def getCondEntropy(self,attribute,data):
        attributeCategories,_ = getCategoryProportions(data[attribute])
        condEntropy = 0
        outputCol = data[self.outputAttribute] # assuming output column
        # always being at the end
        condEntropy = 0
        for category in attributeCategories:
            indices = attributeCategories[category][0]
            outputCond = []
            for index in indices:
                outputCond.append(outputCol[index])
            entropyYCondX = self.getEntropy(outputCond)
            condEntropy += attributeCategories[category][1]*entropyYCondX
        return condEntropy
    # getMutualInformation takes an attribute and data dictionary
    # and returns the mutual information by definition I(Y;X)=H(Y)-H(Y|X)
    def getMutualInformation(self,attribute,data):
        return self.getEntropy(data[self.outputAttribute])-\
               self.getCondEntropy(attribute,data)
    # getPartition is responsible for the splitting process in the decision
    # tree.
    def getPartition(self,data,indices):
        partitionData = copy.deepcopy(data) # avoid aliasing while modifying
        for column in partitionData:
            newColData = []
            for index in indices: # extract only the indices that are
            # partitioned per column for the data
                newColData.append(partitionData[column][index])
            partitionData[column] = newColData
        return partitionData
    # makeDecisionTree calls makeDecisionTree' to modify
    # the self.decisionTree model that it builds up.
    def makeDecisionTree(self): # assuming input for inputData
        inputData = copy.deepcopy(self.trainData) # using python documentation
        # on shallow and deep copy operations - to modify self.trainData
        # reference

        # initializes the root
        root = Node(children=[],splitFeature=None,
                    splitChildren=[],
                    output=inputData[self.outputAttribute],
                    depth=0)
        # to later be modified
        self.decisionTree = self.makeDecisionTree_(root,inputData,0,
                                                   self.maxDepth)
    # makeDecisionTree' recursively calls itself to build the self.decisionTree
    def makeDecisionTree_(self,currNode,inputData,depth,maxDepth):
        outputCol = inputData[self.outputAttribute]
        outputClasses,_ = getCategoryProportions(outputCol)
        # No data
        if (len(outputCol) == 0):
            return Node(label=None,children=[],splitFeature=None,
                        splitChildren=[],
                        output=inputData[self.outputAttribute],
                        depth=depth)
        # One class
        if (len(outputClasses) == 1):
            return Node(label=list(outputClasses.keys())[0],
                        children=[],
                        splitFeature=None,
                        splitChildren=[],
                        output=inputData[self.outputAttribute],
                        depth=depth)
        # depth already at maximum
        if (depth >= maxDepth):
            return Node(label=self.majorityVote(outputCol),
                        children=[],
                        splitFeature=None,splitChildren=[],
                        output=inputData[self.outputAttribute],
                        depth=depth)
        else:
            # otherwise, split on feature that reduces uncertainty the
            # most. Highest mutual information.
            maxMutualInformation = 0
            bestAttribute = None
            for attribute in self.attributes:
                mutualInfo = self.getMutualInformation(attribute,inputData)
                if (mutualInfo > maxMutualInformation): # selecting the first
                # feature with "equal" mutual information to break ties
                    maxMutualInformation = mutualInfo
                    bestAttribute = attribute
            if (maxMutualInformation == 0):
                return Node(label=self.majorityVote(outputCol),
                            children=[],splitFeature=None,
                            splitChildren=[],
                            output=inputData[self.outputAttribute],
                            depth=depth)
            propsAndIndices,_ = \
                getCategoryProportions(inputData[bestAttribute])
            bestAttributeCategories = list(propsAndIndices.keys())
            splitChildrenList = []
            childrenList = []
            for i in range(len(bestAttributeCategories)):
                childrenList.append(None)
                splitChildrenList.append(bestAttributeCategories[i])
            currNode = Node(label=None,children=childrenList,
                            splitFeature=bestAttribute,
                            splitChildren=splitChildrenList,
                            output=inputData[self.outputAttribute],
                            depth = depth)
            for j in range(len(bestAttributeCategories)):
                currNode.children[j] = self.makeDecisionTree_(\
                                        currNode.children[j],
                                        self.getPartition(inputData,
                                  propsAndIndices[splitChildrenList[j]][0]),
                                  depth+1,maxDepth)
            return currNode
    # getClassCounts takes in a tree node and behaves as a helper
    # function for printTree in obtaining the number of observations
    # per category and returns the correct string to display in the
    # tree.
    def getClassCounts(self,node):
        categoryProps,_ = getCategoryProportions(node.output)
        numPerCategory = dict()
        row = "["
        for i in range(len(self.outputClasses)):
            if (self.outputClasses[i] in categoryProps):
                row += str(len(categoryProps[self.outputClasses[i]][0]))
            else:
                row += "0"
            row += " " + self.outputClasses[i] + \
                   ("/" if i < len(self.outputClasses)-1 else "]")
        return row
    # printTree copies the decision tree in the class and initially
    # prints the proportion
    def printTree(self):
        root = copy.deepcopy(self.decisionTree)
        counts = self.getClassCounts(root)
        print(counts)
        self.printTree_(root)
    def printTree_(self,root):
        for i in range(len(root.children)):
            row = "| "*(root.children[i].depth)
            row += (root.splitFeature + " = " + \
                    root.splitChildren[i] + ": " \
                    if root.splitFeature != None else "")
            row += self.getClassCounts(root.children[i])
            print(row)
            if (not self.isLeaf(root.children[i])):
                self.printTree_(root.children[i])
    # returns whether a tree node is a leaf: doesn't have any children
    def isLeaf(self,T):
        return T.children == []
    # plinkoDownModel takes in an observation and
    # classifies the observation by taking plinkoing
    # the observation down each branch that is associated
    # with a specific category that an attribute takes on
    def plinkoDownModel(self,observation,root):
        if (self.isLeaf(root)):
            return root.label, root.output
        category = observation[root.splitFeature]
        for i in range(len(root.children)):
            if root.splitChildren[i] == category:
                return self.plinkoDownModel(observation,root.children[i])
        return None, root.output
    # getErrorRate takes in two lists and returns the mismatch
    # rate between the corresponding elements
    def getErrorRate(self,L1,L2):
        mismatches = 0
        for i in range(len(L1)):
            if (L1[i] != L2[i]):
                mismatches += 1
        return float(mismatches)/len(L1)
    # writeToTrain is a helper function that getLabels uses to get the
    # training labels predicted. writeToTrain writes to a training output
    # file and returns a list of predicted training labels.
    def writeToTrain(self,write,columns,decisionModel):
        trainPredicted = []
        if write:
            trainOutputLabels = open(self.trainOutputLabels,"w")
        for i in range(len(self.trainData[columns[0]])):
            observation = dict()
            for key in self.trainData:
                observation[key] = self.trainData[key][i]
            outcome,output = self.plinkoDownModel(observation,decisionModel)
            if write:
                if outcome is not None:
                    trainOutputLabels.write(outcome)
                else:
                    trainOutputLabels.write(self.majorityVote(output))
            trainPredicted.append(outcome)
            if (write and i != len(self.trainData[columns[0]])-1):
                trainOutputLabels.write("\n")
        if write:
            trainOutputLabels.close()
        return trainPredicted
    # writeToTest is a helper function that getLabels calls and takes
    # in the argument columns, which are the columns of the input data.
    # the length of a column is iterated over (assuming an equal number
    # of rows per column). writeToTest returns a list of test labels
    # based off the decision tree predictions. It also writes to a test output
    # file that contains all the labels.
    def writeToTest(self,write,columns,decisionModel):
        testPredicted = []
        if write:
            testOutputLabels = open(self.testOutputLabels,"w")
        for j in range(len(self.testData[columns[0]])):
            observation = dict()
            for key in self.testData:
                observation[key] = self.testData[key][j]
            outcome,output = self.plinkoDownModel(observation,decisionModel)
            if write:
                if outcome is not None:
                    testOutputLabels.write(outcome)
                else:
                    testOutputLabels.write(self.majorityVote(output))
            testPredicted.append(outcome)
            if (write and j != len(self.testData[columns[0]])-1):
                testOutputLabels.write("\n")
        if write:
            testOutputLabels.close()
        return testPredicted
    # getLabels assumes that the makeDecisionTree function was already called
    # and "plinko"'s the data based on the attributes that the DecisionTree
    # split on - data could either be training or testing data.
    def getLabels(self,write,decisionModel):
        # main function writes to the metrics output file
        # and calls helper functions writeToTrain and writeToTest to
        # obtain the training and testing labels to calculate the error
        # rate.
        columns = list(self.trainData.keys())
        trainPredicted = self.writeToTrain(write,columns,decisionModel)
        testPredicted = self.writeToTest(write,columns,decisionModel)
        errorRateTrain = self.getErrorRate(trainPredicted,
                            self.trainData[self.outputAttribute])
        errorRateTest = self.getErrorRate(testPredicted,
                       self.testData[self.outputAttribute])
        # columns for test and training data
        if write:
            metricsOutput = open(self.metricsOutput,"w")
            metricsOutput.write("error(train): {}".format(errorRateTrain))
            metricsOutput.write("\n")
            metricsOutput.write("error(test): {}".format(errorRateTest))
            metricsOutput.close()
        return (errorRateTrain,errorRateTest)

def populateDict(decisionModel,d,reverseD,terminals):
    root = copy.deepcopy(decisionModel.decisionTree)
    top = root
    modifyDict(decisionModel,root,terminals,d,reverseD)
    return np.array([d, top, reverseD, terminals,decisionModel.outputAttribute,
                     decisionModel.trainData])
def modifyDict(decisionModel,root,terminals,d,reverseD):
    if (root.children == []):
        d[root] = []
    else:
        for i in range(len(root.children)):
            if d.get(root) is None:
                d[root] = [root.children[i]]
            else:
                d[root].append(root.children[i])
            if reverseD.get(root.children[i]) is None:
                reverseD[root.children[i]] = [root]
            else:
                reverseD[root.children[i]].append(root)
            if (not decisionModel.isLeaf(root.children[i])):
                modifyDict(decisionModel,root.children[i],terminals,d,reverseD)
            else:
                modifyDict(decisionModel,root.children[i],
                            terminals.append(root.children[i]),d,reverseD)


if __name__ == "__main__":
    trainFile = sys.argv[1]
    testFile = sys.argv[2]
    maxDepth = sys.argv[3]
    trainOutputLabels = sys.argv[4]
    testOutputLabels = sys.argv[5]
    metricsOutput = sys.argv[6]
    decisionTree = DecisionTree(trainFile,
                                testFile,
                                int(maxDepth),
                                trainOutputLabels,
                                testOutputLabels,metricsOutput)
    decisionTree.makeDecisionTree()
    terminals = []
    d = dict()
    reverseD = dict()
    tuple = populateDict(decisionTree,d,reverseD,terminals)
    pickle.dump(tuple, open('uciPhishGraph.p','wb'))
    #decisionTree.getLabels(True,copy.deepcopy(decisionTree.decisionTree))
    decisionTree.printTree()
