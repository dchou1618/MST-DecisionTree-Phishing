'''
Dylan Chou
dvchou

Multi-Threaded Bayes Minimum Risk Post-Pruning Of Decision Trees:
Loss function - L_k - absolute difference between true class and
predicted class.

Probability - p_k - proportion of C_j in the current node.
Store:

Partitioned class output.
'''
import sys
import pickle
import time

import numpy as np
from functools import reduce
import multiprocessing as mp

'''
Takes in a dictionary representation of node: [list of node's neighbors],
Each node has only 1 parent.

'''

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
    return (categoryDict, outputClasses)

class Node:
    def __init__(self, children, splitFeature,
                 splitChildren, output, label = None, depth = 0):
        self.children = children
        self.splitFeature = splitFeature
        self.splitChildren = splitChildren
        self.output = output
        self.label = label
        self.depth = depth

class Graph(object):
    def __init__(self, dagList, parentList, root, terminals,outAttribute,trainData):
        self.root = root
        self.terminals = terminals
        self.dagList = dagList
        self.parentList = parentList
        self.outAttribute = outAttribute
        self.trainData = trainData
    def preprocessNodesBelow(self):
        nodeDict = dict()
        i = 0
        for node in self.dagList:
            lstNodes = []
            self.lookBelow(node,lstNodes)
            nodeDict[node] = set(lstNodes)
            i += 1
        return nodeDict
    def lookBelow(self, node, lstNodes):
        for child in self.dagList[node]:
            lstNodes.append(child)
            self.lookBelow(child, lstNodes)
    # destructively modifies the graph
    # we prune the graph over all training examples
    # when "turning" a parent into a leaf, the dagList[parent] = []
    def runParPruning(self):
        def modifyDict(node,catProps):
            catProps[node] = getCategoryProportions(node.output)
        currGraph = self.dagList
        parents = self.parentList
        currTerminals = set(self.terminals)
        currFrontier = currTerminals
        nodeBelowDict = self.preprocessNodesBelow()
        catPropsStart = time.time()
        self.catProps = dict()
        _ = [modifyDict(node,self.catProps) \
                     for node in self.dagList]
        print("categoryProportions done: {}".format(time.time()-catPropsStart))
        (currGraph, parents, currTerminals) = \
            self.parPruning(currGraph,
                            currFrontier,
                            parents,
                            currTerminals,
                            nodeBelowDict)
        return (currGraph, parents, currTerminals)
    # leaves should be kept track of.
    def getBayesRisk(self, i, node):
        risk = 0
        trueClass = self.trainData[self.outAttribute][i]
        probDict, outputClasses = self.catProps[node]
        for _,c in enumerate(outputClasses):
            if trueClass != c:
                risk += abs(float(trueClass)-float(c))*(probDict[c][1])
        return risk
    # when the root is reached, all leaves are considered.
    def parPruning(self,currGraph,currFrontier,parents, currTerminals, nodeBelowDict):
        def getRisks(i,node,leaves):
            return(self.getBayesRisk(i,node),\
                   sum(map(lambda leaf: self.getBayesRisk(i,leaf),leaves)))
        def reduceFunc(tuple1,tuple2):
            return (tuple1[0]+tuple2[0],tuple1[1]+tuple2[1])
        def riskFrontier(node, currTerminals,examples, nodeBelowDict):
            leaves = \
                list(nodeBelowDict[node].intersection(currTerminals))
            risksPerI = map(lambda i: getRisks(i,node,leaves),range(examples))
            (parentRisk,leavesRisk) = reduce(reduceFunc, risksPerI)
            if (parentRisk < leavesRisk):
                currGraph[node] = []
                currTerminals = currTerminals.difference(leaves)
                currTerminals.add(node)
        numRound = 0
        examples = len(self.trainData[list(self.trainData.keys())[0]])
        while (len(currFrontier) > 1):
            start = time.time()
            processes = [mp.Process(target=riskFrontier,
                 args=(node, currTerminals, examples, nodeBelowDict)) \
                         for node in currFrontier]
            for p in processes:
                p.daemon = True
                p.start()
            for p in processes:
                p.join()
            nextFrontier = set()
            # moving onto the next frontier from current frontier
            for node in currFrontier:
                if node == self.root:
                    parent = node
                else:
                    parent = parents[node][0]
                nextFrontier.add(parent)
            currFrontier = nextFrontier
            print("{} & {} & {}".format(numRound,\
                                                len(currFrontier),\
                                                time.time()-start))
            numRound += 1
        return (currGraph, parents, currTerminals)
def addToGraph(currGraph, node, d):
    d[node] = currGraph[node]
    for otherNode in currGraph[node]:
        addToGraph(currGraph, otherNode, d)
if __name__ == "__main__":
     print("Benchmarking...")
     if len(sys.argv) <= 1:
     	 print("Not enough arguments")
     else:
         with open(sys.argv[1],'rb') as f:
             graphData = pickle.load(f)
         dagList,root,parentList,terminals,outAttribute,trainData = graphData
         g  = Graph(dagList, parentList, root, terminals, outAttribute, trainData)
         print("Starting...")
         currGraph,_,currTerminals = g.runParPruning()
         newGraph = dict()
         addToGraph(currGraph, g.root, newGraph)
