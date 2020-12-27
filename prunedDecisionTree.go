package main

import (
  "fmt"
  "github.com/bsm/arff"
  "strconv"
  "strings"
  "math"
  "math/rand"
  "time"
  "os"
)


type Label struct  {
  label int
}

type SplitFeature struct {
  splitfeature string
}

type Node struct {
  label *Label
  children []*Node
  splitFeature *SplitFeature
  splitChildren []int
  output []int
  depth int
}

type PrunedTree struct {
  trainData map[string][]int
  features []string
  label string
  testData map[string][]int
  maxDepth int
  decisionTree *Node
}

type Class struct {
  classLabel int;
}

type Pair struct {
  indices []int;
  stat float64;
}

type Risk struct {
  risk float64
}


func randIndices(count int, numTests int) []int {
  indices := []int{}
	for i := 0; i < count; i++ {
	  indices = append(indices,-1)
	}
	iter := 0
	tails := 0
	rand.Seed(time.Now().UnixNano())
	for {
	  if tails == numTests {
	    break
	  }
	  coinFlip := rand.Intn(2)
	  if coinFlip == 0 {
	    if indices[iter%count] == -1 {
	      tails += 1
	      indices[iter%count] = 0
	    }
	  }
	  iter += 1
	}
  return indices
}

/* reference:
   https://github.com/bsm/arff
*/
func (tree *PrunedTree) readFile (filename string, propTest float64, maxDepth int) {
  file, err := arff.Open(filename)
  if err != nil {
		panic("Bad read: " + err.Error())
	}
  data := make(map[string][]int);
  numAttributes := len(file.Attributes)
  attributes := []string{}
  defer file.Close()

  count := 0
	for file.Next() {
    for i := 0; i < numAttributes; i++ {
      feature := file.Attributes[i].Name
      val, err := strconv.Atoi(file.Row().Values[i].(string))
      if count  == 0 {
        attributes = append(attributes, feature)
      }
      if err == nil {
        data[feature] = append(data[feature], val)
      } else {
        panic(err.Error())
      }
    }
    count += 1
	}
	if err := file.Err(); err != nil {
		panic("Failed to read file: " + err.Error())
	}
  numTests := int(float64(count)*propTest)
  indices := randIndices(count, numTests)
  trainData := make(map[string][]int)
  testData := make(map[string][]int)
  for feature,_ := range data {
    for i, val := range data[feature] {
      if indices[i] == 0 {
        testData[feature] = append(testData[feature], val)
      } else {
        trainData[feature] = append(trainData[feature],val)
      }
    }
	}
  tree.trainData = trainData
  tree.testData = testData
  tree.features = attributes[:len(attributes)-1]
  tree.label = attributes[len(attributes)-1]
  tree.maxDepth = maxDepth
}

/* majorityVote takes in the data map */

func majorityVote(data []int) int {
  counts := make(map[int]int)
  for i := 0; i < len(data); i++ {
    counts[data[i]] = counts[data[i]] + 1
  }
  max := math.Inf(-1)
  var maxClass *Class
  for class, count := range counts {
    currCount := float64(count)
    if currCount > max {
      max = currCount
      maxClass = &Class{classLabel:class}
    } else if currCount == max {
      if maxClass != nil && class > maxClass.classLabel {
        maxClass = &Class{classLabel: class}
      }
    }
  }
  if maxClass != nil {
    return maxClass.classLabel
  } else {
    panic("Max Class cannot be nil")
  }
}

/* categorical proportions  */

func categoryProp (colData []int) (map[int]Pair, []int) {
  catDict := make(map[int]Pair)
  for i := 0; i < len(colData); i++ {
    if _, b := catDict[colData[i]]; !b {
      indexSingleton := []int{}
      indexSingleton = append(indexSingleton, i)
      catDict[colData[i]] = Pair{indices: indexSingleton, stat: float64(1)}
    } else {
      catDict[colData[i]] = Pair{indices: append(catDict[colData[i]].indices,i),
                                   stat: catDict[colData[i]].stat + 1}
    }
  }
  outputClasses := []int{}
  for key, _ := range catDict {
    outputClasses = append(outputClasses, key)
    catDict[key] = Pair{indices: catDict[key].indices,
                        stat: float64(catDict[key].stat)/float64(len(colData))}
  }
  return catDict, outputClasses
}

/* getEntropy */

func getEntropy (colData []int) float64 {
  catDict,_ := categoryProp(colData)
  var entropy float64
  for _, val := range catDict {
    prob := val.stat
    if prob != 0 {
      entropy += prob * math.Log2(prob)
    }
  }
  return -1 * entropy
}

/* condEntropy */

func (tree *PrunedTree) condEntropy (attribute string, data map[string][]int) float64 {
  featureCategories,_ := categoryProp(data[attribute])
  outputCol := data[tree.label]
  var condEntropy float64
  for category, _ := range featureCategories {
    indices := featureCategories[category].indices
    outputCond := []int{}
    for _,index := range indices {
      outputCond = append(outputCond, outputCol[index])
    }
    entropyYCondX := getEntropy(outputCond)
    condEntropy += featureCategories[category].stat*entropyYCondX
  }
  return condEntropy
}

/* mutualInfo */

func (tree *PrunedTree) mutualInfo (attribute string, data map[string][]int) float64 {
  return getEntropy(data[tree.label])-tree.condEntropy(attribute, data)
}

/* after splitting, we partition dataset */

func getPartition (data map[string][]int, indices []int) map[string][]int {
  partitionData := make(map[string][]int)
  for column,_ := range data {
      newColData := []int{}
      for _,index := range indices {
          newColData = append(newColData, data[column][index])
      }
      partitionData[column] = newColData
  }
  return partitionData
}

/* mkTree and mkTree_ */

func (tree *PrunedTree) mkTree_(root *Node, inputData map[string][]int, depth int, maxDepth int) *Node {
  outputCol := inputData[tree.label]
  outputDict, outputClasses := categoryProp(outputCol)
  if len(outputCol) == 0 {
    return &Node{label: nil, children: []*Node{},
                 splitFeature: nil, splitChildren: []int{},
                 output: inputData[tree.label],
                 depth: depth}
  }
  if len(outputDict) == 1 {
    return &Node{label: &Label{outputClasses[0]},
                 children: []*Node{},
                 splitFeature: nil,
                 splitChildren: []int{},
                 output: inputData[tree.label],
                 depth: depth}
  }
  if (depth >= maxDepth) {
    return &Node{label: &Label{majorityVote(outputCol)},
                children: []*Node{},
                splitFeature: nil,
                splitChildren: []int{},
                output: inputData[tree.label],
                depth: depth}
  } else {
      var maxMutualInformation float64 = 0.0
      var bestAttribute string
      for _,attribute := range tree.features {
          mutualInfo := tree.mutualInfo(attribute,inputData)
          if (mutualInfo > maxMutualInformation)  {
              maxMutualInformation = mutualInfo
              bestAttribute = attribute
          }
      }
      if (maxMutualInformation == 0.0) {
          return &Node{label: &Label{majorityVote(outputCol)},
                      children: []*Node{},
                      splitFeature: nil,
                      splitChildren: []int{},
                      output: inputData[tree.label],
                      depth: depth}
      }
      propsAndIndices, bestAttributeCategories := categoryProp(inputData[bestAttribute])
      splitChildrenList := []int{}
      childrenList := []*Node{}
      for i,_ := range bestAttributeCategories {
        childrenList = append(childrenList, nil)
        splitChildrenList = append(splitChildrenList,bestAttributeCategories[i])
      }
      currNode := &Node{label: nil,
                      children: childrenList,
                      splitFeature: &SplitFeature{bestAttribute},
                      splitChildren: splitChildrenList,
                      output: inputData[tree.label],
                      depth: depth}
      for j,_ := range bestAttributeCategories {
          currNode.children[j] = tree.mkTree_(currNode.children[j],
                              getPartition(inputData,
                              propsAndIndices[splitChildrenList[j]].indices),
                              depth+1,maxDepth)
      }
      return currNode
  }
}

func (tree *PrunedTree) mkTree() {
  inputData := make(map[string][]int)
  for key,_ := range tree.trainData {
    dataCol := make([]int, len(tree.trainData[key]))
    copy(dataCol,tree.trainData[key])
    inputData[key] = dataCol
  }
  root := &Node{label: nil, children: []*Node{}, splitFeature: nil,
                splitChildren: []int{}, output: inputData[tree.label],
                depth: 0}
  tree.decisionTree = tree.mkTree_(root,inputData, 0, tree.maxDepth)
}

func getClassCounts(node *Node) string {
  categoryProps, outputClasses := categoryProp(node.output)
  row := "["
  for i,_ := range outputClasses {
      if pair, b := categoryProps[outputClasses[i]]; b {
          row += strconv.Itoa(len(pair.indices))
      } else {
          row += "0"
      }
      addOn := ""
      if i < len(outputClasses)-1 {
        addOn = " / "
      } else {
        addOn = "]"
      }
      row += " " + strconv.Itoa(outputClasses[i]) + addOn
  }
  return row
}
func (tree *PrunedTree) printTree() {
    root := &Node{}
    *root = *(tree.decisionTree)
    counts := getClassCounts(root)
    fmt.Println(counts)
    printTree_(root)
}
func printTree_(root *Node) {
  if root != nil {
    for i,_ := range root.children {
        row := strings.Repeat("| ", root.children[i].depth)
        if root.splitFeature != nil {
          row += (root.splitFeature.splitfeature + " = " +
                  strconv.Itoa(root.splitChildren[i]) + ": ")
        }
        row += getClassCounts(root.children[i])
        fmt.Println(row)
        if !isLeaf(root.children[i]) {
          printTree_(root.children[i])
        }
    }
  }
}
func isLeaf(T *Node) bool {
  return len(T.children) == 0
}
func plinkoDownModel(show bool, observation map[string]int, root *Node) (*Label, []int) {
  if (isLeaf(root)) {
      return root.label, root.output
  }
  category := observation[root.splitFeature.splitfeature]
  if show {
    fmt.Println(root.splitFeature.splitfeature)
    fmt.Println(category, root.splitChildren, root.children)
  }
  for i,_ := range root.children {
      if root.splitChildren[i] == category {
          return plinkoDownModel(show, observation,root.children[i])
      }
  }
  return nil, root.output
}

func errorRate(L1 []int, L2 []int) float64 {
  mismatches := 0
  for i,_ := range L1 {
      if (L1[i] != L2[i]) {
          mismatches += 1
      }
  }
  return float64(mismatches)/float64(len(L1))
}

func check(err error) {
  if err != nil {
      panic(err.Error())
  }
}

/* post-processing calculations after training */
func (tree *PrunedTree) getBayesRisk(observation int, node *Node) float64 {
  var risk float64 = 0.0
  trueClass := tree.trainData[tree.label][observation]
  probDict, outputClasses := categoryProp(node.output)
  for _,class := range outputClasses {
      if trueClass != class {
        risk += math.Abs(float64(trueClass-class))*(probDict[class].stat)
      }
  }
  return risk
}

func (tree *PrunedTree) riskLeavesOverTree(observation int,
root *Node, r *Risk) {
  if root != nil {
    if len(root.children) == 0 {
      r.risk += tree.getBayesRisk(observation, root)
    } else {
      for _,child := range root.children {
        tree.riskLeavesOverTree(observation, child, r)
      }
    }
  }
}


func (tree *PrunedTree) pruneTree_(observation int, node *Node) {
  if node != nil {
    if len(node.children) != 0 {
      parentRisk := tree.getBayesRisk(observation, node)
      var leavesRisk float64 = 0.0
      r := &Risk{leavesRisk}
      tree.riskLeavesOverTree(observation, node, r)
      if parentRisk > r.risk {
        node = nil
      } else {
        for _,child := range node.children {
          tree.pruneTree_(observation, child)
        }
      }
    }
  }
}

func (tree *PrunedTree) pruneTree(observation int) {
  root := tree.decisionTree
  tree.pruneTree_(observation, root)
}

func (tree *PrunedTree) pruneTreeOverExamples(columns []string) {
  for i,_ := range tree.trainData[columns[0]] {
    tree.pruneTree(i)
  }
}

func (tree *PrunedTree) writeToFile(data map[string][]int,
labels string, columns []string) []int {
    predicted := []int{}
    outputLabels, err := os.Create(labels)
    check(err)
    defer outputLabels.Close()
    for i,_ := range data[columns[0]] {
        observation := make(map[string]int)
        for key,_ := range data {
            observation[key] = data[key][i]
        }
        outcome,output := plinkoDownModel(false,observation,tree.decisionTree)
        predictedLabel := majorityVote(output)
        if outcome != nil {
          predictedLabel = outcome.label
        }
        _, err := outputLabels.WriteString(strconv.Itoa(predictedLabel))
        check(err)
        predicted = append(predicted, predictedLabel)
        if (i != len(data[columns[0]])-1) {
            _, err := outputLabels.WriteString("\n")
            check(err)
        }
    }
    return predicted
}

func (tree *PrunedTree) getLabels(prune bool, trainFile string, testFile string,
metricsFile string) (float64,float64){
    if prune {
      tree.pruneTreeOverExamples(tree.features)
    }
    trainPredicted := tree.writeToFile(tree.trainData, trainFile,tree.features)
    testPredicted := tree.writeToFile(tree.testData, testFile,tree.features)
    errorRateTrain := errorRate(trainPredicted, tree.trainData[tree.label])
    errorRateTest := errorRate(testPredicted,tree.testData[tree.label])
    metricsOutput, err := os.Create(metricsFile)
    check(err)
    defer metricsOutput.Close()
    _, err = metricsOutput.WriteString("error(train): " +
              strconv.FormatFloat(errorRateTrain,'f',6,64))
    check(err)
    _, err = metricsOutput.WriteString("\n")
    check(err)
    _,err = metricsOutput.WriteString("error(test): " +
             strconv.FormatFloat(errorRateTest,'f',6,64))
    return errorRateTrain,errorRateTest
}

func main() {
  arguments := os.Args
  T := &PrunedTree{}
  trainTestFile := arguments[1]
  propTest,_ := strconv.ParseFloat(arguments[2], 64)
  maxDepth,_ := strconv.Atoi(arguments[3])
  T.readFile(trainTestFile,propTest,maxDepth)
  T.mkTree()

  T.printTree()
  prune,_ := strconv.ParseBool(arguments[4])
  trainOut := arguments[5]
  testOut := arguments[6]
  metricOut := arguments[7]
  T.getLabels(prune,trainOut,testOut,metricOut)
  // fmt.Println(T.decisionTree.children)
}
