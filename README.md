# MST-DecisionTree-Phishing
MSTs between websites that are visually similar, which is based on some metric. Decision trees are done on the UCI dataset.

  - minimumSpanningTree_MHT.go - used for finding tree spanning sites, depicting those that are most similar to one another (using metrics such as cosine similarity)
  - prunedDecisionTree.go:
    - `go build prunedDecisionTree.go`
    - `./prunedDecisionTree [inputFile] [proportionTestData] [maxDepth] [willPrune] [outputTrainFileName] [outputTestFileName] [outputMetricFileName]`
