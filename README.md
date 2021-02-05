# MST-DecisionTree-Phishing
MSTs between websites that are visually similar, which is based on some metric. The UCI dataset was used: https://archive.ics.uci.edu/ml/datasets/phishing+websites

  - minimumSpanningTree_MHT.go - used for finding tree spanning sites, depicting those that are most similar to one another (using metrics such as cosine similarity)
  - prunedDecisionTree.go:
    - Decision Tree uses ID3 - maximize mutual information - and regularizes by restricting depth and post-pruning based on bayes risk from https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0194168#sec006
    - `go build prunedDecisionTree.go`
    - `./prunedDecisionTree [inputFile] [proportionTestData] [maxDepth] [willPrune] [outputTrainFileName] [outputTestFileName] [outputMetricFileName]`
  - Exploration into using parallel BFS alongside the bayes risk pruning algorithm with joblib. There is notable 2 times speedup.
