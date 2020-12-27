/* minimumSpanningTree_MHT.go - implements kruskal's algorithm
  dchou1618
*/
package main

import (
    "fmt"
)

type Edge struct {
    src string;
    dest string;
    cost int;
};

type minHeap struct {
    edges []*Edge;
};

type Map struct {
    edgeNames [][]string;
    costs []int;
}

func (this *minHeap) size() int {
    return len(this.edges);
}

func (this *minHeap) enqueue(edge *Edge) {
    if (this.size() == 0) {
        this.edges = append(this.edges, &Edge{"_","_",0});
    }
    this.edges = append(this.edges, edge);
    var i int = len(this.edges)-1;
    for (i/2 > 0) {
        if (this.edges[i].cost < this.edges[i/2].cost) {
            var temp *Edge = this.edges[i];
            this.edges[i] = this.edges[i/2];
            this.edges[i/2] = temp;
            i /= 2;
        } else {
            break;
        }
    }
}

/* deque approach - greedy */
func (this *minHeap) dequeue() *Edge {
    if (this.size() > 0) {
      front := this.edges[len(this.edges)-1];
      this.edges = this.edges[:len(this.edges)-1];
      if (this.size() == 1) {
          return front;
      }
      dequeued := this.edges[1];
      this.edges[1] = front;
      var i int = 1;
      for (2*i < len(this.edges)) {
          if (2*i+1 < len(this.edges)) {
              if (this.edges[i].cost <= this.edges[2*i].cost && this.edges[i].cost <= this.edges[2*i+1].cost) {
                  break;
              } else if (this.edges[2*i].cost < this.edges[2*i+1].cost) {
                  var temp *Edge = this.edges[2*i+1];
                  this.edges[2*i+1] = this.edges[i];
                  this.edges[i] = temp;
                  i = 2*i + 1;
              } else {
                  var temp *Edge = this.edges[i];
                  this.edges[i] = this.edges[2*i];
                  this.edges[2*i] = temp;
                  i *= 2;
              }
          } else {
              if (this.edges[i].cost <= this.edges[2*i].cost) {
                  break;
              } else {
                  var temp *Edge = this.edges[2*i];
                  this.edges[2*i] = this.edges[i];
                  this.edges[i] = temp;
                  i *= 2;
              }
          }
      }
      return dequeued;
    } else {
        return nil;
    }
}

/* MST - edge weights determined by cosine similarity between websites.
  Metric may also be determined by a linear combination of features
  including hyperlinks, url length, (etc.) */
func _kruskalMST(edges []*Edge) {
    connected := make(map[string]bool);
    mstStruct := [][]string{};
    priorityQueue := &minHeap{};
    for _,edge := range edges {
        priorityQueue.enqueue(edge);
    }
    for _, val := range priorityQueue.edges {
        fmt.Println(val.src, val.dest, val.cost);
    }
    for (priorityQueue.size()-1 != 0) {
        newEdge := priorityQueue.dequeue();
        if (!connected[newEdge.src] || !connected[newEdge.dest]) {
            connected[newEdge.src] = true;
            connected[newEdge.dest] = true;
            mstStruct = append(mstStruct, []string{newEdge.src, newEdge.dest});
        }
    }
    fmt.Println(mstStruct)
}

func main() {
    edges := []*Edge{&Edge{"A","C",3},&Edge{"B","C",3},&Edge{"C","E",3},
                        &Edge{"C","D",2}, &Edge{"E","D",3}, &Edge{"B","D",1},
                        &Edge{"D","F",7}, &Edge{"A","B",1}, &Edge{"D","G",2},
                        &Edge{"E","G",1}};

     _kruskalMST(edges);
}
