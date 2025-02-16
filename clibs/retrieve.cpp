#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>

namespace py = pybind11;
using namespace std;

/* 
 * retrieve: given an edge list (src, dst), and an array of seed nodes,
 * uses BFS to return the union of the shortest paths from seed[0] to every other seed.
 *
 * Parameters:
 *   - src: vector of source endpoints of edges
 *   - dst: vector of destination endpoints of edges
 *   - seed: vector of seed node IDs (at least one seed is assumed)
 *   - num_edge: number of edges in the graph
 *   - num_seed: number of seed nodes
 * 
 * Returns:
 *   - a vector of node IDs that form the subgraph.
 */
vector<int> retrieve(const vector<int>& src, const vector<int>& dst, const vector<int>& seed) {
    if (seed.empty()) {
        return {};
    }

    // 1. Determine the number of nodes (assume nodes are 0-indexed)
    int nNodes = 0;
    for (size_t i = 0; i < src.size(); i++) {
        nNodes = max(nNodes, max(src[i], dst[i]));
    }
    nNodes++;  // because the maximum node id is (nNodes-1)

    // 2. Build the adjacency list.
    vector<vector<int>> adj(nNodes);

    for (size_t i = 0; i < src.size(); i++) {
        adj[src[i]].push_back(dst[i]);
        adj[dst[i]].push_back(src[i]);  // undirected graph
    }

    // 3. Run BFS from the first seed (seed[0]).
    vector<bool> visited(nNodes, false);
    vector<int> parent(nNodes, -1);
    queue<int> q;
    
    int root = seed[0];
    visited[root] = true;
    q.push(root);

    while (!q.empty()) {
        int cur = q.front();
        q.pop();
        for (int nb : adj[cur]) {
            if (!visited[nb]) {
                visited[nb] = true;
                parent[nb] = cur;
                q.push(nb);
            }
        }
    }

    // 4. For each seed, backtrack the path to the root (if reached).
    unordered_set<int> inResult;
    for (int s : seed) {
        if (visited[s]) {
            for (int cur = s; cur != -1; cur = parent[cur]) {
                inResult.insert(cur);
            }
        } else {
            // If not reached by BFS (disconnected), add the seed itself.
            inResult.insert(s);
        }
    }

    // 5. Convert result set to vector and return
    return vector<int>(inResult.begin(), inResult.end());
}

// A simple main function to test retrieve().
int main() {
    // Example graph:
    //   0 --- 1
    //   |     |
    //   2 --- 3 --- 4
    //
    // Let the edges be:
    //   (0,1), (0,2), (1,3), (2,3), (3,4)
    vector<int> src = {0, 0, 1, 2, 3};
    vector<int> dst = {1, 2, 3, 3, 4};

    // Let the seeds be: 1 and 4.
    // We want to return the smallest subgraph connecting 1 and 4.
    vector<int> seed = {1, 4};

    vector<int> subgraph = retrieve(src, dst, seed);

    // Print the retrieved subgraph node IDs.
    cout << "Subgraph contains " << subgraph.size() << " nodes:\n";
    for (int node : subgraph) {
        cout << node << " ";
    }
    cout << endl;

    return 0;
}

// Expose the function using pybind11
PYBIND11_MODULE(libretrieval, m) {
    m.def("retrieve", &retrieve, "Retrieve subgraph using shortest paths");
}
