#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <climits>

namespace py = pybind11;
using namespace std;

/*
 * retrieve: Given an edge list (src, dst) and an array of seed nodes,
 * perform a multi-source BFS starting from all seeds until their BFS
 * waves collide at a common node (the "root"). Then, for each seed, backtrace
 * the shortest path from the seed to the root. The union of all nodes on these
 * paths forms the returned subgraph.
 *
 * Parameters:
 *   - src: vector of source endpoints of edges
 *   - dst: vector of destination endpoints of edges
 *   - seed: vector of seed node IDs (at least one seed is assumed)
 *
 * Returns:
 *   - a vector of node IDs that form the subgraph connecting all seeds.
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
    nNodes++;  // because the maximum node id is (nNodes - 1)

    // 2. Build the undirected graph as an adjacency list,
    //    avoiding duplicate edges and self loops.
    vector<unordered_set<int>> adjSets(nNodes);
    for (size_t i = 0; i < src.size(); i++) {
        int u = src[i];
        int v = dst[i];
        if (u == v)  // Skip self loops.
            continue;
        // Inserting into a set avoids duplicate entries.
        adjSets[u].insert(v);
        adjSets[v].insert(u);
    }
    // Convert the sets to vectors.
    vector<vector<int>> adj(nNodes);
    for (int i = 0; i < nNodes; i++) {
        for (int nb : adjSets[i]) {
            adj[i].push_back(nb);
        }
    }

    // 3. Multi-source BFS from all seeds.
    // Each node maintains a bitmask indicating which seeds have reached it.
    // Assume there are at most 32 seeds; otherwise, use a larger integer type.
    unsigned int full_mask = (seed.size() >= 32) ? 0xffffffff : ((1u << seed.size()) - 1);

    vector<int> dist(nNodes, INT_MAX);
    vector<unsigned int> reachMask(nNodes, 0);

    queue<int> mq;
    // Initialize the queue with all seeds.
    for (size_t i = 0; i < seed.size(); i++) {
        int s = seed[i];
        dist[s] = 0;
        reachMask[s] |= (1u << i);
        mq.push(s);
    }

    // The variable 'root' will store the first node that is reached by all seeds.
    int root = -1;
    while (!mq.empty()) {
        int cur = mq.front();
        mq.pop();

        // Check if current node is reached by all seeds.
        if (reachMask[cur] == full_mask) {
            root = cur;
            // std::cout << root << std::endl;
            break;
        }

        // Propagate BFS from the current node.
        for (int nb : adj[cur]) {
            int nd = dist[cur] + 1;
            // If a shorter distance is found, update neighbor.
            if (nd < dist[nb]) {
                dist[nb] = nd;
                reachMask[nb] = reachMask[cur];
                mq.push(nb);
            }
            // If the neighbor is reached at the same distance, update its mask.
            else if (nd == dist[nb]) {
                unsigned int newMask = reachMask[nb] | reachMask[cur];
                if (newMask != reachMask[nb]) {
                    reachMask[nb] = newMask;
                    mq.push(nb);
                }
            }
        }
    }

    // If no common meeting point is found, return the seeds only.
    if (root == -1) {
        return seed;
    }

    // 4. From the found root, perform a standard BFS to build a shortest-path tree.
    vector<int> parent(nNodes, -1);
    vector<bool> visited(nNodes, false);
    queue<int> bq;
    bq.push(root);
    visited[root] = true;

    while (!bq.empty()) {
        int cur = bq.front();
        bq.pop();
        for (int nb : adj[cur]) {
            if (!visited[nb]) {
                visited[nb] = true;
                parent[nb] = cur;
                bq.push(nb);
            }
        }
    }

    // 5. For each seed, backtrace the path from the seed to the root.
    // Use a set to store the union of nodes along all these paths.
    unordered_set<int> resultSet;
    for (int s : seed) {
        int cur = s;
        // If a seed is unreachable from the root (shouldn't happen in a connected subgraph),
        // we simply add the seed itself.
        if (!visited[s])
            resultSet.insert(s);
        else {
            while (cur != -1) {
                resultSet.insert(cur);
                if (cur == root)
                    break;
                cur = parent[cur];
            }
        }
    }

    // 6. Convert the set to a vector and return.
    return vector<int>(resultSet.begin(), resultSet.end());
}

vector<vector<int>> batch_retrieve(const vector<int>& src, const vector<int>& dst, const vector<vector<int>>& seedBatch) {
    // 1. Determine the number of nodes (assume nodes are 0-indexed)
    int nNodes = 0;
    for (size_t i = 0; i < src.size(); i++) {
        nNodes = max(nNodes, max(src[i], dst[i]));
    }
    nNodes++;  // because the maximum node id is (nNodes - 1)

    // 2. Build the undirected graph as an adjacency list,
    //    avoiding duplicate edges and self loops.
    vector<unordered_set<int>> adjSets(nNodes);
    for (size_t i = 0; i < src.size(); i++) {
        int u = src[i];
        int v = dst[i];
        if (u == v)  // Skip self loops.
            continue;
        // Inserting into a set avoids duplicate entries.
        adjSets[u].insert(v);
        adjSets[v].insert(u);
    }
    // Convert the sets to vectors.
    vector<vector<int>> adj(nNodes);
    for (int i = 0; i < nNodes; i++) {
        for (int nb : adjSets[i]) {
            adj[i].push_back(nb);
        }
    }

    vector<vector<int>> results;
    // Process each seed set in the batch.
    for (const auto& seed : seedBatch) {
        if (seed.empty()) {
            results.push_back({});
            continue;
        }

        // 3. Multi-source BFS from all seeds.
        // Each node maintains a bitmask indicating which seeds have reached it.
        // Assume there are at most 32 seeds; otherwise, use a larger integer type.
        unsigned int full_mask = (seed.size() >= 32) ? 0xffffffff : ((1u << seed.size()) - 1);

        vector<int> dist(nNodes, INT_MAX);
        vector<unsigned int> reachMask(nNodes, 0);
        queue<int> mq;
        // Initialize the queue with all seeds.
        for (size_t i = 0; i < seed.size(); i++) {
            int s = seed[i];
            // (Assuming seed nodes are valid and within [0, nNodes))
            dist[s] = 0;
            reachMask[s] |= (1u << i);
            mq.push(s);
        }

        // The variable 'root' will store the first node that is reached by all seeds.
        int root = -1;
        while (!mq.empty()) {
            int cur = mq.front();
            mq.pop();

            // Check if current node is reached by all seeds.
            if (reachMask[cur] == full_mask) {
                root = cur;
                break;
            }

            // Propagate BFS from the current node.
            for (int nb : adj[cur]) {
                int nd = dist[cur] + 1;
                // If a shorter distance is found, update neighbor.
                if (nd < dist[nb]) {
                    dist[nb] = nd;
                    reachMask[nb] = reachMask[cur];
                    mq.push(nb);
                }
                // If the neighbor is reached at the same distance, update its mask.
                else if (nd == dist[nb]) {
                    unsigned int newMask = reachMask[nb] | reachMask[cur];
                    if (newMask != reachMask[nb]) {
                        reachMask[nb] = newMask;
                        mq.push(nb);
                    }
                }
            }
        }

        // If no common meeting point is found, return the seeds themselves.
        if (root == -1) {
            results.push_back(seed);
            continue;
        }

        // 4. From the found root, perform a standard BFS to build a shortest-path tree.
        vector<int> parent(nNodes, -1);
        vector<bool> visited(nNodes, false);
        queue<int> bq;
        bq.push(root);
        visited[root] = true;

        while (!bq.empty()) {
            int cur = bq.front();
            bq.pop();
            for (int nb : adj[cur]) {
                if (!visited[nb]) {
                    visited[nb] = true;
                    parent[nb] = cur;
                    bq.push(nb);
                }
            }
        }

        // 5. For each seed, backtrace the path from the seed to the root.
        // Use a set to store the union of nodes along all these paths.
        unordered_set<int> resultSet;
        for (int s : seed) {
            int cur = s;
            // If a seed is unreachable from the root, add the seed itself.
            if (!visited[s])
                resultSet.insert(s);
            else {
                while (cur != -1) {
                    resultSet.insert(cur);
                    if (cur == root)
                        break;
                    cur = parent[cur];
                }
            }
        }

        // 6. Convert the set to a vector and add to the results.
        results.push_back(vector<int>(resultSet.begin(), resultSet.end()));
    }
    
    return results;
}

// dense_retrieve: Given an edge list (src, dst) and a seed set,
// compute a connected subgraph that contains the seed nodes and
// is as dense as possible among them using a greedy expansion heuristic.
// The algorithm starts from the subgraph returned by retrieve(), then iteratively
// adds a neighbor that maximizes the induced subgraph density until no improvement is possible.
vector<int> dense_retrieve(const vector<int>& src, const vector<int>& dst, const vector<int>& seed) {
    // Get an initial connected subgraph that connects the seeds using the retrieve() function.
    vector<int> subgraph = retrieve(src, dst, seed);

    // Determine the number of nodes (assume nodes are 0-indexed).
    int nNodes = 0;
    for (size_t i = 0; i < src.size(); i++) {
        nNodes = max(nNodes, max(src[i], dst[i]));
    }
    nNodes++;

    // Build the undirected graph as an adjacency list, avoiding duplicate edges and self loops.
    vector<unordered_set<int>> adjSets(nNodes);
    for (size_t i = 0; i < src.size(); i++) {
        int u = src[i];
        int v = dst[i];
        if (u == v)
            continue;
        adjSets[u].insert(v);
        adjSets[v].insert(u);
    }
    // Convert sets to vectors.
    vector<vector<int>> adj(nNodes);
    for (int i = 0; i < nNodes; i++) {
        for (int nb : adjSets[i]) {
            adj[i].push_back(nb);
        }
    }

    // Helper lambda to compute the density of the induced subgraph on a set S.
    // Density is defined as (number of edges in S) divided by the maximum possible number of edges.
    auto compute_density = [&](const unordered_set<int>& S) -> double {
        int edge_count = 0;
        for (int u : S) {
            for (int nb : adj[u]) {
                if (S.find(nb) != S.end()) {
                    edge_count++;
                }
            }
        }
        edge_count /= 2; // each edge is counted twice
        int n = S.size();
        if (n < 2) return 0.0;
        double max_edges = n * (n - 1) / 2.0;
        return edge_count / max_edges;
    };

    // Convert the initial subgraph to a set for easier manipulation.
    unordered_set<int> S(subgraph.begin(), subgraph.end());
    double density = compute_density(S);

    bool improved = true;
    while (improved) {
        improved = false;
        int best_candidate = -1;
        double best_density = density;
        // Collect candidate nodes: neighbors of S not already in S.
        unordered_set<int> candidates;
        for (int u : S) {
            for (int nb : adj[u]) {
                if (S.find(nb) == S.end()) {
                    candidates.insert(nb);
                }
            }
        }
        // Evaluate each candidate.
        for (int candidate : candidates) {
            unordered_set<int> newS = S;
            newS.insert(candidate);
            double new_density = compute_density(newS);
            if (new_density > best_density) {
                best_density = new_density;
                best_candidate = candidate;
            }
        }
        // If a candidate improves the density, add it to S.
        if (best_candidate != -1) {
            S.insert(best_candidate);
            density = best_density;
            improved = true;
        }
    }

    return vector<int>(S.begin(), S.end());
}

// dense_batch_retrieve: Given an edge list (src, dst) and a batch of seed sets,
// compute for each seed set a connected subgraph that contains the seed nodes and
// is as dense as possible among them using the dense_retrieve algorithm.
vector<vector<int>> dense_batch_retrieve(const vector<int>& src, const vector<int>& dst, const vector<vector<int>>& seedBatch) {
    vector<vector<int>> results;
    for (const auto &seed : seedBatch) {
        if (seed.empty()) {
            results.push_back({});
        } else {
            results.push_back(dense_retrieve(src, dst, seed));
        }
    }
    return results;
}

// steiner_batch_retrieve: Given an edge list (src, dst) and a batch of seed sets,
// compute a heuristic Steiner tree for each seed set. For each seed set, the algorithm
// runs BFS from each seed to compute pairwise shortest path distances, constructs a
// complete graph on the seeds with these distances as weights, computes a minimum
// spanning tree (MST) on this complete graph, and then retrieves the union of the
// shortest paths corresponding to the MST edges. This is a heuristic solution for the
// NP-hard Steiner tree problem.
vector<vector<int>> steiner_batch_retrieve(const vector<int>& src, const vector<int>& dst, const vector<vector<int>>& seedBatch) {
    // 1. Build the undirected graph as an adjacency list, avoiding duplicate edges and self loops.
    int nNodes = 0;
    for (size_t i = 0; i < src.size(); i++) {
        nNodes = max(nNodes, max(src[i], dst[i]));
    }
    nNodes++;  // because the maximum node id is (nNodes - 1)

    vector<unordered_set<int>> adjSets(nNodes);
    for (size_t i = 0; i < src.size(); i++) {
        int u = src[i];
        int v = dst[i];
        if (u == v) // Skip self loops.
            continue;
        adjSets[u].insert(v);
        adjSets[v].insert(u);
    }
    // Convert the sets to vectors.
    vector<vector<int>> adj(nNodes);
    for (int i = 0; i < nNodes; i++) {
        for (int nb : adjSets[i]) {
            adj[i].push_back(nb);
        }
    }

    vector<vector<int>> results;

    // Process each seed set in the batch.
    for (const auto &seed : seedBatch) {
        if (seed.empty()) {
            results.push_back({});
            continue;
        }
        if (seed.size() == 1) {
            results.push_back(seed);
            continue;
        }

        int m = seed.size();
        // For each seed, run BFS to compute distances and parent pointers.
        // distList[i] and parentList[i] correspond to BFS starting from seed[i].
        vector<vector<int>> distList(m, vector<int>(nNodes, INT_MAX));
        vector<vector<int>> parentList(m, vector<int>(nNodes, -1));

        for (int i = 0; i < m; i++) {
            int start = seed[i];
            queue<int> q;
            q.push(start);
            distList[i][start] = 0;
            parentList[i][start] = -1;
            while (!q.empty()) {
                int cur = q.front();
                q.pop();
                for (int nb : adj[cur]) {
                    if (distList[i][nb] > distList[i][cur] + 1) {
                        distList[i][nb] = distList[i][cur] + 1;
                        parentList[i][nb] = cur;
                        q.push(nb);
                    }
                }
            }
        }

        // Build a complete graph on the seeds, where the weight of edge (i, j) is the distance
        // from seed[i] to seed[j] computed from BFS starting at seed[i].
        vector<vector<int>> completeGraph(m, vector<int>(m, 0));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                if (i == j) continue;
                completeGraph[i][j] = distList[i][seed[j]];
            }
        }

        // Compute the MST of the complete graph using Prim's algorithm.
        vector<int> key(m, INT_MAX);
        vector<bool> inMST(m, false);
        vector<int> parentMST(m, -1);
        key[0] = 0;
        
        for (int count = 0; count < m - 1; count++) {
            int u = -1;
            int minVal = INT_MAX;
            for (int v = 0; v < m; v++) {
                if (!inMST[v] && key[v] < minVal) {
                    minVal = key[v];
                    u = v;
                }
            }
            if (u == -1) break;
            inMST[u] = true;
            for (int v = 0; v < m; v++) {
                if (!inMST[v] && completeGraph[u][v] < key[v]) {
                    key[v] = completeGraph[u][v];
                    parentMST[v] = u;
                }
            }
        }

        // Retrieve the union of nodes along the shortest paths corresponding to the MST edges.
        unordered_set<int> steinerNodes;
        // Include all seed nodes.
        for (int s : seed) {
            steinerNodes.insert(s);
        }
        // For each MST edge, recover the path using the BFS result from the parent seed.
        for (int v = 1; v < m; v++) {
            int u = parentMST[v]; // u is the parent of v in the MST
            if (u == -1) continue; // Skip if not connected
            int target = seed[v];
            int current = target;
            // Recover path from seed[u] to seed[v] using the BFS tree from seed[u]
            while (current != -1 && current != seed[u]) {
                steinerNodes.insert(current);
                current = parentList[u][current];
            }
            if (current == seed[u]) {
                steinerNodes.insert(current);
            }
        }

        // Convert the set of nodes to a vector and add to the results.
        vector<int> steinerResult(steinerNodes.begin(), steinerNodes.end());
        results.push_back(steinerResult);
    }
    return results;
}

// A simple main function to test retrieve().
int main() {
    // Example graph 1:
    //    0 --- 1
    //    |     
    //    2 --- 3 --- 4
    //
    // Edges: (0,1), (0,2), (2,3), (3,4)
    // vector<int> src = {0, 0, 2, 3};
    // vector<int> dst = {1, 2, 3, 4};


    // Example graph 2:
    //    0 --- 1
    //    |     |
    //    2 --- 3 --- 4
    //
    // Edges: (0,1), (0,2), (1,3), (2,3), (3,4)
    vector<int> src = {0, 0, 1, 2, 3};
    vector<int> dst = {1, 2, 3, 3, 4};

     // Define a graph with 10 nodes (0 to 9) and 25 edges.
    vector<int> src_large = {0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8};
    vector<int> dst_large = {1, 2, 3, 4, 5, 2, 3, 5, 6, 3, 4, 7, 4, 6, 7, 5, 8, 6, 7, 9, 8, 9, 8, 9, 9};

    // Seed nodes: 1 and 4.
    // The algorithm will start BFS from both seeds, and the first node where their
    // BFS waves collide is taken as the root. Then, the union of the shortest paths
    // from the root to each seed is returned.
    vector<int> seed = {1, 4};

    vector<int> subgraph = retrieve(src_large, dst_large, seed);

    cout << "Subgraph contains " << subgraph.size() << " nodes:\n";
    for (int node : subgraph) {
        cout << node << " ";
    }
    cout << endl;
    
    // Additional test with a larger graph (10 nodes, 25 edges)
   
    
    // Test retrieve() on the large graph with seeds {2, 9}
    vector<int> seed_large = {2, 9};
    vector<int> subgraph_large = retrieve(src_large, dst_large, seed_large);
    cout << "\nLarge Graph Test: retrieve with seeds {2, 9}" << endl;
    cout << "Subgraph contains " << subgraph_large.size() << " nodes:" << endl;
    for (int node : subgraph_large) {
        cout << node << " ";
    }
    cout << endl;
    
    // Test steiner_batch_retrieve() on the large graph with two seed sets: {2, 9} and {0, 4, 7}
    vector<vector<int>> seedBatch_large = {
        {2, 9},
        {0, 4, 7}
    };
    vector<vector<int>> steinerResults_large = steiner_batch_retrieve(src_large, dst_large, seedBatch_large);
    cout << "\nLarge Graph Test: steiner_batch_retrieve" << endl;
    for (size_t i = 0; i < steinerResults_large.size(); i++) {
        cout << "Steiner subgraph for seed set " << i << " contains " << steinerResults_large[i].size() << " nodes:" << endl;
        for (int node : steinerResults_large[i]) {
            cout << node << " ";
        }
        cout << endl;
    }
    
    // Test dense_batch_retrieve() on the large graph with two seed sets: {2, 9} and {0, 4, 7}
    vector<vector<int>> seedBatch_dense = {
        {2, 9},
        {0, 4, 7}
    };
    vector<vector<int>> denseResults = dense_batch_retrieve(src_large, dst_large, seedBatch_dense);
    cout << "\nLarge Graph Test: dense_batch_retrieve" << endl;
    for (size_t i = 0; i < denseResults.size(); i++) {
        cout << "Dense subgraph for seed set " << i << " contains " << denseResults[i].size() << " nodes:" << endl;
        for (int node : denseResults[i]) {
            cout << node << " ";
        }
        cout << endl;
    }
    
    return 0;
}

// Expose the function using pybind11
PYBIND11_MODULE(libretrieval, m) {
    m.def("retrieve", &retrieve, "Retrieve subgraph using shortest paths");
    m.def("batch_retrieve", &batch_retrieve, "Retrieve subgraph for a batch of seed vectors using shortest paths");
    m.def("steiner_batch_retrieve", &steiner_batch_retrieve, "Retrieve subgraph for a batch of seed vectors using a heuristic Steiner tree algorithm");
    m.def("dense_batch_retrieve", &dense_batch_retrieve, "Retrieve subgraph for a batch of seed vectors that contains seed nodes and is as dense as possible");
}
