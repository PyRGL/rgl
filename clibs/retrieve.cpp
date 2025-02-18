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

    // 2. Build the undirected graph as an adjacency list.
    vector<vector<int>> adj(nNodes);
    for (size_t i = 0; i < src.size(); i++) {
        adj[src[i]].push_back(dst[i]);
        adj[dst[i]].push_back(src[i]);
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

    // 2. Build the undirected graph as an adjacency list.
    vector<vector<int>> adj(nNodes);
    for (size_t i = 0; i < src.size(); i++) {
        adj[src[i]].push_back(dst[i]);
        adj[dst[i]].push_back(src[i]);
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

    // Seed nodes: 1 and 4.
    // The algorithm will start BFS from both seeds, and the first node where their
    // BFS waves collide is taken as the root. Then, the union of the shortest paths
    // from the root to each seed is returned.
    vector<int> seed = {1, 4};

    vector<int> subgraph = retrieve(src, dst, seed);

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
    m.def("batch_retrieve", &batch_retrieve, "Retrieve subgraph for a batch of seed vectors using shortest paths");
}
