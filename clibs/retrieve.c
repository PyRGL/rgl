#include <stdio.h>
#include <stdlib.h>

/* 
 * retrieve: given an edge list (src, dst), and an array of seed nodes,
 * uses BFS to return the union of the shortest paths from seed[0] to every other seed.
 *
 * Parameters:
 *   - src: array of source endpoints of edges
 *   - dst: array of destination endpoints of edges
 *   - seed: array of seed node IDs (at least one seed is assumed)
 *   - num_edge: number of edges in the graph
 *   - num_seed: number of seed nodes
 *   - num_retrieved: output parameter, set to the number of nodes in the subgraph
 *
 * Returns:
 *   - an array of node IDs (dynamically allocated) that form the subgraph.
 */
int* retrieve(int* src, int* dst, int* seed, int num_edge, int num_seed, int* num_retrieved) {
    if (num_seed == 0) {
        *num_retrieved = 0;
        return NULL;
    }

    // 1. Determine the number of nodes (assume nodes are 0-indexed)
    int nNodes = 0;
    for (int i = 0; i < num_edge; i++) {
        if (src[i] > nNodes) nNodes = src[i];
        if (dst[i] > nNodes) nNodes = dst[i];
    }
    nNodes++;  // because the maximum node id is (nNodes-1)

    // 2. Build the adjacency list.
    // First, count degree for each node.
    int *deg = calloc(nNodes, sizeof(int));
    for (int i = 0; i < num_edge; i++) {
        deg[src[i]]++;
        deg[dst[i]]++;   // undirected graph: add both endpoints.
    }
    // Allocate the neighbor lists.
    int **adj = malloc(nNodes * sizeof(int*));
    for (int i = 0; i < nNodes; i++) {
        adj[i] = malloc(deg[i] * sizeof(int));
        deg[i] = 0;  // reuse deg[] as an insertion index.
    }
    // Fill the neighbor lists.
    for (int i = 0; i < num_edge; i++) {
        int u = src[i], v = dst[i];
        adj[u][deg[u]++] = v;
        adj[v][deg[v]++] = u;
    }

    // 3. Run BFS from the first seed (seed[0]).
    int *visited = calloc(nNodes, sizeof(int));
    int *parent = malloc(nNodes * sizeof(int));
    for (int i = 0; i < nNodes; i++)
        parent[i] = -1;
    int *queue = malloc(nNodes * sizeof(int));
    int head = 0, tail = 0;
    int root = seed[0];
    visited[root] = 1;
    queue[tail++] = root;
    while (head < tail) {
        int cur = queue[head++];
        // Look at all neighbors of cur.
        for (int i = 0; i < deg[cur]; i++) {
            int nb = adj[cur][i];
            if (!visited[nb]) {
                visited[nb] = 1;
                parent[nb] = cur;
                queue[tail++] = nb;
            }
        }
    }

    // 4. For each seed, backtrack the path to the root (if reached).
    // We mark all nodes on these paths.
    int *inResult = calloc(nNodes, sizeof(int));
    for (int i = 0; i < num_seed; i++) {
        int s = seed[i];
        if (visited[s]) {
            // Follow the parent chain from s to root.
            for (int cur = s; cur != -1; cur = parent[cur])
                inResult[cur] = 1;
        } else {
            // If not reached by BFS (disconnected), add the seed itself.
            inResult[s] = 1;
        }
    }

    // 5. Count and collect the nodes in the result subgraph.
    int count = 0;
    for (int i = 0; i < nNodes; i++) {
        if (inResult[i])
            count++;
    }
    int *result = malloc(count * sizeof(int));
    int j = 0;
    for (int i = 0; i < nNodes; i++) {
        if (inResult[i])
            result[j++] = i;
    }
    *num_retrieved = count;

    // 6. Free temporary memory.
    free(inResult);
    free(visited);
    free(parent);
    free(queue);
    for (int i = 0; i < nNodes; i++) {
        free(adj[i]);
    }
    free(adj);
    free(deg);

    return result;
}


// A simple main function to test retrieve().
int main(void) {
    // Example graph:
    //   0 --- 1
    //   |     |
    //   2 --- 3 --- 4
    //
    // Let the edges be:
    //   (0,1), (0,2), (1,3), (2,3), (3,4)
    int src[] = {0, 0, 1, 2, 3};
    int dst[] = {1, 2, 3, 3, 4};
    int num_edge = sizeof(src) / sizeof(src[0]);

    // Let the seeds be: 1 and 4.
    // We want to return the smallest subgraph connecting 1 and 4.
    int seed[] = {1, 4};
    int num_seed = sizeof(seed) / sizeof(seed[0]);

    int num_retrieved = 0;
    int *subgraph = retrieve(src, dst, seed, num_edge, num_seed, &num_retrieved);

    // Print the retrieved subgraph node IDs.
    printf("Subgraph contains %d nodes:\n", num_retrieved);
    for (int i = 0; i < num_retrieved; i++) {
        printf("%d ", subgraph[i]);
    }
    printf("\n");

    free(subgraph);
    return 0;
}
