/* 
    Preprocesses the simple bipartite graph for induced 6-cycle counting
*/

#include "main.h"

/* 
Given a graph (G), output a vector of new IDs (newID)
s.t. all nodes in a 2-core have a value of 1 with the exception of the first node, 
which has a value of 0 if it is in the 2-core (or -1 if not)
*/
void Compute2Core(graph& G, uint32_t& vLeft, uint32_t& vRight, std::vector<int>& newID) {
    std::vector<uint32_t> deg_vec(vLeft + vRight);
    newID.resize(vLeft + vRight, 1);
    std::queue<uint32_t> deleting_q;
    uint32_t newVLeft = vLeft;
    uint32_t newVRight = vRight;
    for (uint32_t x = 0; x < vLeft + vRight; x++) {
        deg_vec[x] = G[x].size();
        if (deg_vec[x] < 2) {
            deleting_q.push(x);
            newID[x] = 0;
            if (x < vLeft) {
                --newVLeft;
            }
            else {
                --newVRight;
            }
        }
    }
    
    while (!deleting_q.empty()) {
        uint32_t x = deleting_q.front();
        deleting_q.pop();
        for (uint32_t y : G[x]) {
            if (newID[y] == 0) continue;
            deg_vec[y]--;
            if (deg_vec[y] < 2) {
                deleting_q.push(y);
                newID[y] = 0;
                if (y < vLeft) {
                    --newVLeft;
                }
                else {
                    --newVRight;
                }
            }
        }
    }
    vLeft = newVLeft;
    vRight = newVRight;
    --newID[0];
}

/* 
Given a graph (G), filter out nodes which are not in a 2-core and rename accordingly 
(i.e. if node 2 is removed, then node 3 is renamed to node 2, node 4 is renamed to node 3, and so on)
*/
void Obtain2Core(graph& G, uint32_t& vLeft, uint32_t& vRight) {
    uint32_t newVLeft = vLeft;
    uint32_t newVRight = vRight;
    std::vector<int> newID;
    Compute2Core(G, newVLeft, newVRight, newID);
    
    /*
    newID:  - same size as the original (unfiltered) graph
            - first node corresponds to a value of 0 (if node is in a 2-core) or -1 (if not in a 2-core)
            - a node is in a 2-core if its value is greater than the previous node's value (except the first node)
            - if it is in a 2-core, then its value is the updated node's ID
    */
    tbb::parallel_scan(tbb::blocked_range<uint32_t>(0, vLeft + vRight), 0,
		[&](tbb::blocked_range<uint32_t> r, uint32_t sum, bool is_final_scan) {
			uint32_t tmp = sum;
			for (uint32_t u = r.begin(); u < r.end(); ++u) {
                tmp += newID[u];
				if (is_final_scan) {
					newID[u] = tmp;
				}
			}
			return tmp;
		},
		[](uint32_t a, uint32_t b) {
			return a + b;
	    }
    );

    graph newG(newVLeft + newVRight);

    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, vLeft + vRight), [&](tbb::blocked_range<uint32_t> r) {
        for (uint32_t x = r.begin(); x < r.end(); ++x) {
            const uint32_t x_id = newID[x];
            if ((x == 0 && x_id == 0) || (x > 0 && x_id != newID[x - 1])) {
                for (uint32_t y : G[x]) {
                    const uint32_t y_id = newID[y];
                    if ((y == 0 && y_id == 0) || (y > 0 && y_id != newID[y - 1])) {
                        newG[x_id].emplace_back(y_id);
                    }
                }
            }
        }
    });
    G = std::move(newG);
    vLeft = newVLeft;
    vRight = newVRight;
}

/*
Preprocesses the graph (G) by:
    - Filtering out nodes not in a 2-core
    - Swaps left and right sets s.t. the left set has the smaller number of nodes
    - Sorts left set by increasing 2-path counts
*/
void preProcessing(graph& G, uint32_t& vLeft, uint32_t& vRight) {

    // Filtering out nodes not in a 2-core
    Obtain2Core(G, vLeft, vRight);

    // Swaps left and right sets s.t. the left set has the smaller number of nodes
    if (vLeft > vRight) {
        graph G2(vLeft + vRight);
        tbb::parallel_for(tbb::blocked_range<uint32_t>(0, vLeft + vRight), [&](tbb::blocked_range<uint32_t> r) {
            for (uint32_t i = r.begin(); i < r.end(); ++i){
                if (i < vLeft) {
                    G2[i + vRight].reserve(G[i].size());
                    for (uint32_t n : G[i])
                        G2[i + vRight].emplace_back(n - vLeft);
                }
                else {
                    G2[i - vLeft].reserve(G[i].size());
                    for (uint32_t n : G[i])
                        G2[i - vLeft].emplace_back(n + vRight);
                }
            }
        });
        std::swap(vLeft, vRight);
        G = std::move(G2);
    }

    // Sorts left set by increasing 2-path counts
    std::vector<uint32_t> idx(vLeft);
    std::vector<uint64_t> path2Cnts(vLeft);

    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, vLeft), [&](tbb::blocked_range<uint32_t> r) {
        for (uint32_t i = r.begin(); i < r.end(); ++i){
            idx[i] = i;
            const uint32_t v_size = G[i].size();
            uint64_t count = 0;
            #pragma omp simd reduction(+:count)
            for (uint32_t idx = 0; idx < v_size; ++idx)
                count += (uint64_t) G[G[i][idx]].size() - 1;
            path2Cnts[i] = count;
        }
    });

    tbb::parallel_sort(idx.begin(), idx.end(), [&path2Cnts](uint32_t i1, uint32_t i2) {return path2Cnts[i1] < path2Cnts[i2];});

    std::vector<uint32_t> rank(vLeft);

    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, vLeft), [&](tbb::blocked_range<uint32_t> r) {
        for (uint32_t i = r.begin(); i < r.end(); ++i){
            rank[idx[i]] = i;
        }
    });

    graph newG(vLeft + vRight);

    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, vLeft + vRight), [&](tbb::blocked_range<uint32_t> r) {
        for (uint32_t u = r.begin(); u < r.end(); ++u) {
            if (u < vLeft) {
                newG[rank[u]].reserve(G[u].size());
                for (uint32_t v : G[u]) {
                    newG[rank[u]].emplace_back(v);
                }
            }
            else {
                newG[u].reserve(G[u].size());
                for (uint32_t v : G[u]) {
                    newG[u].emplace_back(rank[v]);
                }
            }
        }
    });

    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, vLeft + vRight), [&](tbb::blocked_range<uint32_t> r) {
        for (uint32_t i = r.begin(); i < r.end(); ++i){
            std::sort(newG[i].begin(), newG[i].end(), std::greater<uint32_t>());
        }
    });

    G = std::move(newG);
}

/*
Preprocesses the graph (G) by:
    - Filtering out nodes not in a 2-core
    - Swaps left and right sets s.t. the left set has the smaller number of nodes
    - Sorts left set by increasing 2-path counts
    - Outputs a hashmap of edges (E)
*/
void preProcessing(graph& G, uint32_t& vLeft, uint32_t& vRight, edges& E) {

    // Filtering out nodes not in a 2-core
    Obtain2Core(G, vLeft, vRight);

    // Swaps left and right sets s.t. the left set has the smaller number of nodes
    if (vLeft > vRight) {
        graph G2(vLeft + vRight);
        tbb::parallel_for(tbb::blocked_range<uint32_t>(0, vLeft + vRight), [&](tbb::blocked_range<uint32_t> r) {
            for (uint32_t i = r.begin(); i < r.end(); ++i){
                if (i < vLeft) {
                    G2[i + vRight].reserve(G[i].size());
                    for (uint32_t n : G[i])
                        G2[i + vRight].emplace_back(n - vLeft);
                }
                else {
                    G2[i - vLeft].reserve(G[i].size());
                    for (uint32_t n : G[i])
                        G2[i - vLeft].emplace_back(n + vRight);
                }
            }
        });
        std::swap(vLeft, vRight);
        G = std::move(G2);
    }

    // Sorts left set by increasing 2-path counts
    std::vector<uint32_t> idx(vLeft);
    std::vector<uint64_t> path2Cnts(vLeft);

    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, vLeft), [&](tbb::blocked_range<uint32_t> r) {
        for (uint32_t i = r.begin(); i < r.end(); ++i){
            idx[i] = i;
            const uint32_t v_size = G[i].size();
            uint64_t count = 0;
            #pragma omp simd reduction(+:count)
            for (uint32_t idx = 0; idx < v_size; ++idx)
                count += (uint64_t) G[G[i][idx]].size() - 1;
            path2Cnts[i] = count;
        }
    });

    tbb::parallel_sort(idx.begin(), idx.end(), [&path2Cnts](uint32_t i1, uint32_t i2) {return path2Cnts[i1] < path2Cnts[i2];});

    std::vector<uint32_t> rank(vLeft);

    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, vLeft), [&](tbb::blocked_range<uint32_t> r) {
        for (uint32_t i = r.begin(); i < r.end(); ++i){
            rank[idx[i]] = i;
        }
    });

    graph newG(vLeft + vRight);
    E.resize(vLeft);

    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, vLeft + vRight), [&](tbb::blocked_range<uint32_t> r) {
        for (uint32_t u = r.begin(); u < r.end(); ++u) {
            if (u < vLeft) {
                newG[rank[u]].reserve(G[u].size());
                E[rank[u]].reserve(G[u].size());
                for (uint32_t v : G[u]) {
                    newG[rank[u]].emplace_back(v);
                    E[rank[u]].emplace(v);
                }
            }
            else {
                newG[u].reserve(G[u].size());
                for (uint32_t v : G[u]) {
                    newG[u].emplace_back(rank[v]);
                }
            }
        }
    });

    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, vLeft + vRight), [&](tbb::blocked_range<uint32_t> r) {
        for (uint32_t i = r.begin(); i < r.end(); ++i){
            std::sort(newG[i].begin(), newG[i].end(), std::greater<uint32_t>());
        }
    });

    G = std::move(newG);
}
