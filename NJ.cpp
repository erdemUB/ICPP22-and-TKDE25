/* 
    Prints the induced 6-cycle count of simple bipartite graphs

    To run:
        g++ -o NJ NJ.cpp preProcessing.cpp utilities.cpp -std=c++17 -O3 -fopenmp -ltbb
        ./NJ path_to_dataset
    
    Dataset format:
        |E| |U| |V|
        u1 v1
        u2 v1
        u2 v2
    
    Example dataset:
        3 2 2
        0 0
        0 1
        1 0
*/

#include "main.h"

// returns number of induced 6 cycles
uint64_t getCount(const graph& G, const uint32_t vLeft, const edges& E) {

    // counts number of induced 6-cycles associated with each node
    std::vector<uint64_t> counts(vLeft - 2);
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, vLeft - 2), [&](tbb::blocked_range<uint32_t> r) {
        for (uint32_t u = r.begin(); u < r.end(); ++u) {
            phmap::flat_hash_map<uint64_t, uint64_t> VWs;
            uint32_t size = G[u].size();
            for (uint32_t i = 0; i < size; ++i) {
                uint32_t a = G[u][i];
                for (uint32_t i2 = i + 1; i2 < size; ++i2) {
                    uint32_t b = G[u][i2];
                    std::vector<uint32_t> AB;
                    AB.reserve(G[a].size());
                    // N(A) \ N(b)
                    for (uint32_t v : G[a]) {
                        if (v == u)
                            break;
                        if (!E[v].contains(b)) {
                            AB.emplace_back(v);
                        }
                    }
                    for (uint32_t w : G[b]) {
                        if (w == u)
                            break;
                        if (!E[w].contains(a)) {
                            uint32_t w_size = G[w].size();
                            for (uint32_t v : AB) {
                                uint64_t vw = ((uint64_t) v + w) * (v + w + 1) / 2 + std::min(v, w);
                                if (!VWs.contains(vw)) {
                                    uint64_t count = 0;
                                    if (G[v].size() < w_size)
                                        for (uint32_t c : G[v]) {
                                            if (E[w].contains(c) && !E[u].contains(c)) {
                                                ++count;
                                            }
                                        }
                                    else
                                        for (uint32_t c : G[w]) {
                                            if (E[v].contains(c) && !E[u].contains(c)) {
                                                ++count;
                                            }
                                        }
                                    VWs[vw] = count;
                                    counts[u] += count;
                                }
                                else {
                                    counts[u] += VWs[vw];
                                }
                            }
                        }
                    }
                }
            }
        }
    });

    // sum over all nodes' associated induced 6-cycle counts to obtain total induced 6-cycle count
    Sum total;
    tbb::parallel_reduce(tbb::blocked_range<std::vector<uint64_t>::iterator>(counts.begin(), counts.end()), total);

    return total.value;
}

int main(int argc, char *argv[]) {

    if (argc < 2) {
		std::cout << "Usage: " << argv[0] << " path_to_dataset" << "\n";
        return 1;
	}

    char *filename = argv[1];

    uint32_t nEdge, vLeft, vRight;

    graph G = readGraph(filename, nEdge, vLeft, vRight);

    edges E;

    auto start = get_time();

    preProcessing(G, vLeft, vRight, E);

    uint64_t c = getCount(G, vLeft, E);
    
    std::cout << "Number of induced 6 cycles: " << c << "\n";

    auto finish = get_time();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish-start);
    std::cout << "Elapsed time = " << duration.count() << " milliseconds\n";

    return 0;
}
