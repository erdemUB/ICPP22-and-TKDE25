/* 
    Prints the induced 6-cycle count of simple bipartite graphs

    To run:
        g++ -o TJ TJ.cpp preProcessing.cpp utilities.cpp -std=c++17 -O3 -fopenmp -ltbb
        ./TJ path_to_dataset
    
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

uint64_t ab_c(const edges& E, const std::vector<uint32_t>& AB, const uint32_t c) {
    uint64_t count = 0;
    for (uint32_t x : AB) {
        if (!E[c].contains(x)) {
            ++count;
        }
    }
    return count;
}

// returns number of induced 6 cycles
uint64_t getCount(const graph& G, const uint32_t vLeft, const edges& E) {

    std::vector<phmap::flat_hash_map<uint32_t, std::vector<uint32_t>>> ABs(vLeft - 1);
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, vLeft - 1), [&](tbb::blocked_range<uint32_t> r) {
        for (uint32_t a = r.begin(); a < r.end(); ++a) {
            for (uint32_t u : G[a]) {
                for (uint32_t b : G[u]) {
                    if (b > a) {
                        ABs[a][b].emplace_back(u);
                    }
                    else {
                        break;
                    }
                }
            }
        }
    });

    // counts number of induced 6-cycles associated with each node in the left set
    std::vector<uint64_t> counts(vLeft - 2);

    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, vLeft - 2), [&](tbb::blocked_range<uint32_t> r) {
        for (uint32_t a = r.begin(); a < r.end(); ++a) {
            phmap::flat_hash_set<uint32_t> S;
            for (uint32_t v : G[a])
                for (uint32_t b : G[v])
                    if (b > a)
                        S.emplace(b);
                    else
                        break;
                
            for (uint32_t b : S)
                for (uint32_t c : S)
                    if (b < c) {
                        if (ABs[b].contains(c)) {
                            counts[a] += ab_c(E, ABs[a].at(b), c) * ab_c(E, ABs[a].at(c), b) * ab_c(E, ABs[b].at(c), a);
                        }
                    }
        }
    });

    // sum over all left set nodes' associated induced 6-cycle counts to obtain total induced 6-cycle count
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
