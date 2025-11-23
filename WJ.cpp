/* 
    Prints the induced 6-cycle count of simple bipartite graphs

    To run:
        g++ -o WJ WJ.cpp preProcessing.cpp utilities.cpp -std=c++17 -O3 -fopenmp -ltbb
        ./WJ path_to_dataset
    
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

// finds location of a wedge with endpoint u in the vector of wedges (Wedges)
bool getm (const std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>& Wedges, uint32_t start, uint32_t end, uint32_t u, uint32_t& m) {

    uint32_t l = start;
    uint32_t r = end - 1;
    while (l <= r) {
        m = (l + r) / 2;

        uint32_t u2 = std::get<1>(Wedges[m]);
        
        if (u2 < u)
            l = m + 1;
        else if (u2 > u)
            r = m - 1;
        else
            return true;
    }
    return false;
}

// returns number of induced 6 cycles
uint64_t getCount(const graph& G, const uint32_t nEdge, const uint32_t vLeft, const edges& E) {

    // assign wedge partitions to each node in U
    std::vector<uint64_t> partitions(vLeft);
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, vLeft - 1), [&](tbb::blocked_range<uint32_t> r) {
        for (uint32_t u = r.begin(); u < r.end(); ++u) {
            uint64_t val = 0;
            for (uint32_t v : G[u]) {
                for (uint32_t u2 : G[v]) {
                    if (u2 > u) {
                        ++val;
                    }
                }
            }
            partitions[u + 1] = val;
        }
    });
    tbb::parallel_scan(tbb::blocked_range<uint32_t>(0, vLeft), 0,
		[&](tbb::blocked_range<uint32_t> r, uint64_t sum, bool is_final_scan) {
			uint64_t tmp = sum;
			for (uint32_t u = r.begin(); u < r.end(); ++u) {
                tmp += partitions[u];
				if (is_final_scan) {
					partitions[u] = tmp;
				}
			}
			return tmp;
		},
		[](uint64_t a, uint64_t b) {
			return a + b;
	    }
    );

    // obtain wedges with endpoints in U and sort by endpoints
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> Wedges(partitions[vLeft - 1]);
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, vLeft - 1), [&](tbb::blocked_range<uint32_t> r) {
        for (uint32_t u1 = r.begin(); u1 < r.end(); ++u1) {
            uint64_t i = 0;
            for (uint32_t v1 : G[u1])
                for (uint32_t u2 : G[v1])
                    if (u2 > u1) {
                        for (uint64_t w = partitions[u1]; w < partitions[u1] + i + 1; ++w) {
                            if (w == partitions[u1] + i) {
                                Wedges[w] = std::make_tuple(u1, u2, v1);
                                break;
                            }
                            if (u2 < std::get<1>(Wedges[w])) {
                                for (uint64_t w2 = partitions[u1] + i; w2 >= w + 1; --w2) {
                                    Wedges[w2] = Wedges[w2 - 1];
                                }
                                Wedges[w] = std::make_tuple(u1, u2, v1);
                                break;
                            }
                        }
                        ++i;
                    }
                    else
                        break;
        }
    });

    // counts number of induced 6-cycles associated with each wedge
    std::vector<uint64_t> counts(partitions[vLeft - 1]);
    tbb::parallel_for(tbb::blocked_range<uint64_t>(0, partitions[vLeft - 1]), [&](tbb::blocked_range<uint64_t> r) {
        for (uint64_t w1_idx = r.begin(); w1_idx < r.end(); ++w1_idx) {
            uint64_t c;
            uint64_t count = 0;
            uint64_t idx = vLeft;
            uint32_t u3 = vLeft;
            bool skip = false;
            // wedge w1: u1 -> v1 -> u2
            uint32_t u1 = std::get<0>(Wedges[w1_idx]);
            uint32_t u2 = std::get<1>(Wedges[w1_idx]);
            if (u2 == vLeft - 1)
                continue;
            uint32_t v1 = std::get<2>(Wedges[w1_idx]);
            // wedge w2: u2 -> v2 -> u3
            for (uint64_t w2_idx = partitions[u2]; w2_idx < partitions[u2 + 1]; ++w2_idx) {
                uint32_t v2 = std::get<2>(Wedges[w2_idx]);
                // speedup #1
                if ((skip && u3 == std::get<1>(Wedges[w2_idx])) || E[u1].contains(v2))
                    continue;
                u3 = std::get<1>(Wedges[w2_idx]);
                // inducedness check: u3 -> v1
                skip = E[u3].contains(v1);
                // inducedness check: u1 -> v2
                if (!skip) {
                    // speedup #2
                    if (idx != u3) {
                        c = 0;
                        uint32_t m;
                        if (getm(Wedges, partitions[u1], partitions[u1 + 1], u3, m)) {
                            idx = m;
                            while (idx < partitions[u1 + 1]) {
                                const std::tuple<uint32_t, uint32_t, uint32_t>& curr = Wedges[idx];
                                if (std::get<1>(curr) != u3)
                                    break;
                                // wedge w3: u1 -> v3 -> u3
                                uint32_t v3 = std::get<2>(curr);
                                // inducedness check: u2 -> v3
                                if (!E[u2].contains(v3))
                                    ++c;
                                ++idx;
                            }
                            idx = m - 1;
                            uint32_t min_idx = 0;
                            if (u1 > 0) {
                                min_idx = partitions[u1 - 1];
                            }
                            while (idx >= min_idx) {
                                const std::tuple<uint32_t, uint32_t, uint32_t>& curr = Wedges[idx];
                                if (std::get<1>(curr) != u3)
                                    break;
                                // wedge w3: u1 -> v3 -> u3
                                uint32_t v3 = std::get<2>(curr);
                                // inducedness check: u2 -> v3
                                if (!E[u2].contains(v3))
                                    ++c;
                                --idx;
                            }
                        }
                        idx = u3;
                    }
                    count += c;
                }
            }
            counts[w1_idx] = count;
        }
    });

    // sum over all wedges' associated induced 6-cycle counts to obtain total induced 6-cycle count
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

    uint64_t c = getCount(G, nEdge, vLeft, E);
    
    std::cout << "Number of induced 6 cycles: " << c << "\n";

    auto finish = get_time();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish-start);
    std::cout << "Elapsed time = " << duration.count() << " milliseconds\n";

    return 0;
}
