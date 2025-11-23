/* 
    Prints the induced 6-cycle count of simple bipartite graphs

    To run:
        g++ -o BTJ BTJ.cpp preProcessing.cpp utilities.cpp -std=c++17 -O3 -fopenmp -ltbb
        ./BTJ <path_to_dataset> <partition_size?>
    
    Note that partition_size is optional; it defaults to processing all nodes.

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

uint64_t intersection_size (const std::vector<uint32_t>& s1, const std::vector<uint32_t>& s2) {
  uint64_t result = 0;

  std::vector<uint32_t>::const_iterator first1 = s1.begin();
  std::vector<uint32_t>::const_iterator last1 = s1.end();
  std::vector<uint32_t>::const_iterator first2 = s2.begin();
  std::vector<uint32_t>::const_iterator last2 = s2.end();

  while (first1 != last1 && first2 != last2)
  {
    if (*first1 > *first2) ++first1;
    else if (*first2 > *first1) ++first2;
    else {
      result += 1;
      ++first1;
      ++first2;
    }
  }
  return result;
}

// returns number of induced 6 cycles
uint64_t getCount(const graph& G, const uint32_t vLeft, const uint32_t partition_size) {
    
    // initial bounds
    uint32_t a_end = std::min(partition_size - 1, vLeft - 3);

    uint32_t b_start = 1;
    uint32_t b_end = std::min(partition_size, vLeft - 2);

    uint32_t c_start = 2;

    std::vector<phmap::flat_hash_map<uint32_t, uint32_t>> ACs(vLeft - 2);

    std::vector<uint64_t> counts(vLeft - 2);

    int num_threads = tbb::this_task_arena::max_concurrency();

    std::cout << "Operating on " << num_threads << " threads" << std::endl;

    std::vector<std::vector<uint32_t>> ABCs(num_threads * vLeft);

    std::vector<uint32_t> ABCs_counts(num_threads * vLeft);

    std::vector<uint32_t> localIDs(num_threads * vLeft);

    std::vector<uint32_t> reverseIDs(num_threads * vLeft);

    while (b_start <= vLeft - 2) {

        tbb::parallel_for(tbb::blocked_range<uint32_t>(0, a_end + 1), [&](tbb::blocked_range<uint32_t> r) {
            for (uint32_t a = r.begin(); a < r.end(); ++a) {
                if (a < b_start - 1) {
                    // update existing a
                    phmap::flat_hash_map<uint32_t, uint32_t>& S = ACs[a];
                    for (phmap::flat_hash_map<uint32_t, uint32_t>::iterator it = S.begin(); it != S.end();) {
                        if ((it->first) < c_start) {
                            it = S.erase(it);
                        }
                        else {
                            ++it;
                        }
                    }
                }
                else {
                    for (uint32_t u : G[a]) {
                        for (uint32_t c : G[u]) {
                            if (c > a && c >= c_start) {
                                ACs[a][c] += 1;
                            }
                            else {
                                break;
                            }
                        }
                    }
                }
            }
        });

        tbb::parallel_for(tbb::blocked_range<uint32_t>(b_start, b_end + 1), [&](tbb::blocked_range<uint32_t> r) {
            for (uint32_t b = r.begin(); b < r.end(); ++b) {

                int pid = tbb::task_arena::current_thread_index();
                const uint32_t globalStart = pid * vLeft;

                const uint32_t max = (pid + 1) * vLeft;

                uint32_t numAs = 0;
                uint32_t numCs = 0;

                for (uint32_t u : G[b]) {
                    for (uint32_t c : G[u]) {
                        if (c > b) {
                            if (localIDs[globalStart + c] == 0) {
                                ++numCs;
                                localIDs[globalStart + c] = numCs;
                                reverseIDs[globalStart + b + numCs - 1] = c;
                            }
                            const uint32_t localId = localIDs[globalStart + c];
                            ABCs[globalStart + b + localId - 1].emplace_back(u);
                            ABCs_counts[globalStart + b + localId - 1] += 1;
                        }
                        // c = a
                        else if (b > c) {
                            if (localIDs[globalStart + c] == 0) {
                                ++numAs;
                                localIDs[globalStart + c] = numAs;
                                reverseIDs[globalStart + numAs - 1] = c;
                            }
                            const uint32_t localId = localIDs[globalStart + c];
                            ABCs[globalStart + localId - 1].emplace_back(u);
                            ABCs_counts[globalStart + localId - 1] += 1;
                        }
                    }
                }

                #pragma omp simd
                for (uint32_t a = 0; a < numAs; ++a) {
                    const uint32_t origA = reverseIDs[globalStart + a];
                    phmap::flat_hash_map<uint32_t, uint32_t>& S = ACs.at(origA);
                    const std::vector<uint32_t>& AB = ABCs[globalStart + a];
                    const uint32_t ab_val = ABCs_counts[globalStart + a];
                    for (uint32_t c = 0; c < numCs; ++c) {
                        const uint32_t origC = reverseIDs[globalStart + b + c];
                        const auto iter = S.find(origC);
                        if (iter != S.end()) {
                            const uint32_t ac_val = iter->second;

                            const std::vector<uint32_t>& BC = ABCs[globalStart + b + c];

                            const uint32_t bc_val = ABCs_counts[globalStart + b + c];

                            // notin = |N(a) and N(b) and N(c)|
                            const uint32_t notin = intersection_size(AB, BC);

                            counts[b - 1] += ((uint64_t) ab_val - notin) * (ac_val - notin) * (bc_val - notin);
                        }
                    }
                    ABCs[globalStart + a].clear();
                    ABCs_counts[globalStart + a] = 0;
                    localIDs[globalStart + origA] = 0;
                }

                #pragma omp simd
                for (uint32_t c = 0; c < numCs; ++c) {
                    ABCs[globalStart + b + c].clear();
                    ABCs_counts[globalStart + b + c] = 0;
                    localIDs[globalStart + reverseIDs[globalStart + b + c]] = 0;
                }

            }
        });

        // update bounds
        a_end = std::min(a_end + partition_size, vLeft - 3);
        b_start += partition_size;
        b_end = std::min(a_end + 1, vLeft - 2);
        c_start = b_start + 1;
    }
    
    // sum over all left set nodes' associated induced 6-cycle counts to obtain total induced 6-cycle count
    Sum total;
    tbb::parallel_reduce(tbb::blocked_range<std::vector<uint64_t>::iterator>(counts.begin(), counts.end()), total);

    return total.value;
}

int main(int argc, char *argv[]) {

    if (argc < 2) {
		std::cout << "Usage: " << argv[0] << " <path_to_dataset> <partition_size?>" << std::endl;
        return 1;
	}

    char *filename = argv[1];

    uint32_t nEdge, vLeft, vRight;

    long long int partition_size;

    if (argc == 3) {
        partition_size = atoi(argv[2]);

        if (partition_size <= 0) {
            std::cout << "Error: invalid partition size" << std::endl;
            return 1;
        }
    }

    graph G = readGraph(filename, nEdge, vLeft, vRight);

    if (argc == 2) {
        partition_size = vLeft;
    }

    auto start = get_time();
    
    preProcessing(G, vLeft, vRight);

    std::cout << "New vLeft: " << vLeft << "; new vRight: " << vRight << std::endl;

    if (vLeft < 3 || vRight < 3) {
        std::cout << "Number of induced 6 cycles: 0" << "\n";
    }
    else {
        uint64_t c = getCount(G, vLeft, partition_size);
        std::cout << "Number of induced 6 cycles: " << c << "\n";
    }

    auto finish = get_time();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish-start);
    std::cout << "Elapsed time = " << duration.count() << " milliseconds\n";

    return 0;
}
