/* 
    Header file for induced 6-cycle counting
*/

#ifndef MAIN_H
#define MAIN_H

#include <chrono>
#include <iostream>
#include <vector>
#include <queue>
#include "parallel_hashmap/phmap.h"
#include "parallel_hashmap/phmap_utils.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
#include "tbb/parallel_sort.h"
#include "tbb/parallel_scan.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>

typedef std::vector<std::vector<uint32_t>> graph;

typedef std::vector<phmap::flat_hash_set<uint32_t>> edges;

struct Sum {
    uint64_t value;
    Sum() : value(0) {}
    Sum(Sum& s, tbb::split) {value = 0;}
    void operator()(const tbb::blocked_range<std::vector<uint64_t>::iterator>& r) {
        uint64_t temp = value;
        for(std::vector<uint64_t>::iterator it = r.begin(); it != r.end(); ++it) {
            temp += *it;
        }
        value = temp;
    }
    void join(Sum& rhs) {value += rhs.value;}
};

std::chrono::high_resolution_clock::time_point get_time();

graph readGraph(const char *filename, uint32_t& nEdge, uint32_t& vLeft, uint32_t& vRight);

void preProcessing(graph& G, uint32_t& vLeft, uint32_t& vRight);
void preProcessing(graph& G, uint32_t& vLeft, uint32_t& vRight, edges& E);

#endif
