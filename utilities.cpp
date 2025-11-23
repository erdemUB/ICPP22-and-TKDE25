/* 
    Functions for reading the simple bipartite graph text file and getting the current time
*/

#include "main.h"

// reads graph file and converts to adjacency list
graph readGraph(const char *filename, uint32_t& nEdge, uint32_t& vLeft, uint32_t& vRight) {

    char *f;
    uint32_t size;
    struct stat s;
    int fd = open (filename, O_RDONLY);

    /* Get the size of the file. */
    fstat (fd, &s);
    size = s.st_size;

    f = (char *) mmap (0, size, PROT_READ, MAP_PRIVATE, fd, 0);

    int headerEnd = 0;
    nEdge = 0, vLeft = 0, vRight = 0;
    bool nE = true, vL = false;
    char c = f[headerEnd];
    while (c != '\n') {
        if (isdigit(c)) {
            if (nE) {
                nEdge = nEdge * 10 + c - '0';
            }
            else if (vL) {
                vLeft = vLeft * 10 + c - '0';
            }
            else {
                vRight = vRight * 10 + c - '0';
            }
        }
        else {
            if (nE) {
                nE = false;
                vL = true;
            }
            else {
                vL = false;
            }
        }
        headerEnd += 1;
        c = f[headerEnd];
    }

    graph G(vLeft + vRight);

    uint32_t u = 0, v = 0;
    bool left = true;
    for (uint32_t i = headerEnd + 1; i < size; ++i) {
        c = f[i];

        if (isdigit(c)) {
            if (left) {
                u = u * 10 + c - '0';
            }
            else {
                v = v * 10 + c - '0';
            }
        }
        else {
            if (c == ' ') {
                left = false;
            }
            // if edge has been processed
            else if (!left) {
                left = true;
                v += vLeft;
                G[u].emplace_back(v);
                G[v].emplace_back(u);
                u = 0; v = 0;
            }
        }
    }

    return G;
}

std::chrono::high_resolution_clock::time_point get_time() {return std::chrono::high_resolution_clock::now();}
