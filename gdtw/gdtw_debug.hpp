
#pragma once

#include <iostream>

class GDTW {

private:
    int verbosity;
   
public:
    GDTW(int verbosity) : verbosity(verbosity) {
        std::cout << "Solver init..." << std::endl;
        // printf("Solver init...\n");
    }
};