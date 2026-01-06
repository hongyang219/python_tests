
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <climits>

// LCG generator
uint32_t lcg_next(uint32_t& state) {
    const uint32_t a = 1664525;
    const uint32_t c = 1013904223;
    state = a * state + c;
    return state;
}

// Kadane's algorithm for maximum subarray sum (O(n))
int64_t kadane_max_subarray_sum(const std::vector<int>& arr) {
    int64_t max_sum = arr[0];
    int64_t current_sum = arr[0];
    
    for (size_t i = 1; i < arr.size(); ++i) {
        current_sum = std::max(static_cast<int64_t>(arr[i]), current_sum + arr[i]);
        max_sum = std::max(max_sum, current_sum);
    }
    return max_sum;
}

// Generate array and compute max subarray sum for one seed
int64_t max_subarray_sum_fast(int n, uint32_t seed, int min_val, int max_val) {
    std::vector<int> random_numbers;
    random_numbers.reserve(n);
    
    uint32_t state = seed;
    int range = max_val - min_val + 1;
    
    for (int i = 0; i < n; ++i) {
        uint32_t val = lcg_next(state);
        random_numbers.push_back(static_cast<int>(val % range) + min_val);
    }
    
    return kadane_max_subarray_sum(random_numbers);
}

// Total sum over 20 runs
int64_t total_max_subarray_sum(int n, uint32_t initial_seed, int min_val, int max_val) {
    int64_t total_sum = 0;
    uint32_t state = initial_seed;
    
    for (int run = 0; run < 20; ++run) {
        uint32_t seed = lcg_next(state);
        total_sum += max_subarray_sum_fast(n, seed, min_val, max_val);
    }
    
    return total_sum;
}

int main() {
    const int n = 10000;
    const uint32_t initial_seed = 42;
    const int min_val = -10;
    const int max_val = 10;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    int64_t result = total_max_subarray_sum(n, initial_seed, min_val, max_val);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::cout << "Total Maximum Subarray Sum (20 runs): " << result << std::endl;
    std::cout << "Execution Time: " << std::fixed << std::setprecision(6) 
              << elapsed.count() << " seconds" << std::endl;
    
    return 0;
}
