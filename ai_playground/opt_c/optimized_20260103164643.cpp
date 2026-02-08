
#include <iostream>
#include <iomanip>
#include <chrono>

double calculate(int iterations, double param1, double param2) {
    double result = 1.0;
    for (int i = 1; i <= iterations; ++i) {
        double j = static_cast<double>(i * param1 - param2);
        if (j != 0) result -= (1 / j);
        j = static_cast<double>(i * param1 + param2);
        if (j != 0) result += (1 / j);
    }
    return result;
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    double result = calculate(100'000'000, 4.0, 1.0) * 4;
    auto end_time = std::chrono::high_resolution_clock::now();

    std::cout << "Result: " << std::setprecision(12) << std::fixed << result << std::endl;
    std::cout << "\nExecution Time: " 
              << std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end_time - start_time).count() 
              << " seconds" << std::endl;

    return 0;
}
