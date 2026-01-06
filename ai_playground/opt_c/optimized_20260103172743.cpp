
#include <iostream>
#include <cmath>

// Check if a given number n is a power of another number x.
bool isPowerOf(int n, int x) {
    if (n <= 0)
        return false;
    double power = log2(n) / log2(x);
    std::cout << n << "是" << x << "的" << power << "次方\n";
    std::cout << n << "是" << x << "的" << std::round(power * 10) / 10 << "次方\n";
    return fmod(power, 1.0) == 0;
}

int main() {
    double start_time = 0, end_time = 0;

#ifndef __APPLE__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-security"
#endif
    printf("\nResult: %f\nExecution Time: %f seconds", isPowerOf(243, 3), end_time - start_time);
#ifndef __APPLE__
#pragma clang warning pop
#endif

    return 0;
}
