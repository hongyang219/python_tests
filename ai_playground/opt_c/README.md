# Optimized C++ Code

此目录存放从Python代码自动转换为的高性能C++代码，使用LLM（本地或在线模型）进行转换。

## 文件列表

- `max_sub_array_optimized_20260106161537.cpp`: 优化后的最大子数组和算法。使用Kadane算法（O(n)时间复杂度）计算随机数组的最大子数组和，支持20次运行的总和。使用LCG生成随机数，避免溢出。
- `optimized_20260103164643.cpp`: 优化后的Pi计算代码。使用Leibniz公式计算Pi值，支持大迭代次数（1亿次），使用高精度定时。
- `optimized_20260103172743.cpp`: 优化后的幂判断代码。使用对数函数判断一个数是否为另一个数的整数次幂。

## 转换过程

这些文件由`python_to_cpp.py`脚本生成，使用AI模型将Python代码重写为高效的C++实现。

## 使用方法

编译并运行C++文件，例如：
```
clang++ -Ofast -std=c++17 max_sub_array_optimized_20260106161537.cpp -o max_subarray
./max_subarray
```

## 注意

- 针对M1 Mac优化（如果适用）。

