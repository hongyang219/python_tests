# Python Tests - Programming Exercises

这是一个Python编程练习项目，涵盖了从基础数据结构操作到高级算法问题的全面练习。
项目分为基础和高级两个模块，帮助开发者提升Python编程技能和算法思维。

## 项目结构

```
python_tests/
├── decorators.py              # 装饰器工具
├── test_data.py               # 测试数据
├── python_exercises_basic.py  # 基础编程练习
├── python_exercises_advanced.py # 高级算法练习
└── README.md                  # 项目说明
```

## 内容概述

### 基础练习 (python_exercises_basic.py)

包含 `BasicRunner` 类，实现以下基础Python操作：

- **列表操作**: 追加、删除、排序、去重
- **字典操作**: 键值操作、合并、排序、键值反转
- **字符串操作**: 分割、反转、去重、排序、计数
- **生成器**: 平方数生成器
- **其他工具**: 列表配对、范围展示、列表转字典
- **LeetCode基础题**:
  - LC58: 最后一个单词的长度
  - LC66: 加一
  - LC231: 判断是否为x的幂

### 高级练习 (python_exercises_advanced.py)

继承 `BasicRunner`，包含 `CodeRunner` 类，实现高级算法问题：

- **经典算法问题**:
  - LC1: 两数之和
  - LC3: 无重复字符的最长子串
  - LC4: 寻找两个正序数组的中位数
  - LC5: 最长回文子串
  - LC6: Z字形变换
  - LC9: 回文数
  - LC13: 罗马数字转整数
  - LC14: 最长公共前缀
  - LC49: 字母异位词分组
  - LC128: 最长连续序列

- **链表操作**:
  - LC21: 合并两个有序链表

- **数组操作**:
  - LC26: 删除有序数组中的重复项
  - LC27: 移除元素
  - LC35: 搜索插入位置
  - LC1512: 好数对的数目

- **字符串算法**:
  - LC28: 实现 strStr()
  - LC67: 二进制求和

- **数学算法**:
  - LC69: x的平方根（整数部分）

- **其他挑战**:
  - 斐波那契数列生成
  - 唯一字符位置查找


## 环境要求

- Python 3.x
- 标准库依赖：
  - `collections.Counter`
  - `typing` (用于类型注解)
  - `datetime`, `time`, `traceback`

## 使用方法

1. 克隆项目到本地：
   ```bash
   git clone https://github.com/hongyang219/python_tests.git
   cd python_tests
   ```

2. 运行基础练习：
   ```python
   from python_exercises_basic import BasicRunner

   runner = BasicRunner()
   runner.list_basic()  # 运行列表基础操作
   runner.dict_basic()  # 运行字典基础操作
   ```

3. 运行高级练习：
   ```python
   from python_exercises_advanced import CodeRunner

   runner = CodeRunner()
   result = runner.twoSum([2, 7, 11, 15], 9)  # 两数之和
   result = runner.longestCommonPrefix(["flower", "flow", "flight"])  # 最长公共前缀
   ```

4. 每个方法都有详细的打印输出，帮助理解算法执行过程。

## 特性

- **渐进式学习**: 从基础数据结构到复杂算法
- **详细注释**: 代码中包含详细的执行过程打印
- **LeetCode集成**: 包含大量LeetCode经典题目解法
- **面向对象设计**: 使用类封装相关方法，便于管理和扩展
- **装饰器支持**: 集成自定义装饰器用于方法通知和返回值处理

## 学习建议

1. 先从基础练习开始，熟悉Python基本数据结构操作
2. 逐步过渡到高级算法，理解常见编程问题的解法模式
3. 运行代码观察输出，理解算法的执行流程
4. 尝试修改参数，测试边界情况
5. 参考LeetCode题目，比较不同解法的优劣

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证
Please contact hongyang219@hotmail.com
