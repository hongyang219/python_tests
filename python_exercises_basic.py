# from envs.llms.Lib.importlib.metadata import pass_none
from decorators import *
from test_data import *
from typing import Callable, List, Dict, Any, Optional
from collections import Counter
import time
import traceback
from datetime import datetime
import inspect


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class BasicRunner:
    def __init__(self):
        self.test_suite_name = "Test Suite"
        self.tests = []  # 存储所有测试方法
        self.results = []  # 存储测试结果
        self.setup_methods = []  # 存储setup方法
        self.teardown_methods = []  # 存储teardown方法
        self.before_all_methods = []  # 在所有测试前执行的方法
        self.after_all_methods = []  # 在所有测试后执行的方法

    @notify
    def list_basic(self, list = sample_list):
        print(list)
        list.append(6)
        print(list)
        list.remove(2)
        print(list)
        print(list[2])

    def dict_basic(self, dict = sample_dict):
        print(dict)
        print(dict.keys(), ", dict .keys() or .values() is not iterable, need be transferred to list")
        k = list(dict.keys())
        print(k[0])
        print(dict.values(), ", dict .keys() or .values() is not iterable, need be transferred to list")
        dict["new_key"] = "new_value"
        print(dict)
        del dict["new_key"]
        print(dict)

    def str_split(self, str=sample_string, splitter=' '):
        print(str)
        return str.split(splitter)

    def str_reverse(self, str = sample_string):
        print(str)
        return str[::-1]

    def str_dedup(self, str=sample_string_dup):
        print(str)
        str_dictKeys = dict.fromkeys(str)
        print(str_dictKeys)
        return "".join(str_dictKeys)

    def str_sort(self, str=sample_string):
        print(str)
        print(sorted(str))
        return "".join(sorted(str))

    def str_count(self, str=sample_string):
        print(str)
        return Counter(str)

    def list_dedup(self, lst=sample_list_dup, order=False):
        print(lst)
        list_to_set = set(lst)
        print(list_to_set)
        return list(list_to_set)

    def list_sort(self, lst=sample_list):
        print(lst)
        return sorted(lst)

    def dict_merge(self, dct1=dict_for_merge_1, dct2=dict_for_merge_2):
        print(dct1, dct2)
        print("Left dict will be overwitten by Right dict")
        merged_dict = {**dct1, **dct2}
        return merged_dict

    def show_range(self, start=0, end=3):
        print(range(start, end))
        print(*range(start, end), sep=', ')
        print(list(range(start, end)))
        return list(range(start, end))

    def list_pairs(self, range1=2, range2=3):
        print(range1, range2)
        pairs = [(x, y) for x in range(range1) for y in range(range2)]
        print(pairs)
        return pairs

    def square_generator(self, n=5):
        gen = (x ** 2 for x in range(n))
        return gen

    def dict_sort(self, d: dict = {}):
        print(d)
        print(sorted(d.items()))
        sorted_d = dict(sorted(d.items()))
        print(sorted_d,'\n')
        return sorted_d

    def dict_reverse_kv(self, d: dict = {}):
        print(d)
        reversed_d = {v: k for k, v in d.items()}
        print(reversed_d,'\n')
        return reversed_d

    @notify()
    def list_to_dict(self, lst1: list = key_list, lst2: list = value_list):
        print(lst1)
        print(lst2)
        print(list(zip(lst1, lst2)))
        print(dict(zip(lst1, lst2)))

#   lc58.求包含空格的字符串的最后一个单词的长度
    def lengthOfLastWord(self, s: str) -> int:
        # print(s.strip())
        return len(s.strip().split(" ")[-1])

#   lc66.求数组形式代表的整数+1的值
    def plusOne(self, digits: List[int]) -> List[int]:
        num_str = ''
        for d in digits:
            num_str+=str(d)
        num_str = str(int(num_str)+1)
        print(num_str)

        digits_plus = []
        for n in num_str:
            digits_plus.append(int(n))
        return digits_plus



tc = BasicRunner()
# tc.lengthOfLastWord("hello     world     ")
# tc.list_to_dict()

# d1 = {'c': 333, 'a': 1, 'b': 22}
# tc.dict_sort(d=d1)
# d2 = tc.dict_reverse_kv(d=d1)
# tc.dict_sort(d=d2)

# tc.show_range(3,8)
# gen = tc.square_generator(10)
# while True:
#     print(next(gen))
# print(list(gen))

# print(tc.list_dedup(order=True))
# print(tc.list_sort())
# print(tc.dict_merge())

# tc.dict_basic()
# dedup_str = tc.str_dedup()
# print(dedup_str,'\n')
# sorted_str = tc.str_sort(dedup_str)
# print(sorted_str)
# print(tc.str_count('baanaanaa'))













