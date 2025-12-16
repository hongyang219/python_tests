from os import remove

from jupyter_core.version import pattern

from python_exercises_basic import *


class CodeRunner(BasicRunner):
    def __init__(self):
        self.test_suite_name = "Test Suite"
        self.tests = []  # 存储所有测试方法
        self.results = []  # 存储测试结果
        self.setup_methods = []  # 存储setup方法
        self.teardown_methods = []  # 存储teardown方法
        self.before_all_methods = []  # 在所有测试前执行的方法
        self.after_all_methods = []  # 在所有测试后执行的方法

    @notify_and_return()
    def fibonacci(self, max=21):
        try:
            a, b = 0, 1
            while b <= max:
                a, b = b, a+b
                # print(a)
            return a
        except Exception as e:
            print(e)

    @notify()
    def unique_char(self, s="mississippi"):
        try:
            unique_chars = []
            count = Counter(s)
            print(count)
            enm = list(enumerate(s))
            print(enm)
            for pos, char in enumerate(s):
                if count[char] == 1:
                    unique_chars.append(pos)
            return unique_chars
        except Exception as e:
            print(e)

# lc1.两数之和
    def twoSum(self, nums=[1,2,7,9,13], target=9) -> List[int]:
        # 求给定列表中两数和等于目标值的下标。只存在一组解。
        result = []
        hash = dict(enumerate(nums))
        print (hash)
        for h in hash:
            print (h, hash[h])
        for h in hash:
            if target-hash[h] in nums:
                if target-hash[h] == hash[h]:
                    print('Value is half of the target and count is',nums.count(hash[h]))
                    if nums.count(hash[h])==1:
                        print('No duplicated number')
                        continue
                    if nums.count(hash[h])>1:
                        print('Found duplicated number')
                        result.append(h)
                        continue
                result.append(h)
        return result

#   lc9.判断回文数
    def isPalindrome(self, x:int=5):
        return str(x) == str(x)[:: -1]

#   lc13.计算罗马数
    def romanToInt(self, s: str = 'MCMXCIV') -> int:
        S = s
        value = 0
        dct1 = {
            'I':1,
            'V':5,
            'X':10,
            'L':50,
            'C':100,
            'D':500,
            'M':1000,
        }
        dct2 = {
            'IV':4,
            'IX':9,
            'XL':40,
            'XC':90,
            'CD':400,
            'CM':900
        }
        for k, v in dct2.items():
            if k in s:
                print (k, "in s")
                value += v
                s = s.replace(k,'')
                print(s)

        for _s in s:
            if _s in dct1:
                print(dct1[_s])
                value += dct1[_s]
        print(f"{S} equals {value}")

#   lc14.最大公共前缀
    def longestCommonPrefix(self, strs: List[str] = ["flower","flow","flight"]) -> str:
        # 先计算最短字符
        strlen_list = []
        for str in strs:
            strlen_list.append(len(str))
        print(strlen_list)
        str_len_min = min(strlen_list)
        print(f"min str length is {str_len_min}")

        # 处理空字符串表如[""]
        if str_len_min == 0:
            return ""
        # 处理单个字符表如["a"]
        if len(strs) == 1:
            return strs[0]

        # 根据最短字符长度截取第一个字符串的前缀表
        prefix_list = []
        for i in range(1, str_len_min + 1):
            prefix_list.append(strs[0][0:i])
        print(f"prefix list of strs[0] is {prefix_list}")

        # 遍历前缀表于strs
        # 如果遍历到不符合前缀的字符，则返回前缀表索引上一个前缀
        for p in prefix_list:
            if len(p) == 0:
                return ""
            for str in strs:
                if str.startswith(p) is False:
                    if len(p) == 1:
                        return ""
                    else:
                        return prefix_list[prefix_list.index(p) - 1]
                target_p = p
        return target_p

#   lc21.合并两个有序链表
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if list1 is None:
            return list2
        elif list2 is None:
            return list1

        lst = []
        while list1:
            lst.append(list1.val)
            list1 = list1.next
        while list2:
            lst.append(list2.val)
            list2 = list2.next
        print(lst)

        lst = sorted(lst)
        print(lst)
        head = ListNode(lst[0])
        cur = head
        lst.pop(0)
        for l in lst:
            cur.next = ListNode(l)
            cur = cur.next
        return head

#   lc26.数组去重
    def removeDuplicates(self, nums: List[int]) -> int:
        d = dict.fromkeys(nums)
        nums[:] = d.keys()    # 使用切片复制 - 创建新对象
        print(nums, len(nums))

#   lc27.移除数组指定值的元素
    @notify()
    def removeElement(self, nums: List[int], val: int) -> int:
        _nums = []
        for n in nums:
            if n != val:
                _nums.append(n)
        nums[:] = _nums
        print(nums[:])
        return len(nums[:])

#   lc28.找出字符串中第一个匹配项的下标
    @notify()
    def strStr(self, haystack: str, needle: str) -> int:
        print(haystack)
        print(needle)
        # 相同的字符串则直接返回0
        if len(haystack) == len(needle) and needle in haystack:
            print("Equal string!")
            return 0

        # 用一个临时字符串当栈用，边压边比较
        stack_str = ""
        for h in haystack:
            if needle not in stack_str:
                print(f"{needle} not in {stack_str}, push {h}")
                stack_str += h
            else:
                print(f"{needle} in {stack_str}！")
                return len(stack_str)-len(needle)
        # 如果needle在hay的末尾，上述循环在扫到needle前就结束了...
        if needle in stack_str:
            print(f"{needle} in {stack_str}！")
            return len(stack_str) - len(needle)
        return -1

#   lc35.搜索插入位置
    @notify()
    def searchInsert(self, nums: List[int]=[1,3,5,6], target: int=2) -> int:
        print(nums)
        print(f"where is {target}")
        if target not in nums:
            nums.append(target)
            _nums = sorted(nums)
            print(f"=>{_nums}")
            return _nums.index(target)
        else:
            print(f"where is {target}")
            return nums.index(target)

#   lc3. 无重复字符的最长子串
    @notify()
    def lengthOfLongestSubstring(self, s: str) -> int:
        # 空字符或单个字符直接返回
        if len(s) <= 1:
            return len(s)
        stack = []              # 用一个栈列表放遍历的字符串
        s_len = len(s)
        for i in range(s_len):
            print(f"round{i}")
            stack_str = ""      # 用一个栈字符串存放不重复字符
            for _s in s:
                if _s not in stack_str:
                    stack_str += _s
                    # print(stack_str)
                else:
                    print(f"Got a unique char {stack_str}")
                    stack.append(stack_str)
                    break
            # 如果是
            stack.append(stack_str)
            # 切掉首字符继续循环
            s = s[1:]
            # 字符串过长的时候打印会导致超出力扣输出限制
            # print(f"Cut head - {s}")
        print(stack)
        return len(sorted(stack, key=len)[-1])

#   思科面试题1
    def remove_camel(self, s: str) -> str:
        # Input a string, if it contains pattern like 'aAa' or 'zZz', remove the pattern string
        print(s)
        for i in range(len(s)-2):
            if (s[i].isalpha() and
                s[i+1].isupper() and
                s[i] == s[i+1].lower() and
                s[i] == s[i+2]):
                    pattern = s[i]+s[i+1]+s[i+2]
                    print(f"Found pattern {pattern}, remove\n")
                    s = s.replace(pattern, "")
                    return self.remove_camel(s)
        return s

#   lc.4 中位数
    @notify()
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        m = len(nums1)
        n = len(nums2)
        o = m+n
        nums3 = sorted(nums1+nums2)
        print(nums3)

        print(f"Length:{o}")
        # 如果合并数组是偶数个，二分取中位的两个数除2
        if o%2 == 0:
            mid1 = nums3[int(o/2-1)]
            mid2 = nums3[int(o/2)]
            print("~", mid1, mid2, "~")
            return float((mid1+mid2)/2)
        # 如果合并数组是奇数个，长度除2是x.5，取整的index即是中位
        else:
            return float(nums3[int(o/2)])




tc_adv = CodeRunner()
tc_adv.findMedianSortedArrays(nums1=[1,2,5], nums2=[4,4])
# print(tc_adv.remove_camel("abc==aAa==defghij#@$%^&*(ZZZ)_klahHhAamnop==bBb==777qrstu~~ppp~~vwxyz"))
# tc_adv.lengthOfLongestSubstring("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ abc")
# tc_adv.searchInsert()
# tc_adv.unique_char()
# f = tc_adv.fibonacci()
# assert f==20, "Not expect value"
# tc_adv.removeDuplicates([0,0,1,1,1,2,2,3,3,4])
# tc_adv.removeElement([0,0,1,1,1,2,2,3,3,4], 1)
# tc_adv.strStr("hahasadnotsad", "sad")



