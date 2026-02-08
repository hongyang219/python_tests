from os import remove
from python_exercises_basic import *
import matplotlib.pyplot as plt


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
    @notify()
    def twoSum(self, nums=[1,2,7,9,13], target=9) -> List[int]:
        # 求给定列表中两数和等于目标值的下标。只存在一组解。
        result = []
        hash = dict(enumerate(nums))
        print (hash)
        # for h in hash:
        #     print (h, hash[h])
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
    @notify()
    def longestCommonPrefix(self, strs: List[str] = ["flower","flow","flight"]) -> str:
        # 先计算最短字符
        min_len_str = min(strs, key=len)
        str_len_min = len(min_len_str)
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
        print(f"prefix list of minimum str is {prefix_list}")

        # 遍历前缀表于strs
        # 如果遍历到不符合前缀的字符，则返回前缀表索引上一个前缀
        for p in prefix_list:
            if len(p) == 0:
                return ""
            for s in strs:
                if s.startswith(p) is False:
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
        return len(max(stack,key=len))

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

#   lc.5 最长回文子串
    @notify()
    def longestPalindrome(self, s: str = 'aaabbaabbabadccchhhjiosd334455655443hgfwuoiejrweoif') -> str:
        '''
        Be careful!!!
        In below case, complexity is O(N3)...
        Execution may result in TIMEOUT in leetcode(9000~10000 ms...)
        Also print() will EXCEED output
        "xaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffgggggggggghhhhhhhhhhiiiiiiiiiijjjjjjjjjjkkkkkkkkkk
        llllllllllmmmmmmmmmmnnnnnnnnnnooooooooooppppppppppqqqqqqqqqqrrrrrrrrrrssssssssssttttttttttuuuuuuuuuuvvvvvvvvvv
        wwwwwwwwwwxxxxxxxxxxyyyyyyyyyyzzzzzzzzzzyyyyyyyyyyxxxxxxxxxxwwwwwwwwwwvvvvvvvvvvuuuuuuuuuuttttttttttssssssssss
        rrrrrrrrrrqqqqqqqqqqppppppppppoooooooooonnnnnnnnnnmmmmmmmmmmllllllllllkkkkkkkkkkjjjjjjjjjjiiiiiiiiiihhhhhhhhhh
        ggggggggggffffffffffeeeeeeeeeeddddddddddccccccccccbbbbbbbbbbaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeee
        ffffffffffgggggggggghhhhhhhhhhiiiiiiiiiijjjjjjjjjjkkkkkkkkkkllllllllllmmmmmmmmmmnnnnnnnnnnoooooooooopppppppppp
        qqqqqqqqqqrrrrrrrrrrssssssssssttttttttttuuuuuuuuuuvvvvvvvvvvwwwwwwwwwwxxxxxxxxxxyyyyyyyyyyzzzzzzzzzzyyyyyyyyyy
        xxxxxxxxxxwwwwwwwwwwvvvvvvvvvvuuuuuuuuuuttttttttttssssssssssrrrrrrrrrrqqqqqqqqqqppppppppppoooooooooonnnnnnnnnn
        mmmmmmmmmmllllllllllkkkkkkkkkkjjjjjjjjjjiiiiiiiiiihhhhhhhhhhggggggggggffffffffffeeeeeeeeeeddddddddddccccccccccbbbbbbbbbbaaaa"
        '''
        temp = ""
        i = 0
        # Calculate all palindrome strings
        while i <= len(s):
            stack = ""
            for j in range(i, len(s)):
                stack += s[j]
                if stack == stack[::-1]:
                    temp = max(temp, stack, key=len)
            i += 1
        return temp

#   lc.67 二进制字符串求和
    @notify()
    def addBinary(self, a: str='10101', b: str='1011') -> str:
        la = a[::-1]
        lb = b[::-1]
        # 确保la是较长的字符串
        if len(la) < len(lb):
            la, lb = lb, la
        # 补零使两个字符串长度相同
        lb += '0' * (len(la) - len(lb))

        result = []
        carry = 0

        for i in range(len(la)):
            # 计算当前位的和/进位
            total = int(la[i]) + int(lb[i]) + carry
            cur = total % 2             # 当前位
            carry = total // 2          # 进位
            result.append(str(cur))

        # 别忘了补上最后的进位
        if carry:
            result.append(str(carry))

        # 反转结果并返回
        bin_str = ''.join(reversed(result))

        # 确保没有前导零（除非结果是"0"）
        if bin_str == "0":
            return bin_str
        # 去除可能的前导零
        return bin_str.lstrip('0') or "0"

#   lc.6 Z字形
    @notify()
    def convert_to_Zstring(self, s: str, numRows: int) -> str:
        stack = []
        period = numRows * 2 - 2    # 每个“Z”的“上半部分”的周期，包含一个竖向排列和一个斜向排列

        if numRows == 1:
            return s
        temp = ""

        for i in range(len(s)):
            # 计算当前字符在Z周期中的相对位置
            pos = i % period
            # print(pos)
            # 竖向排列
            if pos < numRows:
                temp += s[i]
                if len(temp) % numRows == 0:
                    stack.append(temp)
                    temp = ""
            # 斜向排列的情况，需要补空格
            else:
                down_blank = ' ' * (pos-numRows+1)          # 超过竖向排列(行数)的距离
                up_blank = ' ' * (period-pos)               # 距離周期末端的距离
                stack.append(up_blank + s[i] + down_blank)
        # 如果循环末尾还有字符未处理，补空格，否则下一部会list out of range
        if len(temp)>0:
            stack.append(temp + ' '*(numRows-len(temp)))
        print(stack)

        # 逐行读取最终结果并去空格
        result = ""
        i = 0
        while i < numRows:
            for item in stack:
                result += item[i]
            i+=1

        return result.replace(' ', '')

#   lc.? 计算平方根
    def mySqrt(self, x: int) -> int:
        i=int(0)
        while (i*i)<=x:
            # print(i, i*i)
            i+=1
        i = i-1
        # print(f"Integer:{i}")

        # 原题要求计算整数部分就可以了。。多写了一步计算到第一位小数的
        # d=float(i+0.1)
        # while (d*d)<=x:
        #     print(d, d*d)
        #     d+=0.1
        # d = d - 0.1
        # print(f"Decimal:{d}")
        return int(i)

#   lc.1512 求数组中相同的数对（好数对）的数量
    @notify()
    def numIdenticalPairs(self, nums: List[int]=[1,2,3,1,1,3]) -> int:
        # 找出所有不唯一的数对，放到一个数组
        cnt = dict(Counter(nums))
        print(cnt)
        lst = []
        for key in cnt:
            # print(cnt[key])
            if cnt[key]>1:
                lst.append(cnt[key])
        print(lst)
        # 因为是每次取两个元素，可以用组合计算所有项的组合数(C n|2)，再求和
        result = 0
        for n in lst:
            result+=n*(n-1)/2
        return int(result)

#   lc.49 找出数组中的异位词：包含相同字母组合但是排列不同的单词
    @notify()
    def groupAnagrams(self, strs: List[str] = ["eat", "tea", "tan", "ate", "nat", "bat"]) -> List[List[str]]:
        # 使用defaultdict性能更优，也无需进行if sorted_s not in h的判断
        h = defaultdict(list)
        for s in strs:
            # 将排序后的字符作为键值，如果有相同排列的只要append键值对应的数组中即可
            # sorted_s = "".join(sorted(s))
            h[tuple(sorted(s))].append(s) # 可以直接使用tuple作为键值，省略排序
        print(h)
        return list(h.values())
        # h = {}
        # for s in strs:
        #     sorted_s = "".join(sorted(s))
        #     if sorted_s not in h:
        #         h[sorted_s] = [s]
        #     else:
        #         h[sorted_s].append(s)
        # return list(h.values())

#   lc.128 最长连续序列
    @notify()
    def longestConsecutive(self, nums: List[int] = [100,4,200,1,3,2,1]) -> int:
        if len(nums) == 0:
            return 0
        # 先把数组排序好
        # 如果这时直接去重 sorted(set(nums), key=int)会超时(?
        sorted_nums = sorted(nums, key=int)
        print(sorted_nums)

        # 用集合储存临时序列，这样在有连续相同数字的情况下可以默认去重
        temp = set()
        result = []
        # 判断n-1是否在临时序列中，如果中断则储存上一个序列。并重置集合。
        for n in sorted_nums:
            if n - 1 in temp:
                temp.add(n)
            else:
                if temp:
                    result.append(temp)
                temp = set()
                temp.add(n)
        result.append(temp)

        print(result)
        return len(max(result, key=len))

#   lc.11 最大面积（盛水）
    @notify()
    def maxArea(self, height: List[int] = [1,8,6,2,5,4,8,3,7]) -> int:
        # 计算面积的闭包
        def area(height: List[int], left: int, right: int) -> int:
            return (right-left)*min(height[left], height[right])

        # 绘制xy轴坐标系以及高度线的闭包
        def plot_heights():
            plt.figure(figsize=(8, 5))
            plt.bar(range(len(height)), height, color='skyblue')
            plt.xlabel('Index')
            plt.ylabel('Height')
            plt.title('Heights Visualization')
            plt.grid(True, alpha=0.3)
            plt.savefig('heights_plot.png')
            plt.close()

        # 在控制台输出同样效果的闭包
        def print_heights_console():
            print("Heights Visualization (Console):")
            max_h = max(height) if height else 0
            for level in range(max_h, 0, -1):
                line = ""
                for h in height:
                    if h >= level:
                        line += "█ "
                    else:
                        line += "  "
                print(line.rstrip())
            # 打印x轴
            print("-" * (len(height) * 2))
            indices = " ".join(str(i) for i in range(len(height)))
            print(indices, '\n')
            heights_str = " ".join(str(h) for h in height)

        # 调用绘制闭包
        # plot_heights()
        print_heights_console()

        # 指向数组两端的双指针
        left, right = 0, len(height)-1
        result = 0
        while left<right:
            result = max(area(height, left, right), result)
            # 由于最大面积由高度较小的柱子决定，所以位移小的那根的指针就可以了
            if height[left]<= height[right]:
                # print("左边低，左指针右移")
                left+=1
            else:
                # print("右边低，右指针左移")
                right-=1

        return result

#   lc.15 求和为0且下标不同的三元数组
    @notify()
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        print(nums)
        n = len(nums)
        result = []
        for i in range(n-2):
            # 如果当前值和上一轮值相同则跳过循环
            if nums[i] == nums[i-1] and i>0:
                continue

            l, r = i+1, n-1

            while l < r:
                total = nums[i]+nums[l]+nums[r]
                if total == 0:
                    result.append([nums[i], nums[l], nums[r]])
                # 跳过重复的第二个数
                    while l < r and nums[l] == nums[l+1]:
                        l += 1
                # 跳过重复的第三个数
                    while l < r and nums[r] == nums[r-1]:
                        r -= 1
                    l+=1
                    r-=1
                elif total > 0:
                    r-=1
                else:
                    l+=1
        return result

#   lc.438 返回字符串中所有字母异位词的首字母位置
    @notify()
    def findAnagrams(self, s: str="cbaebabacd", p: str="abc") -> List[int]:
        lp = len(p)
        if len(s)<lp:
            return []
        # 异位词字母相同且数量相同（顺序可忽略），因此使用Counter()同时存储字符和计数。
        cnt_p = Counter(p)
        cnt_wd = Counter()
        result = []
        # 使用滑动窗口，枚举之后窗口从左往右移动扫描并计数（右指针先进，先进先出）
        for right, char in enumerate(s):
            cnt_wd[char] += 1
            left = right-lp+1
            if left<0:
                continue
            if cnt_wd == cnt_p:
                result.append(left)
            # 左指针对应字符“移出”
            cnt_wd[s[left]]-=1
        return result

#   lc.94 中序遍历二叉树，深度优先
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def dfs(node) -> List[int]:
            if not node:                # 1. 到达终点，返回
                return                  # 注意，递归的return不会结束程序，只会结束当前调用，dfs(None) -> []
            dfs(node.left)              # 2. 先探索整个左子树
            result.append(node.val)     # 3. 访问当前节点（根）
            dfs(node.right)             # 4. 再探索整个右子树

        result = []
        dfs(root)
        return result

#   lc.104 二叉树最大深度
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        # 闭包dfs，每次向下探索时传入上一层深度
        def dfs(node: Optional[TreeNode], cnt):
            if not node:
                return
            cnt+=1
            nonlocal res                # nonlocal用于维护闭包外变量
            res = max(res, cnt)
            dfs(node.left, cnt)
            dfs(node.right, cnt)

        res = 0
        dfs(root,0)
        return res

        # 等同于如下写法：
        # return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1 if root else 0

#   lc.560 和为k的子数组
    @notify()
    def subarraySum(self, nums: List[int]=[-1,-1,1], k: int=0) -> int:
        # 建立前缀和字典，初始值s[0]=0
        pre_cnt = defaultdict(int)
        pre_sum = 0
        res = 0

        for n in nums:
            # 维护∑。如果满足∑=k的子数组存在，则字典中应有满足当前【前缀和-k】的item
            pre_cnt[pre_sum] += 1
            pre_sum += n
            res += pre_cnt[pre_sum-k]

        print(pre_cnt)
        return res

#   lc.53 最大子数组和
    @notify()
    def maxSubArray(self, nums: List[int]=[-2,1,-3,4,-1,2,1,-5,4]) -> int:
        # 在遍历前缀和的同时，减去最小前缀和，即是最大的子数组和
        ans = float('-inf') # Python内置的负无穷
        pre_sum = 0
        pre_min = 0

        for n in nums:
            pre_sum += n
            ans = max(pre_sum - pre_min, ans)
            pre_min = min(pre_min, pre_sum)
        return ans

#   lc.121 买卖股票最大收益
    @notify()
    def maxProfit(self, prices: List[int]=[7,1,5,3,6,4]) -> int:
        min_price = float('inf')
        profit = 0
        # 每日维护最低价，最大收益=当前价-历史最低价(so far)
        for p in prices:
            profit = max(p-min_price, profit)
            min_price = min(p, min_price)
        return profit

#   lc.238 除自身的其他元素乘积
    @notify()
    def productExceptSelf(self, nums: List[int]=[1,2,3,4]) -> List[int]:
        length = len(nums)
        left = []
        right = []

        # 每一项等于前/后一项的“前/后缀积”, p表示两端积默认为1
        p = 1
        for n in nums:
            left.append(p)
            p *= n
        print(left)
        p = 1
        for n in reversed(nums):
            right.append(p)
            p *= n
        right = list(reversed(right))
        print(right)

        # 用列表推导式输出
        res = [left[i] * right[i] for i in range(length)]
        return res

#   lc.73 矩阵置0
    def setZeroes(self, matrix: List[List[int]]=[[0,1,2,0],[3,4,5,2],[1,3,1,5]]):
        length = len(matrix[0])

        # 先记录带0的行
        line_with_zero = []
        for i in range(len(matrix)):
            if matrix[i].count(0) > 0:
                line_with_zero.append(i)
        print(line_with_zero)

        # 根据‘带0行’中0的位置记录所有0所在的列，使用set()保证去重
        zero_column = set()
        for l in line_with_zero:
            enum = enumerate(matrix[l])
            for k, v in enum:
                if v == 0:
                    zero_column.add(k)
        print(zero_column)

        # 根据带0行列更新对应位置为0
        for m in matrix:
            for zc in zero_column:
                m[zc] = 0
        for l in line_with_zero:
            matrix[l] = [0] * length

        for l in matrix:
            print(l)

#   lc.48 旋转矩阵
    def rotate(self, matrix: List[List[int]]=[[1,2,3],[4,5,6],[7,8,9]]) -> None:
        for m in matrix:
            print(m)
        print('')

        # # 使用numpy的rot方法直接旋转
        # m_rotated = np.rot90(matrix, -1).tolist()
        # for m in m_rotated:
        #     print(m)
        # print('')

        # 用zip缝合每一列，得到翻转矩阵，注意此时元素为tuple
        res = list(zip(*matrix))
        # 遍历res将每一行并反转即顺时针90°
        for i in range(len(res)):
            r = list(reversed(list(res[i])))
            matrix[i] = r

        for m in matrix:
            print(m)

#   lc.46 全排列
    @notify()
    def permute(self, nums: List[int]=[1,2,3]) -> List[List[int]]:
        '''
        Call itertools.permutations to calculate permutations of a list
        '''
        perm = it.permutations(nums)    # permutations输出为tuple, e.g. (1,2,3), (1,3,2)...
        res = [list(p) for p in perm]
        return res

adv = CodeRunner()
# adv.permute()
# adv.longestCommonPrefix()
# adv.rotate()
# adv.setZeroes()
# adv.productExceptSelf()
# adv.maxProfit()
# adv.maxSubArray()
# adv.subarraySum()
# adv.findAnagrams( )
# adv.threeSum(lc15_testdata)
# adv.maxArea()
# adv.longestConsecutive()
# adv.groupAnagrams()
# adv.numIdenticalPairs([1,2,3,1,1,3])
# adv.mySqrt(2052228396)
# adv.convert_to_Zstring("PAYPALISHIRING", 4)
# adv.addBinary()
# adv.longestPalindrome()
# adv.findMedianSortedArrays(nums1=[1,2,5], nums2=[4,4])
# print(adv.remove_camel("abc==aAa==defghij#@$%^&*(ZZZ)_klahHhAamnop==bBb==777qrstu~~ppp~~vwxyz"))
# adv.lengthOfLongestSubstring("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ abc")
# adv.searchInsert()
# adv.unique_char()
# adv.removeDuplicates([0,0,1,1,1,2,2,3,3,4])
# adv.removeElement([0,0,1,1,1,2,2,3,3,4], 1)
# adv.strStr("hahasadnotsad", "sad")
# adv.twoSum()