from collections import Counter

import numpy as np
import math
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution(object):
    def longestCommonPrefix(self, strs):
        if not strs:
            return ""

        for i, letter_group in enumerate(zip(*strs)):
            if len(set(letter_group)) > 1:
                return strs[0][:i]
        else:
            return min(strs)

    def twoSum(self, nums, target):
        dic = dict()
        for index, fig in enumerate(nums):
            if target - fig in dic:
                return dic[target - fig], index
            dic[fig] = index

    def searchInsert(self, nums, target):
        if target in nums:
            return (nums.index(target))
        else:
            nums.append(target)
            nums.sort()
            return (nums.index(target))

    def plusOne(self, digits):
        ans = 0
        digits = digits[::-1]
        ans = digits[0] + 1
        for i in range(len(digits)):
            if i != 0:
                ans = ans + digits[i] * (10 ** i)

        digits = str(ans)
        x = list(digits[::1])
        return (x)

    def singleNumber(self, nums):
        dic = dict()
        numset = set(nums)
        for i in numset:
            dic[i] = 0
            return(dic)
        for i in nums:
            dic[i] += 1
        for i in numset:
            if dic[i] == 1:
                return (i)

    def search(self, nums, target):
        l = 0
        r = len(nums)
        while (l < r):
            m = l + (r - l) // 2
            # return(m)
            if (nums[m] == target):
                return m
            elif (nums[m] < target):
                l = m + 1
            else:
                r = m
        return -1

    def sortedSquares(self, nums):
        x = list()
        for i in nums:
            x.append(i ** 2)
        x.sort()
        return (x)

    def rotate(self, nums, k):
        l = len(nums)
        nums_new = nums[l - k:] + nums[:l - k]
        return (nums_new)

    def moveZeroes(self, nums):
        nums.sort()
        l = len(nums)
        amount = nums.count(0)
        num_new = nums[l - amount - 1:] + nums[:l - amount - 1]
        return (num_new)

    def reverseString(self, s):
        return (s[::-1])

    def reverseWords(self, s):
        new = list()
        x = s.split(" ")
        for i in range(len(x)):
            new.append(x[i][::-1])
        new = " ".join((new))
        return (new)

    def rotate1(self, matrix):
        n = len(matrix)
        for i in range(n):
            for j in range(i, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        for i in range(n):
            matrix[i].reverse()
        return matrix

    def spiralOrder(self, matrix):
        return matrix and [*matrix.pop(0)] + self.spiralOrder([*zip(*matrix)][::-1])

    def minDistance(self, word1, word2) :
        list1 = list(word1)
        list2 = list(word2)
        return(list(set(list1).intersection(set(list2))))

    def trailingZeroes(self, n) :
        # count = 0
        # ans = math.factorial(n)
        # m = ans
        # while ans >= 10:
        #     i, x = divmod(m, 10)
        #     if x == 0:
        #         count += 1
        #         m = i
        #     else:
        #         break
        # return (count)
        x = 5
        res = 0
        while x <= n:
            res += n // x
            x *= 5
        return res

    def minSubsequence(self, nums) :
        nums.sort()
        res = []
        while sum(res) <= sum(nums):
            res.append(nums.pop())
        return res

    def maxDistance(self, colors):
        i, j = 0, len(colors) - 1
        while colors[0] == colors[j]:
            j -= 1
        while colors[-1] == colors[i]:
            i += 1
        return (max(j, len(colors) - 1 - i))

    def numDecodings(self, s):
        pre_way, now_way, digit = 0, int(s > ""), ""
        for load in s:
            pre_way, now_way, digit = now_way, (10 <= int(digit + load) <= 26) * pre_way + int(
                load != "0") * now_way, load
        return (now_way)

    def containsDuplicate(self, nums) :
        set_nums = list(set(nums))
        if len(set_nums) < len(nums) :
            return True
        else :
            return False

    def maxSubArray(self, nums) :
        for i in range(1, len(nums)):
            if nums[i - 1] >= 0:
                nums[i] += nums[i - 1]
        return (max(nums))

    def fib(self, n) :
        a,b = 0,1
        for i in range(n):
            a,b = b,a + b
        return a

    def tribonacci(self, n) :
        if n == 0:
            return 0
        elif n < 3:
            return 1
        a, b, c = 0, 1, 1
        for i in range(3,n+1):
            a, b, c = b, c, a + b + c
        return c

    def runningSum(self, nums) :
        ans = []
        temp = 0
        for i in range(len(nums)):
            temp = temp + nums[i]
            ans.append(temp)
        return (ans)

    def pivotIndex(self, nums) :
        for i in range(len(nums)):
            if sum(nums[0:i]) == sum(nums[i + 1::]):
                return i
        return -1

    def average(self, salary) :
        salary.sort()
        salary.pop(0)
        salary.pop()
        return sum(salary) / len(salary)

    def merge(self, nums1, m, nums2, n) :
        nums1[m:] = nums2[:n]
        nums1.sort()

    def isPowerOfTwo(self, n) :
        return (n > 0 and not (n & n - 1))

    def intersect(self, nums1, nums2) :
        a, b = Counter(nums1), Counter(nums2)
        return (list((a & b).elements()))

    def maxProfit(self, prices) :
        buy_in, profit = float("inf"), 0
        for p in prices:
            if p < buy_in:
                buy_in = p
            if p - buy_in > profit:
                profit = p - buy_in
        return (profit)

    def matrixReshape(self, nums, r, c):
        try:
            x = np.array(nums)
            return x.reshape(r, c).tolist()
        except:
            return nums

    def searchMatrix(self, matrix, target) :
        new = (np.array(matrix).reshape(-1)).tolist()
        try:
            if new.index(target) >= 0:
                return True
        except:
            return False

    def firstUniqChar(self, s) :
        x = Counter(list(s))
        for i in (x):
            if x[i] == 1:
                return (s.index(i))
        return -1

    def isAnagram(self, s, t) :
        x, y = Counter(s), Counter(t)
        return (x == y)

    def subtractProductAndSum(self, n) :
        # x = list(str(n))
        # product, sum = 1, 0
        # for i in range(len(x)):
        #     product *= int(x[i])
        #     sum += int(x[i])
        # return (product - sum)

        a = [int(x) for x in str(n)]
        return np.prod(a) - np.sum(a)

    def nearestValidPoint(self, x, y, points) :
        tar, min = [], float("inf")
        for i in range(len(points)):
            if points[i][0] == x:
                tar.append(sum(points[i]))
            if points[i][1] == y:
                tar.append(sum(points[i]))
        tar.sort()
        for i in range(1, len(tar)):
            if tar[i] - tar[i - 1] < min:
                min = tar[i] - tar[i - 1]
        if min == float("inf"): min = -1
        return (min)

    def arraySign(self, nums) :
        x = np.array([nums])
        return (1 if np.prod(x) > 0 else -1 if np.prod(x) < 0 else 0)

    def areAlmostEqual(self, s1, s2) :
        change = 2
        x, y = [ord(i) for i in s1], [ord(i) for i in s2]
        if sorted(x) == sorted(y):
            for i in range(len(x)):
                if x[i] != y[i]: change -= 1

            return (change >= 0)
        return False

    def uncommonFromSentences(self, s1, s2) :
        x = s1.split(" ") + s2.split(" ")
        y = Counter(x)
        ans = []
        for i, k in y.items():
            if k == 1:
                ans.append(i)
        return (ans)

    def constructMaximumBinaryTree(self, nums):
        def construct(l, r):
            if l > r:
                return
            elif l == r:
                return TreeNode(nums[l])
            maximum = -float('inf')
            for i in range(l, r + 1):
                if maximum < nums[i]:
                    maximum = nums[i]
                    max_index = i
            left = construct(l, max_index - 1)
            right = construct(max_index + 1, r)
            root = TreeNode(maximum)
            root.left = left
            root.right = right
            return root
        return construct(0, len(nums) - 1)

    def arithmeticTriplets(self, nums, diff) :
        cnt = 0
        for x in nums:
            if (x + diff) in nums and (x + 2 * diff) in nums:
                cnt = cnt + 1
        return (cnt)

    def isHappy(self, n) :
        num = set()
        while n != 1:
            n = sum(int(i) ** 2 for i in str(n))
            if n in num:
                return False
            else:
                num.add(n)
        else:
            return True

    def diagonalSum(self, mat) :
        sum = 0
        long = len(mat)
        if long % 2 == 0:
            for i in range(long):
                sum += (mat[i][i] + mat[i][long - 1 - i])
            return (sum)
        else:
            for i in range(long):
                sum += (mat[i][i] + mat[i][long - 1 - i])
            sum -= mat[long // 2][long // 2]
            return (sum)

    def findNumbers(self, nums) :
        cnt = 0
        for i in nums:
            cnt += 1 - len(str(i)) % 2
        return (cnt)

    def carPooling(self, trips, capacity) :
        process = [];man = 0
        for num, start, end in trips:
            process.append((start, num))
            process.append((end, -num))
        process.sort()

        for i in process:
            man += i[1]
            if man > capacity:
                return 0
        return 1

    def maxDepth(self, s) :
        cnt, num = 0, 0
        for i in range(len(s)):
            if s[i] == "(":
                cnt += 1
                num = max(cnt, num)
            elif s[i] == ")":
                cnt -= 1
        return (num)

    def hasAlternatingBits(self, n):
        """
        :type n: int
        :rtype: bool
        """
        n_bin = "{0:b}".format(n)
        for i in range(len(n_bin)-1):
            if n_bin[i] == n_bin[i+1]:
                return(False)
        return(True)

    def findTheLongestBalancedSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        maxs = 0
        zeros = 0
        ones = 0
        # Initialize the variables
        for i in s:
            # Loop through the list
            if i == "0" and ones == 0:
                zeros += 1
            # Only increase zeros if ones is 0
            # to make sure all zeros are before ones.
            elif zeros != 0 and i == "1":
                ones += 1
            # Now find the consecutive ones
            elif zeros != 0 and ones != 0 and i == "0":
                maxs = max(maxs, min(ones, zeros) * 2)
                zeros, ones = 1, 0
            # Finally, check to see if the current substring is
            # greater than the max substring
        return max(maxs, min(ones, zeros) * 2)

    def validPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        s = s + " "
        s_set = set(s)
        for i in s_set:
            s_spilt = s.replace(i,"")
            s_spilt = s_spilt.replace(" ","")
            print(s_spilt,"".join(reversed(s_spilt)))
            if s_spilt == "".join(reversed(s_spilt)):
                return(True)
        return(False)

    # ...........................................................
ss = Solution()

# .........................data..............................


# .........................main..............................
if __name__ == '__main__':
    print(ss.validPalindrome(s))
