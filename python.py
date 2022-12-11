from heapq import heappop, heappush
import math
from collections import Counter

class TreeNode:
    def __init__(self, val, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right
class ListNode:
    def __init__(self, val, next=None) -> None:
        self.val = val
        self.next = next

# 26. remove duplicates from sorted array

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        k = 1
        for (i) in range(len(nums)):
            if nums[i]!=nums[k-1]:
                nums[k] = nums[i]
                k+=1
        return k




# 122. best time to buy and sell stock ii

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        buy = prices[0]
        sell = 0
        for price in prices[1:]:
            sell = max(sell, sell + price - buy)
            buy = price
        
        return sell




# 189. rotate array
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        k = k%len(nums)
        moved = 0
        
        for i in range (0,k):
            if moved == len(nums): 
                break
            j = (i + k)%len(nums)
            next_val = nums[i]
            temp = nums[j]
            nums[i] = nums[i-k]
            moved += 1

            while j != i:
                if moved == len(nums): 
                    break
                temp = nums[j]
                nums[j] = next_val
                next_val = temp
                j = (j+k)%len(nums)
                moved += 1
                



# 217. contains duplicate

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        no_dups = set(nums)
        return len(no_dups) != len(nums)



# 136 single number
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        xor = 0
        for num in nums: 
            xor ^= num
        return xor
# 349. intersection of two arrays
# no dups
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1 = set(nums1)
        nums2 = set(nums2)
       
        return nums1.intersection(nums2)

# 350. intersection of two arrays ii
#dups
def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        dict = {}
        
        for num in nums1:
            if num not in dict: 
                dict[num] = 0
            dict[num] += 1
        
        overlap = []
        
        for num in nums2: 
            if num in dict:
                overlap.append(num)
                dict[num] -= 1
                if dict[num] == 0:
                    del dict[num]
                    
        return overlap

# 66. plus one

class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        carry = 1
        i = len(digits) - 1
        
        while carry and 0 <= i:
            digits[i] += 1
            carry = 0
            if digits[i] == 10:
                digits[i] = 0
                carry = 1
            i -= 1
        if carry:
            digits.insert(0,1)
        
        return digits


# 283. move zeroes

class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        zero = 0
        
        for i in range(len(nums)):
            if nums[i] == 0:
                continue
            if i == zero:
                zero += 1
                continue

            nums[zero] = nums[i]
            nums[i] = 0
            zero += 1


# 1. two sum

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        prev_sums = {}
        
        for i in range(len(nums)):
            num = nums[i]
            
            if target - num in prev_sums:
                return [i, prev_sums[target-num]]
            
            prev_sums[num] = i


# 48. rotate image
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        min = 0
        max = len(matrix[0]) - 1
        
        while min < max:
            matrix[min], matrix[max] = [matrix[max], matrix[min]]
            min += 1
            max -= 1
        
        for i in range(len(matrix[0])):
            for j in range(i+1,len(matrix[0])):
                matrix[i][j], matrix[j][i] = [matrix[j][i], matrix[i][j]]

# 345. reverse vowels of a string

class Solution:
    def reverseVowels(self, s: str) -> str:
        vowels = {'a', 'e', 'i', 'o', 'u', 'A', 'E', "I", "O", "U"}
        s = list(s)
        l = 0
        r = len(s) - 1

        while l < r:
            while l < len(s) and s[l] not in vowels:
                l+=1
            while 0 < r and s[r] not in vowels:
                r-=1
            if r <= l:
                break
            s[l], s[r] = s[r], s[l]
            l += 1
            r -= 1
        return ''.join(s)


# 7. reverse integer 
class Solution:
    def reverse(self, x: int) -> int:
        sign = -1 if x < 0 else 1
        x = int(str(abs(x))[::-1])*sign
        
        return x if x < 2147483647 and -2147483648 < x else 0


# 344. reverse string
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        lo = 0
        hi = len(s) - 1
        while lo < hi:
            s[lo], s[hi] = s[hi], s[lo]
            lo+= 1
            hi-= 1

# 387. first unique character in a string
class Solution:
    def firstUniqChar(self, s: str) -> int:
        dict = {}
        
        for char in s:
            if char not in dict:
                dict[char] = 0
            dict[char] += 1
        for i in range(len(s)):
            if dict[s[i]] == 1:
                return i
        return -1

# 242. valid anagram
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        dict = {}
        if len(s) != len(t):
            return False
        for c in s:
            if c not in dict:
                dict[c] = 0
            dict[c] += 1
        for c in t: 
            if c not in dict:
                return False
            dict[c] -= 1
            if dict[c] == 0:
                del dict[c]
                
        return True


# 125. valid palindrome

class Solution:
    def isPalindrome(self, s: str) -> bool:
        l = 0
        r = len(s) -1
       
        while l < r:
            while l < len(s) and not s[l].isalnum():
                l+=1
            while 0 < r and not s[r].isalnum():
                r-=1
            if r < l:
                break
            if s[l].lower() != s[r].lower():
                return False
            l+=1
            r-=1
            
        return True

# 8. string to integer (atoi)
class Solution:
    def myAtoi(self, s: str) -> int:
        num = 0 
        sign = 1
        
        i = 0
        s = s.lstrip()
        if not s:
            return 0
        elif s[i] == '-':
            sign = -1
            i+= 1
        elif s[i] == '+':
            i+= 1
        while i < len(s) and s[i].isnumeric():
            num *= 10
            num += int(s[i])
            i+=1
        num *= sign
        if num < -2147483648:
            return -2147483648
        if 2147483647 < num:
            return 2147483647
        return num

# 28. find the index of the first occurence in a string
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        for i in range(len(haystack)):
            
            if haystack[i] == needle[0]:
                j = 0
                k = i
                while j < len(needle) and k < len(haystack) and haystack[k] == needle[j]:
                    k += 1
                    j += 1
                if j == len(needle):
                    return i
        return -1
# 14. longest common prefix
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        common = strs[0]
        
        for word in strs:
            i = 0
            while i < len(word) and i < len(common) and word[i] == common[i]:
                i += 1
            common = common[:i]
        return common

# 237. delete node in a linked list
# only given the node to be deleted 
class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next

# 19. remove nth node from end of list
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        if not head:
            return
        fast = head
        slow = head
        for i in range(n):
            fast = fast.next
        
        if not fast:
            return slow.next
        while fast.next:
            n -= 1
            fast = fast.next
            slow = slow.next
        
        slow.next = slow.next.next
        return head


# 206. reverse linked list 
# iterative
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = None
        while head:
            temp = head.next
            head.next = cur
            cur = head
            head = temp
        return cur

# recursive
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        
        def rec (head, node):
            if not head:
                return node
            temp = head.next
            head.next = node
            return rec(temp, head)
        return rec(head,None)

# 21 merge two sorted lists
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        sentinel = ListNode('sentinel')
        cur = sentinel
        
        while list1 and list2:
            if list1.val <= list2.val:
                cur.next = list1
                list1 = list1.next
            else:
                cur.next = list2
                list2 = list2.next
            cur = cur.next
        
        cur.next = list1 or list2
        return sentinel.next

# 234. palindrome linked list
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        reversed = reverse(slow)
        slow = head
        while reversed:
            if reversed.val != slow.val:
                return False
            reversed = reversed.next
            slow = slow.next
        return True
def reverse(head):
    cur = None
    
    while head:
        tmp = head.next
        head.next = cur
        cur = head
        head = tmp
    return cur



# 141. linked list cycle 

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head:
            return False
        slow = head
        fast = head.next
        
        while slow != fast and fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        return True if slow == fast else False


# 104. maximum depth of binary tree
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right))+1



# 98. validate binary search tree

class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def helperValid(min, max, root):
            if not root:
                return True
            if root.val <= min or max <= root.val:
                return False
            return helperValid(min, root.val, root.left) and helperValid(root.val, max, root.right)
        return helperValid(-inf, inf, root)

# 101. symmetric tree

class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        
        def helperSym(left,right):
            if not left and not right:
                return True
            if not left or not right:
                return False
            if left.val != right.val:
                return False
            return helperSym(left.left, right.right) and helperSym(left.right, right.left)
        return helperSym(root.left, root.right)



# 102. binary tree level order traversal

class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        level = [root]
        res = []
        while level:
            nextLevel = []
            currLevel = []
            for node in level:
                currLevel.append(node.val)
                if node.left:
                    nextLevel.append(node.left)
                if node.right:
                    nextLevel.append(node.right)
            level = nextLevel
            if currLevel:
                res.append(currLevel)
        return res

# 108. converted sorted array to binary search tree

class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def helper(left, right):
            if right < left:
                return None
            mid = left + (right-left)//2
            root = TreeNode(nums[mid])
            root.left = helper(left, mid-1)
            root.right = helper(mid+1, right)
            return root
        return helper(0, len(nums)-1)



# 88. merge sorted array

class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        k = m+n-1
        n -=1
        m -=1
        while 0 <= k:
            if m < 0:
                nums1[k] = nums2[n]
                n-=1
            elif n < 0:
                nums1[k] = nums1[m]
                m-=1
            elif nums1[m] < nums2[n]:
                nums1[k] = nums2[n]
                n-=1
            else:
                nums1[k] = nums1[m]
                m-=1
            k-=1

# 278. first bad version
class Solution:
    def firstBadVersion(self, n: int) -> int:
        l = 1
        r = n
        
        while l < r:
            mid = l + (r-l)//2
            if isBadVersion(mid):
                r = mid
            else:
                l = mid + 1
        return l





# 70. climbing stairs
# bottom up
class Solution:
    def climbStairs(self, n: int) -> int:
        dp = [0,1,2]
        
        for i in range(2,n):
            dp.append(dp[i] + dp[i-1])
        return dp[n]
# recurssion and memo
class Solution:
    def climbStairs(self, n: int) -> int:
        dp = {}
        
        def helpClimb(n):
            if n <= 3:
                return n
            if n in dp:
                return dp[n]
            
            dp[n] = helpClimb(n-1) + helpClimb(n-2)
            return dp[n]
        
        return helpClimb(n)


# 121. best time to buy and sell stock
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        buy = prices[0]
        
        for num in prices:
            profit = max(profit, num - buy)
            buy = min(buy, num)
        return profit
            
# 53. maximum subarray
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        res = -inf
        curMax = 0
        
        for num in nums:
            curMax += num
            res = max(res, curMax)
            
            curMax = max(0, curMax)
            
        return res

# 198. house robber

class Solution:
    def rob(self, nums: List[int]) -> int:
        res = 0
        dp = [0,0,0]
        for num in nums:
            dp.append(max(dp[-2]+num, dp[-3]+num))
            res = max(res, dp[-1])
        return res


# 384. shuffle an array

class Solution:

    def __init__(self, nums: List[int]):
        self.og = nums
        
    def reset(self) -> List[int]:
        return self.og
    def shuffle(self) -> List[int]:
        shuffled = []
        used = set()
        used.add(-1)
        for i in range(len(self.og)):
            j = -1
            while j in used:
                j = randint(0, len(self.og)-1)
            used.add(j)
            shuffled.append(self.og[j])
        return shuffled
    
# 155. min stack
class MinStack:

    def __init__(self):
        self.stack = []

    def push(self, val: int) -> None:
        if len(self.stack) == 0:
            self.stack.append([val, val])
        else:
            self.stack.append([val, min(self.getMin(), val)])

    def pop(self) -> None:
        return self.stack.pop()[0]

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]



# 1323. maximum 69 number

class Solution:
    def maximum69Number (self, num: int) -> int:
        changed = False
        number = 0
        for n in str(num):
            n = int(n)
            number *= 10
            if n == 6 and not changed:
                number += 9
                changed = True
            else:
                number += n
        return number

# 412. fizz buzz
# no division
class Solution:
    def fizzBuzz(self, n: int) -> List[str]:
        res = []
        fizz = 1
        buzz = 1
        
        for i in range(1,n+1):
            if fizz == 3 and buzz == 5:
                res.append('FizzBuzz')
                fizz = 0
                buzz = 0
            elif fizz == 3:
                res.append('Fizz')
                fizz = 0
            elif buzz == 5:
                res.append('Buzz')
                buzz = 0
            else:
                res.append(str(i))
            fizz += 1
            buzz += 1
        return res


# 204. count primes

class Solution:
    def countPrimes(self, n: int) -> int:
        if n < 3:
            return 0
        nums = [1] * n
        nums[0] = nums[1] = 0
        res = 0    
        for i in range(2,int(sqrt(n)) + 1):
            if nums[i]:
                
                for k in range(i*i,n,i):
                    nums[k] = 0
        return sum(nums)


# 1544. make the string great

class Solution:
    def makeGood(self, s: str) -> str:
        str = []

        for c in s:
            if len(str) == 0:
                str.append(c)
            elif str[-1].lower() == c.lower() and str[-1] != c:
                str.pop()
            else:
                str.append(c)
        return ''.join(str)

# 326. power of three
# if n is a power of three, then n will divide 3**19 
class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        return 0 < n and 1162261467%n == 0

# 13. roman to integer

class Solution:
    def romanToInt(self, s: str) -> int:
        translate = {'M': 1000, 'CM': 900, 'D': 500, 'CD': 400, 'C': 100, 'XC': 90, 'L':50,'XL': 40, 'X': 10, 'IX': 9,'V': 5, 'IV':4, 'I':1}
        
        res = 0
        i = 0
        while i < len(s):
            if i < len(s)-1 and s[i] + s[i+1] in translate:
                res += translate[s[i]+s[i+1]]
                i += 2
            else:
                res+= translate[s[i]]
                i += 1
        return res


# 191. number of 1 bits

class Solution:
    def hammingWeight(self, n: int) -> int:
        bits = [int(i) for i in bin(n)[2:]]
        return sum(bits)

# 461. hamming distance

class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        res = 0
        while x or y:
            if y & 1 and not x & 1:
                res += 1
            elif x & 1 and not y & 1:
                res += 1
            y = y >> 1
            x = x >> 1
        return res

# 190. reverse bits
class Solution:
    def reverseBits(self, n: int) -> int:
        str = ''
        while n:
            if n & 1: 
                str += '1'
            else:
                str += '0'
            n = n >> 1
        while len(str) < 32:
            str += '0'
        return int(str,2)

# 118. pascal's triangle
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        rows = [[1]]
        numRows -= 1
        while numRows:
            nextRow = []
            for i in range (0, len(rows[-1]) + 1):
                if i == 0 or i == len(rows[-1]):
                    nextRow.append(1)
                else:
                    nextRow.append(rows[-1][i] + rows[-1][i-1])
            rows.append(nextRow)
            numRows -= 1
        return rows

# 20. valid parentheses
class Solution:
    def isValid(self, s: str) -> bool:
        convert = {'[':']', '{':'}', '(':')'}
        stack = []
        
        for c in s:
            if c in convert:
                stack.append(convert[c])
            elif len(stack) and c == stack[-1]:
                stack.pop()
            else:
                return False
        return not len(stack)

# 268. missing number
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        res = len(nums)
        for i in range(len(nums)):
            res ^= i
            res ^= nums[i]
        return res

# 15. 3sum
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        
        for i in range(len(nums)):
            if 0 < nums[i]:
                break
            if 0 < i and nums[i] == nums[i-1]:
                continue
            j = i+1
            k = len(nums) -1
            while j < k:
                sum = nums[i] + nums[j] + nums[k]
                if sum == 0:
                    res.append([nums[i], nums[j], nums[k]])
                    j += 1
                    k -= 1
                    while j < k and nums[j-1] == nums[j]:
                        j+=1
                    while j < k and nums[k+1] == nums[k]:
                        k-=1
                elif sum < 0:
                    j+=1
                else:
                    k-=1
        return res

# 73. set matrix zeroes
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        firstCol = False
        firstRow = False
        
        for i in range(len(matrix)):
            if matrix[i][0] == 0:
                firstCol = True
                break
        for j in range(len(matrix[0])):
            if matrix[0][j] == 0:
                firstRow = True
                break

        for i in range(1,len(matrix)):
            for j in range(1, len(matrix[0])):
                if matrix[i][j] == 0:
                    if matrix[0][j] != 0:
                        for k in range(0,i):
                            matrix[k][j] = 0
                    if matrix[i][0] != 0:
                         for k in range(0,j):
                            matrix[i][k] = 0

                elif matrix[0][j] == 0 or matrix[i][0] == 0:
                    matrix[i][j] = 0
        if firstCol:
            for i in range(len(matrix)):
                matrix[i][0] = 0
        if firstRow:
            for j in range(len(matrix[0])):
                matrix[0][j] = 0

# 49. group anagrams
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        dict = {}
        res = []
        for c in strs:
            sortedC = ''.join(sorted(c))
            if sortedC in dict:
                res[dict[sortedC]].append(c)
            else:
                dict[sortedC] = len(res)
                res.append([c])
        return res

# 3. longest substring without repeating characters
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        l = 0
        r = 0
        freq = {}
        maxLen = 0
        while r < len(s):
            
            while s[r] in freq:
                del freq[s[l]]
                l+=1
            freq[s[r]] = True
            maxLen = max(maxLen, r-l+1)
            r+=1
        return maxLen

# 5. longest palindromic substring
class Solution:
    def longestPalindrome(self, s: str) -> str:
        
        def tryPali(l, r):
            
            while 0 <= l and r < len(s) and s[l] == s[r]:
                l-=1
                r+=1
            return s[l+1:r]
        maxPali = ''
        for i in range(len(s)):
            
            even = tryPali(i, i+1)
            odd = tryPali(i, i)
            
            if len(maxPali) < len(even):
                maxPali = even
            if len(maxPali) < len(odd):
                maxPali = odd
        return maxPali

# 334. increasing triplet subsequence 
# remove all nums that dont have a max element to the right
# go through every number again keeping track of the min value
# if current number is greater than min then we know that there exists a triplet subsequence
# otherwise it would be marked as invalid
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        ma = nums[-1]
        
        for j in range(len(nums)-1,-1,-1):
            if nums[j] < ma: continue
            ma = nums[j]
            nums[j] = False
        mi = inf
        
        for i in range(len(nums)-1):
            if type(nums[i]) is not int:
                continue
            if mi < nums[i]:
                return True
            mi = nums[i]
        return False

# 901. online stock spanner

class StockSpanner:

    def __init__(self):
        self.stack = []

    def next(self, price: int) -> int:
        count = 1
        while len(self.stack) and self.stack[-1][0] <= price:
            count += self.stack.pop()[1]
        self.stack.append([price, count])
        return count

# 38. count and say

class Solution:
    def countAndSay(self, n: int) -> str:
        res = '1'
        
        n -= 1
        while n:
            curr = ''
            currNum = res[0]
            count = 0
            for c in res:
                if c == currNum:
                    count += 1
                else:
                    curr += str(count)+currNum
                    currNum = c
                    count = 1
                    
            curr += str(count)+currNum
            n-=1
            res = curr
        return res

# 328. odd even linked list
# keep pointer for current odd and current even
# also keep pointer for the first even so we can connect the two lists at the end
class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        even = head.next
        odd = head
        evenStart = head.next
        while odd.next and even.next:
            odd.next = even.next
            odd = odd.next

            even.next = odd.next
            even = even.next
            
        odd.next = evenStart
        return head

# 1047. remove all adjacet duplicates in string
class Solution:
    def removeDuplicates(self, s: str) -> str:
        word = []
        for c in s:
            if len(word) and c == word[-1]:
                word.pop()
            else:
                word.append(c)
        return ''.join(word)


# 103. binary tree zigzag level order traversal

class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        res = []
        level = [root]
        leftToRight = True
        while level:
            res.append([])
            next_level = []
            
            while level:
                node = level.pop()
                res[-1].append(node.val)
                if leftToRight:
                    if node.left: next_level.append(node.left)
                    if node.right: next_level.append(node.right)
                else:
                    if node.right: next_level.append(node.right)
                    if node.left: next_level.append(node.left)
            leftToRight = not leftToRight
            level = next_level
        return res
                    
# 116. populating next right pointers in each node
class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if not root or not root.left:
            return root
        root.left.next = root.right
        if root.next:
            root.right.next = root.next.left
        self.connect(root.left)
        self.connect(root.right)
        return root

# 200. number of islands 

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        def expandIsland(grid, i,j):
            adj = [[1,0], [-1,-0], [0,1], [0,-1]]
            
            paths= [[i,j]]
            
            while paths:
                
                next_paths = []
                
                while paths:
                    [p,q] = paths.pop()
                    
                    grid[p][q] = '2'
                    for dir in adj:
                        if p+dir[0] < len(grid) and 0 <= p+dir[0] and q+dir[1] < len(grid[0]) and 0 <= q+dir[1] and grid[p+dir[0]][q+dir[1]] == '1':
                            grid[p+dir[0]][q+dir[1]] = '2'
                            next_paths.append([p+dir[0], q+dir[1]])
                
                paths = next_paths
            
        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
               if grid[i][j] == '1':
                res += 1
                expandIsland(grid, i, j)
        return res


# 17. letter combinations of a phone number

class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if len(digits) == 0:
            return []
        
        letters = {'2': ['a', 'b','c'], '3': ['d','e','f'], '4': ['g','h','i'], '5': ['j','k','l'], '6': ['m','n','o'], '7': ['p','q','r','s'], '8': ['t', 'u', 'v'], '9':['w','x','y','z']}
        
        def backtrack(i, digits, combination, res, letters):
            if i == len(digits):
                res.append(combination)
                return res
            
            for c in letters[digits[i]]:
                backtrack(i+1, digits, combination + c, res, letters)
            return res
        
        return backtrack(0, digits, '', [], letters) 


# 22. generate parentheses
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        
        def backtrack(combination, o, c, res):
            if o == 0 and c == 0:
                res.append(combination)
                return
            
            if c == o:
                backtrack(combination+'(', o-1, c, res)
            else:
                if o == 0:
                    backtrack(combination+')', o, c-1,res)
                else:
                    backtrack(combination+')', o, c-1,res)
                    backtrack(combination+'(', o-1, c, res)
        
        res = []
        backtrack('', n, n, res)
        return res
        

# 78. subsets

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        dp = [[]]
        
        for num in nums:
            for i in range(len(dp)):
                new = dp[i][:]
                new.append(num)
                dp.append(new)
        return dp

# 79. word search

class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        
        
        def travel(p, i, j, board):
            if p == len(word):
                return True
            
            neigh = [[1,0], [0,1], [-1,0], [0,-1]]
            
            for adj in neigh:
                r, c = i+adj[0], j+adj[1]
                if 0 <= c and 0 <= r and r < len(board) and c < len(board[0]) and board[r][c] == word[p]:
                    board[r][c] = '#'
                    res = travel(p+1, r, c, board)
                    if res: return True
                    board[r][c] = word[p]
            return False
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == word[0]:
                    board[i][j] = '#'
                    if travel(1, i,j, board):
                        return True
                    board[i][j] = word[0]
        return False

# 75. sort colors 
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        blue = 0
        red = len(nums)-1
        i = 0
        while i < len(nums):
            num = nums[i]
            if num == 1:
                i+=1 
                continue
            elif num == 0 and blue < i:
                nums[i], nums[blue] = nums[blue], nums[i]
                blue += 1
                i -= 1
            elif num == 2 and i < red:
                nums[i], nums[red] = nums[red], nums[i]
                red -= 1
                i -= 1
            i += 1

# 295. find median from data stream 
# using 2 priority queues, numbers on the left are sorted using the * -1 trick because heaps in python keep track of min element
# left list is sorted as max queue
# right list is sorted as min queue
# middle element will be top left element or will be the sum of both 
class MedianFinder:

    def __init__(self):
        self.left = []
        self.right = []

    def addNum(self, num: int) -> None:
        heappush(self.right, num)
        if len(self.left) < len(self.right):
            heappush(self.left, -1 * heappop(self.right))
        if len(self.right) and self.right[0] < self.left[0]*-1:
            heappush(self.right, -1 * heappop(self.left))
            heappush(self.left, -1 * heappop(self.right))
    def findMedian(self) -> float:
        if len(self.left) == len(self.right):
            return (self.left[0]*-1 + self.right[0])/2
        else:
            return self.left[0]*-1

# 215. kth largest element in an array
# using pivot sort to find an element with k-1 elements larger than it 
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        

        def pivotSort(low, high):
            pivot = nums[high]
            j = high
            high = high - 1
            
            while low <= high:
                if pivot <= nums[low]:
                    nums[low], nums[high] = nums[high], nums[low]
                    high -= 1
                else:
                    low += 1
            nums[low], nums[j] = pivot, nums[low]
            return low
        
        l = 0
        r = len(nums) - 1
        n = r - k + 1
        
        j = None
        while j != n:
            j = pivotSort(l,r)
            if j < n:
                l = j+1
            else:
                r = j-1
        return nums[n]

# 162. find peak element
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        if len(nums) == 1: return 0
        l = 0
        r = len(nums) - 1
        mid = None
        while l <= r:
            mid = l + (r-l)//2
            if (mid == 0 and nums[mid+1] < nums[mid]) or (mid == len(nums)- 1 and nums[mid-1] < nums[mid]) or (nums[mid+1] < nums[mid] and nums[mid-1] < nums[mid]):
                return mid
            elif mid != len(nums)-1 and nums[mid] < nums[mid+1]:
                l = mid+1
            else:
                r = mid-1
        return mid
                
# 34. find first and last position of element in sorted array

class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if not nums:
            return [-1,-1]
        l = 0
        r = len(nums) - 1
        mid = None
        while l <= r:
            mid = l + (r-l)//2
            
            if nums[mid] == target and (mid == 0 or nums[mid-1] != target):
                break
            elif nums[mid] < target:
                l = mid+1
            else:
                r = mid-1
        
        if mid < 0 or mid == len(nums) or nums[mid] != target:
            return [-1,-1]
    
        start = mid
        
        l = mid
        r = len(nums) - 1
        
        while l <= r:
            mid = l + (r-l)//2
            if nums[mid] == target and (mid == len(nums) - 1 or nums[mid+1] != target):
                return [start, mid]
            elif target < nums[mid]:
                r = mid-1
            else: 
                l = mid+1
        return [start, l]

# 151. reverse words in a string
class Solution:
    def reverseWords(self, s: str) -> str:
        s = s.split()
        s.reverse()
        
        return ' '.join(s)

# 2470. number of subarrays with lcm equal to k

class Solution:
    def subarrayLCM(self, nums: List[int], k: int) -> int:
        def track(array, j, res, l):
            if j == len(nums) or k%nums[j] != 0 or k < l:
                return res
            array.append(nums[j])
            newL = math.lcm(l, nums[j])
            if newL == k:
                res += 1
            return track(array, j+1, res, newL)
    
        res = 0
        for i in range(len(nums)):
            res += track([], i, 0, 1)
        return res


# 2471. minimum number of operations to sort a binary tree by level
class Solution:
    def minimumOperations(self, root: Optional[TreeNode]) -> int:
        
        res = 0
        
        
        level = [root]
        
        while level:
            next_level = []
            
            for node in level:
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
            
            x = sorted(next_level, key=lambda x: x.val)
            index = {}
            for i in range(len(x)):
                index[x[i]] = i
            temp = 0
            
            for i in range(len(x)):
                if (x[i] != next_level[i]):
                    res += 1
                    temp = x[i]
                    
                    x[i], x[index[next_level[i]]] = x[index[next_level[i]]], x[i]
                    index[temp] = index[next_level[i]]
                    index[next_level[i]] = i
            
            level = next_level
        return res

# 2469. convert the temperature 

class Solution:
    def convertTemperature(self, celsius: float) -> List[float]:
        return [celsius+273.15, celsius*1.8 + 32]

# 1424. diagonal traverse ii

class Solution:
    def findDiagonalOrder(self, nums: List[List[int]]) -> List[int]:
        store = {}
        longest = len(max(nums, key=len))
        
        for i in range(len(nums)-1, -1, -1):
            for j in range(len(nums[i])-1, -1, -1):

                    if i+j not in store:
                        store[i+j] = []
                    
                    store[i+j].append(nums[i][j])
        i = 0
        res = []
        while i in store:
            res.extend(store[i])
            i+= 1
        return res

# 347. guess number higher or lower


class Solution:
    def guessNumber(self, n: int) -> int:
        low = 1
        high = n
        while low <= high:
            mid = low + (high - low)//2
            g = guess(mid)
            if g == 0:
                return mid
            elif g == -1:
                high = mid - 1
            else:
                low = mid + 1

# 23. merge k sorted lists
# nodes dont accept comparison and heaps dont allow altering the native comparison
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        sentinel = ListNode('sent')
        curr = sentinel
        hp = []

        for i in range(len(lists)):
            if lists[i]:
                heappush(hp, (lists[i].val, i, lists[i]))
        while hp:
            obj = heappop(hp)
            curr.next = obj[2]
            if obj[2].next:
                heappush(hp, (obj[2].next.val, obj[1], obj[2].next))
            curr = curr.next
        return sentinel.next

# 263. ugly number
# divide number until its no longer a factor of 2,3 or 5 
# at which case it must be equal to 1 otherwise it has another prime divisor
class Solution:
    def isUgly(self, n: int) -> bool: 
        if n < 1: return False 
        while n%5 == 0: n /= 5
        while n%3 == 0: n /= 3
        while n%2 == 0: n /= 2

        return n == 1


# 947. most stones removed with same row or column
# using ~j and i to have unique keys for row and column
class Solution:
    def removeStones(self, stones: List[List[int]]) -> int:
        parents = {}

        def find(x):
            if parents[x] == x:
                return x
            else:
                return find(parents[x])
        
        for [i,j] in stones:
            parents.setdefault(i,i)
            parents.setdefault(~j,~j)
            pi = find(i)
            pj = find(~j)
            parents[pi] = pj
        seen = set()
        res = 0
        
        for key in parents:
            seen.add(find(key))
        
        return len(stones) - len(seen)

# 264. ugly number ii
# every ugly number after 1 is of the form 2^x * 3^y * 5^z
# enumerate all possiblities in order => using heap to keep the order correct
# 0 is technically not an ugly but its used so we can keep the first conditional simple hence the n+1 as the exit condition
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        hp = [1]
        uglies = [0]

        while len(uglies) < n+1:
            num = heappop(hp)
            if num <= uglies[-1]:
                continue
            uglies.append(num)
            if num%5 == 0:
                heappush(hp, num*5)
            if num%3 == 0:
                heappush(hp, num*3)
                heappush(hp, num*5)
            else:
                heappush(hp, num*2)
                heappush(hp, num*3)
                heappush(hp, num*5)
                
        return uglies[-1]

# 313. super ugly number
# same logic as ugly number ii
# using num%prime == 0 to break out of prime list and add unique numbers to heap
class Solution:
    def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
        uglies = [0]
        hp = [1]

        while len(uglies) < n+1:
            num = heappop(hp)
            if num <= uglies[-1]:
                continue
            uglies.append(num)
            for prime in primes:
                heappush(hp, num * prime)
                if num%prime == 0:
                    break

        return uglies[-1]



# 224. basic calculator 

class Solution:
    def calculate(self, s: str) -> int:
        curr_sum = 0
        sign = 1
        stack = []
        i = 0
        while i < len(s):
            c = s[i]
            if c == ' ':
                pass
            elif c == '(':
                stack.append(curr_sum)
                stack.append(sign)
                curr_sum = 0
                sign = 1
            elif c == ')':
                sign = stack.pop()
                val = stack.pop()
                curr_sum *= sign 
                curr_sum += val
            elif c == '-':
                sign = -1
            elif c == '+':
                sign = 1
            else:
                num = ''
                while i < len(s) and c.isdigit():
                    num += c
                    i += 1
                    if i != len(s):
                        c = s[i]
                i -= 1
                num = int(num) * sign
                sign = 1
                curr_sum += num
            i += 1
        return curr_sum

# 1909. remove one element to make the array strictly increasing

class Solution:
    def canBeIncreasing(self, nums: List[int]) -> bool:        
        res = 0
        mx =  nums[0]
        for i in range(1, len(nums)):
            if nums[i] <= mx:
                res += 1
                if i > 1 and nums[i] <= nums[i-2]:
                    continue
            mx = nums[i]
        return res < 2 

# 1926. nearest exit from entrance in maze
# classic bfs problem just extra careful to not exit through entrance
class Solution:
    def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
        steps = 0

        moves = [[entrance[0],entrance[1]]]

        maze[entrance[0]][entrance[1]] = '+'
        adj = [[1,0], [0,1], [-1,0], [0,-1]]


        while moves:
            next_step = []

            for i,j in moves:
                for x,y in adj:
                    r,c = [i+x, j+y]
                    if r < 0 or c < 0 or r == len(maze) or c == len(maze[0]):
                        if (i == entrance[0] and j == entrance[1]):
                            continue
                        return steps
                    if maze[r][c] != '+':
                        maze[r][c] = '+'
                        next_step.append([r,c])
            moves = next_step
            steps += 1
        return -1

# 279. perfect squares
class Solution:
    def numSquares(self, n: int) -> int:
        res = []

        for i in range(n+1):
            res.append(i)
        
        for i in range(2,floor(math.sqrt(n))+1):
            num = i*i
            for x in range(num, n+1):
                res[x] = min(res[x], res[x-num]+1)
        return res[n]

# 673. number of longest increasing subsequence

class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        lengths = {}
        counts = {}
        max_length = 1

        for i in range(len(nums)):
            lengths[i] = 1
            counts[i] = 1
            for j in range(0, i):

                if nums[j] < nums[i]:

                    if lengths[i] < lengths[j]+1:
                        counts[i] = counts[j]
                        lengths[i] = lengths[j]+1
                    elif lengths[i] == lengths[j]+1:
                        counts[i] += counts[j]
            max_length = max(lengths[i], max_length)
        res = 0
        for i in range(len(nums)):
            if lengths[i] == max_length:
                res += counts[i]
        return res



# 2272. substring with largest variance
# consider each unique letter pairing 
# during the iteration we increment or decrement our variance depending on those 2 letters
# edge case when our sequence starts with our 'b' letter => we can remove this letter if we find another 'b'
class Solution:
    def largestVariance(self, s: str) -> int:
        letters = set(s)
        max_v = 0
        for a in letters:
            for b in letters:
                if a == b: continue
                var = 0
                found_b = 0
                first_b = 0
                for c in s:
                    if c == a:
                        var += 1
                    elif c == b:
                        var -= 1
                        found_b = 1
                        if var < -1 or (var == -1 and not first_b):
                            first_b = 1
                            var = -1
                        elif first_b:
                            first_b = 0
                            var += 1

                    if found_b:
                        max_v = max(max_v, var)
                
                    
        return max_v

# 36. valid sudoku

class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:

        for i in range(9):
            row = set()
            col = set()
            square = set()
            for j in range(9):
                _row = board[i][j]
                _col = board[j][i]
                _square = board[3*(i//3) + j//3][3*(i%3) + j%3]

                if _row != '.' and _row in row:
                    return False
                if _col != '.' and _col in col:
                    return False
                if _square != '.' and _square in square:
                    return False
                row.add(_row)
                col.add(_col)
                square.add(_square)
        return True


# 37. sudoku solver

class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        rows = [set() for i in range(9)]
        cols = [set() for i in range(9)]
        squares = [set() for i in range(9)]
        
        #square => 3*i//3 + j//3
        # row => i
        # col => j
        queue = []
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    queue.append([i,j])
                else:
                    rows[i].add(board[i][j])
                    cols[j].add(board[i][j])
                    squares[3*(i//3) + j//3].add(board[i][j])
        
        def backtrack(ind):
            if ind == len(queue):
                return True
            i,j = queue[ind]
            for x in range(1,10):
                x = str(x)
                if x in rows[i] or x in cols[j] or x in squares[3*(i//3) + j//3]:
                    continue
                board[i][j] = x
                rows[i].add(x), cols[j].add(x), squares[3*(i//3) + j//3].add(x)
                res = backtrack(ind+1)
                if res:
                    return True
                rows[i].remove(x), cols[j].remove(x), squares[3*(i//3) + j//3].remove(x)
            board[i][j] = '.'
            return False
        
        backtrack(0)
        return board


# 980. unique paths iii
# count number of squares we need to travel, once we reach the end square check if weve traveled the required squares and increment result accordingly
# simple backtracking algo to get all possible paths
class Solution:
    def uniquePathsIII(self, grid: List[List[int]]) -> int:
        num_squares = 1
        start_ind = -1
        end_ind = -1
        m = len(grid)
        n = len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    start_ind = [i,j]
                elif grid[i][j] == 0:
                    num_squares += 1
                elif grid[i][j] == 2:
                    end_ind = [i,j]

        self.res = 0

        def track(squares, i,j):
            if i == end_ind[0] and j == end_ind[1]:
                if squares == num_squares:
                    self.res += 1
                return
            
            dir = [[1,0], [-1,0], [0,1], [0,-1]]
            grid[i][j] = -1
            for adj in dir:
                x,y = i+adj[0], j+adj[1]

                if x < 0 or y < 0 or x == m or y == n or grid[x][y] == -1:
                    continue
                track(squares+1, x,y)
            grid[i][j] = 0
            return
        
        track(0,start_ind[0], start_ind[1])
        return self.res    


# 63. unique paths ii
# on each tile that is not an obstacle we can grab the step count by looking up and to the left
# only adding their step count if they are not an obstacle
# using negative numbers to not override obstacle numbers 
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        if obstacleGrid[-1][-1] == 1 or obstacleGrid[0][0] == 1:
            return 0
        
        self.res = 0
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        obstacleGrid[0][0] = -1
        dir = [[1,0], [0,1]]

        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j] == 1:
                    continue
                if i == 0 and j == 0:
                    obstacleGrid[i][j] = -1
                elif i == 0 and obstacleGrid[i][j-1] != 1:
                    obstacleGrid[i][j] += obstacleGrid[i][j-1]
                elif j == 0 and obstacleGrid[i-1][j] != 1:
                    obstacleGrid[i][j] += obstacleGrid[i-1][j]
                else:
                    if obstacleGrid[i][j-1] != 1:
                        obstacleGrid[i][j] += obstacleGrid[i][j-1]
                    if obstacleGrid[i-1][j] != 1:
                        obstacleGrid[i][j] += obstacleGrid[i-1][j]
       

        return -1*obstacleGrid[-1][-1]
        

# 498. diagonal traverse
# get all diagonals in order from bottom up
# if its an even diagonal we must reverse the list because we will travel top down

class Solution:
    def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
        m = len(mat)
        n = len(mat[0])
        digs = [[] for i in range((m) + (n-1))]
        
        
        for i in range(m):
            for j in range(n):
                print(i,j)
                digs[i+j].append(mat[i][j])
        
        res = []

        for i in range(len(digs)):
            if i%2:
                for j in digs[i]:
                    res.append(j)
            else:
                for j in reversed(digs[i]):
                    res.append(j)
        return res

# 474. ones and zeroes

class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        def getCount(x):
            s = 0
            for c in x:
                if c == '1':
                    s += 1
            return [s, len(x)-s] 

        dp = [[0 for j in range(n+1)] for i in range(m+1) ]
        cache = {}

        for s in strs:
            if s not in cache:
                cache[s] = getCount(s)
            o,z = cache[s]
            for i in range(m,z-1,-1):
                for j in range(n, o-1,-1):
                    dp[i][j] = max(dp[i][j], dp[i-z][j-o] + 1)
        return dp[m][n]


# 900. rle iterator
# keep track of how many elements weve called for the current pointer

class RLEIterator:

    def __init__(self, encoding: List[int]):
        self.arr = encoding
        self.pointer = 0
        self.count = 0
    def next(self, n: int) -> int:
        self.count += n 

        while self.pointer < len(self.arr) and self.arr[self.pointer] < self.count:
            self.count -= self.arr[self.pointer]
            self.pointer += 2
        return self.arr[self.pointer+1] if self.pointer < len(self.arr) else -1


# 2059. minimum operations to convert number

class Solution:
    def minimumOperations(self, nums: List[int], start: int, goal: int) -> int:
        visited = set([start])
        queue = [start]
        steps = 0

        while queue:
            next_steps = []
            for x in queue:
                if x == goal:
                    return steps
                if x < 0 or 1000 < x:
                    continue
                for num in nums:
                    if x+num not in visited:
                        visited.add(x+num)
                        next_steps.append(x+num)
                    if x-num not in visited:
                        visited.add(x-num)
                        next_steps.append(x-num)
                    if x ^ num not in visited:
                        visited.add(x^num)
                        next_steps.append(x ^ num)
            queue = next_steps
            steps += 1
        return -1


# 907. sum of subarray minimums
# x is subarrays starting at index j and going up to index i
# y is all subarrays before index j where arr[j] would be the min that include j as min ele
# up to index i 
class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        res = 0
        arr.append(0)
        
        stack = []

        for i in range(len(arr)):
            while len(stack) and arr[i] < arr[stack[-1]]:
                j = stack.pop()
                x = ((i-j) * arr[j])

                z = stack[-1]+1 if len(stack) else 0
                y = (i-j) * (j-z) * arr[j]
                res += x + y
            stack.append(i)

        return res%1000000007



#1235 maximum profit in job scheduling

class Solution:
    def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        x = [[startTime[i], endTime[i], profit[i]] for i in range(len(startTime))]
        
        x =sorted(x, key=lambda job:job[1])
        
        dp = [[0,0]]
        
        def binaryOnJobs(limit):
            lo = 0
            hi = len(dp) - 1

            while lo <= hi:
                mid = lo + (hi-lo)//2
                if dp[mid][0] <= limit:
                    lo = mid+1
                else:
                    hi = mid-1
            return lo-1
            
        for job in x:
            most_profitable = binaryOnJobs(job[0])
            profit = dp[most_profitable][1] + job[2]

            if dp[-1][1] < profit:
                dp.append([job[1], profit])

        return dp[-1][1]


# 2225. find players with zero or on loss
# move players from zero => one to => many depending on loses
class Solution:
    def findWinners(self, matches: List[List[int]]) -> List[List[int]]:
        zero =  set()
        one = set()
        many = set()


        for [x,y] in matches:

            if y in zero:
                zero.remove(y)
                one.add(y)
            elif y in one:
                one.remove(y)
                many.add(y)
            elif y not in many:
                one.add(y)
            if x not in many and x not in zero and x not in one:
                zero.add(x)
                
        return [sorted(list(zero)), sorted(list(one))]



# 2488 count subarrays with median k
# find where k, iterate from index of k to the left side
# keeping track of if we are over or under the median of k
# then iterate right side and see if any of the sub sequences to the left have -over as their over/under
class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        i = nums.index(k)
        res = 1
        dp = {}
        over = 0
        x = i-1
        
        while 0 <= x:
            if nums[x] < k:
                over -= 1
            else:
                over += 1
            if over not in dp:
                dp[over] = 0
            dp[over] += 1
            if over == 0 or over == 1:
                res+=1
            x -= 1
        
        x = i+1
        over = 0
        
        while x < len(nums):
            if nums[x] < k:
                over -= 1
            else:
                over += 1
            
            y = -1*over
            if over == 1 or over == 0:
                res += 1
            if y in dp:
                res += dp[y]
            if y+1 in dp:
                res += dp[y+1]
            
            x += 1

            
        return res


# 2487. remove nodes from linked list
# reverse the list and keep track of the current maximum removing any nodes that are less than the max
# unreverse the list and this is our result
class Solution:
    def removeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        
        def reverse(head):
            
            curr_node = None
            
            while head:
                temp = head
                head = head.next
                temp.next = curr_node
                curr_node = temp
        
            return curr_node
        curr_node = reverse(head)
        curr_max = curr_node.val
        
        sent = ListNode('sent', curr_node)
        
        while curr_node:
            curr_max = curr_node.val
            while curr_node.next and curr_node.next.val < curr_max:
                curr_node.next = curr_node.next.next
            curr_node = curr_node.next
        return reverse(sent.next)


# 2486 append characters to string to make subsequence

class Solution:
    def appendCharacters(self, s: str, t: str) -> int:
        i = 0
        
        for c in s:
            if c == t[i]:
                i += 1
            if i == len(t):
                return 0
        return len(t) - i

# 2485 find the pivot integer
class Solution:
    def pivotInteger(self, n: int) -> int:
        x = (n*(n+1))//2
        y = 1
        i = 1
        while y <= x:
            if y == x:
                return i
            x -= i
            i += 1
            y += i
        
        return -1
            

# 446. arithmetic slices ii - subsequence 
# keep track of all possible numbers to the right of our current index
# take every number and every possible pair for the respective number and check if there are any numbers to the right that would make an arithmetic sequence 
# if there is a number try that numbers index for continuing the arithmetic sequence
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        n = len(nums)
        dp = []
        for i in range(n):
            dp.append({})

        for j in range(n-2,-1,-1):
            dp[j] = copy.deepcopy(dp[j+1])
            
            if nums[j+1] not in dp[j]:
                dp[j][nums[j+1]] = []
            dp[j][nums[j+1]].append(j+1)
        

        res = 0
        
        def arthimeticTravel(index, diff):
            num = nums[index]
            next_num = num - diff
            count = 0
            if next_num in dp[index]:
                for i in dp[index][next_num]:
                    count += 1
                    count += arthimeticTravel(i,diff)
            return count
        for i in range(n-2):
            a = nums[i]
            for b in dp[i]:
                if b==a:
                    m = len(dp[i][b])
                    c = 2
                    while c <= m:
                        res += math.comb(m,c)
                        c += 1
                else:
                    for j in dp[i][b]:
                        res += arthimeticTravel(j, a-b)
        return res



# 2484. count palindromic subsequences
# start off with all possible length 2 subsequences in our right dictionary
# on each iteration remove the current number from the right dictionary
# check if our left dictionary contains any subsequences that are also in the right dictionary
class Solution:
    def countPalindromes(self, s: str) -> int:
        if len(s) < 5:
            return 0
        res = 0

        comboLeft = {s[0]+s[1]:1}
        leftNums = {s[0]:1}
        comboRight = {}
        rightNums = {}

        if s[1] == s[0]:
            leftNums[s[0]] += 1
        else:
            leftNums[s[1]] = 1

        
        for k in range(2, len(s)):
            
            for c in rightNums:
                x = s[k] + c
                if x not in comboRight:
                    comboRight[x] = 0
                comboRight[x] += rightNums[c]
                
            if s[k] not in rightNums:
                rightNums[s[k]] = 0
            rightNums[s[k]]+=1
                
        mid = len(s)//2
                
        for i in range(2,len(s)-2):
            j = s[i]
            
            for num in rightNums:
                if num == j:
                    rightNums[num] -= 1
                y = num+j
                if y in comboRight:
                    comboRight[y] -= rightNums[num]
            
            if i < mid:
                for num in comboLeft:
                    if num in comboRight:
                        res += max(0,comboLeft[num]*comboRight[num])
            else:
                for num in comboRight:
                    if num in comboLeft:
                        res+= max(0,comboLeft[num]*comboRight[num])
            
            
            for num in leftNums:
                x = num+s[i]
                if x not in comboLeft:
                    comboLeft[x] = 0
                comboLeft[x] += leftNums[num]
            if s[i] not in leftNums:
                leftNums[s[i]] = 0
            leftNums[s[i]] += 1
            
        return res%1000000007


# 2483. minimum penalty for a shop
# count penalty for having shop closed at specific time
# add to closing penalty the penalty for staying open
# keep track of minimum on second iteration 
class Solution:
    def bestClosingTime(self, customers: str) -> int:
        customers += 'X'
        penalty = {}
        running_penalty = 0
        for j in range(len(customers)-1,-1,-1):
            if customers[j] == 'Y':
                running_penalty += 1
            penalty[j] = running_penalty
        running_penalty = 0
        curr_min = 0
        for i in range(len(customers)):
            penalty[i] += running_penalty
            if penalty[i] < penalty[curr_min]:
                curr_min = i
            if customers[i] == 'N':
                running_penalty += 1
        print(penalty)
        return curr_min

# 2482. difference between ones and zeroes in row and column
# count the number of 1's in each row and column
# at each point we just perform the calculation for the respective row and column
class Solution:
    def onesMinusZeros(self, grid: List[List[int]]) -> List[List[int]]:
        rows = {}
        cols = {}
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if j not in cols:
                    cols[j] = 0
                if i not in rows:
                    rows[i] = 0
                if grid[i][j] == 1:
                    rows[i] += 1
                    cols[j] += 1
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                x = rows[i] + cols[j] - (len(grid)-rows[i] + len(grid[0])-cols[j])
                grid[i][j] = x
        return grid


# 2481. minimum cuts to divide a circle

class Solution:
    def numberOfCuts(self, n: int) -> int:
        if n == 1:
            return 0
        if n%2 == 0:
            return n//2
        else:
            return n



# 907. sum of subarray minimums
# keep mono-stack of increasing numbers
# when we reach a number if it is smaller than any number in the stack then we can no longer use that number in our subarrays
# so we pop it and check how long the number has been in the mono stack (i-j) 
# and we also track how long the number can extend to the left (j-z) * (i-j)  note subarrays extending to the left can also extend up to the current index
# we calculate z by seeing if there is a number in the stack... this means that number is a blocker and we can only take subarrays that dont have that number
# if there is not a number in the stack then we can take every previous number in our subarray

class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        res = 0
        arr.append(0)
        
        stack = []

        for i in range(len(arr)):
            while len(stack) and arr[i] < arr[stack[-1]]:
                j = stack.pop()
                x = ((i-j) * arr[j]) 
                z = stack[-1]+1 if len(stack) else 0
                y = (i-j) * (j-z) * arr[j]
                res += x + y
            stack.append(i)

        return res%1000000007



# 380. insert delete getrandom o(1)
# array to get random value 
# dictionary to keep location of val
# when we remove ele we swap the last value into the removed values location then remove last element so its not in the array twice
# we also delete the remove ele from our dictionary 
class RandomizedSet:

    def __init__(self):
        self.location = {}
        self.rand = []

    def insert(self, val: int) -> bool:
        if val in self.location:
            return False
        self.location[val] = len(self.rand)
        self.rand.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.location:
            return False
        self.rand[self.location[val]] = self.rand[-1]
        self.location[self.rand[-1]] = self.location[val]
        self.rand.pop()
        del self.location[val] 
        return True
    def getRandom(self) -> int:
        x = random.randint(0,len(self.rand)-1)
        return self.rand[x]


# 1306. jump game iii

class Solution:
    def canReach(self, arr: List[int], start: int) -> bool:
        visited = set()

        queue = [start]

        while queue:

            next_jumps = []

            for x in queue:
                if arr[x] == 0:
                    return True
                y = x + arr[x]
                z = x - arr[x]
                if y < len(arr) and y not in visited:
                    next_jumps.append(y)
                    visited.add(y)
                if 0 <= z and z not in visited:
                    next_jumps.append(z)
                    visited.add(z)
            queue = next_jumps
        return False    


# 1471. the k strongest values in an array

class Solution:
    def getStrongest(self, arr: List[int], k: int) -> List[int]:
        arr = sorted(arr)

        m = arr[(len(arr)-1)//2]

        res = []

        l = 0
        h = len(arr)-1

        while len(res) < k:
            x = abs(arr[l]-m)
            y = abs(arr[h]-m)
            if x <= y:
                res.append(arr[h])
                h -= 1
            else:
                res.append(arr[l])
                l += 1
        return res


# 207. course schedule
# every classes starts off with x number of prereqs
# start off by taking every class with 0 prereqs
# when we take a class we can reduce the degree of all classes that depend on it by 1
# if any of these classes have degree 0 we add them to our queue
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        taken = 0
        
        degree = [0 for i in range(numCourses)]
        requiredFor = {}
        for a,b in prerequisites:
            degree[a] += 1
            if b not in requiredFor:
                requiredFor[b] = []
            requiredFor[b].append(a)

        queue = []

        for i in range(len(degree)):
            if degree[i] == 0:
                queue.append(i)
        
        for x in queue:
            taken += 1

            if x in requiredFor:
                for y in requiredFor[x]:
                    degree[y] -= 1
                    if degree[y] == 0:
                        queue.append(y)
        return taken == numCourses
        


# 1905. count sub islands

class Solution:
    def countSubIslands(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:

        def isSubIsland(i,j):
            adj = [[0,1], [1,0], [0,-1], [-1,0]]
            isSub = True
            queue = [[i,j]]
            grid2[i][j] = 2

            for x,y in queue:
                if grid1[x][y] == 0:
                    isSub = False
                for x1,y1 in adj:
                    r,c = x+x1, y+y1
                    if r < 0 or c < 0 or r == len(grid1) or c == len(grid1[0]) or grid2[r][c] != 1:
                        continue
                    grid2[r][c] = 2
                    queue.append([r,c])
            return isSub
        
        res = 0
        for i in range(len(grid1)):
            for j in range(len(grid1[0])):

                if grid2[i][j] == 1:
                    if isSubIsland(i,j):
                        res += 1
        return res


# 1207. unique number of occurrences

class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        freqs = {}
        counts = {}
        for num in arr:
            if num not in freqs:
                freqs[num] = 1
                counts[1] = counts.get(1,0) + 1
            else:
                counts[freqs[num]] -= 1
                freqs[num] += 1
                counts[freqs[num]] = counts.get(freqs[num], 0) + 1

        for key in counts:
            if 1 < counts[key]:
                return False
        return True


# 1704. determine if string halves are alike

class Solution:
    def halvesAreAlike(self, s: str) -> bool:
        vowels = {"a", "A", "e", "E", "i", "I", "o", "O", "u", "U"}
        x = 0
        y = 0

        for i in range(0, len(s)//2):
            c = s[i]
            if c in vowels:
                x += 1
        for j in range(len(s)//2, len(s)):
            c = s[j]
            if c in vowels:
                x -= 1
        return x == 0

# 1755. closest subsequence sum
# finding all possible subsequences takes too long
# so we split our array in half and find the subsequences for each half
# then we use binary search on the sorted first half subsequences 
# for every subsequence in the second half
class Solution:
    def minAbsDifference(self, nums: List[int], goal: int) -> int:

        def createSubs(sums,i,total, stopper):
            sums.add(total)
            if i == stopper:
                return

            createSubs(sums, i+1, total+nums[i],stopper)
            createSubs(sums, i+1, total, stopper)
        def binarySearch(target, nums):
            lo = 0
            hi = len(l1)-1

            while lo <= hi:
                mid = lo + (hi - lo)//2
                if nums[mid] == target:
                    return mid
                if nums[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid - 1
            return lo
    
        l1 = set()
        l2 = set()
        createSubs(l1, 0, 0, len(nums)//2)
        createSubs(l2,len(nums)//2,0,len(nums))
        
        l1 = sorted(list(l1))
        closest = abs(goal)
        for sum2 in l2:
            if closest == 0:
                return 0
            target = goal - sum2
            idx = binarySearch(target, l1)
            
            if idx < len(l1) and 0 <= idx:
                closest = min(closest, abs(l1[idx] + sum2 - goal))
            if 0 < idx:
                closest = min(closest, abs(l1[idx-1] + sum2 - goal))
            if idx < len(l1)-1:
                closest = min(closest, abs(l1[idx+1] + sum2 - goal))
        return closest

# 1657. determine if two strings are close
# check that the strings both have the same characters, and the same counts for any type of character
class Solution:
    def closeStrings(self, word1: str, word2: str) -> bool:
        c1 = Counter(word1)
        c2 = Counter(word2)
        c = {}

        for x in c1:
            c[c1[x]] = c.get(c1[x], 0) + 1
        for x in c2:
            if c2[x] not in c or x not in c1:
                return False
            c[c2[x]] -= 1
            if c[c2[x]] == 0:
                del c[c2[x]]
                
        return len(c) == 0


# 451. sort characters by frequency 

class Solution:
    def frequencySort(self, s: str) -> str:
        c = Counter(s)
        x = []
        for k,v in c.items():
            heappush(x, (-v,k))
        z = ''
        while x:
            a = heappop(x)
            z += a[1]*(a[0]*-1)
        return z


# 2415. reverse odd levels of binary tree
# classic iterative traversal of binary tree
# only caveat when we are on even level we add our node values to the val_store
# and when we are on odd level we change our node value
# if we are dealing with a large tree val_store might get very large because it contains every odd row's values
# to solve this we could prune it on at the end of every odd iteration
class Solution:
    def reverseOddLevels(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        level = 'even'
        val_store = []
        queue = [root]

        while queue:
            next_queue = []
            j = -1
            for node in queue:
                if node.left:
                    if level == 'even':
                        val_store.append(node.left.val)
                        val_store.append(node.right.val)
                    next_queue.append(node.left)
                    next_queue.append(node.right)
                if level == 'odd':
                    node.val = val_store[j]
                    j -= 1
            if level == 'odd':
                level = 'even'
            else:
                level = 'even'
            queue = next_queue
        return root
                    



# 1125. smallest sufficient team
# for each skill get a list of people who have this skill
# 1. start off with the skill that has the least amount of people 
# 2. try every person with that skill 
# 3. removing every skill that the person has 
# 4. go back to step 1 until all skills have been removed from our set
class Solution:
    def smallestSufficientTeam(self, req_skills: List[str], people: List[List[str]]) -> List[int]:
        skills = {}
        for skill in req_skills:
            skills[skill] = []
        for i in range(len(people)):
            person_skills = people[i]

            for skill in person_skills:
                skills[skill].append(i)
        
        min_skill = min(skills, key=lambda x:len(x))
        

        req_skills = set(req_skills)
        self.min_people = [i for i in range(len(people))]
        
        def track(curr_skill, curr_people, skills_needed):
            for person_index in skills[curr_skill]:
                if len(self.min_people) <= len(curr_people):
                    return 

                removed = []
                curr_people.append(person_index)

                for person_skill in people[person_index]:
                    if person_skill in skills_needed:
                        removed.append(person_skill)
                        skills_needed.remove(person_skill)

                if len(skills_needed) == 0:
                    if len(curr_people) < len(self.min_people):
                        self.min_people = curr_people.copy()
                else:
                    next_skill = min(skills_needed, key=lambda x:len(skills[x]))
                    track(next_skill, curr_people, skills_needed)
                curr_people.pop()
                
                for skill in removed:
                    skills_needed.add(skill)

        track(min_skill, [], req_skills)
        return self.min_people



# 473. matchsticks to square 
class Solution:
    def makesquare(self, matchsticks: List[int]) -> bool:
        total = sum(matchsticks)
        if total%4 != 0:
            return False
        side = total//4
        matchsticks = sorted(matchsticks, reverse = True)
        
        def make(a,b,c,d,index):
            if a == side and b == side and c == side and d == side:
                return True
            if index == len(matchsticks):
                return False
            
            for i in range(index, len(matchsticks)):
                size = matchsticks[i]
                if a+size <= side and make(a+size,b,c,d,i+1):
                    return True
                elif b+size <= side and a != b and make(a,b+size,c,d,i+1):
                    return True
                elif c+size <= side and a != c and b != c and make(a,b,c+size,d,i+1):
                    return True
                elif d+size <= side and a != d and b != d and c != d and make(a,b,c,d+size,i+1):
                    return True
                else:
                    return False
        return make(0,0,0,0,0)

# 2256. minimum average difference
class Solution:
    def minimumAverageDifference(self, nums: List[int]) -> int:

        right = sum(nums) - nums[0]
        left = nums[0]
        j = 1
        min_diff = inf
        min_ind = -1
        n = len(nums)

        while j < n:
            if min_diff == 0:
                return min_ind
            diff = abs(right//(n-j)-left//j)
            if diff < min_diff:
                min_ind = j-1
                min_diff = diff
            left += nums[j]
            right -= nums[j]
            j += 1
            
        if left//n < min_diff:
            return n-1
        return min_ind

# 876. middle node of the linked list

class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = head
        fast = head.next

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        if fast:
            slow = slow.next
        return slow


# 341. flatten nested list iterator 
# the check goes => if hasNext => next
# on every hasNext we prepare our stack then check if we have any elements left
class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        self.arr = nestedList[::-1]
        
    
    def next(self) -> int:
        return self.arr.pop()
    def hasNext(self) -> bool:
        self.prepareStack()
        return len(self.arr) != 0
        

    def prepareStack(self):
        if len(self.arr) == 0 or self.arr[-1].getInteger() != None:
            return
        else:
            x = self.arr.pop().getList()
            for i in range(len(x)-1,-1,-1):
                self.arr.append(x[i])
            self.prepareStack()


# 1961. check if string is a prefix of array

class Solution:
    def isPrefixString(self, s: str, words: List[str]) -> bool:
        i = 0
        words.append('1')
        for word in words:
            if i == len(s):
                return True
            if s[i:i+len(word)] == word:
                i+=len(word)
            else:
                return False

# 938 range sum of bst

class Solution:
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        if not root:
            return 0
        if low <= root.val and root.val <= high:
            return root.val + self.rangeSumBST(root.left,low,high) + self.rangeSumBST(root.right, low, high)
        elif root.val < low:
            return self.rangeSumBST(root.right, low, high)
        else:
            return self.rangeSumBST(root.left, low, high)

# 872. leaf-similar trees
# travel all possilbe nodes until we reach their leaf 
# appending leaf values from left to right 
# compare arrays 
class Solution:
    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        arr1 = []
        arr2 = []

        def findLeaf(arr, root):
            if not root.left and not root.right:
                arr.append(root.val)
            if root.left:
                findLeaf(arr, root.left)
            if root.right:
                findLeaf(arr, root.right)
        findLeaf(arr1, root1)
        findLeaf(arr2, root2)
        return arr1 == arr2


# 2193. minimum number of moves to make palindrome 
# super naive solution works but really slow need to optimize and prune so the code is more readable + faster
class Solution:
    def minMovesToMakePalindrome(self, s: str) -> int:
        if not s or len(s) <= 1:
            return 0
        i = 0
        j = len(s) - 1

        while s[i] == s[j] and i < j:
            i += 1 
            j -= 1
        
        if j <= i:
            return 0
        
        x = -1
        y = -1

        for k in range(i+1,j):
            if s[k] == s[i]:
                x = k
            elif s[k] == s[j] and y == -1:
                y = k

        swap_i = math.inf
        swap_j = math.inf
        if x != -1 and y != -1:
            if j-x < y-i:
                new_sx = s[i+1:x] + s[x+1:j+1]
                swap_i = self.minMovesToMakePalindrome(new_sx) + (j-x)
            else:
                new_sy = s[i:y] + s[y+1:j]
                swap_j = self.minMovesToMakePalindrome(new_sy) + (y-i)
        elif x != -1:
            new_sx = s[i+1:x] + s[x+1:j+1]
            swap_i = self.minMovesToMakePalindrome(new_sx) + (j-x)
        elif y != -1:
            new_sy = s[i:y] + s[y+1:j]
            swap_j = self.minMovesToMakePalindrome(new_sy) + (y-i)
        return min(swap_i, swap_j)
# the extremely smart solution for 2193. 
# looks at last letter and finds the first location of it, swaps it to the front then pops it 
# if last letter == first letter we dont add anything to the res 
# if there is only 1 index for the last letter we move it to the middle 
# always popping the last letter
class Solution:
    def minMovesToMakePalindrome(self, s: str) -> int:
        res = 0
        s = list(s)
        while s:
            i = s.index(s[-1])
            if i == len(s) - 1:
                res += i//2
            else:
                res += i
                s.pop(i)
            s.pop()
        return res


# 2493. divide nodes into the maximum number of groups
# divide into connected components 
# for each connected component we look for the maximum number of groups possible 
# we do this by each node in the connected component and starting a bfs from it
# the starting node will be group 1, each node connected to it will be group 2 each node connected to group 2 will be group 3 etc etc
# if we run into a node thats been seen before its in previous group unless its in the current group then we have a conflict ie =>
# 2 nodes being placed into the same group
class Solution:
    def magnificentSets(self, n: int, edges: List[List[int]]) -> int:
        components = []
        seen = set()
        paths = {i+1:[] for i in range(n)}

        for a,b in edges:
            paths[a].append(b)
            paths[b].append(a)
        
        for i in range(1,n+1):
            if i in seen:
                continue
            queue = [i]
            components.append([i])
            seen.add(i)
            while queue:
                next_level = []
                node = queue.pop()
                for neighbor in paths[node]:
                    if neighbor in seen:
                        continue
                    next_level.append(neighbor)
                    components[-1].append(neighbor)
                    seen.add(neighbor)
                    queue.append(neighbor)
        def bfs(node):
            groups = 0
            seen = {node:1}
            queue = [node]

            while queue:
                groups += 1
                next_group = []
                for node in queue:
                    for neighbor in paths[node]:
                        if neighbor not in seen:
                            next_group.append(neighbor)
                            seen[neighbor] = groups + 1
                        elif seen[neighbor] == groups:
                            return -1
                queue = next_group
            return groups


        longest = [-1] * len(components)
        for i in range(len(components)):
            arr = components[i]
            for node in arr:
                longest[i] = max(longest[i], bfs(node))
        if min(longest) < 0:
            return -1
        return sum(longest)




# 2392. minimum score of a path between two cities
# travel every possible path starting from 1st city stopping if current min value is greater than one we have seen already
class Solution:
    def minScore(self, n: int, roads: List[List[int]]) -> int:
        paths = {}
        tried = {}
        res = 0 
        for a,b,c in roads:
            if a not in paths:
                paths[a] = []
                tried[a] = inf
            if b not in paths:
                paths[b] = []
                tried[b] = inf
            paths[a].append([b,c])
            paths[b].append([a,c])
        
        
        def travel(node, val):
            tried[node] = val
            for city,cost in paths[node]:
                m = min(cost,val)
                if tried[city] <= m:
                    continue
                travel(city, m)
                
        travel(1, 100000)
        return tried[n]

# 2491. divide players into teams of equal skill
# since we have to divide players into groups of 2 of equal value we can use two pointer adding low + high values
class Solution:
    def dividePlayers(self, skill: List[int]) -> int:
        skill = sorted(skill)
        
        x = skill[0] + skill[-1]
        res = 0
        
        i = 0
        j = len(skill) - 1
        
        while i < j:
            if skill[i] + skill[j] != x:
                return -1
            
            res += (skill[i] * skill[j])
            i += 1
            j -= 1
        return res

# 2490. circular sentence 
# check the current word's first letter is the same as the previous word's last letter
# first index i=0 checks if the first and last word are circular because of negative indexing in python
class Solution:
    def isCircularSentence(self, sentence: str) -> bool:
        s = sentence.split(' ')

        for i in range(0, len(s)):
            if s[i][0] != s[i-1][-1]:
                return False
        return True

# 1026. maximum difference between node and ancestor

class Solution:
    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:

        def maxDiff(node, mx, mn):
            if not node:
                return 0
            x = max(abs(node.val - mx), abs(node.val - mn))
            mx = max(node.val, mx)
            mn = min(node.val, mn)
            return max(x, maxDiff(node.left,mx,mn), maxDiff(node.right,mx,mn))

        return maxDiff(root, root.val, root.val)


# 2352. equal row and column pairs

class Solution:
    def equalPairs(self, grid: List[List[int]]) -> int:
        row_trie = {}

        for row in grid:

            trie = row_trie

            for c in row:
                if c not in trie:
                    trie[c] = {}
                trie = trie[c]
            if 'done' not in trie:
                trie['done'] = 0
            trie['done'] += 1
        res = 0
        for c in range(len(grid)):
            trie = row_trie
            for r in range(len(grid)):
                l = grid[r][c]
                if l not in trie:
                    break
                trie = trie[l]
            if 'done' in trie:
                res += trie['done']

        return res

# 1765. map of highest peak
# using 0 extra space 
# start with water expand outwards
class Solution:
    def highestPeak(self, isWater: List[List[int]]) -> List[List[int]]:
        queue = []

        for i in range(len(isWater)):
            for j in range(len(isWater[0])):
                if isWater[i][j] == 1:
                    queue.append([i,j])
                    isWater[i][j] = 0
                else:
                    isWater[i][j] = -1

        neighbors = [[1,0],[0,1],[-1,0], [0,-1]]

        while queue:

            next_level = []

            for i,j in queue:

                for rc,cc in neighbors:
                    r,c = i+rc, j+cc
                    if r < 0 or c < 0 or r == len(isWater) or c == len(isWater[0]) or 0 <= isWater[r][c]:
                        continue
                    isWater[r][c] = isWater[i][j] + 1
                    next_level.append([r,c])
            queue = next_level
        return isWater

# 2166. design bitset
# using an offset to track wether we are in a flipped or not flipped state
# tracking count of 1's 

class Bitset:

    def __init__(self, size: int):
        self.arr = [0 for i in range(size)]
        self.offset = False
        self.cnt = 0
    def fix(self, idx: int) -> None:
        if not self.offset and self.arr[idx] == 0:
            self.arr[idx] = 1
            self.cnt += 1
        elif self.offset and self.arr[idx] == 1:
            self.arr[idx] = 0
            self.cnt += 1

    def unfix(self, idx: int) -> None:
        if not self.offset and self.arr[idx] == 1:
            self.arr[idx] = 0
            self.cnt -= 1
        elif self.offset and self.arr[idx] == 0:
            self.arr[idx] = 1
            self.cnt -= 1

    def flip(self) -> None:
        self.offset = not self.offset
        self.cnt = len(self.arr) - self.cnt
    def all(self) -> bool:
        return self.cnt == len(self.arr)

    def one(self) -> bool:
        return self.cnt >= 1

    def count(self) -> int:
        return self.cnt

    def toString(self) -> str:
        res = ''
        for b in self.arr:
            if (b == 0 and not self.offset) or (b == 1 and self.offset):
                res += '0'
            else:
                res += '1'
        return res


# 1022. sum of root to leaf binary number

class Solution:
    def sumRootToLeaf(self, root: Optional[TreeNode]) -> int:
        res = 0

        def sumPath(node,val):
            val += str(node.val)
            if node.left :
                sumPath(node.left, val)
            if node.right:
                sumPath(node.right, val)
            if not node.right and not node.left:
                nonlocal res
                res += int(val, 2)
        sumPath(root, '')
        return res


# 2278. percentage of letter in string

class Solution:
    def percentageLetter(self, s: str, letter: str) -> int:
        return math.trunc(s.count(letter)/len(s)*100)

# 1839. longest substring of all vowels in order

class Solution:
    def longestBeautifulSubstring(self, word: str) -> int:
        
        vow = {'a':0, 'e':1, 'i':2, 'o':3, 'u':4}
        

        mx = 0
        i = 0

        while i < len(word):
            if word[i] == 'a':

                length = 0
                k = 0
                while i < len(word) and (vow[word[i]] == k or vow[word[i]] == k + 1):
                    k = vow[word[i]]
                    i += 1
                    length += 1
                if k == 4:
                    mx = max(mx, length)
            else:
                i+= 1
        return mx

# 124. binary tree maximum path sum
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        def helper(node):
            if not node: return [-10000,-10000] 

            left = helper(node.left)
            right = helper(node.right)

            path = max(left[0], right[0], 0) + node.val

            max_path = max(path, left[1], right[1], left[0] + right[0] + node.val)

            return [path, max_path]
        return helper(root)[1]