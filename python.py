from heapq import heappop, heappush
import math


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