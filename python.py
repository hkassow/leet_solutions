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







