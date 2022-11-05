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