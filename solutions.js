// 322. Coin Change
// DP 
// if your solution is slower than o(n) can you sort array?
var coinChange = function(coins, amount) {
    let dp = Array(amount + 1).fill(Infinity)
    coins.sort()
    
    dp[0] = 0
    
    
    for (let i=0; i< coins.length; i++) {
        let coin = coins[i]
        
        for (let j = coin; j <= amount; j++){
            dp[j] = Math.min(dp[j], dp[j-coin] + 1)
        }
    }
    
   return (dp[amount] === Infinity)? -1: dp[amount]
};


// 238. product of array except self
//if problem asks for o(n) time remmeber 2 n loops is still o(n) time
//first count the running product going forward then backward
// think of the two endpoints 
// after first loop 
// last number will get product of nums[0]....nums[n-2] 
// first number will have prdocut of 1 
// so we count the product from the back
// last number will still have same product
// first number will now have apporpriate product
// middle case will have count = nums[0]...nums[mid- 1] * backwardsCount = nums[mid + 1] ... nums[n - 1]
var productExceptSelf = function(nums) {
    let answer = Array(nums.length)
    
    let count = 1
    for (let i=0; i< nums.length; i++) {
        answer[i] = count
        count *= nums[i]
    }
    let backwardsCount = 1
    for (let i=nums.length - 1; 0 <= i; i--){
        answer[i] *= backwardsCount
        backwardsCount *= nums[i]
    }
    return answer
};

// 155. min stack 
// storing things as a node keeping the most recent important value in each node
// this allows us to always know what the important value is
class MinStack {
    constructor() {
        this.stack = []
    }

    push() {
        this.stack.push({
            val: val,
            min: this.stack.length? Math.min(val, this.getMin()) : val
        })
    }
    pop() {
        this.stack.pop()
    }
    top() {
        return this.stack[this.stack.length - 1]?.val
    }
    getMin() {
        return this.stack[this.stack.length - 1]?.min
    }
}


// 98. Validate Binary Search Tree
// recurrsion
var isValidBST = function(root, max= Infinity, min= -Infinity) {
    if (!root) {
        return true
    }
    
    if ((root.val <= min)) {
        return false
    }
    if ((max <= root.val)) {
        return false
    }
    return (isValidBST(root.left, root.val, min) && isValidBST(root.right, max, root.val))
};

// 200. Number of Islands
// recurssion to turn each island

var numIslands = function(grid) {
    let count = 0 
    
    for (let i = 0; i < grid.length; i++) {
        for (let j = 0; j < grid[0].length; j++){
            if (grid[i][j] == 1) {
                count++
                turnIsland(i, j, grid)
            } 
        }
    }
    return count
};

const turnIsland = (i, j, grid) => {
    
    if (grid[i][j] == 1) {
        grid[i][j] = 0
        
        if (i+1 < grid.length) {
            turnIsland(i+1, j, grid)
        }
        if (0 <= i-1) {
            turnIsland(i-1, j, grid)
        }
        if (j+1 < grid[0].length) {
            turnIsland(i, j+1, grid)
        }
        if (0 <= j-1) {
            turnIsland(i, j-1, grid)
        }
    }
    
}


// 994. Rotting Orange
// add all the already rotten then spread rot to all nearby oranges 
// 1 day at a time we progress through our list and increment accordingly
// island problem

var orangesRotting = function(grid) {
    let day = 0
    let count = 0
    let stack = []
    for (let i=0; i< grid.length; i++) {
        for (let j=0; j< grid[0].length; j++) {
            if (grid[i][j] === 2) {
                stack.push([i, j])
            }
            if (grid[i][j] === 1) {
                count++
            }
        }
    }
    while (stack.length) {
        let nextDay = []
        while (stack.length) {
            let nextOrange = stack.pop()
            
            
            if (nextOrange[0]+1 <grid.length) {
                if(grid[nextOrange[0] + 1][nextOrange[1]] === 1){
                    grid[nextOrange[0] + 1][nextOrange[1]] = 2
                    count --;
                    nextDay.push([nextOrange[0] + 1, nextOrange[1]])
                }
            }
            if (0 <= nextOrange[0]-1) {
                if (grid[nextOrange[0] - 1][nextOrange[1]] === 1) {
                    grid[nextOrange[0] - 1][nextOrange[1]] = 2
                    count --;
                    nextDay.push([nextOrange[0] -1 , nextOrange[1]])
                }
            }
            if (nextOrange[1]+1 <grid[0].length) {
                if(grid[nextOrange[0]][nextOrange[1] + 1] === 1){
                    grid[nextOrange[0]][nextOrange[1] + 1] = 2
                    count --;
                    nextDay.push([nextOrange[0], nextOrange[1] + 1])
                }
            }
            if (0 <= nextOrange[1]-1) {
                if (grid[nextOrange[0]][nextOrange[1] - 1] === 1) {
                    grid[nextOrange[0]][nextOrange[1] - 1] = 2
                    count --;
                    nextDay.push([nextOrange[0], nextOrange[1] - 1])
                }
            }
            
        }
        if (nextDay.length) {
            stack = nextDay
            day++ 
        }
    }
    return count? -1: day
};

// 33. Search in rotated sorted array
// on each loop we do an extra check to see if our left or right sections can possibly contain our target
var search = function(nums, target) {
    let min = 0
    let max = nums.length - 1
    
    while( (min <= max) && (0 <= max) && (max < nums.length) && (0 <= min) && (min < nums.length)) {
        let mid = Math.floor((max+min)/2)
        if (nums[mid] === target) {
            return mid
        }
        if (target < nums[mid]){
            if ((nums[max] <= nums[mid]) && (target < nums[min])) {
                min = mid + 1 
            } else {
                max = mid - 1
            }
            
        }
        if (nums[mid] < target) {
            if ((nums[mid] <= nums[min]) && (nums[max] < target)) {
                max = mid - 1
            } else {
                min = mid + 1
            }
        }
    }
    return -1
};

// 39. combination sum
// finding all paths that sum up to the targt... each path has a unique array if we hit the target return ans
var combinationSum = function(candidates, target) {
    let ans = []
    
    const travel = (path, index, sum) => {
        if (target < sum ) return
        if (sum === target) { ans.push(path) }
        
        while (index < candidates.length) {
            travel([...path, candidates[index]], index, sum + candidates[index])
            index++
        }
    }
    travel([], 0, 0)
    return ans
};

// 46. permutations 
// following a path very similar to problem above (39. combination sum)
var permute = function(nums) {
    let ans = []
    
    const arrayBuilder = (array, added ) => {
        if (array.length === nums.length) {
            ans.push([...array])
        } else {
            for (let num of nums) {
                if (added[num]) {
                    continue
                }
                array.push(num)
                added[num] = true
                arrayBuilder(array, added)
                array.pop()
                added[num] = false
            }
        }
    }
    arrayBuilder([], {})  
    return ans
};

// 236. lowest common ancestor of a binary tree 
// if root === value either it is the ancestor or q is in a different path thats why we can kick out on the first return 
// if we end up in case where we are [1,2,3] looking for [2,3] we would find both left and right values truthy 
// we return root upstream as either left or right (depending on the root that called) 
// and then the other left or right respectively will always be falsey value and our root will travel to our top call and return itself once more 
var lowestCommonAncestor = function(root, p, q) {
    if (!root || root === p || root === q) {
        return root
    }
    let left = lowestCommonAncestor(root.left,p,q)
    let right = lowestCommonAncestor(root.right,p,q)
    return (left && right)? root: (left || right)
};


// 721. accounts merge
// using travel function to find the parent node for each email
// basically linked lists? 
// merge function takes first email travels down it adding onto end
var accountsMerge = function(accounts) {
    let email2Group = {}
    let email2Name = {}
    
    const travel = (x) => {
        if (email2Group[x] !== x) {
            email2Group[x] = travel(email2Group[x])
        }
        return email2Group[x]
    }
    const merge = (x, y) => {    
        email2Group[travel(y)] = email2Group[travel(x)]  
    }
    for (const [name, ...emails] of accounts) {
        
        for (const email of emails) {
            email2Name[email] = name
            if (!email2Group[email]) {
                email2Group[email] = email
            }
            merge(email, emails[0])
        }
    }
    const uniqueAccounts = {}
    for (const email of Object.keys(email2Group)) {
        const parent = travel(email)
        if (!uniqueAccounts[parent]) {
            uniqueAccounts[parent] = []
        }
        uniqueAccounts[parent].push(email)
    }
    return Object.entries(uniqueAccounts).map(([headEmail, array]) => [email2Name[headEmail], ...array.sort()])
};

// 75. sort color
// fill in 2 on every number 
// if number was 0 or 1 we place it back in the array and increment accordingly
var sortColors = function(nums) {
    let [zeroPointer, onePointer] = [0, 0]
    
    for (let i = 0; i < nums.length; i++) {
        const x = nums[i]
        nums[i] = 2
        if (x === 1) {
            nums[onePointer] = 1
            onePointer++
        } else if (x === 0) {
            nums[onePointer] = 1
            nums[zeroPointer] = 0
            zeroPointer++
            onePointer++
        }
    }
};

// 139. word break
// we travel along every possible slice of our word if its in our word dic
// if we reach target length bail out

var wordBreak = function(s, wordDict) {
    let wordObj = new Set(wordDict)
    let wordLens = new Set(wordDict.map(word => word.length))
    let starts = new Set([0])
    
    for (let start of starts) {
        for (let len of wordLens) { 
            if (wordObj.has(s.slice(start, start+len))) {
                if (start+len === s.length) {
                    return true
                }
                starts.add(start+len)
            }
        }
    }
    return false
};

// 140. word break II
// very similar to word break I traveling down paths keeping track of sentence
var wordBreak = function(s, wordDict) {
    var ans = []
    var wordSet = new Set(wordDict)
    var wordLen = new Set(wordDict.map(word => word.length))
    
    const travel = (start, sentence) => {
        if (start === s.length) {
            ans.push(sentence)
            return
        }
        for (let len of wordLen) {
            let sub = s.slice(start, start+len)
            if (wordSet.has(sub)) {
                travel(start+len, sentence?`${sentence} ${sub}`: sub)
            }
        }
    }
    travel(0, "")
    return ans
};


// 416. partition equal subset sum
// dp is simply a boolean array asking if we can make an array w/ this sum
// dp[0] sum of 0 always true, no elements
// first loop will only have dp[0] be true 
// so will place a value at dp[nums[0]] = true
// then every num after that will either place a new sum start and/or add onto previous ones

var canPartition = function(nums) {
    let sum = nums.reduce((a,b) => a+b)
    
    if (sum%2) return false
    
    let half = sum/2
    let dp = Array(half+1).fill(false)
    dp[0] = true
    
    for (let num of nums) {
        for (let i = half; num <= i; i--) {
            dp[i] = dp[i] || dp[i-num]
        }
    }
    return dp[half]
};

// 8. string to integer (atoi)
// could just do parseInt and check max/min bounds
// also can use trim to eliminate white spacing (tried to do it as naively as possible without any additional functions)
// return Math.max(Math.min(parseInt(s) || 0, 2147483647), -2147483648)
// this one liner also suffices... however parseInt is essentially atoi?
var myAtoi = function(s) {    
    let sign
    let number = ""
    let regex = /[0-9]/
    let min = Math.pow(-2, 31)
    let max = (-1 * min) - 1 
    
    for (let chara of s) {
        if (chara.match(regex)) {
            number += chara
        } else if (number || sign) {
            break
        } else if (chara === "-" || chara === "+") {
            sign = (chara === "-")? -1:1
        } else if (chara === " ") {
            continue
        } else {
            break
        }
    }
    if (!number) return 0
    number = (sign)? sign* +number: +number
    if (number < min) return min
    if (number > max) return max
    return number
};


// 54. spiral matrix
// first solution uses simple boundary... slowly tracing the outer matrix until our boundaries close on eachother
// 2nd solution involves removing the "top level array" then popping out right sided elements
// then inverting our matrix and repeating these 2 steps 
// i think this solution would breakdown on larger matrix sizes because repeated calls to shift + reverse
var spiralOrder = function(matrix) {
    var [rowMax, colMax, rowMin, colMin, ans] = [matrix.length - 1, matrix[0].length - 1, 0, 0, []]
    
    
    while (rowMin <= rowMax && colMin <= colMax) {
        for (let i = colMin; i <= colMax; i++) {
            ans.push(matrix[rowMin][i])
        }
        rowMin ++
        
        for (let j = rowMin; j <= rowMax; j++) {
           ans.push(matrix[j][colMax]) 
        }
        colMax --
        
        if (rowMax < rowMin || colMax < colMin) break
        
        for (let i = colMax; colMin <= i; i--) {
            ans.push(matrix[rowMax][i])
        }
        rowMax -- 
        
        for (let j = rowMax; rowMin <= j; j--) {
            ans.push(matrix[j][colMin])
        }
        colMin ++
    }
    return ans
};

var spiralOrder = function(matrix) {
    let ans = []
    
    while (matrix.length) {
        ans.push(...matrix.shift())
        
        for (let array of matrix) {
            if(array.length) {
                ans.push(array.pop())
                array.reverse()
            }
        }
        
        matrix.reverse()
    }
     return ans
 };


// 78. subsets 
// builds all subsets starting from []
// first num produces [[], [1]]
// then we take next num and concat it with every subset already in our subset and add those to the subsets
// [[], [1], [2], [1,2]]
// also there is a recursive solution using backtracking (push + pop your dependency array so you dont re-add numbers)

var subsets = function(nums) {
    let powerSet = [[]]
    
    for (let num of nums) {
        let len = powerSet.length
        for (let i = 0; i < len; i++) {
            let x = powerSet[i]
            powerSet.push([...powerSet[i], num])
        }
    }
    return powerSet
};