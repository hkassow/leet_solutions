// 57. insert interval
// check boundary conditions on this one
var insert = function(intervals, newInterval) {
    let ans = []
    let set = false
    
    for (let interval of intervals) {
        if ((interval[1] < newInterval[0])) {
            ans.push(interval)
        } else if (newInterval[1] < interval[0]){
            if (!set) { 
                ans.push(newInterval)
                set = true
            }
            ans.push(interval)
        } else {
            if (!set) {
                ans.push(newInterval)
                set = true
            }
            ans[ans.length - 1] = [Math.min(ans[ans.length - 1][0], interval[0]), Math.max(ans[ans.length - 1][1], interval[1])]
        }
    }
    if (!set) {
        ans.push(newInterval)
    }
    return ans
};
 
// 542. 01 matrix
// expand from the 0 nodes 

var updateMatrix = function(mat) {
    let [m, n] = [mat.length, mat[0].length] 
    let queue = new Queue()
    let ans  = Array(m).fill().map(() => Array(n).fill(-1))
    
    
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            if (mat[i][j] === 0) {
                ans[i][j] = 0
                queue.enqueue([i, j])
            }
        }
    }
    
    while (!queue.isEmpty()) {
        let node = queue.dequeue()
        let dir = [[0,1], [1,0], [0,-1], [-1,0]]
        
        for (let dirNode of dir) {
            let adjNode = [node[0] + dirNode[0], node[1] + dirNode[1]]
            if (( 0 <= adjNode[0]) & (adjNode[0] < m) & (0 <= adjNode[1]) & (adjNode[1] < n)) {
                if ((mat[adjNode[0]][adjNode[1]] === 1) & (ans[adjNode[0]][adjNode[1]] === -1)) {
                    ans[adjNode[0]][adjNode[1]] = ans[node[0]][node[1]] + 1
                    queue.enqueue(adjNode)
                }
            
            }
        }
    }
    return ans
};

// 973. k closest points to origin
// maxProirity (heap) with prority set by distance from origin
// to avoid checking if the array size currently doesnt have enough elements
// push k elements into our heap that are infinitly distances from the origin
// then on every new point we always throw out an element and always add an element


 var kClosest = function(points, k) {
    let x = new MaxPriorityQueue({priority: (num) => dist(num)})

    while (heap.size() < k) {
        heap.enqueue([Infinity, 0])
      }
    
      for (const point of points) {
        const dis = dist(point)
        if (dis < heap.front().priority) {
            heap.dequeue()
            heap.enqueue(point)
        }
      }
      return heap.toArray().map(a => a.element)
};
    
    
const dist = (num) => {
    return (num[0]*num[0] + num[1]*num[1])
}

// 3. longest substring without repeating characters
// instead of using includes can use an object
var lengthOfLongestSubstring = function(s) {
    let ans = 0
    let currentSub = ''
    
    for (let letter of s) {
        while (currentSub.includes(letter)){
            currentSub = currentSub.substring(1)
        }
        
        
        
        currentSub = currentSub + letter
        if (ans < currentSub.length) {
                ans =  currentSub.length
            }
    }
    return ans
};

//15. 3sum
var threeSum = function(nums) {
    nums = nums.sort((a,b) => a - b ) 
    
    let i = 0
    let n = nums.length
    let ans = []
    
    while (i < n ) {
        if (!(nums[i] <= 0)) {
            break
        } else if (nums[i] === nums[i-1]) {
            i++ 
            continue
        }
        let j = i + 1
        let k = n - 1
        while (j < k) {
            let sum = nums[i] + nums[j] + nums[k]
            if (sum < 0) {
                j++;
            } else if (0 < sum) {
                k--
            } else {
                ans.push([nums[i],nums[j],nums[k]])
                while (j < k & (nums[j] === nums[j+1])){ 
                    j++
                }
                while (j < k & (nums[k] === nums[k-1])) {
                    k--
                }
                j++
                k--
            }
        }
        i++
    }
    return ans
};

// 102. binary tree level order traversal

var levelOrder = function(root) {
    if (!root) return []
    let res = []
    
    const addLevel = (depth, root) => {
        if (!root) return
        if (!res[depth]) {
            res.push([])
        }
        res[depth].push(root.val)
        addLevel (depth+1, root.left)
        addLevel (depth+1, root.right)
    }
    addLevel(0, root)
    return res
};

// 133. clone graph 


var cloneGraph = function(node, visited ={}) {
    
    if (!node) {
        return
    } 
    if (visited[node.val]) {
        return visited[node.val]
    }
    
    let clone = new Node(node.val)
    visited[clone.val] = clone
    
    for (let neighbor of node.neighbors) {
        if (visited[neighbor.val]) {
            clone.neighbors.push(visited[neighbor.val])
        } else {
            clone.neighbors.push(cloneGraph(neighbor, visited))
        }
    }
    return clone
};



// 150. evaluate reverse polish notation
// storing functions inside a hash 
// Math.trunc removes decimals no rounding
// only reach an operation after we've reached at least two integers
// other wise it would be an invalid RPN
var evalRPN = function(tokens) {
    const operations = {'+': (a,b) => (a+b), "-": (a,b) => (a-b), "*": (a,b) => (a*b), "/": (a,b) => Math.trunc(a/b) }
    let stack = []
    for (const token of tokens) {
        if (operations[token]) {
            a = parseInt(stack.pop())
            b = parseInt(stack.pop())
            stack.push(operations[token](b,a))
        } else {
            stack.push(token)
        }
    }
     return stack.pop()
 };

// 207. course schedule 
// graph stores the depencies 
// degree stores how many prereqs a class has (if it has noreqs degree is 0)
var canFinish = function(numCourses, prerequisites) {
    const order = []
    const graph = new Map()
    const degree = Array(numCourses).fill(0)
    for (let [a, b] of prerequisites) {
        if (!graph.get(b)) {
            graph.set(b, [])
        }
        graph.get(b).push(a)
        degree[a]++ 
    }
    const queue = []
    for (let i = 0; i < numCourses; i++) {
        if (degree[i] === 0) {
            queue.push(i)
        }
    }
    while (queue.length) {
        let nextCourse = queue.shift()
        if (graph.get(nextCourse)) {
            
            for (let course of graph.get(nextCourse)) {
                degree[course]--
                if (degree[course] === 0) {
                    queue.push(course)
                }
            }
        }
        order.push(nextCourse)
    }
    return (order.length === numCourses)
};





// 208. implement trie (prefix tree)
// implement insert, search, and prefix 
// root = {a: {p: {p: {l: {e: {lastLetter: true }}}}}}
// trie containing apple

var Trie = function() {
    this.root = {} 
};

Trie.prototype.insert = function(word) {
    let node = this.root
    
    for (let c of word) {
        if (!node[c]) {
            node[c] = {}
        }
        node = node[c]
    }
    
    node.lastLetter = true
};
Trie.prototype.search = function(word) {
    let node = this.root
    
    for (let c of word) {
        node = node[c]
        if (!node) {
            return false
        }
     }
    return node.lastLetter? true : false
};
Trie.prototype.startsWith = function(prefix) {
    let node = this.root
    
    for (let c of prefix) {
        node = node[c]
        if (!node) {
            return false
        }
     }
    return true
};


// 322. coin change
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
// if number is 2 we move it forward
// if number is 0 we move it back
// continue until we reach the 2s 
var sortColors = function(nums) {
    let leftP = 0
    let rightP = nums.length - 1
    
    for (let i = 0; i<=rightP; i++) {
        while (i < rightP && nums[i] === 2) {
            [nums[i], nums[rightP]] = [nums[rightP], nums[i]]
             rightP--
        }
        while (leftP < i && nums[i] === 0) {
            [nums[i], nums[leftP]] = [nums[leftP], nums[i]]
            leftP++
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
// second solution 
// looks at bits
// for length 3 array like [1,2,3] there are 8 subarrays 2^3
// 000
// 001
// 010.... etc using this we can generate all sub arrays
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

// using bits
var subsets = function(nums) {
    let length = 1<<nums.length
    let sets = Array(length).fill().map(() => Array())
    for (let i = 1; i < length; i++) {
        let bits = i.toString(2)
        let pointer = 0
        for (let j = bits.length - 1; 0 <= j; j--) {
            if (bits[j] === '1') {
                sets[i].push(nums[pointer])
            }
            pointer++
        }
    }
    return sets
};


// 199. binary tree right side view
// fills out left most paths first 
// then any "right" node will override previous node 
var rightSideView = function(root) {
    var ans = []
    
    const travel = (root,h) => {
        if (!root) return
        
        ans[h] = root.val
        travel(root.left, h+1)
        travel(root.right, h+1)
    }
    
    travel(root, 0)
    return ans
};

// 5. longest palindromic substring 
// expanding about the even and odd centers 

var longestPalindrome = function(s) {
    let [lp, rp] = [0, 0]
    
    const findLongestPali = (i,j) => {
        while ((s[i] === s[j]) && (0 <= i) && (j < s.length)) {
            i--
            j++
        }
        return [i+1, j]
    }
    
    for (let center=0; center < s.length; center++ ) {
        
        let odd = findLongestPali(center, center)
        let even = findLongestPali(center, center+1)
        
        if ((rp-lp) < (odd[1] - odd[0])) {
            [lp, rp] = odd
        }
        if ((rp-lp) < (even[1] - even[0])) {
            [lp, rp] = even
        }
    }
    return s.slice(lp, rp)
};

// 62. unique paths
// dp tabulation bottom up 
var uniquePaths = function(m, n) {
    var count = 0
    const dp = Array(m).fill().map(() => Array(n).fill(1))

    for (let i = 1; i < m; i++) {
        for (let j = 1; j < n; j++) {
            
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
        }
    }
    return dp[m-1][n-1]
};

// 105. construct binary tree from preorder and inorder traversal 
//  preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
// tree [3,9,20,null,null,15,7]
// we stop traveling down the node if we reach our stop value
var buildTree = function(preorder, inorder) {
    let [p, i] = [0, 0]
    
    const helpBuild = (stop) => {
        if (inorder[i] !== stop) {
            let node = new TreeNode(preorder[p])
            p++
            node.left = helpBuild(node.val)
            i++
            node.right = helpBuild(stop)
            return node
        }
        return null
    }
    return helpBuild()
};

// 106. construct binary tree from inorder and postorder traversal
// inorder = [9,3,15,20,7], postorder = [9,15,7,20,3]
//parse both lists backwards using inorder as our "stopper" 
var buildTree = function(inorder, postorder) {
    let i = p = inorder.length - 1
    
    const helpBuild = (stop) => {
        if (inorder[i] !== stop) {
            let node = new TreeNode(postorder[p])
            p--
            node.right = helpBuild(node.val)
            i--
            node.left = helpBuild(stop)
            return node    
        }
        return null
    }
    return helpBuild()
};



// 11. container with the most water
// two pointer, shrinking our container

var maxArea = function(height) {
    let leftP = 0
    let rightP = height.length - 1
    let max = 0
    let runningArea = 0
    while (leftP < rightP) {
        runningArea = (rightP - leftP) * Math.min(height[leftP], height[rightP])
        
        max = Math.max(runningArea, max)
        
        if (height[leftP] < height[rightP]) {
            leftP++
        } else {
            rightP--
        }
    }
    
    return max
};



// 17. letter combination of a phone number
// pretty simple building problem 
var letterCombinations = function(digits) {
    if (!digits.length) return []
    let digit2Letter = {
        2: ["a" , "b" , "c"],
        3: ["d", "e", "f"],
        4: [ "g", "h", "i"],
        5: ["j", "k", "l"],
        6: ["m", "n", "o"],
        7: ["p", "q", "r", "s"], 
        8: ["t", "u", "v"],
        9: ["w", "x", "y", "z"]
    }
    let res = []
    
    const helpBuildLetter = (i, string) => {
        if (i === digits.length) {
            res.push(string)
            return
        }
        let digit = digits[i]
        for (let letter of digit2Letter[digit]) {
            let newString = string + letter
            helpBuildLetter(i+1, newString)
        }
    }
    helpBuildLetter(0, "")
    return res
}; 


// 79. word search
// use the matrix and set position to "*" if we have traveled 
// this prvenets any unnecessary space use
var exist = function(board, word) {
    let adjVector = [[0,1], [1,0] , [0, -1] , [-1,0]]
    let ans = false
    let stack = []
    
    
    //p = word pointer
    const helpTravel = (i, j, p) => {
        if (p === word.length) {
            return true
        } 
        for (let adj of adjVector) {
            
            let newPoint = [i + adj[0], j + adj[1]]
            if (0 <= newPoint[0] && 0 <= newPoint[1] && newPoint[0] < board.length && newPoint[1] < board[0].length) {
                if (board[newPoint[0]][newPoint[1]] === word[p]) {
                    board[newPoint[0]][newPoint[1]] = "*"
                    if (helpTravel(newPoint[0], newPoint[1], p+1)) return true
                    board[newPoint[0]][newPoint[1]] = word[p]
                }
            }
        }
        
    }
    for (let i = 0; i < board.length; i++) {
        for (let j = 0; j < board[0].length; j++) {
            if (board[i][j] === word[0]) {
                board[i][j] = "*"
                if (helpTravel(i, j, 1)) return true
                board[i][j] = word[0]
                
            } 
        }
    }
    
    
    return ans
};



// 438. find all anagrams in a string
// sliding window
var findAnagrams = function(s, p) {
    
    let letterFreq = {}
    let ans = []
    let left = 0 
    let right = 0
    let count = p.length
    
    for (let i = 0; i < p.length; i++) {
        if (!letterFreq[p[i]]) {
            letterFreq[p[i]] = 0
        }
        letterFreq[p[i]] ++
    }
    
    while (right < s.length) {
        if (0 < letterFreq[s[right]]) count--
        letterFreq[s[right]]--
        right++
        
        if (count === 0) ans.push(left)
        if (right-left === p.length){ 
            
            if (0 <= letterFreq[s[left]]) count++
            letterFreq[s[left]]++
            left++
        }
    }
    return ans
};





// 310. minimum height trees
// used set for o(1) delete vs array that would have o(n) because of searching index
 var findMinHeightTrees = function(n, edges) {
    if (n < 2) return [0]
    let graph = Array(n).fill().map(() => new Set())
    for (const [i,j] of edges) {
        graph[i].add(j)
        graph[j].add(i)
    }
    
    let outerLeaves = []
    //index for leaf
    graph.forEach((connections, index) => {
        if (connections.size === 1) {
            outerLeaves.push(index)
        }
    })
    while (2 < n) {
        n = n - outerLeaves.length
        let newLeaves = []
        for (const leaf of outerLeaves) {
            let innerLeaf = graph[leaf].values().next().value
            graph[innerLeaf].delete(leaf)
            graph[leaf].delete(innerLeaf)
            if (graph[innerLeaf].size === 1) {
                newLeaves.push(innerLeaf)
            }
        }
        outerLeaves = newLeaves
    }
    return outerLeaves
};



// 621. task scheduler 
// take the biggest limiter and use that to create your interals
// you can stretch your interval to be longer than the cooldown period to achieve the minimum number of idles
var leastInterval = function(tasks, n) {
    let max = 0
    let maxCount = 0
    let taskCount = {}
    
    
    for (const task of tasks) {
        if (!taskCount[task]) {
            taskCount[task] = 0
        }
        taskCount[task]++
        
        if (taskCount[task] === max) {
            maxCount++
        } else if (max < taskCount[task]) {
            max = taskCount[task]
            maxCount = 1
        }
     }
     const partLength = n - (maxCount - 1)
     const partCount = max - 1
     const emptySlots = partLength * partCount
     const availableTasks = tasks.length - max * maxCount 
     const idleSlots = Math.max(0, emptySlots - availableTasks) 
 
     return tasks.length + idleSlots
 };



 // 146. lru cache
 // using dumby head and tail to prevent checking boundaries
 var LRUCache = function(capacity) {
    this.head = {val: "head"}
    this.tail = {val: "tail"}
    this.hash = {}
    this.size = 0
    this.max = capacity
    this.head.next = this.tail
    this.tail.prev = this.head
};

LRUCache.prototype.get = function(key) {
    let node = this.hash[key]
    if (!node) return (-1)
    
    node.next.prev = node.prev
    node.prev.next = node.next
    this.head.next.prev = node
    node.prev = this.head
    node.next = this.head.next
    this.head.next = node
   
    
    return node.val
};

LRUCache.prototype.put = function(key, value) {
    if (this.hash[key]) {
        this.hash[key].val = value
        this.get(key)
    } else {
        if (this.size === this.max) {
            delete this.hash[this.tail.prev.key]
            this.tail.prev = this.tail.prev.prev
            this.tail.prev.next = this.tail
        } else { this.size ++ }
        
        let newNode = {key: key,
                      val: value}
        
        newNode.next= this.head.next
        newNode.prev = this.head
        this.head.next.prev = newNode
        this.head.next = newNode
        
        this.hash[key] = newNode
    }
};


// 230. kth smallest element in a bst

var kthSmallest = function(root, k) {
    let kthSmall = 0
    
    const travelOver = (node, element = 0) => {
        if (!node || kthSmall) return element
        
        element = travelOver(node.left, element) + 1 
        
        if (element === k) {
            kthSmall = node.val
            return;
        }
        
        element = travelOver(node.right, element)
        return element
    }
    
    travelOver(root, 0)
    return kthSmall
};

// 76. minimum window substring
// sliding window 
var minWindow = function(s, t) {
    let letterFreq = {}
    let l = 0
    let r = 0
    let count = t.length
    let min = [-1,-1]
    for (const c of t) {
        letterFreq[c] = (letterFreq[c] || 0) + 1
    }
    while (r < s.length) {
        if (letterFreq[s[r]] !== undefined) {
            letterFreq[s[r]] --
            if (0 <= letterFreq[s[r]]) {
                count--
            }
        }
        
        while (count === 0) {
            if (min[0] === -1 || r-l < min[1] - min[0]) {
                min = [l,r]
            }
            if (letterFreq[s[l]] !== undefined) {
                letterFreq[s[l]]++
                if (0 < letterFreq[s[l]]) {
                    count++
                }
            }
            l++
        }
        r++
    }
    if (min[0] === -1) return ""
    return s.substring(min[0], min[1] + 1)
};



// 297. serialize and deserialize binary tree
// function takes in a root node
// travels down the left side creating node.left values
// once we reach a null value adds a null stopper
// once we reach end of just left nodes we look at the leftmost node's right node 

var serialize = function(root) {
    if (!root) return null
    let str = ''
    
    const serializeDown = (node) => {
        if (!node) {
            str += ` null`
        } else {
            str += ` ${node.val}`
            serializeDown(node.left)
            
            serializeDown(node.right)
        } 
    }
    serializeDown(root)
    return str
};

var deserialize = function(data) {
    if (!data) return null
    data = data.trim().split(' ')
    
    let p = 0
    
    const helper = () => {
        if (data[p] === 'null') {
            p++
            return null  
        } 
        let node = new TreeNode(data[p])
        p++
        node.left = helper()
        
        node.right = helper()
        return node
    }
    
    
    return helper()
};

// 42. trapping rain water
//two pointer moving one boundary at a time until they meet


var trap = function(height) {
    let trapped = 0
    let lp = 0
    let rp = height.length - 1
    let leftMax = height[lp]
    let rightMax = height[rp]
    
    while (lp < rp) {
        leftMax = Math.max(height[lp], leftMax)
        rightMax = Math.max(height[rp], rightMax)
        
        trapped += leftMax - height[lp];
        trapped += rightMax - height[rp];
        
        (rightMax < leftMax)? rp-- : lp++
    }
    
    return trapped
};


// 295. find median from data stream 
// keeping track of all the numbers left of our median in maxHeap (1,2,3) < 3 is highest priority
// all numbers right of our median in min heap (5,4,3) < 3 is highest priority  
// rebalancing heaps so that theyre always equal or offset by 1

var MedianFinder = function() {
    this.rightHeap = new MinPriorityQueue({priority: (num) => num})
    this.leftHeap = new MaxPriorityQueue({priority: (num) => num})
};

MedianFinder.prototype.addNum = function(num) {
    if (num < this.leftHeap.front()?.element) {
        this.leftHeap.enqueue(num)
    } else {
        this.rightHeap.enqueue(num)
    }
    
    if (this.rightHeap.size() - this.leftHeap.size() > 1) {
        this.leftHeap.enqueue(this.rightHeap.dequeue().element)
    } else if (this.leftHeap.size() - this.rightHeap.size() > 1) {
        this.rightHeap.enqueue(this.leftHeap.dequeue().element)
    }

};
MedianFinder.prototype.findMedian = function() {
   if (this.rightHeap.size() > this.leftHeap.size()) {
       return this.rightHeap.front().element
   } else if (this.leftHeap.size() > this.rightHeap.size()) {
       return this.leftHeap.front().element
   } else {
       return (this.rightHeap.front().element + this.leftHeap.front().element)/2
   }
};



// 127. word ladder 
// go through each letter placement for each word that gets into our list
// will slowly eliminate all potential matches by removing them from the wordSet
// once we can get to a word we dont want to travel it again so we remove it ^^
// let altWord = word.substring(0, i) + letter + word.substring(i + 1);
// can also destructure string using array
var ladderLength = function(beginWord, endWord, wordList) {
    
    let alphabet = 'abcdefghijklmnopqrstuvwxyz'.split('');
    
    let wordSet = new Set(wordList)
    
    let queue = [beginWord]
    let steps = 1
    
    while (queue.length) {
        let next = []
        
        for (const word of queue) {
            if (word === endWord) {
                return steps
            }
            for (let i = 0; i < beginWord.length; i++) {
                for (const letter of alphabet) {
                    let altWord = word.substring(0,i) + letter + word.substring(i+1)
                    if (wordSet.has(altWord)) {
                        next.push(altWord)
                        wordSet.delete(altWord)
                    }
                }
            }
            
        }
        queue = next
        steps++
    }
    return 0
};


// 224. basic calculator 
// when we reach ( we store previous sum & sign (incase negative before like '-(5)' )
// if no previous sum or sign it will just be sum = 0 and sign = 1
// when we reach ) we take our equation and add it to the previous stored sum 
var calculate = function(s) {
    
    
    let regex = /[0-9]/
    
    let sum = 0
    let sign = 1
    let stack = []
    
    for (let i = 0; i < s.length; i++) {
        if (s[i].match(regex)) {
            let currNum = ''
            while (i < s.length && s[i].match(regex)) {
                currNum += s[i]
                i++
            }
            i--
            sum += currNum * sign
        } else if (s[i] === '(') {
            stack.push(sum)
            stack.push(sign)
            sum = 0
            sign = 1
        } else if (s[i] === ')') {
            sign = stack.pop()
            sum = stack.pop() + sign*sum
        } else if (s[i] === '-') {
            sign = -1
        } else if (s[i] === '+') {
            sign = 1
        }
    }
    return sum
};

// 432. all o'one data structure
// using linked list to keep track of key count 
// each unique key value will get its own node and its own set
// inc and dec remove key from previous node and add it to the new node
// if the previous node is empty we need to remove it from our linked list
// instead of keeping track of maxCount we couldve kept track of the tail of our LL
// but i found keeping track of num to be easier 
var AllOne = function() {
    this.keyToCount = {}
    this.countToNode = {}
    this.head = {
        next: null,
        val : new Set()
    }
    this.head.val.add('')
    this.countToNode[0] = this.head
    this.maxCount = 0
};
AllOne.prototype.inc = function(key) {
    if (!this.keyToCount[key]) {
        this.keyToCount[key] = 0
    }
    let count = this.keyToCount[key]
    let prevNode = this.countToNode[count]
    prevNode.val.delete(key)
    
    if (count !== 0 && !prevNode.val.keys().next().value) {
        prevNode.prev.next = prevNode.next
        if (prevNode.next) {
            prevNode.next.prev = prevNode.prev
        }
        prevNode = prevNode.prev
        delete this.countToNode[count]
    }
    
    count++
    if (!this.countToNode[count]) {
        this.countToNode[count] = {
            prev: prevNode,
            next: prevNode.next,
            val: new Set()
        }
        if (prevNode.next) {
            prevNode.next.prev = this.countToNode[count]
        }
        prevNode.next = this.countToNode[count]
    }
    this.countToNode[count].val.add(key)
    this.keyToCount[key] = count
    if (this.maxCount < count) {
        this.maxCount = count
    }
    
};
AllOne.prototype.dec = function(key) {
    let count = this.keyToCount[key]
    let startNode = this.countToNode[count]
    startNode.val.delete(key)
    count--
    
    if (!this.countToNode[count]) {
        this.countToNode[count] = {
            next: startNode,
            prev: startNode.prev,
            val: new Set()
        }
        startNode.prev.next = this.countToNode[count]
    }
    startNode.prev = this.countToNode[count]
    
    if (!startNode.val.keys().next().value) {
        if (startNode.next) {
            startNode.next.prev = startNode.prev
            startNode.prev.next = startNode.next

        } else {
            startNode.prev.next = null
            if (this.maxCount === count+1) {
                this.maxCount = count
            }
        }
        delete this.countToNode[count+1]
    }
    
    
    this.keyToCount[key] = count
    if (0 < count) {
        this.countToNode[count].val.add(key)
    }
       
}
AllOne.prototype.getMaxKey = function() {

    return this.countToNode[this.maxCount].val.keys().next().value
};


AllOne.prototype.getMinKey = function() {
    if (this.head.next) {
        return this.head.next.val.keys().next().value
    }
    return this.head.val.keys().next().value
};


// 1235. job scheduling 
// use binary search to find the largest profit before our start time 
// start with the earliest end time 

var jobScheduling = function(startTime, endTime, profit) {
    let dp = []
    let jobs = []
    
    for (let i = 0; i < startTime.length; i++ ) {
        jobs.push([startTime[i], endTime[i], profit[i]])
    }
    jobs.sort((a,b) => a[1] - b[1])
    
    dp[0] = [0, 0]
    
    const findHighestProfitBefore = (avaliable) => {
        let low = 0
        let high = dp.length
        let mid
        
        while (low < high) {
            mid = Math.floor((low+high)/2)
            
            if (dp[mid][0] <= avaliable ) {
                low = mid + 1
            } else {
                high = mid
            }
        } 
        return low - 1
    }
   
    
    for (let i = 0; i < startTime.length; i++) {
        let job = jobs[i]
        let profitIdx = findHighestProfitBefore(job[0])
        let currProfit = job[2] + dp[profitIdx][1]
        
        
        if (dp[dp.length -1][1] < currProfit) {
            dp.push([job[1], currProfit])
        }
    }
    
    return dp[dp.length - 1][1]
    
};


// 985. sum of even numbers after queries 

var sumEvenAfterQueries = function(nums, queries) {
    let result = Array(nums.length)
    let sum = 0
    
    for (let i = 0; i < nums.length; i++) {
        if (nums[i]%2 === 0) {
            sum += nums[i]
        } 
    }
    
    
    for (let j = 0; j < queries.length; j++) {
        result[j] = sum
        
        const query = queries[j]
        
        const [val, i] = query 
        
        const [startValDiv, queryValDiv] = [Math.abs(nums[i]%2), Math.abs(val%2)]
        
       
        if (startValDiv === 0) {
            
            if (queryValDiv === 0) {
                result[j] += val
            } else {
                result[j] -= nums[i] 
            }
           
        } 
        else if ( startValDiv === 1 && queryValDiv === 1) {
            result[j] += val + nums[i]
        } 
  
        
        nums[i] += val
        sum = result[j]
    }
    
    return result
};

// 23. merge k sorted lists
// used merging two lists over and over
var mergeKLists = function(lists) {
    while (1 < lists.length) {
        let a = lists.pop()
        let b = lists.pop()
        lists.push(mergeTwoLists(a,b))
    }
    
     
     return lists[0] || null
 };
 var mergeTwoLists = (a, b) => {
     let dumbyHead = new ListNode()
     let node = dumbyHead
     while(a && b) {
         
         
         if (a.val < b.val) {
             node.next = new ListNode(a.val)
             a = a.next
         } else {
             node.next = new ListNode(b.val)
             b = b.next
         }
         node = node.next
     }
     
     
     if (a) {
         node.next = a
     }
     if (b) {
         node.next = b
     }
     
     return dumbyHead.next
 }
// using priority queue
 var mergeKLists = function(lists) {
    let minPrio = new MinPriorityQueue({priority: (a) => a.val})
    
    
    for (const list of lists) {
        if (list) {
            minPrio.enqueue(list)
        }
    }
    const smartHead = new ListNode('sentinel', null)
    let currNode = smartHead
    
    while (minPrio.size()) {
        const node = minPrio.dequeue().element
        if (node.next) {
            minPrio.enqueue(node.next)
        }
        currNode.next = node
        currNode = node
    }
    currNode.next = null
    return smartHead.next
};

 // 84. largest rectangle in histogram
 // use monostack to keep track of rectangle heights
 // if we reach a height that is smaller than top of the stack we must pop rectangles from the stack 
 // increasing the width as we go relative to what we have popped
 // pushed a 0 rectangle to the end to force our stack to dump at the end
 var largestRectangleArea = function(heights) { 
    let maxArea = heights[0]
    heights.push(0)
    let monoStack = []
    for (let i = 0; i < heights.length; i++) {
        let width = 0
        
        while(monoStack.length && heights[i] <= monoStack[monoStack.length - 1][0]) {
            let currRect = monoStack.pop()
            width += currRect[1]

            if (maxArea < width*currRect[0]) {
                maxArea = width*currRect[0]
            }
        }
        if (heights[i]) {
            monoStack.push([heights[i], width+1])
        }   
    }
    return maxArea
};




// 217. contains duplicate 

var containsDuplicate = function(nums) {
    let distinctSet = new Set(nums)

    return distinctSet.size !== nums.length
};


// 242. valid anagram

var isAnagram = function(s, t) {
    let sLetterFreq = {}
    let count = 0
    for (const letter of s) {
        if (!sLetterFreq[letter]) {
            sLetterFreq[letter] = 0
            count += 1
        }
        sLetterFreq[letter] += 1
    }
    
    
    for (const letter of t) {
        if (!sLetterFreq[letter]) {
            return false
        }
        sLetterFreq[letter] -= 1
        if (sLetterFreq[letter] === 0) {
            count -= 1
        }
    }
    return !count
};

// 1. two sum 
// careful of evaluating prevNums[sum] incase the ith position is the 0th position
// this will evaluate to falsey value
var twoSum = function(nums, target) {
    let prevNums = {}
   
   for (let i = 0; i < nums.length; i++) {
       let sum = target - nums[i]
       if (prevNums[sum] !== undefined) {
           return [prevNums[sum], i]
       } 
       prevNums[nums[i]] = i
   }
   
}

// 49. group anagrams 

var groupAnagrams = function(strs) {
    
    let result = {}
    
    for (let str of strs) {
        
        let orderedStr = str.split("").sort().join("")
        if (result[orderedStr]) {
            result[orderedStr].push(str)
        } else {
            result[orderedStr] = [str]
        }
    }
    return Object.values(result)
}


// 557. reverse words in a string III
// reverse each word but maintain order

var reverseWords = function(s) {
    
    let arrayStr = s.split(' ')
    arrayStr = arrayStr.map(word => word.split('').reverse().join(''))
    return arrayStr.join(' ')
};

// 347. top k frequent elements
// first way using SORT
// second way is o(n) with no sort
var topKFrequent = function(nums, k) {
    let numToCount = {}
    
    for (let i = 0; i < nums.length; i++) {
        if (!numToCount[nums[i]]) {
            numToCount[nums[i]] = 0
        }
        numToCount[nums[i]] += 1
        
    }
    
    return Object.keys(numToCount).sort((a,b) => numToCount[a] - numToCount[b]).slice(-k)
   
};
var topKFrequentNoSort = function(nums, k) {
    let numToCount = {}
    let bucket = []
    
    
    for (let i = 0; i < nums.length; i++) {
        if (!numToCount[nums[i]]) {
            numToCount[nums[i]] = 0
        }
        numToCount[nums[i]] += 1
        
    }
    
    for (let num of Object.keys(numToCount)) {
        if (!bucket[numToCount[num]]) {
            bucket[numToCount[num]] = []
        }
        bucket[numToCount[num]].push(num)
    }
    
    let ans = []
    
    for (let j = bucket.length - 1; ans.length !== k; j--) {
        if (bucket[j]) {
            ans.push(...bucket[j])
        }
    }
    return ans
};

// 1680. concatenation of consecutive binary numbers
// think about if we were concatenating decimal numbers
// for numbers 1-9 we would shift/multiply by 10
// then when we reach a new multiple of 10 we increase how much we are shifting by 10
// in binary we shifting by a power of 2 and increasing when we reach new power of 2 
var concatenatedBinary = function(n) {
    let mod = 1e9 + 7
    let twoPow = 2
    
    
    let ans = 1
    for (let i = 2; i <= n; i++) {
        if (i === twoPow) twoPow *= 2
        ans = (ans * twoPow + i) % mod 
    }
    
    return ans
};

// 113. path sum II

var pathSum = function(root, targetSum) {
    
    
    let path = []
    let ans = []
    const travelPath = (sum, node) => {
        if (node) {
            sum += node.val
            path.push(node.val)
            
            if (!node.left && !node.right) {
                if (sum === targetSum) {
                    ans.push([...path])
                }
            }
            
            travelPath(sum, node.left)
        
            travelPath(sum, node.right)
            
            path.pop(node.val)
        
        }
        

    }
    travelPath(0, root)
    return ans
};

// 36. valid sudoku 

var isValidSudoku = function(board) {
    for (let i = 0; i < 9; i++) {
        const checkRow = {}
        const checkCol = {}
        const checkBox = {}
        for (let j = 0; j < 9; j++) {
            let _row = board[i][j]
            let _col = board[j][i]
            let _box = board[3 * Math.floor(i/3) + Math.floor(j/3)][3 * (i%3) + (j%3)]
            
            if (checkRow[_row] && _row !== '.')  {    
                return false
            }
            if (checkCol[_col] && _col !== '.') {
                return false
            }
            if (checkBox[_box] && _box !== '.') {
                return false
            }
        
            checkRow[_row] = true
            checkCol[_col] = true
            checkBox[_box] = true
        }
    }
    
    return true
};

// 128. longest consecutive sequence 
// ends takes in the tail node and returns the head node
// starts takes in the head node and returns the tail node
// for every number im checking if there is a previous list that starts or ends adjacent to number
var longestConsecutive = function(nums) {
    let longest = 0
    
    nums = new Set(nums)
    let ends = {}
    let starts = {}
    let start, end
    
    for (let num of nums) {
        start = num
        end = num
        
        if (starts[num+1] !== undefined) {
            end = starts[num + 1]
            delete starts[num + 1]
        } 
        
        if (ends[num-1] !== undefined) {
            start = ends[num - 1]
            delete ends[num - 1]
        }
        
        longest = Math.max(longest, end - start + 1)
        
        
        starts[start] = end
        ends[end] = start
    }
    
    return longest
};

// 125. valid palindrome
var isPalindrome = function(s) {
    let regex = /[a-zA-Z0-9]/
    let leftP = 0
    let rightP = s.length - 1
    
    while (leftP < rightP) {
        if (!s[leftP].match(regex)) {
            leftP ++
            continue;
        }
        
        if (!s[rightP].match(regex)) {
            rightP --
            continue
        }
        
        if (s[rightP].toLowerCase() !== s[leftP].toLowerCase()) {
            return false
        }
        leftP ++
        rightP --
    }
   
    
    return true
};


// 167. two sum II - input array is sorted
// use two pointers
// since array is already sorted we just keep incrementing one pointer until we find sum

var twoSum = function(numbers, target) {
    let leftP = 0
    let rightP = numbers.length - 1
    let sum = 0

    while (leftP < rightP) {
        sum = numbers[leftP] + numbers[rightP]
        if (sum === target) {
            return [leftP + 1, rightP + 1]
        }
        
        if (sum < target) {
            leftP ++
        } else {
            rightP --
        }
    }
};


// 121. best time to buy and sell stock
var maxProfit = function(prices) {
    let currentBuy = prices[0]
    let max = 0
    
    
    for (const price of prices) {
        max = Math.max(max, price - currentBuy)
        currentBuy = Math.min(price, currentBuy)
    }
    return max
};

// 424. longest repeating character replacement 
// use sliding window
// if we visit distinct characters our maxWindow will stay at 1 
// so our left index will start to move forward once we reach the alloted amount of replaced letters
// if we reach a non-distinct character IN our window our window size increases
// the trick is we never reduce our window size so returning right - left is sufficient
// if our left index is outside the bounds of right index - (max count + k) 
// we move left index forward to maintain the max window
var characterReplacement = function(s, k) {
    let letterFreq = {}
    let maxSize = 0
    let leftP = 0
    let rightP = 0
    
    while (rightP < s.length) {
        const char = s[rightP]
        if (!letterFreq[char]) letterFreq[char] = 0
        letterFreq[char]++
        
        if (maxSize < letterFreq[char]) {
            maxSize = letterFreq[char]
        }
        
        if (leftP <= (rightP - maxSize - k)) {
            letterFreq[s[leftP]]--
            leftP++
        }
        rightP++
    }
    
    
    return rightP - leftP

    //return Math.min(s.length, maxSize + k)
    //this is usually the return value that makes the most sense to me
};

// 567. permutation in string
// sliding window 
// if we reach a letter that is either not in s1 or
// we have too many iterations of the current letter in our sliding window
// then we start to throw out letters until either we are at our right index
// or we need the current letter in our window
//
var checkInclusion = function(s1, s2) {
    let letterCount = {}
    for (const letter of s1) {
        if (!letterCount[letter]) {
            letterCount[letter] = 0
        }
        letterCount[letter] ++
    }
    let leftP = 0
    let rightP = 0
    while (rightP < s2.length) {
        const char = s2[rightP]
        if (!letterCount[char]) { 
            while (leftP <= rightP && !letterCount[char]) {
                letterCount[s2[leftP]]++
                leftP++
            }
        }
        if (letterCount[char]) {
            letterCount[char]--
        }

        rightP++
        if (rightP - leftP === s1.length) {
            return true
        }
    }
    return false
};

// 239. sliding window maximum
// use linked list to keep track of monotonic decreasing nums
// if our current max is out of range we shift to the next max
var maxSlidingWindow = function(nums, k) {
    if (k === 1) {
        return nums
    }
    
    let ans = []
    
    let window = new linkedList()
    
    for (let i = 0; i < nums.length; i++) {
        let num = nums[i]
        if (window.peekFront().val < i - k + 1) {
            
            window.shift()
        }
        while (nums[window.peekLast().val] < num ) {
            window.pop()
        }
        window.push(i)
        if (k - 1 <= i) {
            ans.push(nums[window.peekFront().val])
        }
    }
    return ans
};


class linkedNode {
    constructor(val) {
        this.val = val
        this.next = null
        this.prev = null
    }
}

class linkedList {
    constructor() {
        this.head = new linkedNode()
        this.tail = new linkedNode()
        this.head.next = this.tail
        this.tail.prev = this.head
        
    }
    peekFront = () => {
        return this.head.next
    }
    peekLast = () => {
        return this.tail.prev
    }

    push = (val) => {
        let node = new linkedNode(val)
        node.next = this.tail
        node.prev = this.tail.prev
        this.tail.prev.next = node
        this.tail.prev = node
    }
    shift = () => {
        let node = this.head.next
        node.next.prev = this.head
        this.head.next = node.next
        
        return node
    }
    pop = () => {
        let node = this.tail.prev
        node.prev.next = node.next
        node.next.prev = node.prev
        return node
    }
}


// 622. design circular queue
// linked list but fed into itself
// using sentinel head and tail to prevent checking boundary conditions
// and gives access to front and back nodes
var LLNode = class{
    constructor(val, next, prev) {
        this.val = val
        this.next = next || undefined
        this.prev = next || undefined
    }
}

var MyCircularQueue = class {
    constructor (k) {
        this.head = new LLNode(-1)
        this.tail = new LLNode(-1)
        this.size = 0 
        this.max = k
        this.head.next = this.tail
        this.tail.prev = this.head 
    }
    enqueue (value) {
        if (this.size === this.max) return false
        this.size++
        let node = new LLNode(value)
        node.prev = this.head
        node.next = this.head.next
        node.next.prev = node
        this.head.next = node
        return true
    }
    
    dequeue (){
        if (this.size === 0) return false
        this.size--
        let node = this.tail.prev
        node.prev.next = this.tail
        this.tail.prev = node.prev
        return true
    }

    front(){
        return this.tail.prev.val
    }
    rear() {
        return this.head.next.val
    }
    isEmpty() {
        return this.size === 0
    }
    isFull() {
        return this.size === this.max
    }
};
// 20. valid parentheses
// inserting the opposite symbol 
// when we reach a closing symbol we compare it with the expected closing bracket
var isValid = function(s) {
    
    let stack = []
    
    for (let symbol of s) {
        if (symbol === '(' ) {
            stack.push(')')
        }
        else if (symbol === '{') {
            stack.push('}')
        }
        else if (symbol === '[') {
            stack.push(']')
        } else {
            if (symbol !== stack.pop()) {
                return false
            }
        }
    }
    return !stack.length
};

// 22. generate parentheses
// back tracking

var generateParenthesis = function(n) {
    const res = []
    
    
    
    const travelPath = (open, close, str) => {
        if (str.length === n*2) {
            res.push(str)
        } else {
            if (0 < open) {
                travel(open - 1, close ,str+'(')
            }
            if (open < close) {
                travel(open, close - 1, str+')' )
           
            }
        }
         
    }
    travelPath(n, n, '')
    
    return res
};

// 739. daily temperatures 
// mono stack storing index of days that havent found a warmer day
// once we reach a day warmer than the top of our stack we loop until the top of the stack is empty of warmer than the current day
var dailyTemperatures = function(temperatures) {
    let ans = Array(temperatures.length)
    
    let tempStack = []
    let currTemp = temperatures[0]
    for (let i = 0; i < temperatures.length; i++) {
        ans[i] = 0
        currTemp = temperatures[i]
       
        while (temperatures[tempStack[tempStack.length - 1]] < currTemp) {
            let j = tempStack.pop()
            ans[j] = i-j
                
        }
        tempStack.push(i)
    }
    
    return ans
};


// 853. car fleet
// sort the cars by position
// start from the 2nd closest car to target
// ask if previous car is slower if it is we change our time to match the slower car
// increment if we find a car that is slower than the car ahead of it because this is now the new leading slow car

var carFleet = function(target, position, speed) {
    let ans = 1
    let cars = []
    for (let i = 0; i < position.length; i++) {
        cars.push([position[i], speed[i], (target - position[i])/speed[i]])
    }
    
    cars.sort((a,b) => b[0] - a[0])
    
    for (let i = 1; i < cars.length; i++) {
        
        if (cars[i][2] <= cars[i-1][2]) {
            cars[i][2] = cars[i-1][2]
        } else {
            ans ++
        }
    }
    
    return ans
};

// 704. binary search
// MATH.FLOOR since 3/2 => 1.5 

var search = function(nums, target) {
    let lp = 0
    let rp = nums.length - 1
    let mid
    
    while (lp <= rp) {
        mid = lp + Math.floor((rp - lp)/2)
        if (nums[mid] === target) {
            return mid
        }
        else if (nums[mid] > target) {
            rp = mid - 1
        }
        else {
            lp = mid + 1
        }
    }
    return -1
};

// 74. search a 2d matrix 
// this solution treates the matrix like a sorted list
var searchMatrix = function(matrix, target) {
    let x = matrix[0].length 
    
    let min = 0
    let max = x*matrix.length - 1
    let mid
    let num
    while (min <= max) {
        mid = min + Math.floor((max - min) / 2)
        num = matrix[Math.floor(mid/x)][mid%x]
        
        if (num === target) return true
        else if (num < target) min = mid+1
        else max = mid-1
    }
    return false
};

// 990. satisfiability of equality equations
// uses 2 passes
// first pass assigns every element a parent element
// second pass looks at the inequalities to make sure 
// if "a!=b" a and b do not have shared parent
var equationsPossible = function(equations) {
    let parent = new Map()
    
    const find = (a) => {
        parent.set(a, parent.get(a) || a)
        return (parent.get(a) !== a)? find(parent.get(a)) : a
        
    }
    for (const assignment of equations) {
        if (assignment[1] === '=') {
            const a = assignment[0]
            const b = assignment[3]
            
            parent.set(find(a), find(b))
        }
    } 
    for (const inequality of equations) {
        if (inequality[1] === '!') {
            const a = inequality[0]
            const b = inequality[3]
            
            if (find(a) === find(b)) return false
        }
    }
    return true
};


// 875. koko eating bananas 
// binary search 
var minEatingSpeed = function(piles, h) {
    let minP = 0
    let maxP = findMax(piles)
    let mid
    
    while (minP <= maxP) {
        mid = minP + Math.trunc((maxP - minP)/2)
        let speed = eatSpeed(piles, mid)
        if (speed <= h) {
            maxP = mid - 1
        } else {
            minP = mid + 1
        }
    }
    return minP
};

const findMax = (array) => {
        let max = 0
        for (let i = 0; i < array.length; i++) {
            if (max < array[i]) {
                max = array[i]
            }
        }
        return max
}

const eatSpeed = (piles, k) => {
        let hours = 0
        
        for (const pile of piles) {
            hours += Math.ceil(pile/k)
        }
        
        return hours
}

// 153. find minimum in rotated sorted array
// if we reach a sorted array leftP < mid < rightP return leftP
// else if the pivot is on the right side of our array
// then the middle element will be greater than rightP and we shift to the right window
// removing the mid element because we know our rightP will be smaller
// otherwise we cant be certain that mid is not the smallest element 
// but we know it is some element in the left window including mid
// think small case [3,1,2] [2,3,1] [1,2,3]
var findMin = function(nums) {
    let leftP = 0
    let rightP = nums.length - 1
    let mid
    while (leftP <= rightP) {
        mid = leftP + Math.trunc((rightP - leftP)/2)
        if (nums[leftP] <= nums[mid] && nums[mid] <= nums[rightP]) {
            return nums[leftP]
        }
        if (nums[rightP] < nums[mid]) {
            leftP = mid + 1
        } else {
            rightP = mid
        }
        
    }
};

// 981. time based key-value store
// binary search solution is first 
// second solution uses an array and populates all possible values for our timestamp to be called on
//  
var TimeMap = function() {
    this.timeMap = {}
};

TimeMap.prototype.set = function(key, value, timestamp) {
    if (!this.timeMap[key]) {
        this.timeMap[key] = []
    }
    this.timeMap[key].push([timestamp, value])
};

TimeMap.prototype.get = function(key, timestamp) {
    if (!this.timeMap[key]) return ""
    let leftP = 0
    let rightP = this.timeMap[key].length - 1
    let mid
    let res = ""
    while (leftP <= rightP) {
        mid = leftP + Math.trunc((rightP - leftP)/2)
        
        if (this.timeMap[key][mid][0] <= timestamp) {
            leftP = mid + 1
            res = this.timeMap[key][mid][1]
        } else {
            rightP = mid - 1
        }
    }
    return res
};
// second solution using an array an no binary search
// the memory for this is probably much worse because we have to store based on value of timestamp
// and not how many times we have called set 
var TimeMap = function() {
    this.timeMap = {}
};

TimeMap.prototype.set = function(key, value, timestamp) {
    if (!this.timeMap[key]) {
        this.timeMap[key] = Array(timestamp).fill("")
    }
    
    let pastVal = this.timeMap[key].at(-1)
    for (let i = this.timeMap[key].length; i < timestamp; i++) {
        this.timeMap[key].push(pastVal)
    }
    this.timeMap[key][timestamp] = value
};
TimeMap.prototype.get = function(key, timestamp) {
    if (!this.timeMap[key]) return ""
    
    return this.timeMap[key][Math.min(timestamp, this.timeMap[key].length - 1)]
};


// 4. median of two sorted arrays


var findMedianSortedArrays = function(nums1, nums2) {
    let n1 = nums1.length,
        n2 = nums2.length
        
    if (n1 < n2) return findMedianSortedArrays(nums2, nums1) 
    let leftP = 0,
        rightP = nums2.length*2,
        mid1,
        mid2

    while (leftP <= rightP) {
        mid2 = leftP + Math.floor((rightP - leftP)/2)
        mid1 = n1 + n2 - mid2 
        const left1 = mid1 === 0? -Infinity: nums1[Math.floor((mid1-1)/2)],
              left2 = mid2 === 0? -Infinity: nums2[Math.floor((mid2-1)/2)],
              right1 = mid1 === 2*n1? Infinity: nums1[Math.floor(mid1/2)],
              right2 = mid2 === 2 *n2? Infinity: nums2[Math.floor(mid2/2)]
        
        if (right2 < left1) {
            leftP = mid2 + 1
        } else if (right1 < left2) {
            rightP = mid2 - 1
        } else {
            return (Math.max(left1,left2) + Math.min(right1,right2))/2
        }
    }
};


// 838. push dominoes
// here is two solutions because 
// i think the first one is too long to reproduce without struggling with edge cases

//1st solution
// 3 possibilites
// 1) if we reach a "."
// we look for a Left falling domino 
// if our right pointer then arrives at a Right falling domino first 
// then theres no possibility for this domino to fall
// 2) if we reach a "R" then we look for a "L"
// if our right pointer reachs  a "R" domino before finding a "L" dominio
// we move our left pointer up to our right pointer knocking over every domino to the right
// if we do reach a "L" domino with our right pointer we just need to find the lenght of our inner set
// then we knock over the respective dominos as we move up left pointer
// 3) if we reach a "L" then we can just move our pointers up
// this is because we knock over the "." dominos when we reach them
// either there are no dominos to the left to knock over
// or theyre already knocked over by a previous iteration

var pushDominoes = function(dominoes) {
    let ans = dominoes.split('')
    let leftP = 0
    let rightP = 0
    let leftDomino = ans[leftP]
    let rightDomino = ans[rightP]

    while (leftP < dominoes.length) {
        leftDomino = ans[leftP]
        rightDomino = ans[rightP]
        if (leftDomino === '.') {
            while(rightP < dominoes.length && rightDomino === '.') {
                rightP++
                rightDomino = ans[rightP]
            }
            if (rightDomino === 'R' || rightDomino === '.' || !rightDomino) {
                leftP = rightP
            } else {
                while (leftP < rightP) {
                    ans[leftP] = 'L'
                    leftP++
                }
            }
        } else if (leftDomino === 'R') {
            while (rightP < dominoes.length && (rightP === leftP || rightDomino === '.')) {
                rightP++
                rightDomino = ans[rightP]
            }
            if (rightDomino === 'L') {
                let len = rightP - leftP + 1
                let mid = (leftP + Math.floor(len/2))
                let odd = len%2
                leftP++
                while (leftP < rightP) {
                    if (leftP < mid) {
                        ans[leftP] = 'R'
                    } else {
                        if (odd) {
                            if (mid < leftP) {
                                ans[leftP] ='L'
                            }
                        } else {
                            ans[leftP] = 'L'
                        }
                    }
                    leftP++
                }
            } else {
                while (leftP < rightP) {
                    ans[leftP] = "R"
                    leftP++
                }
            }
        } else {
            leftP++
            rightP++
        }
    }
    return ans.join("")
}

// 2nd solution
// here we do two passes the first pass to count how far every neutral domino is from right falling domino
// on second pass we count how far every neutral domino is from the closest left falling domino
// on the second pass we also compare the two values and react accordingly
var pushDominoes = function(dominoes) {
    let firstPassArray = Array(dominoes.length).fill()
    let ans = Array(dominoes.length)
    let sinceRight = 0
    for (let i = 0; i < dominoes.length; i++) {
        if (dominoes[i] === '.' && sinceRight) {
            firstPassArray[i] = sinceRight
            sinceRight ++
        } else if (dominoes[i] === 'R') {
            sinceRight = 1
        } else {
            sinceRight = 0
        }
    }
    let sinceLeft = 0
    for (let j = dominoes.length - 1; 0 <= j; j--) {
        if (!firstPassArray[j]) {
            if (dominoes[j] === '.' && sinceLeft) {
                ans[j] = 'L'
            } else {
                ans[j] = dominoes[j]   
            }
        } else {
            if (!sinceLeft || firstPassArray[j] < sinceLeft) {
                ans[j] = 'R'
            } else if (sinceLeft === firstPassArray[j]) {
                ans[j] = '.'
            } else {
                ans[j] = 'L'
            }    
        }
        if (dominoes[j] === 'L') {
            sinceLeft = 1 
        } else if (dominoes[j] === 'R') {
            sinceLeft = 0
        } else if (sinceLeft) {
            sinceLeft++
        }
    }
    return ans.join("")
};


// 206. reverse linked list
// recursive
var reverseList = function(head, prevNode = null) {
    if (head === null) return prevNode
    let node = new ListNode(head.val, prevNode)

    return reverseList(head.next, node)
    
};

// iterative
var reverseList = function(head) {
    let prevNode = null
    while (head) {
        let node = new ListNode(head.val, prevNode)
        prevNode = node
        head = head.next
    }
    return prevNode
};


// 21. merge two sorted lists
var mergeTwoLists = function(list1, list2) {
    let mergedList = new ListNode(),
        currNode = mergedList
    
    while (list1 && list2) {
        if (list1.val < list2.val) {
            currNode.next = list1
            list1 = list1.next
        } else {
            currNode.next = list2
            list2 = list2.next
        }
        currNode = currNode.next
    }
    
    currNode.next = list1 || list2
    return mergedList.next
};

// 143. reorder list
// the proper steps are 
// > find middle of list 
// > reverse right side
// > merge left side with reversed right side
// here I use an array to track every part of the linked lists
// then traveling the array backwards gives up our backwards linked list
var reorderList = function(head) {
    let currNode = head
    let backwards = []
    
    while (currNode) {
        backwards.push(currNode)
        currNode = currNode.next
    }
    let j = backwards.length - 1,
        i = 0,
        node1,
        node2
    
    while (i < j) {
        node1 = backwards[i]
        node2 = backwards[j]
        node2.next = node1.next
        node1.next = node2
        i++
        j--
    }
    backwards[i].next = null
};

// 19. remove nth node from end of list
// let fast be n ahead of slow 
// then prog fast and slow at same speed
// once fast reaches end slow will be at nth node from the end which will be where we chop 
var removeNthFromEnd = function(head, n) {
    let smartNode = new ListNode(null, head),
        slow = smartNode,
        fast = smartNode
    
    for (let i = 0; i <= n; i++){
        fast = fast.next
    }
    
    while (fast) {
        fast = fast.next
        slow = slow.next
    }
    slow.next = slow.next.next
    
    return smartNode.next
};

// 138. copy list with random pointer 
// store pointer to node inside old node so if we loop back we can assign our value 

var copyRandomList = function(head) {
    let currOldNode = head,
        smartHead = new Node(),
        currNewNode = smartHead,
        index = 0
    
    while (currOldNode) {
        let node = currOldNode.pointer || new Node(currOldNode.val)
        currOldNode.pointer = node
        if (currOldNode.random) {
            node.random = currOldNode.random.pointer || new Node(currOldNode.random.val)
            currOldNode.random.pointer = node.random
        }
        currNewNode.next = node
        currNewNode = node
        currOldNode = currOldNode.next
        index++
    }
    return smartHead.next
};


// 2. add two numbers

var addTwoNumbers = function(l1, l2) {
    let carry = 0,
        smartHead = new ListNode(),
        currNode = smartHead
    
    while (l1 || l2) {
        const value = (l1? l1.val:0) + (l2? l2.val:0) + carry
        carry = (10<= value)? 1:0
        currNode.next = new ListNode(value%10)
        currNode = currNode.next
        l1 = l1?.next 
        l2 = l2?.next
    }
    if (carry) {
        currNode.next = new ListNode(1)
    }
    return smartHead.next
};

// 141. linked list cycle
// with o(n) extra memory
var hasCycle = function(head) {
    if (!head) return false
    
    if (head.visited === true) return true
    
    head.visited = true
    return hasCycle(head.next)
};
// with o(1) extra memory 
var hasCycle = function(head) {
    let smartNode = new ListNode(),
        fast = smartNode,
        slow = smartNode
    smartNode.next = head
    fast = fast?.next?.next

    while (fast && fast !== slow){
        fast = fast.next?.next
        slow = slow.next
    }
    return (fast)? true: false
};


// 287. find duplicate number
var findDuplicate = function(nums) {
    let lambda = 1,
        fast = 0,
        slow = nums[0],
        twoPow = 1
    
    while (fast !== slow) {
        if (lambda === twoPow) {
            slow = fast
            twoPow *= 2
            lambda = 0
        }
        fast = nums[fast]
        lambda++
    }
    fast = 0
    slow = 0
    
    while (0 < lambda) {
        fast = nums[fast]
        lambda--
    }
    
    while (fast !== slow) {
        fast = nums[fast]
        slow = nums[slow]
    }
    return fast
};

var findDuplicate = function(nums) {
    let slow = nums[0],
        fast = nums[nums[0]]
  
    while (fast !== slow) {
        slow = nums[slow]
        fast = nums[nums[fast]]
    }
    slow = 0
    
    while (slow !== fast) {
        slow = nums[slow]
        fast = nums[fast]
    }
    return fast
};

// 92. reverse linked list II
// using similar logic to reverse linked list for o(1) memory usage
// except we must find the correct starting nodes
var reverseBetween = function(head, left, right) {
    let count = 1,
        currNode = head,
        start = head
    
    while (count < left) {
        start = currNode
        currNode = currNode.next
        count++
    }
    
    let tail = currNode,
        prev = null
    while (count <= right) {
        let node = currNode
        currNode = currNode.next
        node.next = prev
        prev = node
        count++
    }
    start.next = prev
    tail.next = currNode
    return (left === 1)? prev: head
};

// 25. reverse nodes in k-group 
// use function from 92. reverse linked list II
var reverseKGroup = function(head, k) {
    let startNode = head,
        scoutNode = head,
        count = 1
    for (let i = 0; i<k-1; i++){
        scoutNode = scoutNode.next
    }
    while (scoutNode) {
        scoutNode = scoutNode.next
        head = reverseBetween(head,count, count+k-1) // this function from reverse linked list II 
        count = count+k
        for (let i = 0; i<k-1; i++){
            scoutNode = scoutNode?.next || null
        }
    }
    return head
};


// 572. subtree of another tree

var isSubtree = function(root, subRoot) {
    if (!subRoot) return true
    if (!root) return false
    if (isSameIsh(root, subRoot)) {
        return true
    } else {
        let left = isSubtree(root.left, subRoot),
            right = isSubtree(root.right, subRoot)
        return left || right
    }
};

var isSameIsh = function(p, q) {
    if (!q && !p) return true
    else if (!q || !p || p.val !== q.val) return false
    
    return isSameIsh(p.left, q.left) && isSameIsh(p.right, q.right)
};




// 235. lowest common ancestor of a binary search tree
// since its a binary search tree either we arrive at one of the target nodes or we arrive at a node between the targets
var lowestCommonAncestor = function(root, p, q) {
    if ((root.val <= p.val && q.val <= root.val) || (root.val <= q.val && p.val <= root.val)) {
        return root
    } else {
        return (root.val < q.val)? lowestCommonAncestor(root.right, p, q): lowestCommonAncestor(root.left, p, q)
    }
};


// 1448. count good nodes in binary tree
// travel down every path keeping track of max

var goodNodes = function(root) {
    let res = 0
    
    const travel = (max, root) => {
        if (!root) return
        if (max <= root.val) {
            res++
        }
        travel(Math.max(root.val, max), root.right)
        travel(Math.max(root.val, max), root.left)

    }
    
    travel(root.val, root)
    return res
};



// 218. the skyline problem
// using priority queue to keep track of all the buildings currently in our range ordered by height
// when we reach a new building we remove every building in our queue that does not overlap with our new building up to a new local max
// ^ here we are careful when we expunge buildings from our list we check if the last expunged building failed to overlap with the new local max
// if this happens we add our critical point and continue removing elements while pushing our right index forward
// when we finish removing buildings from our list we look at our current right index
// if it does not equal our current buildings index then we must add a 0 because there is a gap between buildings
// last we compare our current building to our local max
var getSkyline = function(buildings) {
    let res = [],
        currBuildings = new MaxPriorityQueue({priority: (a) => a[2]})
    buildings.push([Infinity, Infinity, 0])
    for (let i = 0; i<buildings.length;i++) {
        
        const building = buildings[i]
        let farthestRight = currBuildings.front()?.element[1],
            currHeight
        while (currBuildings.front()?.element[1] < building[0]) {
            const [left, right, height] = currBuildings.front().element
             if (farthestRight < right){
                 if (height !== currHeight){
                     if (building[2] < height || right < building[0]) {
                        res.push([farthestRight, height])
                     }
                 }
                 farthestRight = right
                 currHeight = height
             }
            if (!currHeight) { currHeight = true }
            currBuildings.dequeue()
        }
        
        if (res.length) {
            if (res.at(-1)[0] === building[0]) {
            if (res.at(-1)[1] < building[2]) {
                res.pop()
            }
            } 
        }
        
        if (farthestRight < building[0] && currHeight) {
            if (currBuildings.front()?.element[0] <= farthestRight) {
                res.push([farthestRight, currBuildings.front().element[2]])
            } else {
                res.push([farthestRight, 0])  
            }
        }
        
        
        if (currBuildings.front()) {
            if (currBuildings.front().element[2] < building[2]) {
                res.push([building[0], building[2]])
            }
            
        } else {
            res.push([building[0], building[2]])
        }
        currBuildings.enqueue(building)
    }
    res.pop()
    return res
};


// 124. binary tree maximum path sum 


var maxPathSum = function(root) {
    if (!root) return 0
    
    
    const travel = (node) => {
        if (!node) return [-Infinity, -Infinity]
        let left = travel(node.left),
            right = travel(node.right),
            maxLen = Math.max(node.val, left[0] + node.val, right[0] + node.val),
            maxPath = Math.max(left[1], right[1], maxLen, left[0] + node.val + right[0])
        return [maxLen, maxPath]
    }
    return travel(root)[1]
    
};


// 658. find k closest elements

var findClosestElements = function(arr, k, x) {
    let lp = 0,
        rp = arr.length - k, 
        mid
    
    while (lp < rp) {
        mid = lp + Math.floor((rp - lp) / 2)
        
        if (arr[mid+k] - x < x- arr[mid]) {
            lp = mid+1
        } else{
            rp = mid
        }
    }
    return arr.slice(lp, lp + k);
} 



// 211. design add and search words data structure
// trie data structure 
// handle path correctly
var WordDictionary = function() {
    this.trie = {}
};
WordDictionary.prototype.addWord = function(word) {
    let node = this.trie 
    
    for (const letter of word) {
        if (!node[letter]) {
            node[letter] = {}
        }
        
        node = node[letter]
    }
    node.last = true
};

WordDictionary.prototype.search = function(word) {
    const travel = (idx, store) => {
        const letter = word[idx]
        idx++
    
        
        if (letter === '.') {
            for (const paths of Object.values(store)) {
                if (idx === word.length) {
                    if (paths.last) return true
                } else {
                    if (travel(idx, paths)) return true
                }
                
            }
        }
        
        
        if (!store[letter]) return false 
        
        if (idx === word.length) {
            return store[letter].last? true: false
        } else {
            return travel(idx, store[letter])
        }
    }
    return travel(0, this.trie)
};

// 212. word search II 
// build trie
// travel from every node, adding word if we reach a node.last
// stop if we backtrack or if our trie does not contain any words with the current letter combination
 var findWords = function(board, words) {
    let trie = {},
        n = board[0].length,
        m = board.length,
        res
    
    for (const word of words) {
        let node = trie
        for (const letter of word) {
            if (!node[letter]) {
                node[letter] = {}
            }
            node = node[letter]
        }
        node.last = word
    }
    
    
    const travel = (i, j, store) => {
        if (!store || board[i][j] === '#' || !store[board[i][j]]) return
        
        const letter = board[i][j]
        store = store[letter]
        if (store?.last) {
            res.push(store.last)
            store.last = false  
        }

        board[i][j] = '#'
        if (i+1 < m ) {
            travel(i+1, j, store)
        }
        if (0 < i ) {
            travel(i-1, j, store)
        }
        if (j+1 < n ) {
            travel(i, j+1, store)
        }
        if (0 < j ) {
            travel(i, j-1, store)
        }
        board[i][j] = letter
        
    }
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            travel(i, j, trie)
        }
    }
    return [...res]
};


// 703. kth largest element in a stream
// use priority queue, always keeps k elements front element will be the smallest
var KthLargest = function(k, nums) {
    this.largerK = new MinPriorityQueue({priority: (a) => a})
    
    for (const num of nums) {
        if (this.largerK.size() < k) {
            this.largerK.enqueue(num)
        } else {
            this.add(num)
        }
    }
    while (this.largerK.size() < k) {
        this.largerK.enqueue(- Infinity)
    }
};

KthLargest.prototype.add = function(val) {
    if (this.largerK.front().element < val) {
        this.largerK.enqueue(val)
        this.largerK.dequeue()
    }
    
    
    return this.largerK.front().element
};


// 91. decode ways
var numDecodings = function(s) {
    if (!s.length) return 0
    let dp = Array(s.length + 1).fill(0)
    dp[s.length] = 1
    
    for (let i = s.length - 1; 0 <= i; i--) {
        if (s[i] !== "0") {
            dp[i] = dp[i+1]
            if (i < s.length-1 && (s[i] === '1' || (s[i] === '2' && s[i+1] < 7))) {
                dp[i] += dp[i+2] 
            }
        }
    }
    return dp[0]
}
 


// 1046. last stone weight

var lastStoneWeight = function(stones) {
    let heap = new MaxPriorityQueue({priority: (a) => a})
    
    for (const stone of stones) {
        heap.enqueue(stone)
    }
    
    while (1 < heap.size()) {
        const y = heap.dequeue().element,
              x = heap.dequeue().element
        
        if (y-x) {
            heap.enqueue(y-x)
        }
    }
    
    return (heap.size())? heap.front().element: 0
};


// 215. kth largest element in an array
// use pivot sort to find an alement such that after pivot sort the left side is nums.length - k 
// meaning our pivot element is the kth largest elemetn

var findKthLargest = function(nums, k) {
    const swap = (i, j) => {
        [nums[i], nums[j]] = [nums[j], nums[i]]
    }
    const partition = (lo, hi) => {
        const pivot = nums[hi]
        let i = lo,
            j = hi - 1
        while (i <= j) {
            while (nums[i] < pivot) {
                i++
            }
            while (pivot < nums[j]) {
                j--
            }
            if (i <= j) {
                swap(i,j)
                i++
                j--
            }
        }
        swap(i, hi) 
        return i
    }
    let lo = 0,
        hi = nums.length - 1,
        kth
    while (kth !== nums.length - k) {
        const j = partition(lo, hi)
        if (j < nums.length - k) {
            lo = j + 1
        } else {
            hi = j - 1
        }
        kth = j
    }
    return nums[kth] 
};


// 355. design twitter
// 
class TweetNode{
    constructor(userId, tweetId){
        this.id = userId
        this.tweetId = tweetId
        this.next = null
        this.prev = null
    }
}

class Twitter {
    constructor() {
        this.users = {};
        this.tweets = new TweetNode('sentinel', 'sentinel');
    }
    postTweet(userId, tweetId) {
        if (!this.users[userId]) {
            this.users[userId] = { follows: {} };
            this.users[userId].follows[userId] = true;
        }

        let node = new TweetNode(userId, tweetId);
        node.next = this.tweets;
        this.tweets.prev = node;
        this.tweets = node;
    }
    getNewsFeed (userId) {
        if (!this.users[userId]) return [];
        let allowedIds = this.users[userId].follows;
        let feed = [], currTweet = this.tweets;
         while (feed.length < 10 && currTweet.id !== 'sentinel') {
            if (allowedIds[currTweet.id]) {
               feed.push(currTweet.tweetId);
            }
            currTweet = currTweet.next;
            }
        return feed;
    }
    follow (followerId, followeeId) {
        if (!this.users[followerId]) {
            this.users[followerId] = { follows: {} };
            this.users[followerId].follows[followerId] = true;
        }

        this.users[followerId].follows[followeeId] = true;
    };
    unfollow (followerId, followeeId) {
        if (!this.users[followerId]) {
            this.users[followerId] = { follows: {} };
            this.users[followerId].follows[followerId] = true;
        }
        delete this.users[followerId].follows[followeeId];
    };
   
}

// 1155. number of dice rolls with target sum
// dp[i] will be the sum after the (n-1+i)th dice has been rolled 
// dp[n+1] is starting point
// dp[1] will be the sum of all the dice rolled, which based on our tracking will only have one sum 
// or no sums if it is invalid such as too many dice and not a high enough target (10 dice, target of 9)


// this problem does not need right shifting and probably makes more sense without it 
// ie: dp[n] = starting point
// dp [0] = end point
// dp[i] = sum of n-i dice


// on each new dice roll we bound our index to include only valid sums
// our min will be either 1 or whatever is needed to complete a valid sum
// consider dices = 1 target = 3
// the min would be 3 
// the max our dice can roll depends on how many dice are left to roll, 
// we can assume all these dice left roll 1
// resulting in max being k or target - sum - (#dicesLeft - 1) * 

var numRollsToTarget = function(n, k, target) {
    let dp = Array(n+1).fill([])
    const mod = 10e8 +7
    dp[n+1] = [[0,1]]
    //sum, count
    
    for (let j = n; 0 < j; j--) {
        let sums = dp[j+1],
            newSums = new Map()
        for (const sum of sums) {
            const min = Math.max(1, target - sum[0] - (j-1)*k)
            const max = Math.min(target - sum[0] - (j-1), k)
            let i = min
            while (i <= max) {
                if (!newSums.has(sum[0]+i)) {
                    newSums.set(sum[0]+i, 0)
                }
                newSums.set(sum[0]+i, (newSums.get(sum[0]+i) + sum[1])%mod)
                i++
            }
        }
        dp[j] = Array.from(newSums)
    }

    // check if a sum was possible 
    return dp[1][0]? dp[1][0][1]: 0
};


// 90. subsets II

var subsetsWithDup = function(nums) {
    let powerSet = new Set()
    
    nums = nums.sort((a,b) => a-b)
    powerSet.add('[]')
    for (let j = 0; j < nums.length; j++) {
    
        let prevSets = [...powerSet.values()].map(a => JSON.parse(a))
       
        
        for (let prevSet of prevSets) {
            const newSet = [...prevSet, nums[j]]
            powerSet.add(JSON.stringify(newSet))
        }
    }
    
    
    return [...powerSet.values()].map(a => JSON.parse(a))
};

// 40. combination sum II
// use count 
var combinationSum2 = function(candidates, target) {
    let candidateCount = {}
    
    for (let candidate of candidates) {
        if (!candidateCount[candidate]){
            candidateCount[candidate] = 0
        }
        candidateCount[candidate]++
    }
    let dp = []
    dp[0] = []
    dp[0].push([0, []])
    candidates.sort((a,b) => a-b)
    let res = []
    for (let i = 0; i < candidates.length; i++) {
        let num = candidates[i]
        if (target < num || (0 < i && num === candidates[i-1])) {
            dp[i+1] = dp[i]
            continue
        }
        
        let paths = [...dp[i]];
        dp[i+1] = []
        
        let options = [[num, [num]]],
            sum = num
        for (let i = 2; i <= candidateCount[num]; i++) {
            sum += num
            options.push([sum, [...options.at(-1)[1], num]])
        }

        for (const path of paths) {
            dp[i+1].push([...path])
            for (const option of options) {
                
                const newSum = option[0] + path[0],
                      newArray = [...path[1], ...option[1]]
                if (newSum === target) {
                    res.push(newArray)
                } else if (newSum < target) {
                    dp[i+1].push([newSum, newArray])
                }
            }
        }
    }
    return res
};


// 1578. minimum time to make rope colorful
// on a chain of same color ballons we will pop all but the one with the largest time
// compare two ballons pop one, keep other
var minCost = function(colors, neededTime) {
    let currColor = '#',
        maxTime = 0,
        sum = 0
    
    for (let i = 0; i < colors.length; i++) {
        const color = colors[i]
        
        if (color === currColor) {
            sum += Math.min(neededTime[i], maxTime)
            maxTime = Math.max(neededTime[i], maxTime)
        } else {
            currColor = color
            maxTime = neededTime[i]
        }
    }
    return sum
    
};

// 51. n-queens
// imagine building the board one row at a time we must place at some point
// base case is row = 0 
// start placing queen at all the potential spots then go from there

var solveNQueens = function(n) {
    let res = []
    let row = Array(n).fill('.')
   const dp = (board , k) => {
       if (k === n) {
           res.push(board.map((queenP) => {
               const arr = [...row]
               arr[queenP[1]] = 'Q'
               return arr.join("")
           }))
           return
       }
       let i = k
       for (let j = 0; j < n; j++) {
           const queenPosition = [i, j]
           if (checkIfValid(board,i,j,n)) {
               board.push(queenPosition)
              
               dp(board, k+1)
               board.pop()
           }
       }
   }
   dp([],0)
    return res
};

// 695. max area of island
// same logic as count islands but we keep track of area
var maxAreaOfIsland = function(grid) {
    let maxArea = 0
    
    for (let i = 0; i < grid.length; i++) {
        for (let j = 0; j < grid[0].length; j++) {
            if (grid[i][j] === 1) {
                let max = moveAdj(grid, i, j)
                if (maxArea < max) {
                    maxArea = max
                }
            }
        }
    }
    return maxArea
};

var moveAdj = (grid, i,j) => {
    grid[i][j] = 0
    let area = 1
    if (i+1 < grid.length) {
          if (grid[i+1][j] === 1 ) {
              area += moveAdj(grid, i+1,j)
          }  
    }
    if (0 <= i-1) {
        if (grid[i-1][j] === 1) {
            area += moveAdj(grid, i-1, j)
        }
    }
    if (j+1 < grid[0].length) {
        if (grid[i][j+1] === 1) {
            area += moveAdj(grid, i, j+1)
        }
    }
    if (0 <= j-1) {
        if (grid[i][j-1] === 1) {
            area += moveAdj(grid, i, j-1)
        }
    }
    return area
}


// 417. pacific atlantic water flow
// flow upwards from both oceans 
// one ocean at a time
// at each node we mark it in our sets
// at the end we compare our set keys
var pacificAtlantic = function(heights) {
    let res = []
    let pacificAccess = new Set()
    let atlanticAccess = new Set()
    let dir = [[1,0], [-1,0], [0,1], [0,-1]]
    
    
    let pacificFlow = new Queue(),
        atlanticFlow = new Queue()
    
    for (let i = 0; i < heights.length; i++) {
        pacificFlow.enqueue([i,0])
        atlanticFlow.enqueue([i, heights[0].length - 1])
    }
    for (let j = 0; j < heights[0].length; j++) {
        pacificFlow.enqueue([0,j])
        atlanticFlow.enqueue([heights.length -1,j])
    }
    while (pacificFlow.size()) {
        let node = pacificFlow.dequeue()
        let height = heights[node[0]][node[1]]
        pacificAccess.add(JSON.stringify(node))
        
        for (let adj of dir) {
            const [p,q] = [node[0] + adj[0], node[1] + adj[1]]
            if (0 <= p && 0 <= q && p < heights.length && q < heights[0].length) {
                if (height <= heights[p][q]) {
                    if (!pacificAccess.has(JSON.stringify([p,q]))) {
                        pacificFlow.enqueue([p,q])
                    }
                }
            }
        }
        
        
    }
    while (atlanticFlow.size()) {
        let node = atlanticFlow.dequeue(),
            height = heights[node[0]][node[1]]
        atlanticAccess.add(JSON.stringify(node))
        for (let adj of dir) {
            const [p,q] = [node[0] + adj[0], node[1] + adj[1]]
            if (0 <= p && 0 <= q && p < heights.length && q < heights[0].length) {
                if (height <= heights[p][q]) {
                    if (!atlanticAccess.has(JSON.stringify([p,q]))) {
                        atlanticFlow.enqueue([p,q])
                    }
                }
            }
        }
        
    }
    for (const key of pacificAccess.keys()) {
        if (atlanticAccess.has(key)) {
            res.push(JSON.parse(key))
        }
    }
    return res
};

// 112. path sum

var hasPathSum = function(root, targetSum) {
    if (!root) return false
    if (root.val === targetSum && !root.left && !root.right) return true
    
    if (hasPathSum(root.left, targetSum - root.val)) return true
    if (hasPathSum(root.right, targetSum - root.val)) return true
    return false
};


// 210. course schedule II

var findOrder = function(numCourses, prerequisites) {
    if (!prerequisites.length){
        let arr = []
        for (let i = 0; i < numCourses; i++) {
            arr.push(i)
        }
        return arr
    } 
    let dependencies = {}
    let numDep = {}
    for (let i = 0; i < numCourses;i++) {
        dependencies[i] = []
        numDep[i] = 0
    }
    for (const [a, b] of prerequisites) {
        if (!dependencies[a]) {
            dependencies[a] = []
        }
        if (!dependencies[b]) {
            dependencies[b] = []
            
        }
        dependencies[b].push(a)
        numDep[a] ++
    }
    let stack = []
    for (const [key, value] of Object.entries(dependencies)) {
        if (!numDep[key]) {
            stack.push(key)
            delete numDep[key]
        }
    }
    let count = 0
    let order = []
    while (stack.length) {
        let course = stack.pop()
        order.push(course)
        count++
        for (const depend of dependencies[course]) {
            numDep[depend]--
            if (numDep[depend] <= 0) {
                stack.push(depend)
                delete numDep[depend]
            } 
        }
    }
    if (Object.keys(numDep).length) return []
    return (count <= numCourses)? order: [] 
};

// 322. reconstruct itinerary
// use map to store destinations from each airport and how many times we can travel there
// sort the destinations so our first succesful path is the one lexigraphically the smallest
// then it just becomes backtracking problem
// on each iteration we add to path and remove a ticket from the map
// if its succesful then we can return the path
// if not we remove from the path and add back our ticket

var findItinerary = function(tickets) {
    let outGoing = {}
  
    for (const ticket of tickets) {
        if (!outGoing[ticket[0]]) {
            outGoing[ticket[0]] = new Map()
        }
        outGoing[ticket[0]].set(ticket[1], (outGoing[ticket[0]].get(ticket[1]) || 0) + 1)
    }
    

    for (let keys of Object.keys(outGoing)) {
        let departures = outGoing[keys]
        
        outGoing[keys] = new Map([...departures].sort())
    }
    
    
    const travel = (arrivalAirport, order) => {
        
        if (order.length === tickets.length + 1) return order
        
        let destinationsMap = outGoing[arrivalAirport]
        if (!destinationsMap) return false
        for (let destination of destinationsMap.keys()) {
            if (destinationsMap.get(destination)) {
                let count = destinationsMap.get(destination)
                destinationsMap.set(destination, count-1)
                order.push(destination)
                let path = travel(destination, order)
                if (path) return path
                order.pop()
                destinationsMap.set(destination, count)
            }
        }
        return false
    }
    return travel("JFK", ["JFK"])
};


// 1584. min cost to connect all points

var minCostConnectPoints = function(points) {
    let connections = 0
    let visited = Array(points.length).fill(1000000000000)
    let i = 0
    let sum = 0
    
    while (connections < points.length-1) {
        visited[i] = Infinity
        let min_j = i
        for (let j = 0; j < points.length; j++) {
            if (visited[j] !== Infinity) {
                let dist = distance(points[j], points[i])
                if (dist < visited[j]) {
                    visited[j] = dist
                }
                if (visited[j] < visited[min_j]) {
                    min_j = j
                }
            }
        }
        i = min_j
        sum += visited[min_j]
        connections++
    }
    return sum
};

var distance = (a,b) => {
    return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1])
}

// 743. network delay time
// travel along lowest weight path
// once we reach all nodes we have successful time 
var networkDelayTime = function(times, n, k) {
    let paths = {}
    
    for (const [source,target,time] of times){
        if (!paths[source]) {
            paths[source] = []
        }
        paths[source].push([target, time])
    }
    
    let minPath = new MinPriorityQueue({priority: (a) => a[1]})
    let time = 0
    let visited = {}
    let nodesVisited = 0
    minPath.enqueue([k, 0])
    while (minPath.size() && nodesVisited<n){
        let [node, nodeTime] = minPath.dequeue().element
        if (visited[node]) continue
        
        visited[node] = true
        if (time < nodeTime) {
            time = nodeTime
        }
        nodesVisited++
        const avaliableNodes = paths[node]
        if (!avaliableNodes) continue
        for (const path of avaliableNodes) {
            if (visited[path[0]]) continue
            minPath.enqueue([path[0], path[1] + nodeTime])
        }
    }
    return (nodesVisited < n || time === 0)? -1: time
};

// 778. swin in rising water
// pretty trivial with min heap
// just keep going to the smallest element and checking if its our i,j
// since we are always taking the smallest possible element 
// we record the max height weve had to travel to

var swimInWater = function(grid) {
    let currMax = 0
    let minHeap = new MinPriorityQueue({priority: (a) => a[2]})
    let dir = [[1,0], [-1,0], [0,1], [0,-1]]
    minHeap.enqueue([0,0, grid[0][0]])
    while (minHeap.size()) {
        const [i,j, time] = minHeap.dequeue().element
        if (currMax < time) {
            currMax = time
        }
        if (i === grid.length -1 && j === grid.length -1 ){
            break;
        }
        grid[i][j] = Infinity
        for (const adj of dir) {
            const [p,q] = [adj[0] + i, adj[1] + j]
            if (0 <= q && 0 <= p && p < grid.length && q < grid.length) {
                if (grid[p][q] !== Infinity) {
                    minHeap.enqueue([p,q, grid[p][q]])
                }  
            }
        }
    }
    return currMax
};

// 623. add one row to tree

var addOneRow = function(root, val, depth) {
    if (depth === 1) {
        let node = new TreeNode(val, root)
        return node
    }
    depth--
    depth--
    const helper = (node, val, currDep) => {
        if (!node) { return }
        if (currDep === 0) {
            let left = new TreeNode(val, node.left)
            let right = new TreeNode(val, null, node.right)
            node.left = left
            node.right = right
            return
        } else {
            currDep--
            helper(node.left, val, currDep)
            helper(node.right, val, currDep)
        }
    }
    helper(root,val,depth)
    return root
};

// 787 cheapest flights within k stops
// use heap to travel along min path
// mark traveled airports with how many flights it took to get there
// if we arrive at the same airport with a lower flight count 
var findCheapestPrice = function(n, flights, src, dst, k) {
    let connections = {}
    for (const [from, to, price] of flights) {
        if (!connections[from]) {
            connections[from] = new Map()
        }
        connections[from].set(to, price)
    }
    let visited = {}
    // heap element [current, numFlights, price]
    let minHeap = new MinPriorityQueue({priority: a => a[2]})
    minHeap.enqueue([src, 0, 0])
    while(minHeap.size()) {
        let [currAirport, numFlights, currPrice] = minHeap.dequeue().element
        if (currAirport === dst && numFlights <= k+1) {
            return currPrice
        } else if (k + 1 < numFlights) {
            continue
        }
        if (visited[currAirport] < numFlights) continue
        visited[currAirport] = numFlights
        let destinations = connections[currAirport]?.keys()
        if (!destinations) continue 
        for (const destination of destinations) {
            const price = currPrice + connections[currAirport].get(destination)
            minHeap.enqueue([destination, numFlights+1, price])
        }
    }
    return -1
};



// 70. climbing stairs
// using memoization to prevent having to climb 
// multiple times on high n
var climbStairs = function(n) {
    let dp = []
    const helpClimb = (k) => {
        if (k <= 3) return k
        if (dp[k]) return dp[k]
        
        let oneStep = helpClimb(k-1)
        let twoStep = helpClimb(k-2)
        
        dp[k] = oneStep + twoStep
        return dp[k]
    }
    
    return helpClimb(n)
};

// 746. min cost climbing stairs
var minCostClimbingStairs = function(cost) {
    let dp = Array(cost.length)
    dp[cost.length] = 0
    
    for (let j = cost.length -1; 0 <= j; j--) {
        dp[j] = cost[j] + Math.min(dp[j+1], dp[j+2] || 0)
    }
    return Math.min(dp[0], dp[1])
};

// 300. longest increasing subsequence 

var lengthOfLIS = function(nums) {
    let monoStack = []
    let max = 0 
    
    for (const num of nums) {
        
        if (monoStack.length === 0 || monoStack.at(-1) < num) {
            monoStack.push(num)
        } else {
            let replace = bst(monoStack, num)
            monoStack[replace] = num
        }
        if (max < monoStack.length) {
            max = monoStack.length
        }
    }
    return max
};
var bst = (array, target) => {
    let min = 0
    let max = array.length
    let mid
    
    while (min <= max) {
        mid = min + Math.floor((max - min)/2)
        
        if (array[mid] === target) {
            return mid
        }
        
        if (array[mid] < target ) {
            min = mid + 1
        } else {
            max = mid - 1
        }
    }
    return (target <= array[min])? min: min+1
}



// 732. my calender III
// object in javascrip is a treemap aslong as keys => 0
// we iterate between the keys each key will be the total of how many start - how many end on this key
// so iterate along all keys and record max value

var MyCalendarThree = function() {
    this.tm = {}
    this.max = 0
};

MyCalendarThree.prototype.book = function(start, end) {
    this.tm[start] = (this.tm[start] || 0 ) + 1
    this.tm[end] = (this.tm[end] || 0 ) - 1
    
    let k = 0
    for (const key in this.tm) {
        k += this.tm[key]
        if (this.max < k) {
            this.max = k
        }
    }
    return this.max
    
};


// 309. best time to buy and sell stock with cooldown
// there are only 3 states buy sell cooldown
// buy numbers inserted as negatives so look for min value 

var maxProfit = function(prices) {
    let buyState = -prices[0]
    let sellState = -Infinity
    let restState =  0
    
    for (let j = 0; j < prices.length; j++) {
        
        let newBuy = Math.max(buyState, restState - prices[j])
        let newRest = Math.max(restState, sellState)
        let newSell = buyState + prices[j]
        buyState = newBuy
        restState = newRest
        sellState = newSell
    }
     return Math.max(sellState, restState)
};


// 518. coin change II
var change = function(amount, coins) {
    let dp = Array(amount+1).fill(0)
    dp[0] = 1 
    
    for (const coin of coins) {
        
        for (let j = coin; j <= amount; j++) {
            dp[j] = (dp[j] || 0 ) + dp[j-coin] 
        }
    }
    return dp[amount]
};

// 494. target sum
// use dp so we dont repeat tracks
var findTargetSumWays = function(nums, target) {
    let dp = {}
   
    
    let res = 0
    
    const calculate = (index, sum) => {
        
        if (dp[JSON.stringify([index,sum])] !== undefined) return dp[JSON.stringify([index,sum])]
        if (index === nums.length) return sum === target? 1 : 0
        let min = calculate(index+1, sum-nums[index])
        let max = calculate(index+1, sum+nums[index])
        
        dp[JSON.stringify([index,sum])] = max+min
        return dp[JSON.stringify([index,sum])]
    }
    
    return calculate(0,0)
};

// 16. 3sum closest

var threeSumClosest = function(nums, target) {
    let closest = -Infinity
    nums = nums.sort((a,b) => a-b)
    for (let i = 0; i < nums.length - 2 ; i++) {
        let j = i+1
        let k = nums.length - 1
        
        while ( j < k) {
            let sum = nums[i] + nums[j] + nums[k]
            if (Math.abs(target - sum) < Math.abs(target - closest)) {
                closest = sum
            }
            
            if (closest === target) return target
            
            if (sum < target) {
                j++
            } else {
                k--
            }
        }
    }
     return closest
 };

// 97. interleaving string 
// dp[i][j] currently interleaved s1 up to i-1 and s2 up to j-1
 var isInterleave = function(s1, s2, s3) {
    let dp = Array(s1.length+1).fill().map(() => Array(s2.length+1))
    if (s1.length + s2.length !== s3.length) return false
    for (let i = 0; i <= s1.length; i++) {
        for (let j = 0; j <= s2.length; j++) {
            if (i === 0 && j === 0) {
                dp[i][j] = true
            }
            else if (i === 0) {
                dp[i][j] = dp[i][j-1] && s3[j-1] === s2[j-1]
            }
            else if (j === 0) {
                dp[i][j] = dp[i-1][j] && s3[i-1] === s1[i-1]
            }
            else {
                dp[i][j] = (dp[i-1][j] && s3[i+j-1] === s1[i-1]) || (dp[i][j-1] && s3[i+j-1] === s2[j-1])
            }
        }
    }
     return dp[s1.length][s2.length]
 };


// 329. longest increasing path in a matrix
// use memoization to prevent arriving at a node recalculating its max path
 var longestIncreasingPath = function(matrix) {
    let dp = Array(matrix.length).fill().map(() => Array(matrix[0].length).fill(-1))
    
    
    const travel = (i, j, min) => {
        if (matrix[i][j] <= min) return 0
        if (dp[i][j] !== -1) return dp[i][j]
        
        let longestPathFromPoint = 0
        
        for (const adj of [[1,0], [0,1], [-1,0], [0,-1]]) {
            const [p,q] = [adj[0] + i, adj[1] + j]
            if (0 <= p && 0 <= q && q < matrix[0].length && p < matrix.length) {
                let pathFromAdj = travel(p,q, matrix[i][j])
                if (longestPathFromPoint < pathFromAdj) {
                    longestPathFromPoint = pathFromAdj
                }
            }
        }
        dp[i][j] = longestPathFromPoint + 1
        return dp[i][j]
    }
    let max = 0
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[0].length; j++) {
            let currMaxPathFrom = travel(i,j, -Infinity)
            if (max < currMaxPathFrom) {
                max = currMaxPathFrom
            }
        }
    }
    return max
};


// 72. edit distance 

var minDistance = function(word1, word2) {
    let dp = Array(word2.length+1).fill().map(() => Array(word1.length+1))
    
    for (let i = 0; i <= word2.length; i++) {
        dp[i][0] = i 
    }
    for (let j = 0; j <= word1.length; j++) {
        dp[0][j] = j
    }
    
    
    for (let i = 1; i <= word2.length; i++ ) {
        for (let j = 1; j <= word1.length; j++) {
            if (word1[j-1] === word2[i-1]) {
                dp[i][j] = dp[i-1][j-1]
            } else {
                dp[i][j] = 1 + Math.min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])
            }
        }
    }
    return dp[word2.length][word1.length]
};


// 53. maximum subarray

var maxSubArray = function(nums) {
    let runningSum = 0
    let maxSum = nums[0]
    
    
    for (const num of nums) {
        runningSum = Math.max(runningSum + num, num)
        
        if (maxSum < runningSum) {
            maxSum = runningSum
        } 
    }
    return maxSum
};


// 55. jump game

var canJump = function(nums) {
    let maxInd = 0
    for (let i = 0; i < nums.length; i++) {
        if (maxInd < i) return false
        if (maxInd < nums[i] + i) {
            maxInd = nums[i] + i
        } 
    }
    return true
};

// 45. jump game II

var jump = function(nums) {
    let dp = Array(nums.length).fill(Infinity)
    dp[0] = 0
    
    for (let i = 0; i < nums.length; i++) {
        if (nums[i] !== 0) {
           for (let j = i+1; j <= i+nums[i]; j++) {
               dp[j] = Math.min(dp[j], dp[i]+1)
           } 
        }
        
    }
    return dp[nums.length-1]
};

// 134. gas station
// check if circuit is possible
// find index 
var canCompleteCircuit = function(gas, cost) {
    let gasTotal = gas.reduce((a,b) => a+b)
    let costTotal = cost.reduce((a,b) => a+b)
    if (gasTotal < costTotal) return -1
    
    let start = 0
    let tank = 0
    
    for (let i = 0; i < gas.length; i++) {
        
        tank += gas[i] - cost[i]
        
        if (tank < 0) {
            tank = 0
            start = i+1
        }
    }
    return start
};


// 846. hand of straights
var isNStraightHand = function(hand, groupSize) {
    if (hand.length%groupSize !== 0) return false
    let numFreq = {}
    for (let i = 0; i < hand.length; i++) {
        if (!numFreq[hand[i]]) {
            numFreq[hand[i]] = 0
        }
        numFreq[hand[i]] ++
    }
    let keys = Object.keys(numFreq)
    let i = 0
    while (i < keys.length) {
        for (let j = keys[i]; j < Number(keys[i])+groupSize; j++) {
            if (!numFreq[j]) return false
            numFreq[j]--
        }
        while (numFreq[keys[i]] === 0) {
            i++
        }
    }
    return true
}

// 1899. merge triplets to form target triplet
// since we use the max we will only choose triplets where all values are less than or equal to target triplet
var mergeTriplets = function(triplets, target) {
    let currTrip = [1,1,1]
    
    for (const trip of triplets) {
        if (trip[0] <= target[0] && trip[1] <= target[1] && trip[2] <= target[2]) {
            currTrip = [Math.max(trip[0], currTrip[0]), Math.max(trip[1], currTrip[1]), Math.max(trip[2], currTrip[2])]
        }
        if (currTrip[0] === target[0] && currTrip[1] === target[1] && currTrip[2] === target[2]) {
            return true
        }
    }
    
    return false
};

// 763. partition labels

var partitionLabels = function(s) {
    let letterFreq = {}
    for (let letter of s) {
        letterFreq[letter] = (letterFreq[letter] || 0 ) + 1
    }
    let lp = 0
    let rp = 0
    let partition = []
    while (rp < s.length) {
        let currWindow = []
        while (currWindow.length || lp === rp) {
            currWindow.push(s[rp])
            letterFreq[s[rp]]--
            while (letterFreq[currWindow.at(-1)] === 0 ) {
                currWindow.pop()
            }
            rp++
        }
        partition.push(rp-lp)
        lp = rp
    }
    return partition
};

// 678. valid parenthesis string

var checkValidString = function(s) {
    let minOpenCount = 0
    let maxOpenCount = 0
    for (const char of s) {
        if (char === '(') {
            minOpenCount++
            maxOpenCount++
        }
        if (char === ')') {
            maxOpenCount--
            if (0 < minOpenCount ) {
                minOpenCount--
            }
            if (maxOpenCount < 0) return false
        }
        if (char === '*') {
            if (0 < minOpenCount ) {
                minOpenCount--
            }
            maxOpenCount++
        }
    }
    return minOpenCount === 0 
};


// 653. two sum iv - input is a bst
// traditional two sum approact

var findTarget = function(root, k) {
    let sums = {}
    const travel = (node) => {
        if (!node) return
        if (sums[k-node.val]) return true
        sums[node.val] = true
        if (travel(node.left)) return true
        if (travel(node.right)) return true
        return false
    }
    return travel(root)
};

// 495. teemo attacking

var findPoisonedDuration = function(timeSeries, duration) {
    let timePoisoned = 0
    let expireTime = 0
    let startTime = 0
    
    for (const time of timeSeries) {
        if (time < expireTime) {
            timePoisoned -= (expireTime - time)
        } 
        timePoisoned += duration
        startTime = time
        expireTime = time + duration
    }
    
    return timePoisoned
};


// 435. non-overlapping intervals

var eraseOverlapIntervals = function(intervals) {
    let dp = Array(intervals.length + 1)
    dp[0] = 0
    let pastInt = [-Infinity, -Infinity]
    intervals = intervals.sort((a,b) => a[0] - b[0] === 0? a[1] - b[1]: a[0] - b[0])
    for (let j = 1; j <= intervals.length; j++) {
        if (pastInt[1] <= intervals[j-1][0]) {
            pastInt = intervals[j-1]
            dp[j] = dp[j-1]
        } else {
            dp[j] = dp[j-1] + 1
            pastInt[1] = Math.min(intervals[j-1][1], pastInt[1])
        }
    }
    return dp[intervals.length]
};


// 73. set matrix zeroes
// not constanst space 

var setZeroes = function(matrix) {
    const columns = {}
    const rows = {}
    
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[0].length; j++) {
            if (matrix[i][j] === 0) {
                columns[j] = true
                rows[i] = true
            }
        }
    }
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j <matrix[0].length; j++) {
            if (columns[j] || rows[i]) {
                matrix[i][j] = 0
            }
        }
    }
};


// 202. happy number

var isHappy = function(n) {
    let foundSum = {}
    
    const helpTravel = (sum) => {
        if (sum === 1 || 0) return true
        if (foundSum[sum]) return false
        else {foundSum[sum] = true}
        let str = sum.toString()
        let squaredSum = 0
        for (let i = 0; i < str.length; i++) {
            let num = str[i]
            squaredSum += parseInt(num) * parseInt(num)
        }
        return helpTravel(squaredSum)
    }
    return helpTravel(n)
};


// 66. plus one
// using bigInt to avoid int overflow
// would be possible to also iterate from the back of the array 
// keeping carry over
var plusOne = function(digits) {
    digits = BigInt(digits.join('')) + 1n
    return digits.toString().split('').map(Number)
};

// iterative solution
var plusOne = function(digits) {
    let carry = 1
    for (let i = digits.length - 1; 0 <= i; i--) {
        if (carry) {
            digits[i]++
            carry = 0
        }
        if (digits[i] < 10) {
            return digits
        } else {
            carry = 1
            digits[i] = 0
        }
    }
    digits.unshift(1)
    return digits
};


// 50. pow(x,n)
// iterative solution 
var myPow = function(x, n) {
    if (n === 0 || x === 1 ) return 1
    if (x === -1) return (n%2)? -1: 1
    if (n < 0) return myPow(1/x, -n)
    let pow = 1
    while (0 < n) {
        if (pow === 0) return 0
        pow *= x
        n--
    }
    return pow
};



// 1328. break a palindrome
var breakPalindrome = function(palindrome) {
    if (palindrome.length === 1) return ""
    
    for (let i = 0; i < Math.floor(palindrome.length/2); i++) {
        if (palindrome[i] !== 'a') {
            palindrome = palindrome.substring(0,i) + 'a' + palindrome.substring(i+1)
            return palindrome
        }
    }
    palindrome = palindrome.substring(0, palindrome.length-1) + 'b'
    return palindrome
};



// 1351. count negative numbers in a sorted matrix

var countNegatives = function(grid) {
    let res = 0
    let mid = 0
    for (let i = 0; i < grid.length; i++) {
        let min = 0
        let max = mid || grid[i].length -1
        while (min <= max) {
            mid = (min + Math.floor((max-min)/2))
            if (grid[i][mid] < 0 && (0 <= grid[i][mid-1] || mid === 0 )) {
                res += grid[i].length - mid
                break
            }        
            if (0 <= grid[i][mid]) {
                min = mid + 1
            } else {
                max = mid
            }
        }
    }
    return res
};

// 1837. sum of digits n base k
var sumBase = function(n, k) {
    
    let kbased = (k === 10)? `${n}`:(n >>> 0).toString(k)
    
    let sum = 0
    
    for (let i = 0; i < kbased.length ; i++) {
        sum += parseInt(kbased[i])
    }
    return sum
};

// 1963. minimum number of swaps to make the string balanced
// could use 2 pointers but it is faster to just count the swap when it needs to happen, 
// rebalance our counter and continue without the swap
var minSwaps = function(s) {
    let swaps = 0
    let openBrackets = 0
    //let closestOpen = s.length - 1
    for (let i = 0; i < s.length; i++) {
        //if (closestOpen < i) return swaps
        if (s[i] === '[') {
            openBrackets++
        }  else {
            openBrackets--
        }
        
        if (openBrackets < 0) {
            // let j = closestOpen
            // while (s[j] !== '[') {
            //     j--
            // }
            // closestOpen = j-1
            // s[i] = '['
            // s[j] = ']'
            openBrackets += 2
            swaps++
        }
    }
    return swaps
};

// 334. increasing triplet subsequence
var increasingTriplet = function(nums) {
    let max = nums.at(-1)
    for (let j = nums.length - 1; 0 <= j; j--) {
        if (nums[j] < max) continue
        max = nums[j]
        nums[j] = false
    }
    let min = Infinity
    for (let i =0; i < nums.length - 1; i++) {
        if (nums[i] === false) continue
        if (min < nums[i]) return true
        min = Math.min(min, nums[i])
    }
    return false
};


// 976. largest perimeter triangle
// for 3 lengths 
// successful triangle if c < a+b
var largestPerimeter = function(nums) {
    nums.sort((a,b) => a-b);
    
    for (let j = nums.length -1; 2 <= j; j--) {
        if (nums[j] < (nums[j-1] + nums[j-2])) {
            return nums[j] + nums[j-1] + nums[j-2]
        }
    }
    return 0
}


// 237. delete node in a linked list
// not given head node, and since list is not doubly linked we cant access prev node
// so instead of deleting the node by accessing the prev node, we shift the next node down to become our deleted node
var deleteNode = function(node) {
    node.val = node.next.val
    node.next = node.next?.next || null  
};


// 2013. detect squares
// used JSON.stringify and parse to store arrays directly
// couldve stored arrays by their x index
class DetectSquares {
    constructor() {
        this.points = {}
    }
    add(point) {
        point = JSON.stringify(point)
        this.points[point] = (this.points[point] || 0) + 1
    }
    count(point) {
        let count = 0
        for (let addedPoint of Object.keys(this.points)) {
            addedPoint = JSON.parse(addedPoint)
            if (addedPoint[1] === point[1] && point[0] !== addedPoint[0]) {
                let dist = addedPoint[0] - point[0]
                let corner10 = JSON.stringify([point[0], point[1]-dist])
                let corner11 = JSON.stringify([addedPoint[0], addedPoint[1]-dist])
                let corner20 = JSON.stringify([point[0], point[1]+dist])
                let corner21 = JSON.stringify([addedPoint[0], addedPoint[1]+dist])


                if (this.points[corner10] && this.points[corner11]) {
                    count += (this.points[corner10] * this.points[corner11] * this.points[JSON.stringify(addedPoint)])
                }
                if (this.points[corner20] && this.points[corner21]) {
                    count += this.points[corner20] * this.points[corner21] * this.points[JSON.stringify(addedPoint)]
                }
            }
        }
        return count
    }
};

// 136. single number
// use XOR a^a = 0 a^0 = a
// since XOR is transitive we dont need to sort 
var singleNumber = function(nums) {
    let res = 0
    for (const num of nums) {
        res = res^num
    }
    return res
};


// 191. number of 1 bits
// 
var hammingWeight = function(n) {
    let count = 0
    while (n !== 0) {
        if (n & 1) {
            count++
        }
        n = n >>> 1
        
    }
    return count
};

// 338. counting bits

var countBits = function(n) {
    let ans = [0]
    let twoPow = 1
    let count = 0
    
    for (let i = 1; i <= n; i++) { 
        if (i === twoPow*2) {
            twoPow = twoPow * 2
        }
        ans.push(ans[i-twoPow] + 1)
    }
    return ans
};

// 190. reverse bits
var reverseBits = function(n) {
    n = (((n >> 1) & 0x55555555) | ((n & 0x55555555) << 1));
    n = (((n >> 2) & 0x33333333) | ((n & 0x33333333) << 2));
    n = (((n >> 4) & 0x0f0f0f0f) | ((n & 0x0f0f0f0f) << 4));
    n = (((n >> 8) & 0x00ff00ff) | ((n & 0x00ff00ff) << 8));
    n = ((n >>> 16) | (n << 16));
    return n >>> 0
};
// 268. missing number 
// using bit operations 
// remmember a ^ a = 0
var missingNumber = function(nums) {
    let num = nums.length
    for (let i = 0; i < nums.length; i++) {
        num = nums[i] ^ num ^ i
    }
    return num
};

// sum all numbers in nums
// sum all numbers from 0-n
// compare
var missingNumber = function(nums) {
    let sum = nums.length * (nums.length + 1) / 2
    let numsSum = 0
    for (const num of nums) {
        numsSum += num
    }
    return sum - numsSum
}

// 7. reverse integer

var reverse = function(x) {
    let sign = 1
    if (x < 0) {
        sign = -1
        x *= -1
    }
    let max = 2**31
    let sum = 0
    while (x) {
        let newDigit = x%10
        sum*= 10
        sum+= newDigit
        x = Math.floor(x/10)
    }
    sum *= sign
    return (sum < max && -max < sum)? sum: 0
};


// 2095. delete the middle node of a linked list

var deleteMiddle = function(head) {
    let fast = head.next
    let slow = head
    if (!fast) return null
    while (fast) {
        fast = fast?.next?.next
        if (fast) {
            slow = slow.next
        }
    }
    slow.next = slow.next.next
    return head
}; 


// 1299. replace elements with greatest element on right side

var replaceElements = function(arr) {
    let max = -1
    for (let j = arr.length -1; 0 <= j; j--) {
        let val = arr[j]
        arr[j] = max
        max = Math.max(val, max)
    }
    return arr
};


// 392. is subsequence
// two pointer solution much faster 
var isSubsequence = function(s, t) {
    let sp = 0
    let tp = 0
    
    while (sp < s.length) {
       while (tp < t.length && s[sp] !== t[tp]) {
           tp++
       }
       if (t[tp] === s[sp]) {
           tp++
           sp++
       } else {
           return false
       }
    }
     return true
 };
// dp solution
var isSubsequence = function(s, t) {
    let dp = Array(s.length+1).fill().map(() => Array(t.length+1))
    
    for (let i = 0; i <= s.length; i++) {
        dp[i][0] = false
    }
     for (let j = 0; j <= t.length; j++) {
         dp[0][j] = true
     }
     for (let i = 1; i <= s.length; i++) {
         for (let j = 1; j <= t.length; j++) {
             if (s[i-1] === t[j-1]) {
                 dp[i][j] = dp[i-1][j-1]
             } else {
                 dp[i][j] = dp[i][j-1]
             }
         }
     }
     return dp[s.length][t.length]
    
 };


 // 58. length of last word

 var lengthOfLastWord = function(s) {
    s = s.trim().split(' ')
    return s.at(-1).length
};

// 14. longest common prefix
// using reduce to turn array into single element
var longestCommonPrefix = function(strs) {
    if (strs.length === 1) return strs[0]
    return strs.reduce((prev, next) => {
        let i = 0
        while (i < prev.length && next[i] === prev[i]) {
            i++
        }
        return prev.substring(0,i)
    })
};
// same thing without using reduce and just storing the prefix outside our loop
var longestCommonPrefix = function(strs) {
    if (strs.length === 1) return strs[0]
    let longestPre = strs[0]
    
    for (const str of strs) {
        if (longestPre === '') return longestPre
        let i = 0
        
        while (i < str.length && str[i] === longestPre[i]) {
            i++
        }
        longestPre = longestPre.substring(0, i)
    }
    return longestPre
};


// 118. pascal triangle

var generate = function(numRows) {
    let pascal = [[1]]
    for (let i = 1; i < numRows; i++) {
        let row = Array(i+1)
        for (let j = 0; j <= i; j++) {
            if (j === 0 || j === i) {
                row[j] = 1
            } else {
                row[j] = pascal[i-1][j-1] + pascal[i-1][j]
            }
        }
        pascal.push(row)
    }
    return pascal
};

// 27. remove element
var removeElement = function(nums, val) {
    let k = 0
    for (const num of nums){
        if (num !== val) {
            nums[k] = num
            k++
        }
    }
    return k
};

// 929. unique email addresses

var numUniqueEmails = function(emails) {
    let uniqueEmails = new Set()
    
    for (let email of emails) {
        let index = email.indexOf('+')
        let atIndex = email.indexOf('@')
        let domain = email.substring(atIndex)
        if (index !== -1) {
            email = email.substring(0,index)
        } else {
            email = email.substring(0, atIndex)
        }
        email = email.replace(/\./g, '')
        uniqueEmails.add(email+domain)
    }
    return uniqueEmails.size
};

// 205. isomorphic strings 
// since each chara can only map to 1 other chara we must keep track of the map both ways
// ie. from t: a => b from s: b => a
var isIsomorphic = function(s, t) {
    let map = {}
    let taken = {}
    for (let i = 0 ; i < s.length; i++) {
        if (!map[s[i]] && !taken[t[i]]) {
            map[s[i]] = t[i]
            taken[t[i]] = true
        }
        if (map[s[i]] !== t[i]) return false
    }
    return true
}; 


// 605. can place flowers
// greedy approach
var canPlaceFlowers = function(flowerbed, n) {
    for (let i = 0; i < flowerbed.length; i++) {
        
        if (flowerbed[i] !== 1 && flowerbed[i-1] !== 1 && flowerbed[i+1] !== 1) {
            n--
            flowerbed[i] = 1
        }
        if (n <= 0) return true
    }
    return false
};


// 169. majority element
// since we are looking for element that occurs more than half the time
// if we subtract every other element from the majority elements count it will still have a count greater than 1
var majorityElement = function(nums) {
    let max = nums[0] 
    let count = 0
    
    for (let num of nums) {
        if (num === max) {
            count += 1
        } else if (count === 0) {
            max = num
            count = 1
        } else {
            count -= 1
        }
    }
    return max
};


// 496. next greater element i

var nextGreaterElement = function(nums1, nums2) {
    let nextGreater = {}
    let monoStack = []
    for (let i = 0; i < nums2.length; i++) {
        while (monoStack.length !== 0 && monoStack.at(-1) < nums2[i]) {
            let num = monoStack.pop()
            nextGreater[num] = nums2[i]
        }
        nextGreater[nums2[i]] = -1
        monoStack.push(nums2[i])
    }
    
    let ans = [] 
    for (const num of nums1) {
        ans.push(nextGreater[num])
    }
    return ans
};

// 724. find pivot index 
// 1991. find the middle index in array
// start with sum of every element except the first
var pivotIndex = function(nums) {
    let index = 0
    let rightSum = 0
    let leftSum = 0
    for (let i = 1; i < nums.length; i++) {
        rightSum += nums[i]
    }
    

    while (index <= nums.length) {
        if (leftSum === rightSum) return index
        leftSum += nums[index]
        index++
        rightSum -= nums[index]
    }
    return -1
};


// 448. find all numbers disappeared in an array
// convert every found index to negative
// travel through array and add positive index's to ans array
// this problem is trivial without the extra constraints 
var findDisappearedNumbers = function(nums) {
    for (let i = 0; i < nums.length; i++) {
        let num = Math.abs(nums[i])
        nums[num-1] = - Math.abs(nums[num-1]) 
    }
    let ans = []
    for (let i = 0; i < nums.length; i++) {
        if (0 < nums[i]) {
            ans.push(i+1)
        }
    }
    return ans
}


// 1189. maximum number of balloons
// return the limiting letter
var maxNumberOfBalloons = function(text) {
    let letterFreq = {'a': 0, 'b':0, "l":0, 'n':0, 'o':0}
    for (const letter of text) {
        if (letterFreq[letter] !== undefined) {
            letterFreq[letter]++
        }
    }

    return Math.min(letterFreq['b'], letterFreq['a'], letterFreq['n'], Math.floor(letterFreq['l']/2), Math.floor(letterFreq['o']/2))
};


// 290. word pattern

var wordPattern = function(pattern, s) {
    let patternToString = {}
    let stringToPattern = {}
    
    s = s.split(' ')
    if (pattern.length !== s.length) return false
    for (let j = 0; j < pattern.length; j++) {
        if (patternToString[pattern[j]]) {
            let string = patternToString[pattern[j]]
            if (string !== s[j]) return false
        } else {
            if (stringToPattern[s[j]]) return false
            patternToString[pattern[j]] = s[j] 
            stringToPattern[s[j]] = pattern[j]
        }
    }
    return true
};


// 535. encode and decode tinyurl
// take a random string and use that as your short url
// if the random string is already used by a different url retry
class TinyUrl {
    constructor () {
        this.codeToUrl = {}
        this.urlToCode = {}
        this.map = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    } 
    encode (longUrl) {
        let code = ''
        for (let i = 0; i < 7; i++) {
           let random = Math.floor(Math.random() * 62)
           code += this.map[random]
        }
        if (this.codeToUrl[code]) {
           return this.encode[longUrl]
        }
        this.codeToUrl[code] = longUrl
        this.urlToCode[longUrl] = code
        return code
    }
    decode (shortUrl) {
        return this.codeToUrl[shortUrl]
    }

}


// 554. brick wall
// every path will be height of the wall - how many edges the path travels through
// so we increment the edge count for every edge for every row
// [[1], [1], [1]] will have no edges 
// [[2], [2], [1,1]] will have no edges until the last row which will have an edge at position 1 
var leastBricks = function(wall) {
    let width = wall[0].reduce((sum, next) => sum +next)
    let edges = {}
    let maxEdges = 0
    for (let i = 0; i < wall.length; i++) { 
        let sum = 0
        for (let j = 0; j < wall[i].length; j++) {
            sum += wall[i][j]
            if (sum < width) {
                edges[sum] = (edges[sum] || 0) + 1
                if (maxEdges < edges[sum]) {
                    maxEdges = edges[sum]
                }
             }
        }
    }
    return wall.length - maxEdges
};


// 122. best time to buy and sell stock ii

var maxProfit = function(prices) {
    let currBuy = -Infinity
    let currSell = 0
    
    for (const price of prices) {
        let prevBuy = currBuy
        let prevSell = currSell
        
        currBuy = Math.max(prevBuy, prevSell - price)
        currSell = Math.max(prevSell, prevBuy + price)
    }
    return currSell
};


// 714. best time to buy and sell stock with transaction fee
// very similar to other buy and sell stock problems
// we only need to add the transation fee to one equation
var maxProfit = function(prices, fee) {
    let hold = -Infinity
    let noHold = 0
    
    for (const price of prices) {
        let prevHold = hold
        let prevNoHold = noHold
        
        hold = Math.max(prevHold, prevNoHold - price - fee)
        noHold = Math.max(prevNoHold, prevHold + price)
    }
    return noHold
};


// 560. subarray sum equals to k
var subarraySum = function(nums, k) {
    let sum = 0
    let prevSums = {0: 1}
    let ans = 0
    
    for (const num of nums) {
        sum += num
        
        if (prevSums[sum-k]) {
            ans += prevSums[sum-k]
        }
        prevSums[sum] = (prevSums[sum] || 0) + 1
    }
    return ans
};


// 1930. unique length-3 palindromic subsequences
// record first and last occurence of every letter
// if a letter only has 1 occurence we wont consider
// count all unique characters between first and last occurence of each letter that occurs more than once
var countPalindromicSubsequence = function(s) {
    let letterPosition = {}
    
    for (let i = 0; i < s.length; i++) {
        const letter = s[i]
        if (!letterPosition[letter]) {
            letterPosition[letter] = [i,i]
        }
        letterPosition[letter][1] = i
    }
    let ans = 0

    for (const key of Object.keys(letterPosition)) {
        let min = letterPosition[key][0]
        let max = letterPosition[key][1]
        
        if (min === max) continue
        let x = new Set(s.substring(min+1, max)).size
        ans += x  
    }
    return ans
};

// 2001 number of pairs of interchangeable rectangles

var interchangeableRectangles = function(rectangles) {
    let ratio = {}
    let res = 0
    for (let i = 0; i < rectangles.length; i++) {
        const rectangle = rectangles[i]
        let r = rectangle[0]/rectangle[1]
        
        if (!ratio[r]) {
            ratio[r] = 0
        }
        res += ratio[r]
        ratio[r]++
    }
    return res
};

// 2002. maximum product of the length of two palindromic subsequences
// backtracking solution on each index we explore s[index] for either string 1, string 2, or neither
var maxProduct = function(s) {
    let res = 1
    const isPali = (array) => {
        let lp = 0
        let rp = array.length -1
        while (lp <= rp) {
            if (array[lp] !== array[rp]) {
                return false
            }
            lp++
            rp--
        }
        return true
    }
    const backTrack = (str1, str2, index) => {
        if (index === s.length) {
            if ((str1.length * str2.length > res) && isPali(str1) && isPali(str2)) {
                res = Math.max(res, str1.length * str2.length)
            }
        } else {
            str1.push(s[index])
            backTrack(str1, str2, index+1)
            str1.pop()
           
            str2.push(s[index])
            backTrack(str1, str2, index+1)
            str2.pop()
           
            backTrack(str1, str2, index+1)
        }
    }
    backTrack([], [], 0)
    return res
};


// 28. find the index of the first occurence in a string

var strStr = function(haystack, needle) {
    
    for (let i = 0; i < haystack.length; i++) {
        if (haystack[i] === needle[0]) {
            
            
            if (haystack.substring(i, i+needle.length) === needle) return i
        }
    }
    return -1
};


// 179. largest number 
// sort by combining two numbers and choosing the larger combo
// (31, 30) => (3130 > 3031)
var largestNumber = function(nums) {
    nums = nums.map((a) => a.toString()).sort((a,b) => {
        let ba = a+b
        let ab = b+a
        return ab-ba
    }).join('')
    let trailingZero = 0
    while (nums[trailingZero] == 0) {
        trailingZero++
    }
    nums = nums.substring(trailingZero)
    
    if (nums.length === 0) return "0"
    return nums
};

// 1832. check if the sentence is pangram

var checkIfPangram = function(sentence) {
    //     if (sentence.length < 26) return false
    //     let letters = {}
        
    //     for (const letter of sentence) {
    //         letters[letter] = true
    //     }
    //     return Object.keys(letters).length === 26
    return new Set(sentence.split('')).size === 26
};


// 523. continuous subarray sum
// smart math solution
// if a previous sum matches our current sum at index (i)
// and previous sum was added at index (j)
// that means sum from i to j === multiple of k
// [3,5,4,2] k = 6
// [3,2,0,2] <= prefix sums%k i = 3, j = 1 => sum from 1 to 3 === divisible by 6
var checkSubarraySum = function(nums, k) {
    let sum = 0
    let prefix = 0
    const prevSums = {}
    
    for (let i = 0; i < nums.length; i++) {
        sum += nums[i]
        
        sum %= k
        
        if (prevSums[sum]) return true
        
        
        prevSums[prefix] = true
        prefix = sum
    }
    return false
};
// big backtrack soltuion 
var checkSubarraySum = function(nums, k) {
    let dp = Array(nums.length)
    dp[nums.length -1] = 0
    for (let j = nums.length -2; 0 <= j; j--) {
        dp[j] = (dp[j+1] + nums[j+1])%k
        if (dp[j]%k === 0 && j !== nums.length -2) {
            return true
        }
    }
    const backtrack = (index, sum) => {
        if (index === nums.length) return false
        sum += nums[index]
        if (sum%k === 0) return true
        if (sum + dp[index] < k) return false
        if (backtrack(index+1, sum)) return true
    }
    
    for (let i = 0; i <= nums.length - 2; i++) {
        let sum = nums[i] + nums[i+1]
        if (sum%k === 0) return true
        if (backtrack(i+2, sum)) return true
    }
    return false
};

// 187. repeated dna sequences

var findRepeatedDnaSequences = function(s) {
    let seen = new Set()
    let repeated = new Set()
    for (let i = 0; i+9 < s.length; i++) {
        let str = s.substring(i, i+10) 
        if (seen.has(str)) repeated.add(str)
        
        seen.add(str)
    }
    return [...repeated]
};


// 380. insert delete getRandom o(1)
// array + map 
// array for random output
// map to keep track on index for insert + remove
// on removal of a value we move the last element in our array to the index of the value to remove
// then we remove the last element
class RandomizedSet {
    constructor () {
        this.valToPos = new Map()
        this.posToVal = []
    }
    insert (val) {
        if (this.valToPos.has(val)) return false
        this.valToPos.set(val, this.posToVal.length)
        this.posToVal.push(val)
        return true 
    }
    remove (val) {
        if (!this.valToPos.has(val)) return false
    
        let position = this.valToPos.get(val)
        let tail = this.posToVal.at(-1);
        this.posToVal[position] = tail
        this.valToPos.set(tail, position)
        this.valToPos.delete(val)
        this.posToVal.pop()
        return true
    }
    getRandom () {
        let rand = Math.floor(Math.random() * this.posToVal.length)
        return this.posToVal[rand]
    }
}


// 1461. check if a string contains all binary codes of size k

var hasAllCodes = function(s, k) {
    let codes = 2**k
    let codesHas = new Set()
    
    for (let i = 0; i+k-1 < s.length; i++) {
        let str = s.substring(i, i+k)
        codesHas.add(str)
        if (codesHas.size === codes) return true
    }
    return false
};


// 304. range sum query 2d - immutable

class NumMatrix{
    constructor (matrix) {
        this.dp = Array(matrix.length+1).fill().map(() => Array(matrix[0].length + 1).fill(0))
        for (let i = 1; i <= matrix.length; i++) {
            for (let j = 1; j <= matrix[0].length; j++) {
                this.dp[i][j] = matrix[i-1][j-1] + this.dp[i-1][j] + this.dp[i][j-1] - this.dp[i-1][j-1]
            }
        }  
    }
    sumRegion (row1, col1, row2, col2) {
        return this.dp[row2+1][col2+1] - this.dp[row1][col2+1] - this.dp[row2+1][col1] + this.dp[row1][col1] 
    }
}


// 303. range sum query 
class NumArray {
    constructor (nums) {
        this.nums = nums
        for (let i = 1; i < nums.length; i++) {
            this.nums[i] += this.nums[i-1]
        }   
    }
    sumRange (left, right) {
        if (left === 0) return this.nums[right]
        return this.nums[right] - this.nums[left-1]
    }
}


// 665. non-decreasing array
// greedy approach 
// everytime we run into an instance where the next integer is less than our current integer
// we compare our next int to our prev int
// if next int is equal or larger we can change our current int to behave
// if not we must change our next int
var checkPossibility = function(nums) {
    nums.push(Infinity)
    let prev = -Infinity
    let curr 
    let next
    let changes = 0
    
    for (let i = 0; i < nums.length-1; i++) {
        curr = nums[i]
        next = nums[i+1]
        if (next < curr) {
            if (next < prev) nums[i+1] = curr
            else nums[i] = prev
            changes++
            if (changes === 2) return false
        } else {
            prev = curr
        }
    }
    return true
};



// 38. count and say

var countAndSay = function(n) {
    
    const travel = (i, str) => {
        if (i === 1) return travel(i+1, '1')
        if (n < i) return str
        
        let newStr = ''
        for (let j = 0; j < str.length; j++) {
            let num = str[j]
            let count = 0
            while (j < str.length && str[j] === num) {
                j++
                count++
            }
            j--
            newStr += `${count}${num}`
        }
        return travel(i+1, newStr)
    }
    return travel(1, '')
 };


// 41. first missing positive
// iterate through nums moving positive integers into their respective spots
// note we only consider positive integers that are inside the array length
// this is because we need 1...n+1 inside an array of length n 
var firstMissingPositive = function(nums) {
    for (let i = 0; i < nums.length; i++) {
        if (nums[i] <= 0) continue
        if (nums.length < nums[i]) continue
        let num = nums[i]
        while (0 < num && num <= nums.length && num !== i+1 && num !== nums[num-1]) {
            nums[i] = nums[num-1]
            nums[num-1] = num
            num = nums[i]
        }
    }

    for (let i = 0; i <nums.length; i++) {
        if (nums[i] !== i+1) return i+1
    }
    return nums.length+1
};



// 680. valid palindrome ii
// check if s is palindrome 
// if we run into letters that dont equal eachother 
// try shift either of our pointers by one and checking if that would make a palindrome
// if shifting by one wont result in palindrome then we would need to delete more than one 
var validPalindrome = function(s) {
    const isValiPali = (l, r) => { 
        while (l <= r) {
            if (s[l] !== s[r]) return false
            l++
            r--
        }
        return true
    } 
    let lp = 0
    let rp = s.length-1
    while (lp <= rp) {
        if (s[lp] !== s[rp]) {
            if (isValiPali(lp+1, rp)) return true
            if (isValiPali(lp, rp-1)) return true
            return false
        }
        lp++
        rp--
    }
    return true
};


// 1984. minimum difference between highest and lowest of k scores
// trivial after sorting the array just compare every window's max and min 
var minimumDifference = function(nums, k) {
    if (k === 1) return 0
    let ans = Infinity
    nums = nums.sort((a,b) => a-b)
    for (let i = 0; i + k -1 < nums.length; i++) {
        ans = Math.min(ans, nums[i+k-1] - nums[i])
    }
    return ans
}; 



// 344. reverse string
var reverseString = function(s) {
    for (let i = 0; i < Math.floor(s.length/2); i++) {
        [s[i], s[s.length - i - 1]] = [s[s.length - i - 1], s[i]];
    }
};


// 88. merge sorted array


var merge = function(nums1, m, nums2, n) {
    let p1 = m - 1
    let p2 = n - 1
    let sortedP = m + n - 1
    
    while (0 <= p1 && 0 <= p2) {
        if (nums1[p1] < nums2[p2]) {
            nums1[sortedP] = nums2[p2]
            p2--
        } else {
            nums1[sortedP] = nums1[p1]
            p1--
        }
        sortedP--
    }
    
    while (0 <= p2) {
        nums1[sortedP] = nums2[p2]
        p2--
        sortedP--
    } 
};


// 283. move zeroes 

var moveZeroes = function(nums) {
    let lp = 0
    const swap = (l, r) => {[nums[l], nums[r]] = [nums[r], nums[l]]}
    for (let i = 0; i < nums.length; i++) {
        if (nums[i] != 0){
            swap(lp, i)
            lp++
        }
    }
};


// 26.  remove duplicates from sorted array
var removeDuplicates = function(nums) {
    let dups = 0
    let prev = Infinity
    for (let i = 0; i < nums.length; i++) {
        if (nums[i] === prev) dups++
        else (nums[i-dups] = nums[i])
        prev = nums[i]
    }
    nums.splice(nums.length - dups)
}; 


// 1498. number of subsequences that satisfy the given sum condition
var numSubseq = function(nums, target) {
    nums.sort((a,b) => a-b)
    let lp = 0
    let rp = nums.length - 1
    let res = 0
    while ((nums[lp] + nums[rp]) > target && lp < rp) {
            rp--
    }
    let powers = allPow(2, rp-lp,1000000007)
    
    while (nums[lp]*2 <= target && lp < nums.length) {
        while ((nums[lp] + nums[rp]) > target && lp < rp) {
            rp--
        }
        res += powers[rp-lp];
        lp++
    }
    
   
    return res%1000000007
};

let allPow = function(power, exp, mod) {
    let powers = [1]
    let res = 1
    while (0 < exp) {
        res *= power
        res %= mod
        powers.push(res)
        exp--
    }
    return powers
}


// 189. rotate array
// using reverse

// nums = "----->-->"; k =3
// result = "-->----->";
// reverse "----->-->" we can get "<--<-----"
// reverse "<--" we can get "--><-----"
// reverse "<-----" we can get "-->----->"

var rotate = function(nums, k) {
    const reverse = (l,r) => {
        while (l < r) {
           [nums[l], nums[r]] = [nums[r], nums[l]];
           l++
           r--
        }
    }
    k %= nums.length
    reverse(0, nums.length -1)
    reverse(0, k-1)
    reverse(k, nums.length -1)
};

// keep shifting an element k spaces until we've shifted every element
// consider [1,2,3,4] and k = 1
// on the first pass we shift every element so we can exit [4,1,2,3]
// consider now [1,2,3,4] and k = 2
// on first pass we get [3,2,1,4] since k && length of array arent co-prime we have a cycle
// so we must start another shift cycle starting from the index 1
// this produces [3,4,1,2] and 4 shifts so we can exit
var rotate = function(nums, k) {
    let n = nums.length
    let count = 0
    if (n < k) {
        k = k%n
    }
    if (n === 1 || k === n) return nums  
    const shift = (stop, num, index) => {
        const hold = nums[index]
        if (count === n) return
        count++
        nums[index] = num
        if (stop !== index) {
            shift(stop, hold, (index+k)%n)
        }
    }
    for (let i = 0; i < k; i++) {
        shift(i, nums[i], i+k)
    }
    
};


// 1968. array with elements not equal to average of neighbors

var rearrangeArray = function(nums) {
    for (let i = 1; i < nums.length - 1; i++) {
        if ((nums[i-1] + nums[i+1])/2 === nums[i]) {
            [nums[i], nums[i+1]] = [nums[i+1], nums[i]]
        }
    }
    for (let j = nums.length - 1; 0 < j; j--) {
        if ((nums[j-1] + nums[j+1])/2 === nums[j]) {
            [nums[j], nums[j-1]] = [nums[j-1], nums[j]]
        }
    }
    return nums
};



// 881. boats to save people

var numRescueBoats = function(people, limit) {
    people.sort((a,b) => a-b)
    let l = 0
    let r = people.length - 1
    let res = 0
  
    while ( l < r) {
        if (people[r] + people[l] > limit) {
            res++
            r--
        } else {
            l++
            r--
            res++
        }
    }
    return res + r-l + 1
};



// 1838. frequency of the most frequent element 
// sliding window with criteria that our window satisfies maxElemet * length < window Sum + k
// this criteria is just making sure that we can turn all elements inside our window into the max element
var maxFrequency = function(nums, k) {
    nums.sort((a,b) => a - b)
    
    
    let lp = 0
    let rp = 0
    let sum = 0
    let max = 0
    
    while (rp < nums.length) {
        sum += nums[rp]
        
        while (sum + k < nums[rp]*(rp-lp+1)) {
            sum -= nums[lp]
            lp++
        }
        max = Math.max(max, rp-lp+1)
        rp++
    }
    return max
};


// 692. top k frequent words
// o(n logn) because we are sorting n elements
// can reduce sorts by looping through and creating buckets 
// then loop through and grab the last buckets until we have reached k words
// then sort

var topKFrequent = function(words, k) {
    let frequency = {}
    
    for (const word of words) {
        frequency[word] = (frequency[word] || 0) + 1
    }
    
    let sorted = Object.keys(frequency).sort((a,b) => {
        if (frequency[a] === frequency[b]) {
            if (a < b) return -1
        } else {
            return frequency[b] - frequency[a]
        }
    }).slice(0, k)
    return sorted
};


// using buckets
// o(n logK) speed because now on average we only have to sort k elements
var topKFrequent = function(words, k) {
    let frequency = {}
    let buckets = []
    let mostFreq = []

    for (const word of words) {
        frequency[word] = (frequency[word] || 0) + 1
    }
    
    for (const [key,value] of Object.entries(frequency)) {
        if (!buckets[value]) {
            buckets[value] = []
        }
        buckets[value].push(key)
    }
    
    for (let j = buckets.length; 0 <=j; j--) {
        if (buckets[j]) {
            buckets[j].sort((a,b) => a > b? -1:1 )
            while (buckets[j].length && mostFreq.length < k) {
                mostFreq.push(buckets[j].pop())
            }
        }
    }
    return mostFreq
};



// 209. minimum size subarray sum

var minSubArrayLen = function(target, nums) {
    let l = 0
    let r = 0
    let minSize = Infinity
    let sum = 0
    while (r < nums.length) {
        sum += nums[r]
       
        while (target <= sum ) {
            minSize = Math.min(minSize, r-l+1)
            sum -= nums[l]
            l++
        }
        r++
    }
    return minSize === Infinity? 0: minSize
};


// 682. baseball game

var calPoints = function(operations) {
    let stack = []
    for (const operation of operations) {
        if (operation.match(/([0-9])/)) {
            stack.push(+operation)
        }
        else if (operation === '+') {
            stack.push(stack.at(-1) + stack.at(-2))
        }
        else if (operation === 'D') {
            stack.push(stack.at(-1)*2)
        }
        else {
            stack.pop()
        }
    }
    return stack.length? stack.reduce((sum, next) => sum + next): 0
};

// 225. implement stack using queues
// QUEUE ONLY HAS push to back and remove from front
// so we mush do some magic on pop and top to get the last element
class MyStack {
    constructor() {
        this.stack = []
    }
    push(x) {
        return this.stack.push(x)
    }
    pop() {
        for (let i = 0; i < this.stack.length - 1; i++) {
            this.stack.push(this.stack.shift())
        }
        return this.stack.shift()
    }
    top() {
        let ele = this.pop()
        
        this.stack.push(ele)
        return ele
    }
    empty() {
        return this.stack.length === 0
    }
}


// 901. online stock span

class StockSpanner {
    constructor() {
        this.prices = []
    }
    next (price) {
        let span = [price, 1]
        while (this.prices.length && this.prices.at(-1)[0] <= price) {
            span[1] += this.prices.at(-1)[1]
            this.prices.pop()
        }
        this.prices.push(span)
        return span[1] 
    }
}

// 735. asteroid collision 
var asteroidCollision = function(asteroids) {
    let stack = []
    
    for (const asteroid of asteroids) {
        if (asteroid < 0) {
            while (stack.length && stack.at(-1) < asteroid*-1 && 0 < stack.at(-1)) {
                stack.pop()
            }
            
            if (stack.length && stack.at(-1) === asteroid*-1) {
                stack.pop()
            } 
            else if (!stack.length || stack.at(-1) < 0) stack.push(asteroid)
            
        } else {
            stack.push(asteroid)
        }
    }
    return stack
};

// 71. simplify path 
var simplifyPath = function(path) {
    let stack = ['/']
    path = path.split('/')
    
    for (const directory of path) {
        if (directory === '.' || directory === '') continue
        else if (directory === '..' ) {
            if (stack.length === 1) continue
            else stack.pop()
        }
        else {
            if (stack.length === 1) {
                stack.push(directory)
            } else {
                stack.push('/' + directory)
            }
        }
        
    }
    return stack.join('')
};


// 12. integer to roman

var intToRoman = function(num) {
    let val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    let numeral = ["M", 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    let roman = ''
    
    for (let i = 0; i <numeral.length; i++) {
        
        while (val[i] <= num) {
            num -= val[i]
            roman += numeral[i]
        }
    }
    return roman
};


// 394. decode string 

var decodeString = function(s) {
    let stack = []
    let currStr = ''
    let currNum = 0
    for (const c of s) {
        if (c === '[') {
            stack.push(currStr)
            stack.push(currNum)
            currNum = 0
            currStr = ''
        }
        else if ( c === ']') {
            let val = stack.pop()
            let str = stack.pop()
            let repeated = ''
            for (let i = 0; i < val; i++) {
                repeated += currStr
            }
            currStr = str + repeated
        }
        else if (c.match(/[0-9]/)) {
            currNum = currNum*10 + +c
        }
        else {
            currStr += c
        }
    }
    return currStr
};


// 402. remove k digits
var removeKdigits = function(num, k) {
    let stack = []
    
    for (const c of num) {
        if (!stack.length && c != 0) {
            stack.push(c)
        }
        else {
            while (c < stack.at(-1) && 0 < k) {
                stack.pop()
                k--
            }
            
            if (c != 0) {
                stack.push(c)
            } else if (stack.length) {
                stack.push(c)   
            }
        }
    }
    while (0 < k) {
        stack.pop()
        k--
    }
    if (!stack.length) stack.push(0)
    return stack.join('')
};


// 1209. remove all adjacent duplicates in string ii
// use stack to keep track of current letter, if current letter ever reaches k count we pop it 
var removeDuplicates = function(s, k) {
    let stack = []
    for (let i = 0; i < s.length; i++) {
        if (!stack.length) stack.push([s[i], 1])
        else if (stack.at(-1)[0] === s[i]) {
            stack.at(-1)[1]++
            if (stack.at(-1)[1] === k) stack.pop()
        }
        else {
            stack.push([s[i], 1])
        }
    }
    
    let str = ''
    
    for (const [letter, count] of stack) {
        for (let i = 0; i < count; i++) {
            str += letter
        }
    }
    return str
};


// 456. 132 pattern 
var find132pattern = function(nums) {
    let stack = []
    let middle = -Infinity
    
    for (let j = nums.length - 1; 0 <= j; j--) {
        while (stack.length && stack.at(-1) < nums[j]) {
            middle = Math.max(middle, stack.pop())
        }
        stack.push(nums[j])
        if (nums[j] < middle) return true
    }
    return false
};

// 895. maximum frequency stack
class FreqStack {
    constructor() {
        this.stack = []
        this.valToFreq = {}
    }
    push(val) {
        this.valToFreq[val] = (this.valToFreq[val] || 0) + 1
        if (!this.stack[this.valToFreq[val]-1]) {
            this.stack[this.valToFreq[val] - 1] = []
        }
        this.stack[this.valToFreq[val] - 1].push(val)  
    }
    pop () {
        let popped = this.stack.at(-1).pop()
        this.valToFreq[popped]--
        if (!this.stack.at(-1).length) {
            this.stack.pop()
        }
        return popped 
    }
}

// 35. search insert position

var searchInsert = function(nums, target) {
    let l = 0
    let r = nums.length -1
    let mid
    while (l <= r) {
        mid = l + Math.floor((r-l)/2)
        if (nums[mid] === target) {
            return mid
        } 
        else if (nums[mid] < target) {
            l = mid + 1
        } 
        else {
            r = mid - 1
        }
    }
    return l
};

// 374. guess number higher or lower

var guessNumber = function(n) {
    let low = 1
    let high = n
    let mid 
    while (low <= high) {
        mid = low + Math.floor((high - low)/2)
        let val = guess(mid)
        if (val === 0) return mid
        else if (val < 0 ) high = mid - 1
        else low = mid + 1
    }
    return low
};

// 441. arranging coins

var arrangeCoins = function(n) {
    let low = 1
    let high = n
    let mid 
    let sum
    while (low <= high) {
        mid = low + Math.floor((high - low)/2)
        sum = (mid* (mid+1))/2
        if (sum === n) return mid
        if (sum < n) low = mid + 1
        else high = mid - 1
    }
    return (low* (low+1)/2 < sum)? low: low-1
};


// 977. squares of a sorted array
// since array is sorted we just need to comapre left and right pointer to get the larger element
var sortedSquares = function(nums) {
    let squares = Array(nums.length)
    
    let left = 0
    let right = nums.length - 1
    for (let p = nums.length -1; 0 <= p; p--) {
        if (nums[left]*nums[left] < nums[right] * nums[right]) {
            squares[p] = nums[right] * nums[right]
            right --
        } 
        else {
            squares[p] = nums[left] * nums[left]
            left++
        }
    }
    return squares
};


// 367. valid perfect square
// binary search
var isPerfectSquare = function(num) {
    let low = 0
    let high = num
    let mid 
    while (low <= high) {
        mid = low + Math.floor((high - low)/2)
        if (mid*mid === num) return true
        else if (mid*mid < num) low = mid + 1
        else high = mid - 1
    }
    return false
};
// trying every square up to num
var isPerfectSquare = function(num) {
    let square = 1
    let twos = 3
    while (square <= num) {
        if (square === num) return true
        square += twos
        twos += 2
    }
    return false
};

// 219. contains duplicate ii

var containsNearbyDuplicate = function(nums, k) {
    let dup = {}
    for (let i = 0; i < nums.length; i++) {
        if (dup[nums[i]] !== undefined && i - dup[nums[i]] <= k) return true
        dup[nums[i]] = i
    }
    return false
};


// 1898. maximum number of removable characters
// binary search on removable array
var maximumRemovals = function(s, p, removable) {
    
    const isSubWithRemoved = (removed) => {
        let pp = 0
        
        for (let i = 0; i < s.length; i++) {
            if (s[i] === p[pp] && !removed.has(i)) {
                pp++
            }
            
            if (pp === p.length) return true
        }
        return false
    }
    
    let l = 0
    let r = removable.length - 1
    let mid
    
    while (l <= r) {
        mid = l + Math.floor((r-l)/2)
        
        let removed = new Set(removable.slice(0, mid+1))
        if (isSubWithRemoved(removed)) {
            l = mid + 1
        } else {
            r = mid - 1
        }
    }
    return l
    
};


// 116. populating next right pointers in each node
// bfs 
var connect = function(root) {
    if (!root) return null
    let queue = new Queue()
    queue.enqueue(root)
    while (queue.size()) {
        
        let nextLevel = new Queue()
        let currNode 
        while (queue.size()) {
            currNode = queue.dequeue()
            if (queue.front()) {
                currNode.next = queue.front()
            }
            if (currNode.left) {
                nextLevel.enqueue(currNode.left)
                nextLevel.enqueue(currNode.right)
            }
        }
        queue = nextLevel
    }
    return root
};

// dfs connecting childs before calling recurssion 
var connect = function(root) {
    if (root === null || root.left === null) return root 
    root.left.next = root.right
    root.right.next = root.next? root.next.left: null
    
    connect(root.right)
    connect(root.left)
    return root
};

// 1268 search suggestions system
var suggestedProducts = function(products, searchWord) {
    products.sort()
    let res = Array(searchWord.length).fill().map(() => [])
    let l = 0
    let r = products.length -1 
    for (let i = 0; i < searchWord.length; i++) {
        while (l <= r && products[l][i] !== searchWord[i]) {
            l++    
        }
        while (l <= r && products[r][i] !== searchWord[i]) {
            r--
        }
        
        for (let j = l; j < l+3 && j <= r; j++) {
            res[i].push(products[j])
        }
    }
    return res
};


// 2448. minimum cost to make array equal
// JAVASCRIPT SOLUTION DOES NOT WORK CORRECTLY BECAUSE Of INTEGER OVERFLOW
// EVEN USING BIGINT THE CONVERSION DOES NOT WORK
// the problem uses binary search on range of (min int of nums, max int of nums)
var minCost = function(nums, cost) {
    let max = Math.max(...nums)
    let min = Math.min(...nums) 
    var convexFun = function (x){
        let ans = 0
        for (let i = 0; i < nums.length; i++) {
            ans += Math.abs(x-nums[i]) * cost[i]
        }
       return ans
    }
    let mid
    let res = Infinity
    while (min < max) {
        mid = min + Math.floor((max- min)/2)
       
        let [midf, midf2] = [convexFun(mid), convexFun(mid+1)]
        res = Math.min(midf, midf2, res)
        if (midf < midf2) {
            max = mid
        } 
        else {
            min = mid + 1
        }
    }
    return res
};


// 2449. minimum number of operations to make arrays similar
// consider only odd array = [1,3,3] target [1,1,5]
// since we are guranteed a solution exists then we must have half our array over and half our array under
// we take the abs of every difference between array and target 
// since we can increment by 2 each time the sum can be halved
// and since we do 2 operations at one time the number of operations is actually 1/4th (1/2 * 1/2)
// now the original question has both even and odds in the array, but since we are de/incrementing by 2 each time
// the evens and odds will never turn into eachother, so we can consider each array seperately 
var makeSimilar = function(nums, target) {
    let numsOdd = []
    let numsEven = []
    let targetOdd = []
    let targetEven = []
    nums.sort((a,b) => a-b)
    target.sort((a,b) => a-b)
    
    for (let i = 0; i < nums.length; i++) {
        if (nums[i]%2) {
            numsOdd.push(nums[i])
        } else {
            numsEven.push(nums[i])
        }
        if (target[i]%2) {
            targetOdd.push(target[i])
        } else {
            targetEven.push(target[i])
        }
    }
    let sumOdd = 0
    let sumEven = 0
    for (let i = 0; i < numsOdd.length; i++) {
        sumOdd += Math.abs(numsOdd[i] - targetOdd[i])
    }
    for (let i = 0; i < numsEven.length; i++) {
          sumEven += Math.abs(numsEven[i] - targetEven[i])
    }
    return (sumOdd + sumEven)/4
};



// 2446. determine if two events have conflict

var haveConflict = function(event1, event2) {
    event1[0] = event1[0][0] + event1[0][1] + event1[0][3] + event1[0][4]
    event2[0] = event2[0][0] + event2[0][1] + event2[0][3] + event2[0][4]
    
    event1[1] = event1[1][0] + event1[1][1] + event1[1][3] + event1[1][4]
    event2[1] = event2[1][0] + event2[1][1] + event2[1][3] + event2[1][4]
    
    
    return Math.max(event1[0], event2[0]) <= Math.min(event1[1], event2[1])
                                                                                             
        
};


// 2447 number of subarrays with gcd equal to k
// try every subarray starting from i, if every element is divisible by k
var subarrayGCD = function(nums, k) {
    let ans = 0
    
    for (let i = 0; i < nums.length ; i++) {
        if (nums[i]%k === 0) {
            if (nums[i] === k) ans++
            let lp = i
            let rp = i+1
            let array = [nums[i]]
            while (rp < nums.length) {
                if (nums[rp]%k !== 0) break
                array.push(nums[rp])
                if (subArrayGCD(array, k)) ans++
                rp++
            }
        }
    }
    return ans
};

var subArrayGCD = function(array, k){
    let min = Math.min(...array)
    for (let i = k+1; i <= min; i++) {
        if (array.every((element) => element%i === 0)) return false
    }
    return true
}


// 203. remove linked list elements

var removeElements = function(head, val) {
    while (head && head.val === val) {
        head = head.next
    }
    
    let node = head
    
    while (node && node.next) {
        if (node.next.val === val) {
            node.next = node.next.next
        }
        else {
            node = node.next
        }
    }
    return head
};


// 234. palindrome linked list
// using fast + slow pointer to get slow pointer at mid point
// reverse right half of list
// now compare two linked lists, one starting from head and one from head
// this solution uses o(n) time and o(1) space
var isPalindrome = function(head) {
    let slow = head
    let fast = head
    
     while (fast && fast.next) {
         slow = slow.next
         fast = fast.next.next
     }
     let prev = slow
     slow = slow.next
     prev.next = null
     
     while (slow) {
         let temp = slow.next
         slow.next = prev
         prev = slow
         slow = temp
     }
     fast = head
     slow = prev
     while (slow) {
         if (fast.val !== slow.val) return false
         fast = fast.next
         slow = slow.next
     }
     return true
 };


 // 83. remove dupliaces from sorted list

 var deleteDuplicates = function(head) {
    let node = head
    while (node && node.next) {
        if (node.next.val === node.val) {
            node.next = node.next.next
        } else {
            node = node.next
        }
    }
    return head
};



// 160. intersection of two linked lists


var getIntersectionNode = function(headA, headB) {
    let tail = headA
    while (tail.next) {
        tail = tail.next
    }
    tail.next = headB
    
    let fast = headA
    let slow = headA
    
    while (fast && fast.next) {
        slow = slow.next
        fast = fast.next.next
        if (slow === fast) break
    }
    
    if (!fast || !fast.next) {
        tail.next = null
        return
    }
    slow = headA
    
    while (slow !== fast) {
        slow = slow.next
        fast = fast.next
    }
    tail.next = null
    return slow
};


// 24. swap nodes in pairs
var swapPairs = function(head) {
    if (!head) return head
    let temp = new ListNode('sentinel', head)
    let i = head 
    let j = head.next
    if (j) head = j
    while (j) {
        i.next = j.next;
        j.next = i;
        temp.next = j
        temp = i
        i = i.next
        j = (i)? i.next: null
        
    }
    return head
};

// 1239. maximum length of a concatenated string with unique characters

var maxLength = function(arr) {
    let dp = ['']
    
    for (const str of arr) {
        let a = new Set(str)
        if (a.size < str.length) continue
        for (const sub of dp) {
            if ((str+sub).length === new Set(str+sub).size) {
                dp.push(str+sub)
            }
        }
    }
    let longestStr = dp.reduce((a,b) => {
        if (a.length < b.length) {
            return b
        } else return a})
    return longestStr.length
};



// 148. sort list
//merge sort on linked list
// find middle 
// sort left and right
// merge
var sortList = function(head) {
    if (!head || !head.next) return head
    let slow = head
    let fast = head.next
    while (fast && fast.next) {
        fast = fast.next.next
        slow = slow.next
    }
    
    let mid = slow.next
    slow.next = null
    let left = sortList(head)
    let right = sortList(mid)
    let sentinel = new ListNode('sentinel')
    let currNode = sentinel
    
    while (left && right) {
        if (left.val < right.val) {
            currNode.next = left
            left = left.next
        } else {
            currNode.next = right
            right = right.next
        }
        currNode = currNode.next
    }
    if (left) {
        currNode.next = left
    }
    if (right) {
        currNode.next = right
    }
    return sentinel.next
}


// 86. partition list

var partition = function(head, x) {
    let less = new ListNode('sentinel')
    let lessP = less
    let greater = new ListNode('sentinel')
    let greaterP = greater
    let currNode = head
    while (currNode) {
        if (currNode.val < x) {
            lessP.next = currNode
            lessP = currNode
            
        } else {
            greaterP.next = currNode
            greaterP = currNode
        }
        currNode = currNode.next
    }
    
    
    head = less.next || greater.next
    lessP.next = greater.next
    greaterP.next = null
    return head
};


// 61. rotate list
// find length of list, k mod length
// have a lead pointer and another pointer thats k distance away
// move tail to head
// and set new tail to have no next node
var rotateRight = function(head, k) {
    if (!head || !head.next) return head
    let size = 0
    let currNode = head
    while (currNode) {
        size++
        currNode = currNode.next
    }
    k = k%size
    if (k === 0) return head
    let leadNode = head
    currNode =head
    while (0 < k) {
        leadNode = leadNode.next
        k--
    }
    while (leadNode.next) {
        leadNode = leadNode.next
        currNode = currNode.next
    }
    leadNode.next = head
    head = currNode.next
    currNode.next = null
    return head
};


// 147. insertion sort list

var insertionSortList = function(head) {
    let sentinel = new ListNode('sentinel')
    
    let currNode = head
    let temp = sentinel
    while (currNode) {
        if (!temp.next || currNode.val < temp.next.val) temp = sentinel
        while (temp.next && temp.next.val < currNode.val) {
            temp = temp.next
        }
        let hold = currNode.next
        currNode.next = temp.next
        temp.next = currNode
        currNode = hold
    }
    return sentinel.next
};


// 94. binary tree inorder traversal
// recursive
var inorderTraversal = function(root) {
    const helper = (root, arr) => {
        if (!root) return arr
        helper(root.left, arr)
        arr.push(root.val)
        helper(root.right, arr)
        return arr
    }
    return helper(root, [])
};
//iterative
var inorderTraversal = function(root) {
    if (!root) return []
    let ans = []
    let queue = []
    queue.push(root)
    
    while (queue.length) {
        let node = queue.pop()
        if (node.left) {
            queue.push(node)
            queue.push(node.left)
            node.left = null
        } else {
            ans.push(node.val)
            if (node.right) {
                queue.push(node.right)
            }
        }
    }
    return ans
};


// 108. convert sorted array to binary search tree
// the middle element will always be the root
var sortedArrayToBST = function(nums) {
    if (nums.length === 0) {
        return null
    } else if (nums.length === 1) {
        let root = new TreeNode(nums[0])
        return root
    }
    let mid = Math.floor(nums.length/2)
    let left = sortedArrayToBST(nums.slice(0, mid))
    let right = sortedArrayToBST(nums.slice(mid+1))
    let root = new TreeNode(nums[mid], left, right)
    return root
};


// 617. merge two binary trees
var mergeTrees = function(root1, root2) {
    if (root1 && root2) {
        root1.val += root2.val
        root1.left = mergeTrees(root1.left, root2.left)
        root1.right = mergeTrees(root1.right, root2.right)
    } else if (!root1) {
        return root2
    }
    return root1
};

// 606. construct string from binary tree

var tree2str = function(root) {
    if (!root) return 
    let str = `${root.val}`
    
    if (root.left || root.right) {
        if (root.right && root.left) {
            str += `(${tree2str(root.left)})`
            str += `(${tree2str(root.right)})`
        }
        else if (root.right) {
            str += '()'
            str += `(${tree2str(root.right)})`
        }
        else {
            str += `(${tree2str(root.left)})`
        }
    }
    return str
};

// 96. unique binary search trees
// to find num trees n we consider each potential root 1..n
// consider 1,2,3,4,5
// root at 1 means 2,3,4,5 will be on the right side
// the potental trees for 2,3,4,5 is the same as trees for 1,2,3,4
// thus each root has potential trees of  PT(mid - 1) * PT(n-mid) (where PT === potential trees)
var numTrees = function(n) {
    let dp = Array(n+1).fill(0)
    dp[0] = dp[1] = 1
    
    for (let i = 2; i <= n; i++) {
        for (let j = 1; j <= i; j++ ) {
            dp[i] += dp[j-1]*dp[i-j]
        }
    }
    return dp[n]
}
var numTrees = function(n) {
    let cache = {}
    const countTrees = (low, high) => {
        if (high <= low) return 1
        let total = 0
        if (cache[high-low]) return cache[high-low]
        for (let i = low; i <= high; i++) {
            total += countTrees(low, i-1)*countTrees(i+1, high)
        }
        return cache[high-low] = total
    }
    return countTrees(1,n)
};

// 129. sum root to leaf numbers

var sumNumbers = function(root) {
    let array = []
    
    const travel = (root, num) => {
        if (!root) return
        
        num += root.val
        if (!root.left && !root.right) array.push(num)
        else {
            travel(root.left, num)
            travel(root.right, num)
        }
    }
    travel(root, '')
    return array.reduce((a,b) => +a + +b)
};

// 337. house robber iii
// keep track of largest chain and largest chain that allows current node
// on each recurssion our left and right arrays can mix with eachother because they are not directly linked
var rob = function(root) {
    const helper = (root) => {
        if (!root) return [0, 0]
        if (!root.left && !root.right) return [root.val, 0]
        let left = helper(root.left)
        let right = helper(root.right)
        return [root.val+left[1]+right[1], Math.max(left[1] + right[1], left[0] + right[0], left[1]+right[0], right[1] + left[0])]
    }
    return Math.max(...helper(root))
};

// 951. flip equivalent binary trees

var flipEquiv = function(root1, root2) {
    if (!root1 && !root2) return true
    if (!root1 || !root2) return false
    if (root1.val !== root2.val) return false
    
    if (root1.left?.val !== root2.left?.val) {
        return flipEquiv(root1.left, root2.right) && flipEquiv(root1.right, root2.left)
    } else {
        return flipEquiv(root1.left, root2.left) && flipEquiv(root1.right, root2.right)
    }
};

// 1993. operation on tree
var Node = class {
    constructor (val) {
        this.val = val
        this.child = []
        this.parent = null
        this.locked = false
    }
}
var LockingTree = class {
    constructor (parent) {
        this.parents = Array(parent.length)
        let node = new Node(0)
        this.parents[0] = node
        for (let i = 1; i < parent.length; i++) {
            if (this.parents[i]) {
                node = this.parents[i]
            } else {
                node = new Node(i)
            }
             if (!this.parents[parent[i]]) {
                this.parents[parent[i]] = new Node(parent[i])
            }
            node.parent = this.parents[parent[i]]

            this.parents[i] = node
            node.parent.child.push(node)
        }
    }
    lock (num, user) {
        if (this.parents[num].locked) return false
        this.parents[num].locked = user
        return true
    }
    unlock (num, user) {
        if (this.parents[num].locked !== user) return false
        this.parents[num].locked = false
        return true
    }
    upgrade (num, user) {
        if (this.parents[num].locked) return false
        let node = this.parents[num]
        let ancestor = node.parent
        while (ancestor) {
            if (ancestor.locked) return false
            ancestor = ancestor.parent
        }
        const isChildLocked = (root) => {
            if (!root) return false
            if (root.locked) return true
            if (root.child.length === 0) return false
            for (const node of root.child) {
                if (isChildLocked(node)) return true
            }
            return false
        }
        if (!isChildLocked(node)) return false
        const unlockChild = (root) => {
            if (!root) return
            root.locked = false
            if (root.child.length === 0) return
            for (const node of root.child) {
                unlockChild(node)
            }
        }
        unlockChild(node);
        node.locked = user;
        return true
    }
};


// 513. find bottom left value

var findBottomLeftValue = function(root) {
    let ans = 0
    let count = -1
    const helpFind = (node, depth) => {
        if (!node) return false
        if (count < depth) {
            ans = node.val
            count = depth
        }
        depth++
        helpFind(node.left, depth)
        helpFind(node.right, depth)
    }
    helpFind(root, 0)
    return ans
};

// 669. trim a binary search tree

var trimBST = function(root, low, high) {
    if (!root) return root
    if (root.val < low) {
        return trimBST(root.right, low, high)
    }
    if (high < root.val) {
        return trimBST(root.left, low, high)
    }
    
    root.left = trimBST(root.left, low, high)
    root.right = trimBST(root.right, low, high)
    return root
};

// 538. convert bst to greater tree
var convertBST = function(root) {
    let sum = 0
    const helpTravel = (root) => {
        if (!root) return
        helpTravel(root.right)
        sum += root.val
        root.val = sum
        helpTravel(root.left)
        
    }
    helpTravel(root)
    return root
};


// 1834. single-threaded cpu
// use heap to keep track of element with the smallest processing time and smallest index
// if we reach a point where we need to idle we set our current time to the next soonest element
var getOrder = function(tasks) {
    tasks = tasks.map((a, index) =>  [...a, index])
    tasks.sort((a,b) => b[0] - a[0])
    let ans = []
    let minPrio = new MinPriorityQueue({compare: (a, b) => a[1] - b[1] || a[2] - b[2]})
    let p = tasks.length - 1
    let currTime = tasks[p][0]

    while (tasks.length || minPrio.size()) {
        while (tasks.length && tasks[p][0] <= currTime) {
            minPrio.enqueue(tasks.pop())
            p--
        }
        
        if (minPrio.size() === 0) {
            currTime = tasks[p][0]
        } else {
            let next = minPrio.dequeue()
            ans.push(next[2])
            currTime += next[1]
        }
       
    }
    return ans
};


// 1845. seat reservation manager
// only need the prio queue if we unreserve a seat
// otherwise we can just process the seats in order 1,2,3,4,5....
var SeatManager = class{ 
    constructor(n) {
        this.previousSeats = new MinPriorityQueue({compare: (a,b) => a-b})
        this.currSeat = 1 
    }
    reserve () {
        if (this.previousSeats.size()) {
            return this.previousSeats.dequeue()
        } else {
            return this.currSeat++
        }
    }
    unreserve (seatNumber) {
        this.previousSeats.enqueue(seatNumber)
    }
};



// 1882. process tasks using servers
// idle servers are processed using weight then index
// active servers are processed using time then weight then index
// if we dont have any idle servers we pull from active server
var assignTasks = function(servers, tasks) {
    servers = servers.map((ele, index) => [ele, index, 0])
    let idleServers = new MinPriorityQueue({compare: (a,b) => a[0] - b[0] || a[1] - b[1]})
    let activeServers = new MinPriorityQueue({compare: (a,b) => a[2] - b[2] || a[0] - b[0] || a[1] - b[1]})
    let tasksP = 0
    let ans = []
    for (const server of servers) {
        idleServers.enqueue(server)
    }
    while (tasksP < tasks.length) {
        while (activeServers.size() && activeServers.front()[2] <= tasksP) {
            idleServers.enqueue(activeServers.dequeue())
        }
        
        let nextServer = idleServers.dequeue() || activeServers.dequeue()
        let nextTask = tasks[tasksP]
        nextServer[2] = Math.max(nextServer[2], tasksP) + nextTask
        ans.push(nextServer[1])
        activeServers.enqueue(nextServer)
        tasksP++
    }
    return ans
};


// 1985. find the kth largest integer in the array
// using max prio queue
// faster solution uses quick select, grab an element, check how many elements are smaller if its nums.length - k then we found our answer
var kthLargestNumber = function(nums, k) {
    let mp = new MaxPriorityQueue({compare: (b,a) => {
        if (a.length - b.length) return a.length - b.length
        else {
            let i = 0 
            while (i < a.length && a[i] === b[i]) {
                i++
            }
            return a[i] - b[i] || 0
        }
    }})
    for (const num of nums) {
        mp.enqueue(num)
    }
    while (1 < k) {
        mp.dequeue()
        k--
    }
    return mp.dequeue()
};


// 767. reorganize string
// o(n) time
// find longest character and place it on as many even index as we can 
// then go to every other index and place the next character
// consider 'aaabbbc' => a:3 b:3 c:1
// take a as longests => [a,/,a,/,a,/,/]
// => next b or c does not matter => [a,/,a,/,a,/,c]
// => [a,b,a,b,a,b,c]
var reorganizeString = function(s) {
    let max = -1
    let freq = {}
    for (const c of s) {
        freq[c] = (freq[c] || 0) + 1
        if (max === -1 || freq[max] < freq[c]) {
            max = c
        }
    }
    if ((s.length+1)/2 < freq[max]) return ''
    let str = Array(s.length)
    let idx = 0
    while (freq[max]) {
        str[idx] = max
        freq[max]--
        idx += 2
    }
    for (const key of Object.keys(freq)) {
        while (freq[key]) {
            if (s.length <= idx) {
                idx = 1
            }
            str[idx] = key
            freq[key]--
            idx += 2
        }
    }
    return str.join('')
};


// using maxPrio queue
// takes the most frequent letter and adds it to the string 
// if the last letter is the same as the most frequent we must pop an additional element
var reorganizeString = function(s) {
    let mp = new MaxPriorityQueue({priority: (a) => a[1]})
    let freq = {}
    for (const c of s) {
        freq[c] = (freq[c] || 0) + 1
    }
    for (const [key, value] of Object.entries(freq)) {
        mp.enqueue([key, value])
    }
    let str = ''
    while (mp.size()) {
        let maxLetter = mp.dequeue().element[0]
        freq[maxLetter]--
        if (str[str.length - 1] === maxLetter) {
            if (mp.size() === 0) return ""
            let filler = mp.dequeue().element[0]
            str += filler
            freq[filler]--
            if (freq[filler]) {
                mp.enqueue([filler, freq[filler]])
            }
        }
        str += maxLetter
        if (freq[maxLetter]) {
            mp.enqueue([maxLetter, freq[maxLetter]])
        }
    }
    return str
}



// 1094 car pooling
// heap to keep track of the soonest destination
// process the trips in order of starting point 
// keep track of how many people we have in a car
// each location we check if we can remove people from the car
// then check if we have too many people in the car
var carPooling = function(trips, capacity) {
    trips.sort((a,b) => a[1] - b[1] || a[2] - b[2])
    let mp = new MinPriorityQueue({priority: (a) => a[2]})
    let people = 0
    let location = trips[0][1]
    let p = 0
    while (p < trips.length) {
        location = trips[p][1]
        while (mp.size() && mp.front().priority <= location) {
            people -= mp.dequeue().element[0]
        }
        people += trips[p][0]
        if (capacity < people) return false
        mp.enqueue(trips[p])
        p++
    }
    return true
};



// 1405. longest happy string
var longestDiverseString = function(a, b, c) {
    let str = ''
    let freq = {'a': a, 'b': b, 'c':c}
    while (freq['a'] || freq['b'] || freq['c']) {
        let x = returnHighest(freq['a'], freq['b'], freq['c'])
        if (str.length && str[str.length - 1] === x[0]) {
            if (freq[x[1]] === 0) return str
            freq[x[1]]--
            str += x[1]
        }
        if (freq[x[0]] < 2) {
            freq[x[0]]--
            str += x[0]
        } else {
            freq[x[0]] -= 2
            str += x[0]
            str += x[0]
        }
    }
    return str
};

// manually checking to avoid having to sort, not sure if this saves a lot of time since we would only be sorting 3 elements 
var returnHighest = (a,b,c) => {
    if (b <= a && c <= a) {
        if (c <= b) return ['a', 'b', 'c']
        else return ['a', 'c', 'b']
    }
    if (a <= b && c <= b) {
        if (c <= a) return ['b', 'a', 'c']
        else return ['b', 'c', 'a']
    }
    else {
        if (b <= a) return ['c', 'a', 'b']
        else return ['c', 'b', 'a']
    }
}



// 77. combinations

var combine = function(n, k) {
    let ans = []
    
    const helpCombine = (l, array, k) => {
        if (k === 0) {
            ans.push([...array])
            return 
        }
        for (let i = l; i <= n-k+1; i++) {
            
            array.push(i)
            helpCombine(i+1, array, k-1)
            array.pop()
        }
    }
    helpCombine(1, [], k)
    return ans
};


// 47. permuations ii 
// use nums[i] === nums[i+1] to prevent uniques
// nums[i] = 'x' to prevent using the same value twice
// 47. permuations ii 
// use nums[i] === nums[i+1] to prevent uniques
// nums[i] = 'x' to prevent using the same value twice
var permuteUnique = function(nums) {
    nums.sort((a,b) => a-b)
    let ans = []
    const helpPermute = (arr) => {
        if (arr.length === nums.length) {
            ans.push([...arr])
            return
        }
        for (let i = 0; i < nums.length; i++)  {
            if (nums[i] === 'X' || nums[i] === nums[i+1]) continue
            let num = nums[i]
            nums[i] = 'X'
            arr.push(num)
            helpPermute(arr)
            nums[i] = num
            arr.pop()
        }
    }
    helpPermute([])
    return ans
};


// 93. restore ip addresses
// iterate from the back trying possible values 
var restoreIpAddresses = function(s) {
    let ans = []
    let n = s.length
    
    const helpRestore = (index, arr, k) => {
        if (index < 0 && k === 0) {
            ans.push([...arr].reverse().join('.'))
        } 
        
        if (index < 0 || k === 0) return
        let str = ''
        
        for (let j = index; 0 <= j; j--) {
            if (str.length === 3) break
            str = s[j] + str
            if (s[j] == 0 && str.length !== 1) continue
            if (str.length === 3 && (2 < s[j] || 255 < str)) continue
            arr.push(str)
            helpRestore(j - 1, arr, k-1)
            arr.pop()
        }
    }
    helpRestore(n - 1, [], 4)
    return ans
};


// 473. matchsticks to square
// we only need to check if 3 sides equal 
var makesquare = function(matchsticks) {
    let sum = matchsticks.reduce((a,b) => a+b)
    matchsticks.sort((a,b) => a-b)
    if (sum%4 !== 0) return false
    let side = sum/4
    if (side < matchsticks.at(-1)) return false
    const trySquare = (a,b,c,d) => {
        if (side === a && a === b && a === c) return true
        if (side < a || side < b || side < c || side < d || matchsticks.length === 0) return false
        let value = matchsticks.pop()
        if (trySquare(a+value,b,c,d)) return true
        if (trySquare(a,b+value,c,d)) return true
        if (trySquare(a,b,c+value,d)) return true
        if (trySquare(a,b,c,d+value)) return true
        matchsticks.push(value)
        return false
    }
    let x = matchsticks.pop()
    let y = matchsticks.pop()
    return trySquare(x+y,0,0,0) || trySquare(x,y,0,0)
};
// building each side naturally instead of trying all possiblities
var makesquare = function(matchsticks) {
    let sum = matchsticks.reduce((a,b) => a+b)
    matchsticks.sort((a,b) => b-a)
    if (sum%4 !== 0) return false
    let side = sum/4
    if (side < matchsticks[0]) return false
    const trySquare = (i, space, k) => {
        if (k === 3) return true
        for (; i < matchsticks.length; i++) {
            let val = matchsticks[i]
            let j = i
            if (space < val) continue
            
            matchsticks[j] = side + 1
            if (val === space) {
                if (trySquare(1, side, k+1)) return true
            } else {
                if (trySquare(i+1, space - val, k)) return true
            }
            matchsticks[j] = val
            while(val === matchsticks[i+1]) {
                i++
            }
        }
        return false
    }
    return trySquare(0, side, 0)
};


// 835. image overlap
// get every [i,j] pair with value 1
// then compare every pair from img1 with every pair in img2 storing the [p,q] shift we would need to make img1 pair line up with img2 pair
var largestOverlap = function(img1, img2) {
    let LA = []
    let RA = []
    let n = img1.length 
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            if (img1[i][j] === 1) LA.push([i,j])
            if (img2[i][j] === 1) RA.push([i,j])
        }
    }
    let counts = {}
    let res = 0
    let str
    let val
    for (let i of LA) {
        for (let j of RA) {
            str = `${i[0]-j[0]},${i[1]-j[1]}`
            val = counts[str] = (counts[str] || 0) + 1
            res = Math.max(val, res)
        }
    }
    return res 
};