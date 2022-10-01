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
// if our heap is full, check if the farthest element from the origin
// is closer than our next point

 var kClosest = function(points, k) {
    let x = new MaxPriorityQueue({priority: (num) => dist(num)})
    
    
    for (let point of points) {
        
        if (x.size() < k ) {
            x.enqueue(point)
        } else {
            if (dist(point) < dist(x.front()["element"])) {
                x.dequeue()
                x.enqueue(point)
            }
        }
    }
    return x.toArray().map((hash) => hash["element"])
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

/** 
 * Your Trie object will be instantiated and called as such:
 * var obj = new Trie()
 * obj.insert(word)
 * var param_2 = obj.search(word)
 * var param_3 = obj.startsWith(prefix)
 */
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
    let count = 0
    
    let [leftMin, rightMin] = [-1, s.length]
    
    let [leftW, rightW] = [0, 0]
    
    
    for (const letter of t) {
        if (!letterFreq[letter]) {
            letterFreq[letter] = 0
        }
        letterFreq[letter]++
        count++
    }
    
    
    
    while (rightW < s.length) {
        
        if (t.includes(s[rightW])) {
            letterFreq[s[rightW]]--
            if (0 <= letterFreq[s[rightW]]) {
                count--
                while (count===0) {
                    if (t.includes(s[leftW])) {
                        letterFreq[s[leftW]]++
                        if (0 < letterFreq[s[leftW]]) {
                            count++
                            if ((rightW - leftW) < (rightMin - leftMin)) {
                                
                                [rightMin, leftMin] = [rightW, leftW]
                            }
                        }
                    }
                    leftW++
                }
            }
        }
        rightW++
    }
    if (leftMin === -1) return ""
    return s.substring(leftMin, rightMin+1)
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

// turned the word into an array because thats how i usually remember
// this way is more condense and clear
// let altWord = word.substring(0, i) + letter + word.substring(i + 1);
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
                let wordArray = [...word]
                for (const letter of alphabet) {
                    wordArray[i] = letter
                    let altWord = wordArray.join('')
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


var MyCircularQueue = function(k) {
    this.max = k
    this.size = 0
    this.head
};

/** 
 * @param {number} value
 * @return {boolean}
 */
MyCircularQueue.prototype.enQueue = function(value) {
    if (this.size === this.max) {
        return false
    }
    let node = new Node(value)
    this.size ++
    if (this.size === 1) {
        this.head = node
        this.head.next = node
        this.head.prev = node
    } else {
        node.prev = this.head.prev
        node.next = this.head
        this.head.prev.next = node
        this.head.prev = node
    }
    
    return true
};

/**
 * @return {boolean}
 */
MyCircularQueue.prototype.deQueue = function() {
    if (this.size === 0) {
        return false
    }
    
    this.size--
    let node = this.head
    node.next.prev = node.prev
    node.prev.next = node.next
    this.head = node.next
    
    return true
};

/**
 * @return {number}
 */
MyCircularQueue.prototype.Front = function() {
    if (this.isEmpty()) return -1
    return this.head.val
};

/**
 * @return {number}
 */
MyCircularQueue.prototype.Rear = function() {
    if (this.isEmpty()) return -1
    return this.head.prev.val
};

/**
 * @return {boolean}
 */
MyCircularQueue.prototype.isEmpty = function() {
    return (!this.size)
};

/**
 * @return {boolean}
 */
MyCircularQueue.prototype.isFull = function() {
    return (this.size === this.max)
};


class Node {
    constructor(val) {
        this.next = null
        this.prev = null
        this.val =  val
    }
}

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
 