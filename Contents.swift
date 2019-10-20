import Foundation
let arr = [1,1,2,3,4,4,45]
let set = Set(arr)
set.sorted()

let str = "word"
let chars = Array(str)

func longestConsecutive(_ nums: [Int]) -> Int {
    let set = Set(nums)
    var longestLen = 0
    for num in set {
        if !set.contains(num - 1) {
            var digit = num
            var len = 1
            while set.contains(digit+1) {
                digit += 1
                len += 1
            }
            longestLen = max(longestLen, len)
        }
    }
    return longestLen
}
//longestConsecutive([100,1,200,4,2,3])
class TreeNode {
    var val: Int
    var left: TreeNode?
    var right: TreeNode?
    init(_ val: Int) {
        self.val = val
    }
}

func sumNumbers(_ root: TreeNode?) -> Int {
    var paths = [[Int]]()
    func dfs(_ node2: TreeNode?, _ tmp: [Int]) {
        guard let node = node2, (node.left != nil || node.right != nil) else {
            if let _node = node2 {
                paths.append(tmp)
                paths[paths.count-1].append(_node.val)
            }
            return
        }
        var tmp2 = tmp
        tmp2.append(node.val)
        if node.left != nil {dfs(node.left, tmp2)}
        if node.right != nil {dfs(node.right, tmp2)}
    }
    dfs(root, [])
    var ans = 0
    for path in paths {
        ans += path.reduce(0, { (result, newNum)  in
            return result * 10 + newNum
        })
    }
    return ans
}

//130、被环绕的区域的"O"替换为“X”，与边界连接的O不需要
func solve(_ board: inout [[Character]]) {
    let rows = board.count
    let columns = board[0].count
    if rows == 0 {return}
    func dfs(_ board: inout [[Character]], _ i: Int, _ j: Int) {
        if i < 0 || j < 0 || i >= rows || j >= columns || board[i][j] == "X" || board[i][j] == "#" {
            return
        }
        board[i][j] = "#"
        dfs(&board, i, j+1)
        dfs(&board, i+1, j)
        dfs(&board, i, j-1)
        dfs(&board, i-1, j)
    }

    for i in 0..<rows {
        for j in 0..<columns {
            let isEdge = i==0 || j==0 || i==rows-1 || j==columns-1
            if isEdge && board[i][j] == "O" {
                dfs(&board, i, j)
            }
        }
    }
    for i in 0..<rows {
        for j in 0..<columns {
            if board[i][j] == "#" {
                board[i][j] = "O"
            } else if board[i][j] == "O" {
                board[i][j] = "X"
            }
        }
    }
}

//找并集
class UnionFind { //连通区域的每个点都有
    public var parents: [Int]
    init(totalNodes: Int) {
        self.parents = Array(0..<totalNodes)
    }
    
    public func find(_ node: Int) -> Int {
        var node = node
        while parents[node] != node {
            parents[node] = parents[parents[node]]
            node = parents[node]
        }
        return node
    }
    
    public func union(_ node1: Int, _ node2: Int) {
        let root1 = find(node1)
        let root2 = find(node2)
        if root1 != root2 {
            parents[root2] = root1
        }
    }
    
    public func isConnected(_ node1: Int, _ node2: Int) -> Bool {
        return find(node1) == find(node2)
    }
}

func solve2(_ board: inout [[Character]]) {
    if board.count == 0 { return }
    let rows = board.count
    let columns = board[0].count
    let uf = UnionFind(totalNodes: rows * columns + 1)
    let dummyNode = rows * columns
    func node(_ i: Int, _ j: Int) -> Int {
        return i * columns + j
    }
    for i in 0..<rows {
        for j in 0..<columns {
            if board[i][j] == "O" {
                if i == 0 || i == rows - 1 || j == 0 || j == columns - 1 {
                    uf.union(node(i, j), dummyNode)
                } else {
                    if i > 0 && board[i-1][j] == "O" {
                        uf.union(node(i, j), node(i-1, j))
                    }
                    if i < rows - 1 && board[i+1][j] == "O" {
                        uf.union(node(i, j), node(i+1, j))
                    }
                    if j > 0 && board[i][j-1] == "O" {
                        uf.union(node(i,j), node(i, j-1))
                    }
                    if j < columns - 1 && board[i][j+1] == "O" {
                        uf.union(node(i, j), node(i, j+1))
                    }
                }
            }
        }
    }
    for i in 0..<rows {
        for j in 0..<columns {
            if uf.isConnected(node(i, j), dummyNode) {
                board[i][j] = "O"
            } else {
                board[i][j] = "X"
            }
        }
    }
    print(uf.parents)
}
//var board: [[Character]] = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
//solve2(&board)
//print(board)

func partition(_ s: String) -> [[String]] {
    var ans = [[String]]()
    var tmp = [String]()
    func dfs(_ nowPos: Int, _ tmp: inout [String]) {
        if nowPos == s.count  {
            ans.append(tmp)
            return
        }
        for len in 1...s.count-nowPos {
            let str = (s as NSString).substring(with: NSRange(location: nowPos, length: len))
            if str == String(str.reversed()) {
                tmp.append(str)
                dfs(nowPos+len, &tmp)
                tmp.removeLast()
            }
        }
    }
    dfs(0, &tmp)
    return ans
}
//print(partition("aaba"))

class Node: Hashable, CustomStringConvertible {
    static func == (lhs: Node, rhs: Node) -> Bool {
        return lhs.val == rhs.val && lhs.neighbors == rhs.neighbors
    }
    
    public var val: Int
    public var neighbors = [Node?]()
    init(_ val: Int) {
        self.val = val
    }
    func hash(into hasher: inout Hasher) {
        hasher.combine(val)
    }
    var description: String {
        var str = "\(val): "
        for i in neighbors {
            str += "\((i?.val) ?? -1) "
        }
        return str
    }
}
func cloneGraph(_ node: Node?) -> Node? {
    var lookup = [Node: Node]()
    func dfs(_ node: Node?) -> Node? {
        guard let node = node else { return nil }
        if let target = lookup[node] {
            return target
        }
        let clone = Node(node.val)
        lookup[node] = clone
        for i in node.neighbors {
            clone.neighbors.append(dfs(i))
        }
        return clone
    }
    return dfs(node)
}

func cloneGraphByBfs(_ node: Node?) -> Node? {
    guard let node = node else { return nil }
    var lookup = [Node: Node]()
    let clone = Node(node.val)
    lookup[node] = clone
    var queue = [node]
    while !queue.isEmpty {
        let tmp = queue.removeFirst()
        for n in tmp.neighbors where n != nil {
            if let n = n {
                if lookup[n] == nil {
                    lookup[n] = Node(n.val)
                    queue.append(n)
                }
                lookup[tmp]!.neighbors.append(lookup[n])
            }
        }
    }
    return clone
}

let node1 = Node(1)
let node2 = Node(2)
let node3 = Node(3)
let node4 = Node(4)
node1.neighbors = [node2, node4]
node4.neighbors = [node1, node3]
node2.neighbors = [node1, node3]
node3.neighbors = [node2, node4]
//let clonedNode = cloneGraphByBfs(node1)
//let _node2 = clonedNode?.neighbors[0]

func canCompleteCircuit(_ gas: [Int], _ cost: [Int]) -> Int {
    var sum = 0
    var total = 0
    var start = 0
    for i in 0..<gas.count {
        sum += gas[i] - cost[i]
        total += gas[i] - cost[i]
        if sum < 0 {
            sum = 0
            start = i + 1
        }
    }
    return total >= 0 ? start : -1
}
//canCompleteCircuit([5,1,2,3,4], [4,4,1,5,1])

func candy(_ ratings: [Int]) -> Int {
    var candies = [Int](repeating: 1, count: ratings.count)
    var _candies = candies
    for i in 1..<ratings.count {
        if ratings[i] > ratings[i-1] {
            candies[i] = candies[i-1] + 1
        }
    }
    for i in stride(from: ratings.count-1, through: 1, by: -1) {
        if ratings[i-1] > ratings[i] {
            _candies[i-1] = candies[i] + 1
        }
    }
    var ans = 0
    for i in 0..<candies.count {
        ans += max(candies[i], _candies[i])
    }
    return ans
}
//candy([1,3,2,2,1])
func singleNumber(_ nums: [Int]) -> Int {
    return nums.reduce(0, ^)
}
//print(singleNumber([4,1,2,1,2]))

//func singleNumber2(_ nums: [Int]) -> Int {
//
//}

func postOrder(_ root: TreeNode?) -> [Int] {
    guard let root = root else {
        return []
    }
    var stack: [TreeNode] = [root]
    var ans = [Int]()
    while !stack.isEmpty {
        let node = stack.removeLast()
        ans.insert(node.val, at: 0)
        if node.left != nil {
            stack.append(node.left!)
        }
        if node.right != nil {
            stack.append(node.right!)
        }
    }
    return ans
}
let root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left?.left = TreeNode(4)
root.left?.right = TreeNode(5)
//print(postOrder(root))

func singleNumber2(_ nums: [Int]) -> Int {
    let nums = nums.map{Int32($0)}
    var bitNum = [Int](repeating: 0, count: 32)
    for i in 0..<nums.count {
        var mask: Int32 = 1
        let num = nums[i]
        for j in 0...31 {
            if mask & num != 0 {
                bitNum[j] += 1
            }
            mask <<= 1
        }
    }
    var mask: Int32 = 1
    var res:Int32 = 0
    for i in 0...31 {
        if bitNum[i] % 3 != 0 {
            res ^= mask
        }
        mask <<= 1
    }
    
    return Int(res)
}
//print(singleNumber2([-2,-2,-1,-1,-3,-1,-3,-3,-4,-2]))

func candidates(_ firstChar: Character, words: inout Set<String>) -> [String] {
    var ans = [String]()
    for word in words {
        if word.first! == firstChar {
            ans.append(word)
        }
    }
    return ans
}

func wordBreak(_ s: String, _ wordDict: [String]) -> Bool {//BFS
    let words = Set(wordDict)
    let str = s as NSString
    var queue = [Int]()
    var visited = [Bool](repeating: false, count: s.count)
    queue.append(0)
    while !queue.isEmpty {
        let start = queue.removeFirst()
        if !visited[start] {
            for end in start...s.count {
                if words.contains(str.substring(with: NSRange(start..<end))) {
                    queue.append(end)
                    if end == s.count {
                        return true
                    }
                }
            }
            visited[start] = true
        }
    }
    return false
}

func wordBreak(_ s: String, _ wordDict: [String]) -> [String] {
    let words = Set(wordDict)
    let str = s as NSString
    var map = [Int: [String]]()
    func helper(_ start: Int) -> [String] {
        if let res = map[start] {
            return res
        }
        var res = [String]()
        if start == s.count {
            res.append("")
            return res
        }
        for end in start+1...s.count {
            if words.contains(str.substring(with: NSRange(start..<end))) {
                let list = helper(end)
                for i in list {
                    res.append(str.substring(with: NSRange(start..<end)) + (i == "" ? "" : " ") + i)
                }
            }
        }
        map[start] = res
        return res
    }
    let ans = helper(0)
    print(map)
    return ans
}
//let ans: [String] = wordBreak("pineapplepenapple", ["apple", "pen", "applepen", "pine", "pineapple"])
//print(ans)

class ListNode: Hashable, CustomStringConvertible {
    var description: String {
        return String(self.val)
    }
    static var index = 0
    var _index = 0
    var val: Int
    var next: ListNode?
    init(_ val: Int) {
        self.val = val
        ListNode.index += 1
        _index = ListNode.index
    }
    
    func hash(into hasher: inout Hasher) {
        hasher.combine(_index)
    }
    
    static func == (lhs: ListNode, rhs: ListNode) -> Bool {
        return lhs._index == rhs._index
    }
}

func hasCycle(_ head: ListNode?) -> Bool {
    if head == nil { return false }
    var qPointer: ListNode? = head?.next
    var sPointer: ListNode? = head
    while sPointer !== qPointer {
        qPointer = qPointer?.next?.next
        sPointer = sPointer?.next
        if qPointer == nil || qPointer?.next == nil {
            return false
        }
    }
    return true
}

func detectCycle(_ head: ListNode?) -> ListNode? {
    if head == nil { return head }
    var set: Set<ListNode?> = Set()
    var pointer = head
    while pointer != nil {
        if set.contains(pointer!) {
            return pointer
        }
        set.insert(pointer)
        pointer = pointer?.next
    }
    return nil
}
//print(detectCycle(lNode1)?.val)

//画图，第一步先找出快慢指针相遇的点，第二步两个慢指针，一个指向head一个指向相遇点 d(fast)=2d(slow)
//a + b + c + b = 2(a + b)     a = c
//所以步调一致后会指向入口点
func detectCycle2(_ head: ListNode?) -> ListNode? {
    if head == nil { return nil }
    func getInterset() -> ListNode? {
        var fast = head
        var slow = head
        while fast != nil && fast?.next != nil {
            fast = fast?.next?.next
            slow = slow?.next
            if fast === slow {
                return fast
            }
        }
        return nil
    }
    guard let interset = getInterset() else { return nil }
    var ptr1 = head
    var ptr2: ListNode? = interset
    while ptr1 != ptr2 {
        ptr1 = ptr1?.next
        ptr2 = ptr2?.next
    }
    return ptr1
}
//print(detectCycle2(lNode1)?.val)

func reorderList(_ head: ListNode?) -> ListNode? {
    if head == nil || head?.next == nil { return nil }
    var queue = [ListNode]()
    var stack = [ListNode]()
    var ptr: ListNode? = head
    var len = 0
    while ptr != nil {
        len += 1
        ptr = ptr?.next
    }
    ptr = head
    for _ in 0..<len/2 {
        queue.append(ptr!)
        ptr = ptr?.next
    }
    if len % 2 != 0 {
        queue.append(ptr!)
        ptr = ptr?.next
    }
    for _ in (len%2==0 ? len/2 : len/2+1)..<len {
        stack.append(ptr!)
        ptr = ptr?.next
    }
    var p: ListNode? = ListNode(-1)
    let newHead = p
    for _ in 0..<len/2 {
        let nodeOfQueue = queue.removeFirst()
        let nodeOfStack = stack.removeLast()
        p?.next = nodeOfQueue
        nodeOfQueue.next = nodeOfStack
        p = nodeOfStack
    }
    if len % 2 != 0 {
        p?.next = queue.removeFirst()
        p = p?.next
    }
    p?.next = nil
    return newHead?.next
}
//var ans = reorderList(lNode1)

func preorderTraversal(_ root: TreeNode?) -> [Int] {
    guard let root = root else { return [] }
    var ans = [Int]()
    var arr: [TreeNode] = [root]
    while !arr.isEmpty {
        let node = arr.removeLast()
        ans.append(node.val)
        if node.right != nil {
            arr.append(node.right!)
        }
        if node.left != nil {
            arr.append(node.left!)
        }
    }
    return ans
}

class LRUCache {
    private var dict: [Int: (Int, Int)]
    private let capacity: Int
    init(_ capacity: Int) {
        dict = [:]
        self.capacity = capacity
        self.dict.reserveCapacity(capacity)
    }
    
    func get(_ key: Int) -> Int {
        dict.forEach{dict[$0.key]!.1 += 1}
        if let value = dict[key] {
            dict[key]!.1 = 0
            return value.0
        } else {
            return -1
        }
    }
    
    func put(_ key: Int, _ value: Int) {
        guard capacity > 0 else { return }
        dict.forEach{dict[$0.key]!.1 += 1}
        if dict[key] != nil {
            dict.updateValue((value, 0), forKey: key)
        } else if dict.count < capacity {
            dict[key] = (value, 0)
        } else {
            let target = dict.max(by: {$0.value.1<$1.value.1})
            let index = dict.index(forKey: target!.key)
            dict.remove(at: index!)
            dict[key] = (value, 0)
        }
    }
}

class BListNode {
    var key: Int?
    var value: Int?
    var pre: BListNode?
    var next: BListNode?
}

class LRUCache2 {
    private let capacity: Int
    private var hashmap: [Int: BListNode] = [:]
    private let head: BListNode
    private let tail: BListNode
    
    init(_ capacity: Int) {
        self.capacity = capacity
        head = BListNode()
        tail = BListNode()
        head.next = tail
        tail.pre = head
    }
    
    func moveNodeToTail(_ key: Int) {
        let node = hashmap[key]
        node?.pre?.next = node?.next
        node?.next?.pre = node?.pre
        
        node?.pre = tail.pre
        node?.next = tail
        tail.pre?.next = node
        tail.pre = node
    }
    
    func get(_ key: Int) -> Int {
        if let value = hashmap[key] {
            moveNodeToTail(key)
            return value.value!
        } else {
            return -1
        }
    }
    
    func put(_ key: Int, _ value: Int) {
        if hashmap[key] != nil {
            hashmap[key]!.value = value
            moveNodeToTail(key)
        } else {
            if hashmap.count == capacity {
                hashmap.removeValue(forKey: head.next!.key!)
                head.next = head.next?.next
                head.next?.pre = head
            }
            let new = BListNode()
            new.key = key
            new.value = value
            hashmap[key] = new
            new.pre = tail.pre
            new.next = tail
            tail.pre?.next = new
            tail.pre = new
        }
    }
}

//let lruCache = LRUCache2(2)
//lruCache.put(1, 1)
//lruCache.put(2, 2)
//lruCache.get(1)
//lruCache.put(3, 3)
//lruCache.get(2)
//lruCache.put(4, 4)
//lruCache.get(1)
//lruCache.get(3)
//lruCache.get(4)
//

func insertionSortList(_ head: ListNode?) -> ListNode? {
    guard head != nil && head?.next != nil else { return head }
    let h = ListNode(-1)
    h.next = head
    var pre: ListNode? = h
    var cur = head
    var lat: ListNode?
    while cur != nil {
        lat = cur?.next
        if lat != nil && lat!.val < cur!.val {
            while pre?.next != nil && pre!.next!.val < lat!.val {
                pre = pre?.next
            }
            let tmp = pre?.next
            pre?.next = lat
            cur?.next = lat?.next
            lat?.next = tmp
            pre = h
        } else {
            cur = lat
        }
    }
    return h.next
}
let lNode1 = ListNode(4)
let lNode2 = ListNode(3)
let lNode3 = ListNode(5)
let lNode4 = ListNode(2)
let lNode5 = ListNode(1)
lNode1.next = lNode2
lNode2.next = lNode3
lNode3.next = lNode4
lNode4.next = lNode5

//var ans2 = insertionSortList(lNode1)
//while ans2 != nil {
//    print(ans2!.val)
//    ans2 = ans2?.next
//}
//mergeSortWithList
func sortList(_ head: ListNode?) -> ListNode? {
    if head == nil || head?.next == nil { return head }
    var slow = head //reach the middle of the linklist
    var fast = head?.next
    while fast != nil && fast?.next != nil {
        slow = slow?.next
        fast = fast?.next?.next
    }
    let tmp = slow?.next
    slow?.next = nil
    var left = sortList(head)
    var right = sortList(tmp)
    var h: ListNode? = ListNode(0)
    let res = h
    while left != nil && right != nil {
        if left!.val < right!.val {
            h?.next = left
            left = left?.next
        } else {
            h?.next = right
            right = right?.next
        }
        h = h?.next
    }
    h?.next = left != nil ? left : right
    return res?.next
}

func maxPoints(_ points: [[Int]]) -> Int {
    if points.count <= 2 { return points.count }
    var maxLen = 2
    for i in 0..<points.count - 1 {
        for j in i + 1 ..< points.count {
            var len = 0
            let x1 = points[i][0]
            let y1 = points[i][1]
            var x2 = points[j][0]
            var y2 = points[j][1]
            if x1 == x2 && y1 == y2 {//important如果两个点重叠，不作处理的话那下面一步操作会导致所有点都在直线上（因为斜率可以当做是任意值
                x2 += 1
                y2 += 1
            }
            for point in points {
                let x = point[0]
                let y = point[1]
                if (y-y1)*(x2-x1) == (x-x1)*(y2-y1) {
                    len += 1
                }
            }
            maxLen = max(maxLen, len)
        }
    }
    return maxLen
}

//print(maxPoints([[0,0],[0,0]]))
//print(6/(-132))
func evalRPN(_ tokens: [String]) -> Int {
    var stack: [Int] = []
    let set: [String: (Int, Int) -> Int] = ["+": {$0+$1}, "-": {$0-$1}, "*": {$0*$1}, "/": {$0/$1}]
    for str in tokens {
        if let op = set[str] {
            let a = stack.removeLast()
            let b = stack.removeLast()
            stack.append(op(b, a))
        } else {
            stack.append(Int(str)!)
        }
    }
    return stack.first!
}
//evalRPN(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"])

func reverseWords(_ s: String) -> String {
    let words = s.split(separator: " ")
    return words.reversed().joined(separator: " ")
}
//reverseWords("a good   example")

func maxProduct(_ nums: [Int]) -> Int {
    var imax = 1
    var imin = 1
    var ans = Int.min
    for num in nums {
        if num < 0 {
            (imax, imin) = (imin, imax)
        }
        imax = max(imax * num, num)
        imin = min(imin * num, num)
        
        ans = max(ans, imax)
    }
    return ans
}

func findMin(_ nums: [Int]) -> Int { //No duplicates
    if nums.last! >= nums.first! { return nums.first! }
    var low = 0
    var high = nums.count - 1
    var mid = (low + high) / 2
    while low < high {
        mid = (low + high) / 2
        if nums[mid] > nums[high] {
            low = mid + 1
        } else if nums[mid] < nums[high] {
            high = mid
        }
    }
    return nums[low]
}

//print(findMin([6,7,8,12,15,-14,-1,2,4]))

func findMin2(_ nums: [Int]) -> Int { //有重复元素
    var low = 0
    var high = nums.count - 1
    while low < high {
        let mid = (low + high) / 2
        if nums[mid] > nums[high] {
            low = mid + 1
        } else if nums[mid] < nums[high] {
            high = mid
        } else {
            high -= 1
        }
    }
    return nums[low]
}
//print(findMin2([10,1,10,10,10]))

class MinStack {
    private var stack = [Int]()
    private var helper = [Int]()
    
    func push(_ x: Int) {
        stack.append(x)
        if helper.count == 0 || helper.last! >= x {
            helper.append(x)
        } else {
            helper.append(helper.last!)
        }
    }
    
    func pop() {
        if !stack.isEmpty {
            stack.removeLast()
            helper.removeLast()
        }
    }
    
    func top() -> Int {
        guard !stack.isEmpty else { return -1 }
        return stack.last!
    }
    
    func getMin() -> Int {
        guard !helper.isEmpty else { return -1 }
        return helper.last!
    }
}

func findPeakElement(_ nums: [Int]) -> Int { //可以假设nums[-1]=nums[n]=-∞, 找山峰，当处于上坡，则坡顶在右边，反之在左边
    var left = 0
    var right = nums.count - 1
    while left < right {
        let mid = (left + right) / 2
        if nums[mid] > nums[mid + 1] {
            right = mid
        } else {
            left = mid + 1
        }
    }
    return left
}

func baseSort(_ nums: [Int]) -> [Int] {
    guard nums.count > 1 else { return nums }
    var nums = nums
    var exp = 1
    let maxNum = nums.max()!
    var aux = Array(repeating: 0, count: nums.count)
    while maxNum / exp > 0 {
        var count = Array(repeating: 0, count: 10)
        
        for i in 0..<nums.count {
            count[(nums[i] / exp) % 10] += 1
        }
        
        for i in 1..<10 {
            count[i] += count[i-1]
        }
        for i in stride(from: nums.count-1, through: 0, by: -1) {
            let index = (nums[i] / exp) % 10
            count[index] -= 1
            aux[count[index]] = nums[i]
        }
        for i in 0..<nums.count {
            nums[i] = aux[i]
        }
        exp *= 10
    }
    return nums
}

//print(baseSort([13,26,37,8,15,44,16,21]))
func compareVersion(_ version1: String, _ version2: String) -> Int {
    var v1 = version1.split(separator: ".").map{Int($0)!}
    var v2 = version2.split(separator: ".").map{Int($0)!}
    if v1.count > v2.count {
        v2.append(contentsOf: Array(repeating: 0, count: v1.count-v2.count))
    } else if v1.count < v2.count {
        v1.append(contentsOf: Array(repeating: 0, count: v2.count-v1.count))
    }
    for i in 0..<v1.count {
        if v1[i] > v2[i] {
            return 1
        } else if v1[i] < v2[i] {
            return -1
        }
    }
    return 0
}
//print(compareVersion("7.5.2.4", "7.5.3"))

func fractionToDecimal(_ numerator: Int, _ denominator: Int) -> String {
    if numerator == 0 { return "0" }
    var fraction = ""
    if (numerator < 0 && denominator > 0) || (numerator > 0 && denominator < 0) {
        fraction += "-"
    }
    let numerator = abs(numerator)
    let denominator = abs(denominator)
    fraction += String(numerator / denominator)
    var remainder = numerator % denominator
    if remainder == 0 {
        return fraction
    }
    fraction += "."
    var map = [Int: Int]()
    while remainder != 0 {
        if let index = map[remainder] {
            fraction.insert("(", at: fraction.index(fraction.startIndex, offsetBy: index))
            fraction.append(contentsOf: ")")
            break
        }
        map[remainder] = fraction.count
        remainder *= 10
        fraction += String(remainder / denominator)
        remainder %= denominator
    }
    return fraction
}
//print(fractionToDecimal(1, 6))

func twoSum(_ numbers: [Int], _ target: Int) -> [Int] {
    var dict = [Int: Int]()
    for (index, num) in numbers.enumerated() {
        if let index1 = dict[target-num] {
            return [index1, index + 1]
        } else {
            dict[num] = index + 1
        }
    }
    return []
}
701 % 26
func convertToTitle(_ n: Int) -> String {
    let nums = "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26".split(separator: " ").map{Int($0)!}
    let chars = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".split(separator: " ").map{Character(String($0))}
    var dict = [Int: Character]()
    for i in 0..<nums.count {
        dict[nums[i]] = chars[i]
    }
    var a = n
    var b = 1
    var res = ""
    while a != 0 {
        let c = (Int(pow(Double(26), Double(b-1))))
        var index = (a / c) % 26
        if index == 0 { index = 26 }
        res = String(dict[index]!) + res
        a -= index * c
        b += 1
    }
    return res
}
//print(convertToTitle(731))

func majorityElement(_ nums: [Int]) -> Int {
    var count = 1
    var num = nums[0]
    for i in 1..<nums.count {
        if nums[i] == num {
            count += 1
        } else {
            count -= 1
        }
        if count < 0 {
            num = nums[i]
            count = 1
        }
    }
    return num
}

func titleToNumber(_ s: String) -> Int {
    let nums = "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26".split(separator: " ").map{Int($0)!}
    let chars = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".split(separator: " ").map{String($0)}
    var dict = [String: Int]()
    for i in 0..<nums.count {
        dict[chars[i]] = nums[i]
    }
    var ans = 0
    var i = 0
    for char in s.reversed() {
        let index = dict[String(char)]!
        ans += index * Int(pow(Double(26), Double(i)))
        i += 1
    }
    return ans
}
//print(titleToNumber("ZY"))

func trailingZeroes(_ n: Int) -> Int { //阶乘n!的结尾有多少个零，都是来自于5*2的贡献，计算有多少5即可，如25！，会有5，10，15，20，25，共5个，但是25还可以继续分为5*5，所以要再加上一个，总6个
    var count = 0
    var n = n
    while n >= 5 {
        n /= 5
        count += n
    }
    return count
}
//trailingZeroes(25)

class BSTIterator {
    
    var nums = [Int]()
    let root: TreeNode?
    static var ptr = 0

    init(_ root: TreeNode?) {
        self.root = root
        preTraversal(root)
    }
    
    func preTraversal(_ root: TreeNode?) {
        if root == nil { return }
        preTraversal(root?.left)
        nums.append(root!.val)
        preTraversal(root?.right)
    }
    
    /** @return the next smallest number */
    func next() -> Int {
        let index = BSTIterator.ptr
        BSTIterator.ptr += 1
        return nums[index]
    }
    
    /** @return whether we have a next smallest number */
    func hasNext() -> Bool {
        return BSTIterator.ptr < nums.count
    }
}

func calculateMinimumHP(_ dungeon: [[Int]]) -> Int {
    var ans = Int.min
    let rows = dungeon.count
    let columns = dungeon[0].count
    func helper(_ i: Int, _ j: Int, total: Int, tmp: Int) {
        if i == rows-1 && j == columns-1 {
            ans = max(ans, tmp)
            return
        }
        if tmp < ans { return }
        if i < rows - 1 && j < columns - 1 {
            helper(i + 1, j, total: total+dungeon[i+1][j], tmp: min(tmp, total+dungeon[i+1][j]))
            helper(i, j + 1, total: total+dungeon[i][j+1], tmp: min(tmp, total+dungeon[i][j+1]))
        } else if i == rows - 1 && j < columns - 1 {
            helper(i, j + 1, total: total+dungeon[i][j+1], tmp: min(tmp, total+dungeon[i][j+1]))
        } else if j == columns - 1 && i < rows - 1 {
            helper(i + 1, j, total: total+dungeon[i+1][j], tmp: min(tmp, total+dungeon[i+1][j]))
        }
        print((i,j,tmp))
    }
    helper(0, 0, total: dungeon[0][0], tmp: min(0, dungeon[0][0]))
    return abs(ans) + 1
}
func calculateMinimumHP2(_ dungeon: [[Int]]) -> Int {
    let rows = dungeon.count
    let columns = dungeon[0].count
    var memory = Array(repeating: Array(repeating: -1, count: columns), count: rows)
    func dfs(_ i: Int, _ j: Int) -> Int {
        if i >= rows || j >= columns {
            return Int.max
        }
        if memory[i][j] != -1 {
            return memory[i][j]
        }
        if i == rows - 1 && j == columns - 1 {
            if dungeon[i][j] >= 0 {
                return 0
            } else {
                return -dungeon[i][j]
            }
        }
        let right = dfs(i, j+1)
        let down = dfs(i+1, j)
        let needMin = min(right, down) - dungeon[i][j]
        let res: Int
        if needMin < 0 {
            res = 0
        } else {
            res = needMin
        }
        memory[i][j] = res
        print((i,j,res))
        return res
    }
    return dfs(0, 0) + 1
}
//calculateMinimumHP2([[-2,-3,3],
//                     [-5,-10,1],
//                     [10,30,-5]])
//直接动态规划，从最后一个数字往前。。。

func findRepeatedDnaSequences(_ s: String) -> [String] {
    var dict = [String: Int]()
    var left = 0
    var right = 9
    var res = [String]()
    let str = s as NSString
    while right < s.count {
        let substr = str.substring(with: NSRange(left...right))
        if let count = dict[substr] {
            dict[substr]! += 1
            if count == 1 {
                res.append(substr)
            }
        } else {
            dict[substr] = 1
        }
        left += 1
        right += 1
    }
    return res
}


func findRepeatedDnaSequences2(_ s: String) -> [String] { //前一个hash超时，这个可通过
    var ans = [String]()
    var isExist = [Bool](repeating: false, count: 1 << 20)
    var isAdd = [Bool](repeating: false, count: 1 << 20)
    let k = (1 << 18) - 1
    var key = 0
    let dict: [String: Int] = ["A": 0, "C": 1, "G": 2, "T":3]
    for (i, char) in s.enumerated() {
        key <<= 2
        key += dict[String(char)]!
        if i >= 9 {
            if isExist[key] {
                if !isAdd[key] {
                    isAdd[key] = true
                    ans.append((s as NSString).substring(with: NSRange((i-9)...i)))
                }
            } else {
                isExist[key] = true
            }
        }
        key &= k //相当于滑动窗口，保留住key的最新9位字符（18位），在下一轮循环中<<2后与最新的一个字符（2位）相加
    }
    return ans
}
//print(findRepeatedDnaSequences2("AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"))

//print(String(909, radix: 2))

func rob(_ nums: [Int]) -> Int { //三种解法核心思想都是nums[i]要还是不要，要的话就是nums[i] + dp[i-2], 不要的话就跟前一个一样，dp[i-1]
    var memo = [Int](repeating: -1, count: nums.count)
//    func robTo(_ index: Int) -> Int {
//        if index == 0 { return nums[0] }
//        if index < 0 {
//            return 0
//        }
//        if memo[index] != -1 {
//            return memo[index]
//        }
//        let sum1 = robTo(index - 2) + nums[index]
//        let sum2 = robTo(index - 1)
//        let ans = max(sum1, sum2)
//        memo[index] = ans
//        return ans
//    }
//    return robTo(nums.count - 1)
    
//    func robFrom(_ index: Int) -> Int {
//        if index >= nums.count { return 0 }
//        if index == nums.count - 1 { return nums[index] }
//        if memo[index] != -1 { return memo[index] }
//        let sum1 = nums[index] + robFrom(index + 2)
//        let sum2 = robFrom(index + 1)
//        let ans = max(sum1, sum2)
//        memo[index] = ans
//        return ans
//    }
//    return robFrom(0)
    var dp = [Int](repeating: 0, count: nums.count)
    if nums.count == 0 { return 0 }
    if nums.count == 1 { return nums.first! }
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    for i in 2..<nums.count {
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
    }
    return dp.last!
}
//rob([1,9,2,3,10,6])

func rightSideView(_ root: TreeNode?) -> [Int] {
    if root == nil { return [] }
    var queue: [TreeNode] = [root!]
    var ans = [Int]()
    while !queue.isEmpty {
        let count = queue.count
        for i in 0..<count {
            let node = queue.removeFirst()
            if i == count - 1 {
                ans.append(node.val)
            }
            if node.left != nil { queue.append(node.left!) }
            if node.right != nil { queue.append(node.right!) }
        }
    }
    return ans
}

func numIslands(_ grid: [[Character]]) -> Int {
    if grid.count == 0 {return 0}
    var copy = grid
    copy.insert([Character](repeating: "0", count: grid[0].count), at: 0)
    copy.append([Character](repeating: "0", count: grid[0].count))
    for i in 0..<copy.count {
        copy[i].insert("0", at: 0)
        copy[i].append("0")
    }
    let rows = copy.count
    let columns = copy[0].count
    func index(_ row: Int, _ column: Int) -> Int {
        return row * columns + column
    }
    let unionFind = UnionFind(totalNodes: rows * columns)
    for i in 0..<copy.count {
        for j in 0..<copy[0].count {
            if copy[i][j] == "1" {
                if i != 0 && i != rows - 1 && j != 0 && j != columns - 1 {
                    if i > 0 && copy[i - 1][j] == "1" {
                        unionFind.union(index(i, j), index(i-1, j))
                    }
                    if i < rows - 1 && copy[i + 1][j] == "1" {
                        unionFind.union(index(i, j), index(i+1, j))
                    }
                    if j > 0 && copy[i][j - 1] == "1" {
                        unionFind.union(index(i, j), index(i, j-1))
                    }
                    if j < columns - 1 && copy[i][j+1] == "1" {
                        unionFind.union(index(i, j), index(i, j+1))
                    }
                }
            }
        }
    }
    var set: Set<Int> = []
    var ans = 0
    for i in 1..<rows-1 {
        for j in 1..<columns-1 {
            if copy[i][j] == "1" && !set.contains(unionFind.find(index(i, j))) {
                ans += 1
                set.insert(unionFind.find(index(i, j)))
            }
        }
    }
    return ans
}

func numIslands2(_ grid: [[Character]]) -> Int {
    var grid = grid
    
    func dfs(_ i: Int, _ j: Int) {
        guard i >= 0 && i < grid.count && j >= 0 && j < grid[i].count else { return }
        let c = grid[i][j]
        grid[i][j] = "2"
        if c == "1" {
            dfs(i, j-1)
            dfs(i, j+1)
            dfs(i-1, j)
            dfs(i+1, j)
        }
    }
    
    var count = 0
    for i in 0..<grid.count {
        for j in 0..<grid[i].count {
            if grid[i][j] == "1" {
                count += 1
                dfs(i, j)
            }
        }
    }
    return count
}

func rangeBitwiseAnd(_ m: Int, _ n: Int) -> Int { //寻找[m,n]范围内二进制数高位（左边）没有变化的数，后面补上0即为所求的结果。
    var count = 0
    var m = m
    var n = n
    while m != n {
        m >>= 1
        n >>= 1
        count += 1
    }
    n <<= count
    return n
}
//rangeBitwiseAnd(Int.max - 2, Int.max)

func isHappy(_ n: Int) -> Bool {
    var set: Set<Int> = [n]
    var num = n
    while num != 1 {
        var components: [Int] = []
        while num != 0 {
            let x = num % 10
            components.append(x)
            num -= x
            num /= 10
        }
        for i in components {
            num += i * i
        }
        if set.contains(num) {
            return false
        }
        set.insert(num)
    }
    return true
}
//print(isHappy(19))

func removeElements(_ head: ListNode?, _ val: Int) -> ListNode? {
    if head == nil { return head }
    let dummy = ListNode(-1)
    dummy.next = head
    var p = head
    var pre: ListNode? = dummy
    while p != nil {
        if p!.val == val {
            pre?.next = p?.next
        } else {
            pre = p
        }
        p = p?.next
    }
    return dummy.next
}

func countPrimes(_ n: Int) -> Int {
//    func isPrime(_ num: Int) -> Bool {
//        if num == 2 || num == 3 { return true }
//        if num % 6 != 5 && num % 6 != 1 {
//            return false
//        }
//        for i in stride(from: 5, through: sqrt(Double(num)), by: 6) {
//            if num % Int(i) == 0 || num % Int(i + 2) == 0 {
//                return false
//            }
//        }
//        return true
//    }
    //筛选法
    var count = 0
    if n <= 2 { return 0 }
    var isPrimes = [Bool](repeating: true, count: n)
    for i in 2..<n {
        if isPrimes[i] {
            count += 1
            for j in stride(from: i + i, to: n, by: i) {
                isPrimes[j] = false
            }
        }
    }
    return count
}
//countPrimes(150000)

func isIsomorphic(_ s: String, _ t: String) -> Bool {
    var dict1: [Character: Character] = [:]
    var dict2: [Character: Character] = [:]
    let chars1 = Array(s) as [Character]
    let chars2 = Array(t) as [Character]
    for i in 0..<chars1.count {
        if let char = dict1[chars1[i]] {
            if char != chars2[i] {
                return false
            }
        } else {
            dict1[chars1[i]] = chars2[i]
        }
        if let char = dict2[chars2[i]] {
            if char != chars1[i] {
                return false
            }
        } else {
            dict2[chars2[i]] = chars1[i]
        }
    }
    return true
}

//isIsomorphic("ab", "aa")

func reverseList(_ head: ListNode?) -> ListNode? {
    if head == nil || head?.next == nil { return head }
//    let newHead = reverseList(head?.next)
//    head?.next?.next = head
//    head?.next = nil
//    return newHead
    let dummy = ListNode(0)
    dummy.next = head
    var pre: ListNode?
    var cur = head
    var next = head?.next
    while cur != nil {
        cur?.next = pre
        pre = cur
        cur = next
        next = next?.next
    }
    return pre
}

//207
func canFinish(_ numCourses: Int, _ prerequisites: [[Int]]) -> Bool { //检测一个图中是否有循环
    if numCourses <= 0 { return false }
    if prerequisites.count == 0 { return true }
    var preCourses = [Int](repeating: 0, count: numCourses)
    for i in prerequisites {
        preCourses[i[0]] += 1
    }
    var queueOfToLearn = [Int]()
    for i in 0..<numCourses {
        if preCourses[i] == 0 {
            queueOfToLearn.append(i)
        }
    }
    var learned = [Int]()
    while !queueOfToLearn.isEmpty {
        let num = queueOfToLearn.removeFirst()
        learned.append(num)
        for p in prerequisites {
            if p[1] == num {
                preCourses[p[0]] -= 1
                if preCourses[p[0]] == 0{
                    queueOfToLearn.append(p[0])
                }
            }
        }
    }
    return learned.count == numCourses
}

func canFinish2(_ numCourses: Int, _ prerequisites: [[Int]]) -> Bool {
    if numCourses <= 0 { return false }
    if prerequisites.count == 0 { return true }
    var marked = [Int](repeating: 0, count: numCourses)
    var graph: [Set<Int>] = Array(repeating: [], count: numCourses)
    for p in prerequisites {
        graph[p[1]].insert(p[0])
    }
    func dfs(_ i: Int) -> Bool {
        if marked[i] == 1 {
            return true
        }
        if marked[i] == 2 {
            return false
        }
        marked[i] = 1 //开始遍历
        let successorNodes = graph[i]
        for successor in successorNodes {
            if dfs(successor) {
                return true
            }
        }
        marked[i] = 2 //遍历完成
        return false
    }
    for i in 0..<numCourses {
        if dfs(i) {
            return false
        }
    }
    return true
}
//canFinish2(7, [[1,0],[2,1],[3,2],[4,3],[5,4],[6,3],[2,6]])

class Trie {
    class TrieNode {
                
        private var links: [TrieNode?]
        private final let R = 26
        private var end: Bool = false
        init() {
            self.links = [TrieNode?](repeating: nil, count: R)
        }
        
        private func index(_ char1: Character) -> Int {
            return Int(char1.asciiValue! - Character("a").asciiValue!)
        }
        
        public func containsKey(_ char: Character) -> Bool {
            return links[index(char)] != nil
        }
        
        public func get(_ char: Character) -> TrieNode {
            guard containsKey(char) else { fatalError("Does Not Contain Key \(char)")}
            return links[index(char)]!
        }
        
        public func put(_ char: Character, _ node: TrieNode) {
            links[index(char)] = node
        }
        
        public func setEnd() {
            self.end = true
        }
        
        public func isEnd() -> Bool {
            return end
        }
    }
    
    private let root = TrieNode()
    
    public func insert(_ str: String) {
        let chars = Array(str) as [Character]
        var node = root
        for char in chars {
            if !node.containsKey(char) {
                node.put(char, TrieNode())
            }
            node = node.get(char)
        }
        node.setEnd()
    }
    
    public func searchPrefix(_ word: String) -> TrieNode? {
        var node = root
        let chars = Array(word) as [Character]
        for char in chars {
            if node.containsKey(char) {
                node = node.get(char)
            } else {
                return nil
            }
        }
        return node
    }
    
    public func search(_ word: String) -> Bool{
        guard let node = searchPrefix(word) else { return false }
        return node.isEnd()
    }
    
    public func startWith(_ prefix: String) -> Bool {
        return searchPrefix(prefix) != nil
    }
}

func minSubArrayLen(_ s: Int, _ nums: [Int]) -> Int {
    let n = nums.count
    var ans = Int.max
    var left = 0
    var sum = 0
    for i in 0..<n {
        sum += nums[i]
        while sum >= s {
            ans = min(ans, i + 1 - left)
            sum -= nums[left]
            left += 1
        }
    }
    return ans
}
//print(minSubArrayLen(7, [2,3,1,2,4,3]))

