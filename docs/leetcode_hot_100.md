# 1.两数之和

- 给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。
- 你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
- 你可以按任意顺序返回答案。

```swift
func twoSum(_ nums: [Int], _ target: Int) -> [Int] {
    var map = [Int: Int]()
    for (index, num) in nums.enumerated() {
        if let remainIndex = map[target - num] {
            return [remainIndex, index]
        }
        // 利用字典，将下标值存入用num作为key的字典中
        map[num] = index
    }
    return []
}
```

# 2.字母异位词分组

- 输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
- 输出: [["bat"],["nat","tan"],["ate","eat","tea"]]

```Swift
func groupAnagrams(_ strs: [String]) -> [[String]] {
    var anagrams: [String: [String]] = [:]

    for word in strs {
        // 将单词排序，以确保字母异位词具有相同的键
        let sortedWord = String(word.sorted())

        // 将排序后的单词作为键，原始单词添加到对应的值列表中
        if var anagramGroup = anagrams[sortedWord] {
            anagramGroup.append(word)
            anagrams[sortedWord] = anagramGroup
        } else {
            anagrams[sortedWord] = [word]
        }
    }

    // 返回结果，即分好组的字母异位词
    return Array(anagrams.values)
}
```

# 3.最长连续序列

- 给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
- 请你设计并实现时间复杂度为 O(n) 的算法解决此问题。

- 输入：nums = [100,4,200,1,3,2]
- 输出：4
- 解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。

```Swift
func longestConsecutive(_ nums: [Int]) -> Int {
    if nums.isEmpty {
        return 0
    }
    // 转为Set 减少遍历次数
    let numSet = Set(nums)
    var longestStreak = 0
    for num in numSet {
        // 不重复计算最长连续序列的子序列
        if !numSet.contains(num - 1) {
            var currentStreak = 1
            var currentNum = num
            while numSet.contains(currentNum + 1) {
                currentStreak += 1
                currentNum += 1
            }
            longestStreak = max(longestStreak, currentStreak)
        }
    }
    return longestStreak
}
```

# 4.移动零

- 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
- 必须在不复制数组的情况下原地对数组进行操作。

```Swift
func moveZeroes(_ nums: inout [Int]) {
    var slow = 0
    // 移动零 等价于 移动非零后，后面补0
    for num in nums {
        if num != 0 {
            nums[slow] = num
            slow += 1
        }
    }
    if slow < nums.count {
        for index in slow...nums.count - 1 {
            nums[index] = 0
        }
    }
}
```

# 5.盛水最多的容器

- 给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 
- 找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水
- 返回容器可以储存的最大水量。

```Swift
func maxArea(_ height: [Int]) -> Int {
    var left = 0
    var right = height.count - 1
    var capacity = 0
    while left < right {
        let h = min(height[left], height[right])
        let w = right - left
        capacity = max(capacity, h * w)

        if height[left] < height[right] {
            left += 1
        } else {
            right -= 1
        }
    }
    return capacity
}
```

# 6.三数之和

- 给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请
- 你返回所有和为 0 且不重复的三元组。
- 注意：答案中不可以包含重复的三元组。

```Swift
func threeSum(_ nums: [Int]) -> [[Int]] {
    var result: [[Int]] = []

    // 首先对数组进行排序
    let sortedNums = nums.sorted()
    
    for i in 0..<sortedNums.count - 2 {
        // 避免重复的三元组
        if i > 0 && sortedNums[i] == sortedNums[i - 1] {
            continue
        }
        
        var left = i + 1
        var right = sortedNums.count - 1
        
        while left < right {
            let sum = sortedNums[i] + sortedNums[left] + sortedNums[right]
            
            if sum == 0 {
                // 找到一个符合条件的三元组
                result.append([sortedNums[i], sortedNums[left], sortedNums[right]])
                
                // 避免重复的元素
                while left < right && sortedNums[left] == sortedNums[left + 1] {
                    left += 1
                }
                while left < right && sortedNums[right] == sortedNums[right - 1] {
                    right -= 1
                }
                
                // 移动指针
                left += 1
                right -= 1
            } else if sum < 0 {
                left += 1
            } else {
                right -= 1
            }
        }
    }
    
    return result
}
```

# 7.接雨水

- 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

```Swift
func trap(_ height: [Int]) -> Int {
    // 无法形成凹槽，返回0
    guard height.count > 2 else { return 0 }

    var left = 0
    var right = height.count - 1
    
    // 左边的最大高度
    var leftMax = 0 
    // 右边的最大高度
    var rightMax = 0 
    // 储水量结果
    var result = 0 

    while left < right {
        // 更新左边的最大值
        leftMax = max(leftMax, height[left]) 
        // 更新右边的最大值
        rightMax = max(rightMax, height[right])

        if height[left] < height[right] {
            // 如果左边的高度比较小，计算左边的储水量
            result += leftMax - height[left]
            left += 1
        } else {
            // 如果右边的高度比较小，计算右边的储水量
            result += rightMax - height[right]
            right -= 1
        }
    }

    return result
}
```

# 8.无重复字符的最长子串

- 给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。

```Swift
func lengthOfLongestSubstring(_ s: String) -> Int {
    var maxLength = 0
    // 用于存储字符及其最近一次出现的索引
    var charIndexMap = [Character: Int]()
    // 滑动窗口的起始位置
    var start = 0
    for (end, char) in s.enumerated() {
        if let lastIndex = charIndexMap[char], lastIndex >= start {
            // 如果字符已经在窗口中出现过，更新窗口的起始位置
            start = lastIndex + 1
        }
        // 计算当前窗口的长度
        let currentLength = end - start + 1
        maxLength = max(maxLength, currentLength)
        // 更新字符的最近一次出现的索引
        charIndexMap[char] = end
    }
    return maxLength
}
```

# 9.找到字符串中所有字母异位词

- 给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。
- 异位词 指由相同字母重排列形成的字符串（包括相同的字符串）。

```Swift
func findAnagrams(_ s: String, _ p: String) -> [Int] {
    let sChars = Array(s)
    let pChars = Array(p)
    var result: [Int] = []
    
    guard sChars.count >= pChars.count else {
        return result
    }

    var pFrequency: [Character: Int] = [:]
    var windowFrequency: [Character: Int] = [:]

    // 初始化p字符串的字符频率哈希表
    for char in pChars {
        pFrequency[char, default: 0] += 1
    }

    // 初始化窗口的字符频率哈希表
    for i in 0..<pChars.count {
        windowFrequency[sChars[i], default: 0] += 1
    }

    // 滑动窗口
    for i in 0...(sChars.count - pChars.count) {
        // 检查当前窗口是否是字母异位词
        if windowFrequency == pFrequency {
            result.append(i)
        }

        // 移动窗口
        if i + pChars.count < sChars.count {
            // 移除窗口左侧字符的频率
            let leftChar = sChars[i]
            windowFrequency[leftChar]! -= 1
            if windowFrequency[leftChar] == 0 {
                windowFrequency.removeValue(forKey: leftChar)
            }

            // 添加窗口右侧字符的频率
            let rightChar = sChars[i + pChars.count]
            windowFrequency[rightChar, default: 0] += 1
        }
    }

    return result
}
```

# 10.和为K的子数组

- 给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的子数组的个数

```Swift
func subarraySum(_ nums: [Int], _ k: Int) -> Int {
    var prefixSum = 0
    var count = 0
    var sumFrequency: [Int : Int] = [0 : 1] // 用于存储前缀和的频率

    for num in nums {
        prefixSum += num

        // 查找前缀和为 prefixSum - k 的频率
        if let frequency = sumFrequency[prefixSum - k] {
            count += frequency
        }

        // 更新当前前缀和的频率
        if let existingValue = sumFrequency[prefixSum] {
            sumFrequency[prefixSum] = existingValue + 1
        } else {
            sumFrequency[prefixSum] = 1
        }
    }

    return count
}
```

# 11.滑动窗口的最大值

- 给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。
- 返回每次滑动 滑动窗口中的最大值 。

```Swift
func maxSlidingWindow(_ nums: [Int], _ k: Int) -> [Int] {
    guard nums.count > 0 else {
        return []
    }
    
    var result = [Int]()
    var deque = [Int]() // 存储元素在数组中的索引
    
    // 定义函数，用于从队列尾部删除小于等于给定值的元素
    func cleanDeque(_ index: Int) {
        // 窗口大小超出k
        while !deque.isEmpty && deque.first! < index - k + 1 {
            deque.removeFirst()
        }
        // 进来的更大
        while !deque.isEmpty && nums[deque.last!] < nums[index] {
            deque.removeLast()
        }
    }
    
    for i in 0..<nums.count {
        cleanDeque(i)
        
        deque.append(i)
        
        if i >= k - 1 {
            result.append(nums[deque.first!])
        }
    }
    
    return result
}
```

# 12.最小覆盖子串

- 给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。

```Swift
func minWindow(_ s: String, _ t: String) -> String {
    let sArr = [Character](s)
    // 窗口的字典
    var windowDict = [Character: Int]()
    // 所需字符的字典
    var needDict = [Character: Int]()
    for c in t {
        needDict[c, default: 0] += 1
    }

    // 当前窗口的左右两端
    var left = 0, right = 0
    // 匹配次数，等于needDict的key数量时代表已经匹配完成
    var matchCnt = 0
    // 用来记录最终的取值范围
    var start = 0, end = 0
    // 记录最小范围
    var minLen = Int.max
    
    while right < sArr.count {
        // 开始移动窗口右侧端点
        let rChar = sArr[right]
        right += 1
        // 右端点字符不是所需字符直接跳过
        if needDict[rChar] == nil { continue }
        // 窗口中对应字符数量+1
        windowDict[rChar, default: 0] += 1
        // 窗口中字符数量达到所需数量时，匹配数+1
        if windowDict[rChar] == needDict[rChar] {
            matchCnt += 1
        }

        // 如果匹配完成，开始移动窗口左侧断点, 目的是为了寻找当前窗口的最小长度
        while matchCnt == needDict.count {
            // 记录最小范围
            if right - left < minLen {
                start = left
                end = right
                minLen = right - left
            }
            let lChar = sArr[left]
            left += 1
            if needDict[lChar] == nil { continue }
            // 如果当前左端字符的窗口中数量和所需数量相等，则后续移动就不满足匹配了，匹配数-1
            if needDict[lChar] == windowDict[lChar] {
                matchCnt -= 1
            }
            // 减少窗口字典中对应字符的数量
            windowDict[lChar]! -= 1
        }
    }

    return minLen == Int.max ? "" : String(sArr[start..<end])
}
```

# 13.最大子数组和

- 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

```Swift
func maxSubArray(_ nums: [Int]) -> Int {
    guard !nums.isEmpty else {
        return 0
    }

    var maxSum = nums[0]
    var currentSum = nums[0]

    for i in 1..<nums.count {
        // 选择当前元素加入子数组，或者重新开始一个新的子数组
        currentSum = max(nums[i], currentSum + nums[i])

        // 更新最大子数组和
        maxSum = max(maxSum, currentSum)
    }

    return maxSum
}
```

# 14.合并区间

```Swift
func merge(_ intervals: [[Int]]) -> [[Int]] {
    guard intervals.count > 1 else {
        return intervals
    }

    var sortedIntervals = intervals.sorted { $0[0] < $1[0] }
    var mergedIntervals: [[Int]] = []

    for interval in sortedIntervals {
        if mergedIntervals.isEmpty || interval[0] > mergedIntervals.last![1] {
            // 无重叠，直接添加新的区间
            mergedIntervals.append(interval)
        } else {
            // 有重叠，合并区间
            mergedIntervals[mergedIntervals.count - 1][1] = max(mergedIntervals.last![1], interval[1])
        }
    }

    return mergedIntervals
}
```

# 15.轮转数组（不同解法）

```Swift
// 原地算法 时间复杂度为 O(n)，空间复杂度为 O(1)
func rotate(_ nums: inout [Int], _ k: Int) {
    let n = nums.count
    let shift = k % n

    // 将整个数组反转
    reverse(&nums, start: 0, end: n - 1)

    // 反转前 k 个元素
    reverse(&nums, start: 0, end: shift - 1)

    // 反转剩余的元素
    reverse(&nums, start: shift, end: n - 1)
}

func reverse(_ nums: inout [Int], start: Int, end: Int) {
    var start = start
    var end = end

    while start < end {
        nums.swapAt(start, end)
        start += 1
        end -= 1
    }
}

// 逐步右移 时间复杂度：O(k * n) 空间复杂度：O(1)
func rotate(_ nums: inout [Int], _ k: Int) {
    let n = nums.count
    let shift = k % n

    for _ in 0..<shift {
        let lastElement = nums.removeLast()
        nums.insert(lastElement, at: 0)
    }
}

// 使用额外数组 时间复杂度：O(n) 空间复杂度：O(n)
func rotate(_ nums: inout [Int], _ k: Int) {
    let n = nums.count
    var rotatedNums = Array(repeating: 0, count: n)

    for i in 0..<n {
        let newIndex = (i + k) % n
        rotatedNums[newIndex] = nums[i]
    }

    nums = rotatedNums
}
```

# 16.除自身以外数组的乘积

```Swift
func productExceptSelf(_ nums: [Int]) -> [Int] {
    let n = nums.count
    var leftProducts = Array(repeating: 1, count: n)
    var rightProducts = Array(repeating: 1, count: n)
    var result = Array(repeating: 1, count: n)

    // 计算左侧所有元素的乘积
    var leftProduct = 1
    for i in 0..<n {
        leftProducts[i] = leftProduct
        leftProduct *= nums[i]
    }

    // 计算右侧所有元素的乘积
    var rightProduct = 1
    for i in (0..<n).reversed() {
        rightProducts[i] = rightProduct
        rightProduct *= nums[i]
    }

    // 计算最终结果
    for i in 0..<n {
        result[i] = leftProducts[i] * rightProducts[i]
    }

    return result
}
```

# 17.缺失的第一个正数

- 给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。
- 请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。

```Swift
func firstMissingPositive(_ nums: [Int]) -> Int {
    var tmp = nums
    let n = tmp.count

    // Step 1: 将每个负数和大于数组长度的数变成数组长度 + 1
    for i in 0..<n {
        if tmp[i] <= 0 || tmp[i] > n {
            tmp[i] = n + 1
        }
    }

    // Step 2: 标记存在的正整数
    for i in 0..<n {
        let num = abs(tmp[i])
        if num <= n {
            tmp[num - 1] = -abs(tmp[num - 1])
        }
    }

    // 找到第一个正整数的位置
    for i in 0..<n {
        if tmp[i] > 0 {
            return i + 1
        }
    }

    // 如果数组中都是正整数，则返回数组长度 + 1
    return n + 1
}
```

# 18.矩阵置零

- 给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。

```Swift
func setZeroes(_ matrix: inout [[Int]]) {
    guard !matrix.isEmpty else {
        return
    }

    let rows = matrix.count
    let cols = matrix[0].count
    var zeroRows = Set<Int>()
    var zeroCols = Set<Int>()

    // 找到所有包含零的行和列
    for i in 0..<rows {
        for j in 0..<cols {
            if matrix[i][j] == 0 {
                zeroRows.insert(i)
                zeroCols.insert(j)
            }
        }
    }

    // 将零所在的行和列置零
    for row in zeroRows {
        for j in 0..<cols {
            matrix[row][j] = 0
        }
    }

    for col in zeroCols {
        for i in 0..<rows {
            matrix[i][col] = 0
        }
    }
}
```

# 19.螺旋矩阵

```Swift
func spiralOrder(_ matrix: [[Int]]) -> [Int] {
    var left = 0 
    var right = matrix[0].count - 1 
    
    if matrix.count == 0 || matrix[0].count == 0 {
        return [Int]()
    }


    var top = 0 
    var bottom = matrix.count - 1
    var ans = [Int]()
    while left <=  right && top <= bottom {
        for i in left...right {
            ans.append(matrix[top][i])
        }
        if top < bottom {
            for i  in top+1...bottom {
                ans.append(matrix[i][right])
            }
        }
        
        if left < right && top < bottom {
                // 右往左 
            for column in stride(from: right - 1, through: left + 1, by:-1) {
                ans.append(matrix[bottom][column])
            }

            // 下往上
            for row in stride(from: bottom , through:top + 1 , by:-1) {
                ans.append(matrix[row][left])
            }
        }
        
        left += 1
        right -= 1
        top += 1 
        bottom -= 1 
    }
    return ans

}
```

# 20.旋转图像

```Swift
func rotate(_ matrix: inout [[Int]]) {
    let n = matrix.count

    // 先进行矩阵的转置操作
    for i in 0..<n {
        for j in i..<n {
            let temp = matrix[i][j]
            matrix[i][j] = matrix[j][i]
            matrix[j][i] = temp
        }
    }

    // 对每一行进行反转操作
    for i in 0..<n {
        matrix[i].reverse()
    }
}
```

# 21.搜索二维矩阵 II

```Swift
func searchMatrix(_ matrix: [[Int]], _ target: Int) -> Bool {
    var width = matrix.count - 1
    var height = matrix[0].count - 1
    var heightIndex = height
    var widthIndex = 0
    // 从左下角开始搜索
    while widthIndex <= width && heightIndex >= 0 {
        if matrix[widthIndex][heightIndex] > target {
            heightIndex -= 1
        } else if matrix[widthIndex][heightIndex] < target {
            widthIndex += 1
        } else {
            return true
        }
    }
    return false
}
```

# 22.相交链表

```Swift
func getIntersectionNode(_ headA: ListNode?, _ headB: ListNode?) -> ListNode? {
    var currentA = headA
    var currentB = headB
    if currentA == nil || currentB == nil {
        return nil
    }
    while currentA !== currentB {
        currentA = (currentA == nil) ? headB : currentA?.next
        currentB = (currentB == nil) ? headA : currentB?.next
    }
    return currentA
}
```

# 23.反转链表

```Swift
func reverseList(_ head: ListNode?) -> ListNode? {
    var pre: ListNode? = nil
    var current = head

    while current != nil {
        var tmp = current!.next
        current!.next = pre
        pre = current
        current = tmp
    }
    return pre
}
```

# 24.回文链表

```Swift
func isPalindrome(_ head: ListNode?) -> Bool {
    // 要求 O(n) 时间复杂度和 O(1) 空间复杂度
    // 如果链表为空或只有一个节点，认为是回文链表
    if head == nil || head?.next == nil {
        return true
    }
    // 使用快慢指针找到链表的中点
    var slow = head
    var fast = head
    while fast?.next != nil {
        slow = slow?.next
        fast = fast?.next?.next
    }
    // 反转后半部分链表
    var half = reverseList(slow)
    var start = head
    // 比较前半部分和反转后的后半部分是否相等
    while half != nil {
        if half?.val != start?.val {
            return false
        }
        half = half?.next
        start = start?.next
    }
    return true
}

func reverseList(_ head: ListNode?) -> ListNode? {
    var pre: ListNode? = nil
    var current = head

    while current != nil {
        var tmp = current!.next
        current!.next = pre
        pre = current
        current = tmp
    }
    return pre
}
```

# 25.环形链表

```Swift
func hasCycle(_ head: ListNode?) -> Bool {
    var slow = head
    var fast = head?.next
    while slow != nil && fast?.next != nil {
        if slow === fast {
            return true
        }
        slow = slow?.next
        fast = fast?.next?.next
    }
    return false
}
```

# 26.环形链表 II

```Swift
func detectCycle(_ head: ListNode?) -> ListNode? {
    var slow = head
    var fast = head

    // 检查是否存在环，若不存在环，这个while会最终结束
    while fast != nil && fast?.next != nil {
        slow = slow?.next
        fast = fast?.next?.next

        // 快慢指针相遇，说明存在环，此时快指针走过的距离是慢指针的两倍
        // a 头节点到环入口的距离
        // b 环入口到相遇节点的距离
        // c 相遇节点到环入口的距离
        // a + b + c + b = 2 (a + b)
        // 化简后得 a = c 
        // 即当快慢指针相遇时，头节点到环入口的距离 等于 相遇节点到环入口的距离
        if slow === fast {
            // 重置快指针为头节点
            fast = head
            // 快慢指针再次相遇时，就是环的入口节点
            while fast !== slow {
                fast = fast?.next
                slow = slow?.next
            }
            return fast
        }
    }

    return nil
}
```

# 27.合并两个有序链表

```Swift
func mergeTwoLists(_ list1: ListNode?, _ list2: ListNode?) -> ListNode? {
    if list1 == nil { 
        return list2 
    }
    if list2 == nil {
        return list1
    }
    if list1!.val <= list2!.val {
        list1!.next = mergeTwoLists(list1?.next, list2)
        return list1
    } else {
        list2!.next = mergeTwoLists(list1, list2?.next)
        return list2
    }
}
```

# 28.两数相加

- 给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
- 请你将两个数相加，并以相同形式返回一个表示和的链表。
- 你可以假设除了数字 0 之外，这两个数都不会以 0 开头。
- 每一位计算的同时需要考虑上一位的进位问题，而当前位计算结束后同样需要更新进位值
- 如果两个链表全部遍历完毕后，进位值为 1，则在新链表最前方添加节点 1
- 对于链表问题，返回结果为头结点时，通常需要先初始化一个预先指针 pre，该指针的下一个节点指向真正的头结点 head。使用预先指针的目的在于链表初始化时无可用节点值，而且链表构造过程需要指针移动，进而会导致头指针丢失，无法返回结果。

```Swift
func addTwoNumbers(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
    var preNode = ListNode(0)
    var curNode: ListNode? = preNode
    var carry = 0

    var l1 = l1
    var l2 = l2
    while l1 != nil || l2 != nil {
        let x = l1?.val ?? 0
        let y = l2?.val ?? 0
        let sum = x + y + carry
        carry = sum / 10
        let newNode = ListNode(sum % 10)
        curNode?.next = newNode
        // 当前节点判断完成所有链表的节点都向后移一个
        curNode = curNode?.next
        l1 = l1?.next
        l2 = l2?.next
    }

    if carry == 1 {
        curNode?.next = ListNode(1)
    }

    return preNode.next
}
```

# 29.删除链表的倒数第N个结点

```Swift
func removeNthFromEnd(_ head: ListNode?, _ n: Int) -> ListNode? {
    guard let head = head else { 
        return nil 
    }

    var dummy = ListNode(0)
    dummy.next = head
    var fast: ListNode? = dummy
    var slow: ListNode? = dummy

    for _ in 0...n {
        fast = fast?.next
    }

    while fast != nil {
        fast = fast?.next
        slow = slow?.next
    }

    slow?.next = slow?.next?.next

    return dummy.next
}
```

# 30.两两交换链表中的节点

```Swift
func swapPairs(_ head: ListNode?) -> ListNode? {
    let dummy = ListNode(0)
    dummy.next = head
    var pre = dummy

    while let first = pre.next, let second = first.next {
        pre.next = second
        first.next = second.next
        second.next = first

        pre = first
    }

    return dummy.next
}
```

# 31.K个一组翻转链表

- 给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。
- k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
- 你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

```Swift
func reverseKGroup(_ head: ListNode?, _ k: Int) -> ListNode? {
    // 统计链表长度
    var length = 0
    var current = head
    while current != nil {
        length += 1
        current = current?.next
    }

    // 定义一个虚拟头节点
    let dummy = ListNode(0)
    dummy.next = head

    var prevGroupEnd: ListNode? = dummy
    var currentGroupStart: ListNode? = dummy.next

    // 对每一组 k 个节点进行翻转
    for _ in 0..<(length / k) {
        var prev: ListNode? = nil
        var current: ListNode? = currentGroupStart

        // 翻转当前组 k 个节点
        for _ in 0..<k {
            let next = current?.next
            current?.next = prev
            prev = current
            current = next
        }

        // 更新连接关系
        prevGroupEnd?.next = prev
        currentGroupStart?.next = current
        prevGroupEnd = currentGroupStart
        currentGroupStart = current
    }

    return dummy.next
}
```

# 32.随机链表的复制

```Swift
func copyRandomList(_ head: Node?) -> Node? {
    guard let head = head else { return nil }

    var current: Node? = head
    var mapping: [Node : Node] = [:]

    while current != nil {
        mapping[current!] = Node(current!.val)
        current = current?.next
    }
    
    current = head
    while current != nil {
        if let nextNode = current?.next {
            mapping[current!]!.next = mapping[nextNode]
        }
        if let randomNode = current?.random {
            mapping[current!]!.random = mapping[randomNode]
        }
        current = current?.next
    }
    return mapping[head]
}
```

# 33.排序链表

```Swift
func sortList(_ head: ListNode?) -> ListNode? {
    guard head != nil, head?.next != nil else {
        return head
    }
    // 找到链表中间节点
    var slow = head
    var fast = head?.next
    while fast != nil && fast?.next != nil {
        slow = slow?.next
        fast = fast?.next?.next
    }

    let rightHead = slow?.next
    slow?.next = nil

    let sortedLeft = sortList(head)
    let sortedRight = sortList(rightHead)

    return merge(sortedLeft, sortedRight)
}

func merge(_ left: ListNode?, _ right: ListNode?) -> ListNode? {
    if left == nil {
        return right
    }
    if right == nil {
        return left
    }
    if left!.val < right!.val {
        left?.next = merge(left?.next, right)
        return left
    } else {
        right?.next = merge(left, right?.next)
        return right
    }
}
```

# 34.合并K个升序链表

```Swift
func mergeKLists(_ lists: [ListNode?]) -> ListNode? {
    guard !lists.isEmpty else {
        return nil
    }
    
    return merge(lists, 0, lists.count - 1)
}

func merge(_ lists: [ListNode?], _ start: Int, _ end: Int) -> ListNode? {
    if start == end {
        return lists[start]
    }
    
    let mid = (start + end) / 2
    let left = merge(lists, start, mid)
    let right = merge(lists, mid + 1, end)
    
    return mergeTwoLists(left, right)
}

func mergeTwoLists(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
    guard let l1 = l1 else {
        return l2
    }
    guard let l2 = l2 else {
        return l1
    }
    
    if l1.val < l2.val {
        l1.next = mergeTwoLists(l1.next, l2)
        return l1
    } else {
        l2.next = mergeTwoLists(l1, l2.next)
        return l2
    }
}
```

# 35.LRU缓存

- 请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
- 实现 LRUCache 类：
- LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
- int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
- void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。
- 函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。

```Swift
class LRUCache {

    class Node {
        var key: Int
        var value: Int
        var prev: Node?
        var next: Node?
        
        init(_ key: Int, _ value: Int) {
            self.key = key
            self.value = value
            self.prev = nil
            self.next = nil
        }
    }
    
    var capacity: Int
    var cache: [Int: Node]
    var head: Node
    var tail: Node

    init(_ capacity: Int) {
        self.capacity = capacity
        self.cache = [:]
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = tail
        self.tail.prev = head
    }
    
    func get(_ key: Int) -> Int {
        if let node = cache[key] {
            // 将访问的节点移动到链表头部
            moveToHead(node)
            return node.value
        }
        return -1
    }
    
    func put(_ key: Int, _ value: Int) {
        if let existingNode = cache[key] {
            // 如果键已存在，更新值并移动到链表头部
            existingNode.value = value
            moveToHead(existingNode)
        } else {
            // 如果键不存在，创建新节点并插入到链表头部
            let newNode = Node(key, value)
            cache[key] = newNode
            addToHead(newNode)
            
            // 如果缓存超过容量，移除链表尾部节点
            if cache.count > capacity {
                removeTail()
            }
        }
    }

    private func addToHead(_ node: Node) {
        node.next = head.next
        node.prev = head
        head.next?.prev = node
        head.next = node
    }
    
    private func removeNode(_ node: Node) {
        node.prev?.next = node.next
        node.next?.prev = node.prev
    }
    
    private func moveToHead(_ node: Node) {
        removeNode(node)
        addToHead(node)
    }
    
    private func removeTail() {
        if let tailPrev = tail.prev {
            removeNode(tailPrev)
            cache.removeValue(forKey: tailPrev.key)
        }
    }
}
```

# 36.二叉树的中序遍历

```Swift
func inorderTraversal(_ root: TreeNode?) -> [Int] {
    var values: [Int] = []
    dfs(root, &values)
    return values
}

func dfs(_ root: TreeNode?, _ values: inout [Int]) {
    guard let root = root else { return }
    dfs(root.left, &values)
    values.append(root.val)
    dfs(root.right, &values)
}
```

# 37.二叉树的最大深度

```Swift
func maxDepth(_ root: TreeNode?) -> Int {
    guard let root = root else { return 0 }
    return max(maxDepth(root.left), maxDepth(root.right)) + 1 
}
```

# 38.翻转二叉树

```Swift
func invertTree(_ root: TreeNode?) -> TreeNode? {
    guard let root = root else { return nil }

    let left = invertTree(root.left)
    let right = invertTree(root.right)

    root.left = right
    root.right = left

    return root
}
```

# 39.对称二叉树

```Swift
func isSymmetric(_ root: TreeNode?) -> Bool {
    // 辅助函数，判断两个树是否是对称的
    func isMirror(_ left: TreeNode?, _ right: TreeNode?) -> Bool {
        // 两个节点都为空，对称
        if left == nil, right == nil {
            return true
        }
        // 一个节点为空，一个节点非空，不对称
        if left == nil || right == nil {
            return false
        }
        // 两个节点的值不相等，不对称
        if left!.val != right!.val {
            return false
        }
        // 递归判断左子树的左子树与右子树的右子树，左子树的右子树与右子树的左子树是否对称
        return isMirror(left?.left, right?.right) && isMirror(left?.right, right?.left)
    }

    // 根节点为空，对称
    guard let root = root else {
        return true
    }

    // 调用辅助函数判断左右子树是否对称
    return isMirror(root.left, root.right)
}
```

# 40.二叉树的直径

```Swift
class Solution {

    var diameter = 0

    func diameterOfBinaryTree(_ root: TreeNode?) -> Int {
        depth(root)
        return diameter
    }
    
    // 计算节点的深度
    func depth(_ node: TreeNode?) -> Int {
        // 递归基：空节点深度为0
        guard let node = node else {
            return 0
        }

        // 递归计算左右子树深度
        let leftDepth = depth(node.left)
        let rightDepth = depth(node.right)

        // 更新直径
        diameter = max(diameter, leftDepth + rightDepth)

        // 返回当前节点的深度
        return 1 + max(leftDepth, rightDepth)
    }
}
```

# 41.二叉树的层序遍历

- 输入：root = [3,9,20,null,null,15,7]
- 输出：[[3],[9,20],[15,7]]

```Swift
func levelOrder(_ root: TreeNode?) -> [[Int]] {
    guard let root = root else {
        return []
    }

    var result = [[Int]]()
    var queue = [root]

    while !queue.isEmpty {
        let levelSize = queue.count
        var currentLevel = [Int]()
        for _ in 0..<levelSize {
            let node = queue.removeFirst()
            currentLevel.append(node.val)

            if let left = node.left {
                queue.append(left)
            }
            
            if let right = node.right {
                queue.append(right)
            }
        }
        result.append(currentLevel)
    }
    return result
}
```

# 42.将有序数组转换为二叉搜索树

- 给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。
- 高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。

```Swift
func sortedArrayToBST(_ nums: [Int]) -> TreeNode? {
    // 辅助函数，构建高度平衡二叉搜索树
    func buildBST(_ left: Int, _ right: Int) -> TreeNode? {
        // 递归基：左边界大于右边界，返回nil
        if left > right {
            return nil
        }

        // 选择中间位置的元素作为根节点
        let mid = left + (right - left) / 2
        let root = TreeNode(nums[mid])

        // 递归构建左右子树
        root.left = buildBST(left, mid - 1)
        root.right = buildBST(mid + 1, right)

        return root
    }

    // 调用辅助函数开始构建二叉搜索树
    return buildBST(0, nums.count - 1)
}
```

# 43.验证二叉搜索树

```Swift
func isValidBST(_ root: TreeNode?) -> Bool {
    return isValidBST(root, Int.min, Int.max)
}

func isValidBST(_ node: TreeNode?, _ lowerBound: Int, _ upperBound: Int) -> Bool {
    guard let node = node else {
        return true
    }
    if node.val <= lowerBound || node.val >= upperBound {
        return false
    }
    return isValidBST(node.left, lowerBound, node.val) && isValidBST(node.right, node.val, upperBound)
}
```

# 44.二叉搜索树中第K小的元素（不同解法）

```Swift
// 借助栈实现dfs
func kthSmallest(_ root: TreeNode?, _ k: Int) -> Int {
    // 二叉搜索树的中序遍历是升序的
    var values: [Int] = []
    var stack = [TreeNode]()
    var currentNode = root

    while currentNode != nil || !stack.isEmpty {
        while let node = currentNode {
            stack.append(node)
            currentNode = node.left
        }

        if let node = stack.popLast() {
            values.append(node.val)
            currentNode = node.right
        }
    }

    return values[k-1]
}

// 递归实现dfs
func kthSmallest(_ root: TreeNode?, _ k: Int) -> Int {
    // 二叉搜索树的中序遍历是升序的
    var values: [Int] = []
    dfs(root, &values)
    return values[k - 1]
}

func dfs(_ root: TreeNode?, _ values: inout [Int]) {
    guard let root = root else { return }
    dfs(root.left, &values)
    values.append(root.val)
    dfs(root.right, &values)
}
```

# 45.二叉树的右视图

```Swift
func rightSideView(_ root: TreeNode?) -> [Int] {
    // BFS 层序遍历，每一层的最后一个节点
    guard let root = root else { return [] }
    var result: [Int] = []
    var queue: [TreeNode] = [root]

    while !queue.isEmpty {
        let levelSize = queue.count
        for i in 0..<levelSize {
            let node = queue.removeFirst()
            // 只加入每层最后一个节点
            if i == levelSize - 1 {
                result.append(node.val)
            }

            if let left = node.left {
                queue.append(left)
            }

            if let right = node.right {
                queue.append(right)
            }
        }
    }
    return result
}
```

# 46.二叉树展开为链表

```Swift
func flatten(_ root: TreeNode?) {
    guard let root = root else {
        return
    }
    
    flatten(root.left)
    flatten(root.right)

    let left = root.left
    let right = root.right

    root.left = nil
    root.right = left

    var current: TreeNode? = root
    while current?.right != nil {
        current = current?.right
    }

    current?.right = right
}
```

# 47.从前序与中序遍历序列构造二叉树

```Swift
func buildTree(_ preorder: [Int], _ inorder: [Int]) -> TreeNode? {
    guard !preorder.isEmpty && preorder.count == inorder.count else {
        return nil
    }

    let rootVal = preorder[0]
    let root = TreeNode(rootVal)

    if preorder.count == 1 {
        return root
    }
    
    // 在中序遍历中找到根节点的位置
    if let rootIndex = inorder.firstIndex(of: rootVal) {
        // 构造左子树的前序和中序遍历序列
        let leftInorder = Array(inorder[..<rootIndex])
        let leftPreorder = Array(preorder[1..<(1 + leftInorder.count)])
        root.left = buildTree(leftPreorder, leftInorder)
        
        // 构造右子树的前序和中序遍历序列
        let rightInorder = Array(inorder[(rootIndex + 1)...])
        let rightPreorder = Array(preorder[(1 + leftInorder.count)...])
        root.right = buildTree(rightPreorder, rightInorder)
    }
    
    return root
}
```

# 48.路径总和 III

- 给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。
- 路径不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

```Swift
class Solution {
    func pathSum(_ root: TreeNode?, _ targetSum: Int) -> Int {
        guard let root = root else {
            return 0
        }
        
        return dfs(root, targetSum) + pathSum(root.left, targetSum) + pathSum(root.right, targetSum)
    }

    func dfs(_ node: TreeNode?, _ targetSum: Int) -> Int {
        guard let node = node else {
            return 0
        }
        
        var count = 0
        
        // 以当前节点为起点向下搜索
        if node.val == targetSum {
            count += 1
        }
        
        count += dfs(node.left, targetSum - node.val)
        count += dfs(node.right, targetSum - node.val)
        
        return count
    }
}
```

# 49.二叉树的最近公共祖先

```Swift
func lowestCommonAncestor(_ root: TreeNode?, _ p: TreeNode?, _ q: TreeNode?) -> TreeNode? {
    // 如果根节点为空 root == nil，则返回 nil。
    // 如果根节点的值等于 p 或 q 的值，说明当前节点就是其中一个节点或两个节点中的一个，直接返回当前节点。
    if root == nil || root?.val == p?.val || root?.val == q?.val {
        return root
    }
    // 递归调用 lowestCommonAncestor 函数，传入左子树和右子树。返回的结果分别存储在 left 和 right 变量中。
    let left = lowestCommonAncestor(root?.left, p, q)
    let right = lowestCommonAncestor(root?.right, p, q)
    // 如果左右子树的结果都不为 nil，说明 p 和 q 分别位于当前节点的左右子树，那么当前节点就是它们的最低公共祖先.
    if left != nil && right != nil {
        return root
    }
    // 如果其中一个子树的结果为 nil，说明 p 和 q 都在另一侧，返回不为 nil 的那一侧的结果。
    return left ?? right
}
```

# 50.二叉树中的最大路径和

- 二叉树中的 路径 被定义为一条节点序列，序列中每对相邻节点之间都存在一条边。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。
- 路径和 是路径中各节点值的总和。
- 给你一个二叉树的根节点 root ，返回其 最大路径和 。

```Swift
class Solution {
    var maxSum: Int = Int.min
    
    func maxPathSum(_ root: TreeNode?) -> Int {
        maxPathSumHelper(root)
        return maxSum
    }
    
    func maxPathSumHelper(_ root: TreeNode?) -> Int {
        guard let root = root else {
            return 0
        }
        
        // 计算左子树和右子树的最大贡献值
        let leftMax = max(0, maxPathSumHelper(root.left))
        let rightMax = max(0, maxPathSumHelper(root.right))
        
        // 更新全局最大路径和
        maxSum = max(maxSum, root.val + leftMax + rightMax)
        
        // 返回以当前节点为根的子树的最大贡献值
        return root.val + max(leftMax, rightMax)
    }
}
```

# 51.岛屿数量

- 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
- 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
- 此外，你可以假设该网格的四条边均被水包围。

```Swift
class Solution {
    //! 以当前点，扩散方向：左，右，上，下
    let dx = [-1,1,0,0]
    let dy = [0,0,-1,1]
    var grid:[[Character]]!

    func numIslands(_ grid: [[Character]]) -> Int {
      var isLands = 0
      self.grid = grid
      //! 从左到右，从上到下，依次遍历
      for i in 0..<grid.count {
        for j in 0..<grid[i].count {
          if grid[i][j] == "0" {
            continue
          }
          //! 统计陆地+1，并且将附近的元素置为 水
          isLands += sink(i,j)
        }
      }
      return isLands
    }
    //! 扩散
    func sink(_ i:Int, _ j:Int) -> Int {
      if grid[i][j] == "0" {
        return 0
      }

      //! i,j == 1
      grid[i][j] = "0"
      //! 对相邻的点，进行扩散
      for k in 0..<dx.count {
        let x = i + dx[k]
        let y = j + dy[k]
        if x >= 0 
          && x < grid.count 
          && y >= 0 
          && y < grid[i].count {
          if grid[x][y] == "0" {
            continue
          }
          //! 如果是陆地，那么继续扩散，但是不统计
          sink(x, y)
        }
      }
      return 1
  }
}
```

# 52.腐烂的橘子

- 在给定的 m x n 网格 grid 中，每个单元格可以有以下三个值之一：
- 值 0 代表空单元格；
- 值 1 代表新鲜橘子；
- 值 2 代表腐烂的橘子。
- 每分钟，腐烂的橘子 周围 4 个方向上相邻 的新鲜橘子都会腐烂。
- 返回 直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 -1 。

```Swift
class Solution {
    func orangesRotting(_ grid: [[Int]]) -> Int {
        if grid.isEmpty {
            return -1
        }
        // 行
        let m = grid.count
        // 列
        let n = grid[0].count
        // 新鲜水果个数
        var fresh = 0
        var queue = [[Int]]()
        for i in 0..<m {
            for j in 0..<n {
                if grid[i][j] == 2 {
                    queue.append([i, j])
                } else if (grid[i][j] == 1) {
                    fresh += 1
                } 
            }
        }
        if fresh == 0 {
            return 0
        }
        var oranges = grid
        // 记录被橘子感染的四个方向
        let dirs = [[1 ,0], [-1, 0], [0, 1], [0, -1]]
        var minutes = 0
        while !queue.isEmpty && fresh > 0 {
            var size = queue.count 
            while size > 0 {
                let x = queue[0][0]
                let y = queue[0][1]
                queue.removeFirst()
                for i in 0..<4 {
                    let dx = x + dirs[i][0]
                    let dy = y + dirs[i][1]
                    // 越界、或者是新鲜的橘子
                    if dx < 0 || 
                    dx >= m ||
                    dy < 0 || 
                    dy >= n ||
                    oranges[dx][dy] != 1 {
                        continue
                    }
                    oranges[dx][dy] = 2
                    fresh -= 1
                    queue.append([dx, dy])
                }
                size -= 1
            }
            minutes += 1
        }
        return fresh > 0 ? -1:minutes
    }
}
```

# 53.课程表

- 你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。
- 在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。
- 例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
- 请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。

```Swift
func canFinish(_ numCourses: Int, _ prerequisites: [[Int]]) -> Bool {
    // 每个课程需要直接前置课程的个数,写成一个数组
    var degree: [Int] = [Int](repeating: 0, count: numCourses)
    // 每个课程完成之后,可以直接学习的课程,写成一个数组
    var nextCourses: [[Int]] = [[Int]](repeating: [], count: numCourses)
    for item in prerequisites {
        let cur = item[0] // 下一个课程
        let pre = item[1] // 前置课程
        degree[cur] += 1
        nextCourses[pre].append(cur) // 获取每个pre的list,在上面添加cur
    }

    var queue: [Int] = []
    for i in 0..<degree.count {
        if degree[i] == 0 {
            queue.append(i) // 入度为0的课程进入待打印队列
        }
    }
    var res: [Int] = []
    while queue.count>0 {
        let top = queue.popLast()! // 将入度为0的队列的课程最上面的课程取出
        res.append(top) // 打印出来
        let list = nextCourses[top] // top课程为前提的课程列表
        for x in list {
            degree[x] -= 1 // 列表课程前度全部减1
            if degree[x] == 0 { // 如果减为0, 加到队列中
                queue.append(x)
            }
        }
    }
    return res.count == numCourses
}
```

# 54.实现Trie（前缀树）

- Trie（发音类似 "try"）或者说 前缀树 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补完和拼写检查。

- 请你实现 Trie 类：

- Trie() 初始化前缀树对象。
- void insert(String word) 向前缀树中插入字符串 word 。
- boolean search(String word) 如果字符串 word 在前缀树中，返回 true（即，在检索之前已经插入）；否则，返回 false 。
- boolean startsWith(String prefix) 如果之前已经插入的字符串 word 的前缀之一为 prefix ，返回 true ；否则，返回 false 。

```Swift
class TrieNode {
    var children: [Character: TrieNode] = [:]
    var isWord = false
}

class Trie { 
    let root: TrieNode
    init() { 
        root = TrieNode()
    }
    
    func insert(_ word: String) {
        var cur = root
        for c in word {
            if let childNode = cur.children[c] {
                cur = childNode
            } else {
                cur.children[c] = TrieNode()
                cur = cur.children[c]!
            }
        }
        cur.isWord = true
    }
    
    func search(_ word: String) -> Bool {
        var cur = root
        for c in word {
            if let childNode = cur.children[c] {
                cur = childNode
            } else {
                return false
            }
        }
        return cur.isWord
    }
    
    func startsWith(_ prefix: String) -> Bool {
        var cur = root 
        for c in prefix {
            if let childNode = cur.children[c] {
                cur = childNode
            } else {
                return false
            } 
        }
        return true
    }
}
```

# 55.全排列

- 给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。

```Swift
func permute(_ nums: [Int]) -> [[Int]] {
    var result: [[Int]] = []
    var currentPermutation: [Int] = []
    var used: Set<Int> = Set()
    
    backtrack(&result, &currentPermutation, nums, &used)
    
    return result
}

func backtrack(_ result: inout [[Int]], _ currentPermutation: inout [Int], _ nums: [Int], _ used: inout Set<Int>) {
    if currentPermutation.count == nums.count {
        result.append(Array(currentPermutation))
        return
    }
    
    for num in nums {
        if used.contains(num) {
            continue
        }
        
        currentPermutation.append(num)
        used.insert(num)
        
        backtrack(&result, &currentPermutation, nums, &used)
        
        currentPermutation.removeLast()
        used.remove(num)
    }
}
```

# 56.子集

- 给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。
- 解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
- 输入：nums = [1,2,3]
- 输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

```Swift
class Solution {
    func subsets(_ nums: [Int]) -> [[Int]] {
        var result: [[Int]] = []
        var currentSubset: [Int] = []
        
        backtrack(&result, &currentSubset, nums, 0)
        
        return result
    }
    
    func backtrack(_ result: inout [[Int]], _ currentSubset: inout [Int], _ nums: [Int], _ startIndex: Int) {
        result.append(Array(currentSubset))
        
        for i in startIndex..<nums.count {
            currentSubset.append(nums[i])
            backtrack(&result, &currentSubset, nums, i + 1)
            currentSubset.removeLast()
        }
    }
}
```

# 57.电话号码的字母组合

- 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
- 给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

```Swift
class Solution {
    var letterCombinationsResult = [String]()
    var letterCombinationsString = ""

    func letterCombinations(_ digits: String) -> [String] {
        let data = [Character](digits).map { ch -> String in
            switch ch {
            case "2":
                return "abc"
            case "3":
                return "def"
            case "4":
                return "ghi"
            case "5":
                return "jkl"
            case "6":
                return "mno"
            case "7":
                return "pqrs"
            case "8":
                return "tuv"
            case "9":
                return "wxyz"
            default:
                return ""
            }
        }
        
        if data.count == 0 {
            return []
        }
        
        letterCombinationsDFS(data,0)
        
        return letterCombinationsResult
    }

    func letterCombinationsDFS(_ data:[String],_ index:Int) {
        
        if index > data.count - 1 {
            letterCombinationsResult.append(letterCombinationsString)
            return
        }
        
        let value = data[index]
        for c in value {
            letterCombinationsString.append(c)
            letterCombinationsDFS(data, index+1)
            letterCombinationsString.removeLast()
        }
    }
}
```

# 58.组合总和

- 给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。
- candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。 
- 对于给定的输入，保证和为 target 的不同组合数少于 150 个。

```Swift
class Solution {
    var combinationSumResult = [[Int]]()
    var combinationSumPaths = [Int]()
    func combinationSum(_ candidates: [Int], _ target: Int) -> [[Int]] {
        
        for i in 0..<candidates.count {
            combinationSumDFS(candidates, i, target)
        }
        
        return combinationSumResult
    }

    func combinationSumDFS(_ candidates: [Int],_ index:Int, _ target: Int) {
        
        if index > candidates.count - 1 {
            return
        }
        
        // 由于数组的值 > 0, 进行剪枝
        if target < 0 {
            return
        }
        
        combinationSumPaths.append(candidates[index])
        
        if candidates[index] == target {
            combinationSumResult.append(combinationSumPaths)
            combinationSumPaths.removeLast()
            return
        }
        
        let tempTarget = target - candidates[index]
        for i in index..<candidates.count {
            combinationSumDFS(candidates, i, tempTarget)
        }
        
        combinationSumPaths.removeLast()
    }
}
```

# 59.括号生成

- 数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。

```Swift
// MARK: 本题的核心思想在于如何在已知n-1对合法括号的情况下推导到n对合法括号。
/// 本题类似n个骰子的概率题。
/// dp的含义：
///     - dp[i]表示第i对所有的合法括号列表
///     - dp[0] = [""] 且 dp[1] = ["()"]
///     - dp[i]的递推公式为**"(" + dp[p]中任意一个合法括号 + ")" + dp[q]中任意一个合法括号**，其中**p+q=i-1**。
///     - 显然p的取值范围为0～i-1
func generateParenthesis(_ n: Int) -> [String] {
    if n == 1 { return ["()"] }
    var dp: [[String]] = [[""], ["()"]]
    for i in 2...n {
        var cur: [String] = []
        for p in 0..<i {
            /// **取dp[p]中任意一个合法括号**
            for k in dp[p] {
                /// **dp[q]中任意一个合法括号**
                for l in dp[i-1-p] {
                    cur.append("(" + k + ")" + l)
                }
            }
        }
        dp.append(cur)
    }
    return dp[n]
}
```

# 60.单词搜索

- 给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
- 单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

```Swift
class Solution {
        
    var res = false
    func exist(_ board: [[Character]], _ word: String) -> Bool {
        
        guard board.count > 0, word.count > 0 else {
            return false
        }

        var board = board

        var word = Array(word)

        var used = Array(repeating: Array(repeating: 0, count: board[0].count), count: board.count)

        for i in 0..<board.count {
            for j in 0..<board[0].count {

                if backtrack(&board, word, i, j, 0, &used) {
                    res = true
                }
            }
        }

        return res
    }

    func backtrack(_ board: inout [[Character]], _ word: [Character], _ i: Int, _ j: Int, _ index: Int, _ used: inout [[Int]]) -> Bool {

        if index >= word.count {
            return true
        }

        if i < 0 || j < 0 || i >= board.count || j >= board[0].count {
            return false
        }

        if board[i][j] != word[index] {
            return false
        }

        if used[i][j] == 1 {
            return false
        }

        used[i][j] = 1
 
        if backtrack(&board, word, i + 1, j, index + 1, &used) ||
        backtrack(&board, word, i - 1, j, index + 1, &used) ||
        backtrack(&board, word, i, j + 1, index + 1, &used) ||
        backtrack(&board, word, i, j - 1, index + 1, &used) {
            return true
        }

        used[i][j] = 0

        return false
    }
   
}
```

# 61.分割回文串

```Swift
// 主要函数，用于找到字符串的所有可能的回文子串分割方案
func partition(_ s: String) -> [[String]] {
    var result: [[String]] = [] // 存储最终分割结果的数组
    var currentPartition: [String] = [] // 当前正在构建的分割
    
    // 辅助函数，用于检查给定的字符串是否是回文
    func isPalindrome(_ str: String) -> Bool {
        let characters = Array(str)
        var left = 0
        var right = characters.count - 1
        
        // 通过从两端比较来检查字符是否形成回文
        while left < right {
            if characters[left] != characters[right] {
                return false
            }
            left += 1
            right -= 1
        }
        
        return true
    }
    
    // 回溯函数，用于探索所有可能的分割方案
    func backtrack(_ start: Int) {
        // 如果已经到达字符串的末尾，将当前分割添加到结果中
        if start == s.count {
            result.append(currentPartition)
            return
        }
        
        // 从当前位置开始探索所有可能的子串
        for end in start..<s.count {
            let startIndex = s.index(s.startIndex, offsetBy: start)
            let endIndex = s.index(s.startIndex, offsetBy: end)
            let substring = String(s[startIndex...endIndex])
            
            // 如果子串是回文，将其添加到当前分割中
            if isPalindrome(substring) {
                currentPartition.append(substring)
                // 继续从下一个位置开始探索分割
                backtrack(end + 1)
                // 移除最后添加的子串，以进行回溯并探索其他可能性
                currentPartition.removeLast()
            }
        }
    }
    
    // 从字符串的开头开始回溯过程
    backtrack(0)
    
    return result
}
```

# 62.N皇后

- 按照国际象棋的规则，皇后可以攻击与之处在同一行或同一列或同一斜线上的棋子。
- n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
- 给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。
- 每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。

```Swift
class Solution {
    var solveNQueensResult = [[Int]]() // 保存n皇后结果
    var solveNQueensPath = [Int]() // 临时结果

    var col = [Int]()          // 列方向
    var diagonals1 = [Int]()   // 斜线方向
    var diagonals2 = [Int]()   // 斜线方向

    func solveNQueens(_ n: Int) -> [[String]] {
        
        solveNQueensDFS(n, 0)
        
        // 结果转换 [0，0，0，1] -> "...Q"
        var tempResult = [[String]]()
        for result in solveNQueensResult {
            var tempData = [String]()
            for c in result {
                var tempString = ""
                for i in 0..<n {
                    if c == i {
                        tempString.append("Q")
                    }
                    else {
                        tempString.append(".")
                    }
                }
                tempData.append(tempString)
            }
            tempResult.append(tempData)
        }
        
        return tempResult
    }

    func solveNQueensDFS(_ n: Int, _ rowIndex: Int) {
        
        if rowIndex > n - 1 {
            solveNQueensResult.append(solveNQueensPath)
            return
        }
        
        for i in 0..<n {
            if col.contains(i) || diagonals1.contains(rowIndex-i) ||  diagonals2.contains(rowIndex+i){
                continue
            }
            
            col.append(i)
            diagonals1.append(rowIndex - i)
            diagonals2.append(rowIndex + i)
            solveNQueensPath.append(i)
            print(solveNQueensPath)
            solveNQueensDFS(n, rowIndex+1)
            solveNQueensPath.removeLast()
            diagonals2.removeLast()
            diagonals1.removeLast()
            col.removeLast()
        }
    }
}
```

# 63.搜索插入位置

- 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
- 请必须使用时间复杂度为 O(log n) 的算法。

```Swift
func searchInsert(_ nums: [Int], _ target: Int) -> Int {
    var low = 0
    var high = nums.count - 1
    while low <= high {
        let mid = low + (high - low) / 2
        if nums[mid] == target {
            return mid
        } else if nums[mid] < target {
            low = mid + 1
        } else {
            high = mid - 1
        }
    }
    return low
}
```

# 64.搜索二维矩阵

```Swift
func searchMatrix(_ matrix: [[Int]], _ target: Int) -> Bool {
    guard matrix.count > 0 && matrix[0].count > 0 else {
        return false
    }

    let rows = matrix.count
    let cols = matrix[0].count

    // 从右上角开始搜索
    var row = 0
    var col = cols - 1

    while row < rows && col >= 0 {
        let current = matrix[row][col]
        if current == target {
            return true
        } else if (current < target) {
            row += 1
        } else if (current > target) {
            col -= 1
        }
    }

    return false
}
```

# 65.在排序数组中查找元素的第一个和最后一个位置

```Swift
class Solution {
    func searchRange(_ nums: [Int], _ target: Int) -> [Int] {
        var result = [-1, -1]

        // 查找第一个出现的位置
        result[0] = findFirst(nums, target)

        // 查找最后一个出现的位置
        result[1] = findLast(nums, target)

        return result
    }

    func findFirst(_ nums: [Int], _ target: Int) -> Int {
        var left = 0
        var right = nums.count - 1
        var result = -1

        while left <= right {
            let mid = left + (right - left) / 2

            if nums[mid] >= target {
                right = mid - 1
            } else {
                left = mid + 1
            }

            if nums[mid] == target {
                result = mid
            }
        }

        return result
    }

    func findLast(_ nums: [Int], _ target: Int) -> Int {
        var left = 0
        var right = nums.count - 1
        var result = -1

        while left <= right {
            let mid = left + (right - left) / 2

            if nums[mid] <= target {
                left = mid + 1
            } else {
                right = mid - 1
            }

            if nums[mid] == target {
                result = mid
            }
        }

        return result
    }
}
```

# 66.搜索旋转排序数组

```Swift
func search(_ nums: [Int], _ target: Int) -> Int {
    var left = 0
    var right = nums.count - 1

    while left <= right {
        let mid = left + (right - left) / 2

        if nums[mid] == target {
            return mid
        }

        if nums[left] <= nums[mid] {
            // 左半部分有序
            if nums[left] <= target && target < nums[mid] {
                right = mid - 1
            } else {
                left = mid + 1
            }
        } else {
            // 右半部分有序
            if nums[mid] < target && target <= nums[right] {
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
    }
    return -1
}
```

# 67.寻找旋转排序数组中的最小值

```Swift
class Solution {
    func findMin(_ nums: [Int]) -> Int {
        var left = 0
        var right = nums.count - 1

        while left < right {
            let mid = left + (right - left) / 2

            if nums[mid] > nums[right] {
                left = mid + 1
            } else if nums[mid] < nums[right] {
                right = mid
            } else {
                right -= 1
            }
        }

        return nums[left]
    }
}
```

# 68.寻找两个正序数组的中位数

- 给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
- 算法的时间复杂度应该为 O(log (m+n)) 。

```Swift
func findMedianSortedArrays(_ nums1: [Int], _ nums2: [Int]) -> Double {
    var mergeArray = [Int](repeating: 0, count: nums1.count+nums2.count)
    var i = 0
    var j = 0

    for index in 0..<mergeArray.count {
    
    if (j >= nums2.count) || (i < nums1.count && nums1[i] <= nums2[j])  {
        mergeArray[index] = nums1[i]
        i+=1
    } else  {
        mergeArray[index] = nums2[j]
        j+=1
    }
    
    }

    if mergeArray.count % 2 == 0 {
    return Double(mergeArray[mergeArray.count/2-1] + mergeArray[mergeArray.count/2]) / 2.0
    } else {
    return Double(mergeArray[mergeArray.count/2])
    }
}
```

# 69.有效的括号

```Swift
func isValid(_ s: String) -> Bool {
    var left: [String] = []
    let map = ["}" : "{", ")" : "(", "]" : "["]

    for char in s {
        if char == "{" || char == "[" || char == "(" {
            left.append(String(char))
        } else {
            if let last = left.last {
                if last == map[String(char)] {
                    left.removeLast()
                } else { 
                    return false
                }
            } else {
                return false
            }
        }
    }

    return (left.count == 0)
}
```

# 70.最小栈

- 设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。
- 实现 MinStack 类:
- MinStack() 初始化堆栈对象。
- void push(int val) 将元素val推入堆栈。
- void pop() 删除堆栈顶部的元素。
- int top() 获取堆栈顶部的元素。
- int getMin() 获取堆栈中的最小元素。

```Swift
class MinStack {

    var stack: [Int] = []

    var minStack: [Int] = []

    init() {
        stack = []
        minStack = []
    }
    
    func push(_ val: Int) {
        stack.append(val)
        if let currentMin = minStack.last {
            minStack.append(min(val, currentMin))
        } else {
            minStack.append(val)
        }
    }
    
    func pop() {
        stack.removeLast()
        minStack.removeLast()
    }
    
    func top() -> Int {
        return stack.last!
    }
    
    func getMin() -> Int {
        return minStack.last!
    }
}   
```

# 71.字符串解码

```Swift
func decodeString(_ s: String) -> String {
    // 保存字符和重复次数的栈
    var stack: [(String, Int)] = []
    // 当前数字
    var currentNunmber = 0
    // 当前字符串
    var currentString = ""
    for char in s {
        if char.isNumber {
            // *10是为了正确处理多位数
            currentNunmber = currentNunmber * 10 + Int(String(char))!
        } else if char == "[" {
            stack.append((currentString, currentNunmber))
            currentNunmber = 0
            currentString = ""
        } else if char == "]" {
            let (preString, count) = stack.removeLast()
            currentString = preString + String(repeating: currentString, count: count)
        } else {
            currentString.append(char)
        }
    }
    return currentString
}
```

# 72.每日温度

```Swift
func dailyTemperatures(_ temperatures: [Int]) -> [Int] {
    let n = temperatures.count
    var result = Array(repeating: 0, count: n)
    var stack: [Int] = []

    for i in 0..<n {
        // 当栈不为空且当前温度大于栈顶温度时，栈顶温度出栈，计算距离并更新结果
        while !stack.isEmpty && temperatures[i] > temperatures[stack.last!] {
            let lastIndex = stack.removeLast()
            result[lastIndex] = i - lastIndex
        }
        
        // 将当前温度入栈
        stack.append(i)
    }

    return result
}
```

# 73.柱状图中最大的矩形

- 给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。
- 求在该柱状图中，能够勾勒出来的矩形的最大面积。

```Swift
func largestRectangleArea(_ heights: [Int]) -> Int {
    var heights = [0] + heights + [0]
    var maxArea: Int = 0
    var stack: [Int] = []

    for (index, height) in heights.enumerated() {

        while !stack.isEmpty && stack.last != nil && heights[stack.last!] > height {
            let removed = stack.removeLast()   
            if let leftEdge = stack.last {
                var rightEdge = index
                maxArea = max(maxArea, heights[removed] * (rightEdge - 1 - leftEdge))
            }
        }
        stack.append(index)
    }

    return maxArea
}
```

# 74.数组中的第K个最大元素

- 给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。
- 请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
- 你必须设计并实现时间复杂度为 O(n) 的算法解决此问题。

```Swift
class Solution {
    // 主函数，找到数组中第k个最大的元素
    func findKthLargest(_ nums: [Int], _ k: Int) -> Int {
        // 创建一个最小堆，初始包含数组前k个元素
        var minHeap = Array(nums[0..<k])
        heapify(&minHeap) // 对堆进行初始化
        var mn = minHeap[0] // 最小堆的堆顶元素，即目前的第k个最大元素

        // 遍历数组中剩余的元素
        for i in k..<nums.count {
            // 如果当前元素比最小堆的堆顶元素大，替换堆顶元素
            if nums[i] > mn {
                heapReplace(&minHeap, nums[i])
                mn = minHeap[0] // 更新最小堆的堆顶元素
            }
        }

        return mn
    }

    // 将数组转化为最小堆
    func heapify(_ nums: inout [Int]) {
        var i = nums.count / 2 - 1
        while i >= 0 {
            siftDown(&nums, i, nums.count)
            i -= 1
        }
    }

    // 替换最小堆的堆顶元素，并重新调整堆
    func heapReplace(_ nums: inout [Int], _ newValue: Int) {
        nums[0] = newValue
        siftDown(&nums, 0, nums.count)
    }

    // 将指定位置的元素下移，维持最小堆的性质
    func siftDown(_ nums: inout [Int], _ index: Int, _ heapSize: Int) {
        var currentIndex = index
        while true {
            let leftChild = 2 * currentIndex + 1
            let rightChild = 2 * currentIndex + 2
            var smallest = currentIndex

            // 找到左右子节点中较小的那个
            if leftChild < heapSize && nums[leftChild] < nums[smallest] {
                smallest = leftChild
            }

            if rightChild < heapSize && nums[rightChild] < nums[smallest] {
                smallest = rightChild
            }

            // 如果最小的节点就是当前节点，停止下移
            if smallest == currentIndex {
                break
            }

            // 交换当前节点与最小节点的值
            nums.swapAt(currentIndex, smallest)
            currentIndex = smallest
        }
    }
}
```

# 75.前K个高频元素

- 给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。

```Swift
func topKFrequent(_ nums: [Int], _ k: Int) -> [Int] {
    var hashTable: [Int: Int] = [:]
    for num in nums {
        if let count = hashTable[num] {
            hashTable[num] = count + 1
        } else {
            hashTable[num] = 1
        }
    }

    var bucket: [[Int]] = Array(repeating: [], count: nums.count + 1)
    // 桶排序，将出现的频率作为数组的下标位置，存入num
    for key in hashTable.keys {
        if let count = hashTable[key] {
            var array = bucket[count]
            if array.isEmpty {
                bucket[count] = [key]
            } else {
                bucket[count] = array + [key]
            }
        }
    }

    var k = k 

    var targetArray: [Int] = []
    for array in bucket.reversed() {
        if k == 0 { return targetArray }
        if array.isEmpty {
            continue
        } else {
            targetArray.append(contentsOf: array)
            k = k - array.count
        }
    }
    return targetArray
}
```

# 76.数据流的中位数

- 中位数是有序整数列表中的中间值。如果列表的大小是偶数，则没有中间值，中位数是两个中间值的平均值。
- 例如 arr = [2,3,4] 的中位数是 3 。
- 例如 arr = [2,3] 的中位数是 (2 + 3) / 2 = 2.5 。
- 实现 MedianFinder 类:

- MedianFinder() 初始化 MedianFinder 对象。
- void addNum(int num) 将数据流中的整数 num 添加到数据结构中。
- double findMedian() 返回到目前为止所有元素的中位数。与实际答案相差 10-5 以内的答案将被接受。

```Swift
class MedianFinder { 
    var arr = [Int]()
    init() {

    }
    
    func addNum(_ num: Int) {
        if arr.count == 0 || num > arr.last! {
            arr.append(num)
        } else {
            let index = bs(arr, num)
            arr.insert(num, at: index)
        } 
    }
    
    func findMedian() -> Double {
        let mid = (arr.count - 1) / 2
        if arr.count % 2 == 0 {
            return Double(arr[mid] + arr[mid + 1]) / 2.0
        } else {
            return Double(arr[mid])
        }
    } 

    func bs(_ arr: [Int], _ target: Int) -> Int {
        var left = 0, right = arr.count - 1
        while left < right {
            let mid = left + (right - left) / 2
            if arr[mid] < target {
                left = mid + 1
            } else {
                right = mid
            }
        }
        return left
    }
}
```

# 77.买卖股票的最佳时机

```Swift
func maxProfit(_ prices: [Int]) -> Int {
    var cost = prices[0]
    var profit = 0
    for (index, price) in prices.enumerated() {
        cost = min(cost, price)
        profit = max(profit, price - cost)
    }
    return profit
}
```

# 78.跳跃游戏

- 给你一个非负整数数组 nums ，你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。
- 判断你是否能够到达最后一个下标，如果可以，返回 true ；否则，返回 false 。

```Swift
func canJump(_ nums: [Int]) -> Bool {
    var maxReach = 0

    for (index, num) in nums.enumerated() {
        // 当前位置超过最远距离
        if index > maxReach {
            return false
        }
        maxReach = max(maxReach, index + num)
        if maxReach >= nums.count - 1 {
            return true
        }
    }
    return false
}
```

# 79.跳跃游戏 II

- 给定一个长度为 n 的 0 索引整数数组 nums。初始位置为 nums[0]。
- 每个元素 nums[i] 表示从索引 i 向前跳转的最大长度。换句话说，如果你在 nums[i] 处，你可以跳转到任意 nums[i + j] 处:
- 0 <= j <= nums[i] 
- i + j < n
- 返回到达 nums[n - 1] 的最小跳跃次数。生成的测试用例可以到达 nums[n - 1]。

```Swift
func jump(_ nums: [Int]) -> Int {
    if nums.count == 1 {
        return 0
    }

    var steps = 0
    var maxReach = 0
    var lastJump = 0

    for i in 0..<nums.count - 1 {
        maxReach = max(maxReach, i + nums[i])

        if i == lastJump {
            // 当达到上一次跳跃的最大范围时，进行下一次跳跃
            lastJump = maxReach
            steps += 1
        }
    }

    return steps
}
```

# 80.划分字母区间

- 给你一个字符串 s 。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。
- 注意，划分结果需要满足：将所有划分结果按顺序连接，得到的字符串仍然是 s 。
- 返回一个表示每个字符串片段的长度的列表。

```Swift
func partitionLabels(_ s: String) -> [Int] {
    var lastIndices = [Int](repeating: 0, count: 26)
    let asciiOffset = Int(Character("a").asciiValue!)

    // 记录每个字母最后出现的位置
    for (index, char) in s.enumerated() {
        let charIndex = Int(char.asciiValue!) - asciiOffset
        lastIndices[charIndex] = index
    }

    var partitions = [Int]()
    var start = 0
    var end = 0

    // 遍历字符串，找到每个字母的结束位置，形成字母区间
    for (index, char) in s.enumerated() {
        let charIndex = Int(char.asciiValue!) - asciiOffset
        end = max(end, lastIndices[charIndex])

        if index == end {
            // 当遍历到字母的结束位置时，形成一个字母区间
            partitions.append(end - start + 1)
            start = index + 1
        }
    }

    return partitions
}
```

# 81.爬楼梯

```Swift
func climbStairs(_ n: Int) -> Int {
    if n < 3 {
        return n
    }
    var map = [Int : Int]()
    map[1] = 1
    map[2] = 2
    for index in 3...n {
        map[index] = map[index - 1]! + map[index - 2]!
    }
    return map[n]!
}
```

# 82.杨辉三角

```Swift
func generate(_ numRows: Int) -> [[Int]] {
    var triangle = [[Int]]()
    for i in 0..<numRows {
        var row = [Int](repeating: 1, count: i + 1)
        // 从第三行开始迭代
        if i > 1 {
            for j in 1..<i {
                // 计算当前行的非首尾元素。
                // 在第三行及以后的行，非首尾元素的值是上一行对应位置和前一位置的元素之和。
                // triangle[i - 1]代表上一行
                row[j] = triangle[i - 1][j - 1] + triangle[i - 1][j]
            }
        }
        triangle.append(row)
    }
    return triangle
}
```

# 83.打家劫舍

```Swift
func rob(_ nums: [Int]) -> Int {
    guard nums.count > 0 else {
        return 0
    }

    if nums.count == 1 {
        return nums[0]
    }

    // dp 数组用来存储在每个位置上选择或不选择偷窃时的最大金额。
    // dp[i] 表示第 i 个位置的最大金额，nums[i] 表示当前位置的金额。
    var dp = Array(repeating: 0, count: nums.count)
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])

    for i in 2..<nums.count {
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
    }

    return dp[nums.count - 1]
}
```

# 84.完全平方数

- 给你一个整数 n ，返回 和为 n 的完全平方数的最少数量

```Swift
class Solution {
    func numSquares(_ n: Int) -> Int {
        var dp = Array(repeating: Int.max, count: n + 1)
        dp[0] = 0

        // 遍历每个数
        for i in 1...n {
            var j = 1
            // 遍历所有可能的完全平方数
            while i - j * j >= 0 {
                dp[i] = min(dp[i], dp[i - j * j] + 1)
                j += 1
            }
        }

        return dp[n]
    }
}
```

# 85.零钱兑换

```Swift
func coinChange(_ coins: [Int], _ amount: Int) -> Int {
    // 用amount + 1来代表不可能
    var dp = [Int](repeating: amount + 1, count: amount + 1)
    dp[0] = 0
    for i in 0..<dp.count {
        for coin in coins {
            // 硬币值大于总量，直接跳过
            if i < coin {
                continue
            }
            // dp[i]表示凑出总量为 i 的硬币最小值
            dp[i] = min(dp[i], 1 + dp[i - coin])
        }
    }
    return (dp[amount] == amount + 1) ? -1 : dp[amount]
}
```

# 86.单词拆分

```Swift
func wordBreak(_ s: String, _ wordDict: [String]) -> Bool {
    let sArray = Array(s) // 将输入字符串 s 转换为字符数组 sArray
    let wordSet = Set(wordDict) // 将 wordDict 转换为集合 wordSet，方便快速查找单词
    let n = s.count // 获取字符串 s 的长度

    // dp[i] 表示 s 的前 i 个字符是否可以拆分成 wordDict 中的单词
    var dp = Array(repeating: false, count: n + 1)
    dp[0] = true // 空字符串可以被拆分

    for i in 1...n {
        for j in 0..<i {
            let startIndex = s.index(s.startIndex, offsetBy: j) // 获取子字符串的起始索引
            let endIndex = s.index(s.startIndex, offsetBy: i) // 获取子字符串的结束索引
            let word = String(s[startIndex..<endIndex]) // 从 s 中获取子字符串 word

            // 如果 dp[j] 为 true（s 的前 j 个字符可以被拆分）且 wordSet 中包含当前子字符串 word
            if dp[j] && wordSet.contains(word) {
                dp[i] = true // 则表示 s 的前 i 个字符可以被拆分
                break
            }
        }
    }

    return dp[n] // 返回结果，表示整个字符串 s 是否可以被拆分成 wordDict 中的单词
}
```

# 87.最长递增子序列

- 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
- 子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

```Swift
class Solution {
    func lengthOfLIS(_ nums: [Int]) -> Int {
        guard !nums.isEmpty else {
            return 0
        }

        let n = nums.count
        var dp = Array(repeating: 1, count: n)

        for i in 0..<n {
            for j in 0..<i {
                if nums[i] > nums[j] {
                    dp[i] = max(dp[i], dp[j] + 1)
                }
            }
        }

        return dp.max()!
    }
}
```

# 88.乘积最大子数组

- 给你一个整数数组 nums ，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
- 测试用例的答案是一个 32-位 整数。
- 子数组 是数组的连续子序列。

```Swift
func maxProduct(_ nums: [Int]) -> Int {
    guard nums.count > 1 else { return nums.first ?? 0 }

    var result: Int = .min
    var imin: Int = 1
    var imax: Int = 1

    for i in 0 ..< nums.count {
        if nums[i] < 0 { (imin, imax) = (imax, imin) }

        imax = max(nums[i], imax * nums[i])
        imin = min(nums[i], imin * nums[i])

        result = max(result, imax)
    }

    return result
}
```

# 89.分割等和子集

- 给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

```Swift
func canPartition(_ nums: [Int]) -> Bool {
    var sum = 0
    for i in 0..<nums.count {
        let num = nums[i]
        sum += num
    }

    if sum % 2 != 0 {
        return false
    }
    
    let count = nums.count
    sum = sum / 2
    
    var dp = [Bool](repeating: false, count: sum + 1)
    dp[0] = true
    
    for i in 0..<count {
        for j in (0..<dp.count).reversed() {
            if j - nums[i] >= 0 {
                dp[j] = dp[j] || dp[j - nums[i]]
            }
        }
    }
    return dp[sum]
}
```

# 90.最长有效括号

- 给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。

```Swift
func longestValidParentheses(_ s: String) -> Int {
    var strArr = Array(s)
    var stack = [-1]
    var result = 0
    for i in 0 ..< strArr.count {
        if strArr[i] == "(" {
            stack.append(i)
        } else {
            if stack.count > 1 && strArr[stack.last!] == "(" {
                stack.removeLast()
                result = max(result, i - stack.last!)
            } else {
                stack.append(i)
            }
        }
    }
    return result
}
```

# 91.不同路径

- 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
- 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
- 问总共有多少条不同的路径？

```Swift
func uniquePaths(_ m: Int, _ n: Int) -> Int {
    var dp = [Int](repeating: 1, count: m)
    for _ in 1..<n {
        for col in 1..<m {
            dp[col] = dp[col] + dp[col-1]
        }
    }
    return dp[m-1]
}
```


# 92.最小路径和

- 给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
- 说明：每次只能向下或者向右移动一步。

```Swift
func minPathSum(_ grid: [[Int]]) -> Int {
    var dp = [[Int]](repeating:[Int](repeating: 0, count: grid[0].count), count: grid.count)
    
    dp[0][0] = grid[0][0]
    
    for i in 0..<grid.count {
        for j in 0..<grid[0].count {
            if i > 0 && j > 0 {
                dp[i][j] = min(dp[i-1][j],dp[i][j-1]) + grid[i][j]
            } else if i > 0 {
                dp[i][j] = dp[i-1][j] + grid[i][j]
            } else if j > 0{
                dp[i][j] = dp[i][j-1] + grid[i][j]
            }
        }
    }
    
    return dp.last!.last!
}
```

# 93.最长回文子串

- 给你一个字符串 s，找到 s 中最长的回文子串。

```Swift
func longestPalindrome(_ s: String) -> String {
    if s.isEmpty {
        return "" // 如果输入字符串为空，直接返回空字符串
    }

    let n = s.count // 获取输入字符串的长度
    let sArray = Array(s) // 将输入字符串转换为字符数组
    var start = 0 // 记录最长回文子串的起始位置
    var maxLength = 1 // 记录最长回文子串的长度

    // 辅助函数，用于扩展回文中心
    func expandAroundCenter(_ left: Int, _ right: Int) {
        var l = left
        var r = right

        // 扩展回文中心，直到左右两边不相等
        while l >= 0 && r < n && sArray[l] == sArray[r] {
            if r - l + 1 > maxLength {
                start = l
                maxLength = r - l + 1
            }
            l -= 1
            r += 1
        }
    }

    for i in 0..<n {
        // 以当前字符为中心扩展
        expandAroundCenter(i, i)

        // 以当前字符和下一个字符之间的空隙为中心扩展
        expandAroundCenter(i, i + 1)
    }

    let startIndex = s.index(s.startIndex, offsetBy: start)
    let endIndex = s.index(startIndex, offsetBy: maxLength - 1)
    return String(s[startIndex...endIndex]) // 返回最长回文子串
}
```

# 94.最长公共子序列

- 给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。
- 一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
- 例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
- 两个字符串的 公共子序列 是这两个字符串所共同拥有的子序列。

```Swift
func longestCommonSubsequence(_ text1: String, _ text2: String) -> Int {

    var dp: [[Int]] = Array(repeating: Array(repeating: 0, count: text2.count + 1), count: text1.count + 1)

    let array1 = Array(text1)
    let array2 = Array(text2)

    for i in 1...array1.count {
        for j in 1...array2.count {
            if array1[i - 1] == array2[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1
            } else {
                dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
            }
        }
    }

    return dp[text1.count][text2.count]
}
```

# 95.编辑距离
- 给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数  。
- 你可以对一个单词进行如下三种操作：
- 插入一个字符
- 删除一个字符
- 替换一个字符

```Swift
func minDistance(_ word1: String, _ word2: String) -> Int {
    let m = word1.count
    let n = word2.count

    if n == 0 || m == 0 {
        return max(m, n)
    }
    
    // 初始化动态规划数组
    var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)
    
    // 初始化边界条件
    for i in 0...m {
        dp[i][0] = i
    }
    
    for j in 0...n {
        dp[0][j] = j
    }
    
    // 动态规划递推
    for i in 1...m {
        for j in 1...n {
            if word1[word1.index(word1.startIndex, offsetBy: i-1)] == word2[word2.index(word2.startIndex, offsetBy: j-1)] {
                dp[i][j] = dp[i-1][j-1]
            } else {
                dp[i][j] = min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1]) + 1
            }
        }
    }
    
    return dp[m][n]
}
```

# 96.只出现一次的数字

- 给你一个 非空 整数数组 nums ，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
- 你必须设计并实现线性时间复杂度的算法来解决此问题，且该算法只使用常量额外空间。

```Swift
func singleNumber(_ nums: [Int]) -> Int {
    var result = 0
    for num in nums {
        result ^= num
    }
    return result
}
```

# 97.多数元素

- 给定一个大小为 n 的数组 nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。
- 你可以假设数组是非空的，并且给定的数组总是存在多数元素。

```Swift
func majorityElement(_ nums: [Int]) -> Int {
    var count = 0
    var majority = 0
    for num in nums {
        if count == 0 {
            majority = num
            count += 1
        } else if (majority == num) {
            count += 1
        } else {
            count -= 1
        }
    }
    return majority
}
```

# 98.颜色分类

- 给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums ，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
- 我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
- 必须在不使用库内置的 sort 函数的情况下解决这个问题。

```Swift
func sortColors(_ nums: inout [Int]) {
    var low = 0
    var high = nums.count - 1
    var current = 0

    while current <= high {
        if nums[current] == 0 {
            // 交换当前元素和低位元素
            (nums[current], nums[low]) = (nums[low], nums[current])
            low += 1
            current += 1
        } else if nums[current] == 2 {
            // 交换当前元素和高位元素
            (nums[current], nums[high]) = (nums[high], nums[current])
            high -= 1
        } else {
            // 当前元素为1，直接移动到下一位
            current += 1
        }
    }
}
```

# 99.下一个排列

- 整数数组的一个 排列  就是将其所有成员以序列或线性顺序排列。
- 例如，arr = [1,2,3] ，以下这些都可以视作 arr 的排列：[1,2,3]、[1,3,2]、[3,1,2]、[2,3,1] 。
- 整数数组的 下一个排列 是指其整数的下一个字典序更大的排列。更正式地，如果数组的所有排列根据其字典顺序从小到大排列在一个容器中，那么数组的 下一个排列 就是在这个有序容器中排在它后面的那个排列。如果不存在下一个更大的排列，那么这个数组必须重排为字典序最小的排列（即，其元素按升序排列）。
- 例如，arr = [1,2,3] 的下一个排列是 [1,3,2] 。
- 类似地，arr = [2,3,1] 的下一个排列是 [3,1,2] 。
- 而 arr = [3,2,1] 的下一个排列是 [1,2,3] ，因为 [3,2,1] 不存在一个字典序更大的排列。
- 给你一个整数数组 nums ，找出 nums 的下一个排列。
- 必须 原地 修改，只允许使用额外常数空间。

```Swift
func nextPermutation(_ nums: inout [Int]) {
    // 从右向左找到第一个非递增的位置
    var i = nums.count - 2
    while i >= 0 && nums[i] >= nums[i + 1] {
        i -= 1
    }

    // 如果找到了非递增的位置，再从右向左找到第一个大于nums[i]的数
    if i >= 0 {
        var j = nums.count - 1
        while j >= 0 && nums[j] <= nums[i] {
            j -= 1
        }
        // 交换两个数
        nums.swapAt(i, j)
    }

    // 将i后面的部分反转
    var left = i + 1
    var right = nums.count - 1
    while left < right {
        nums.swapAt(left, right)
        left += 1
        right -= 1
    }
}
```

# 100.寻找重复数

- 给定一个包含 n + 1 个整数的数组 nums ，其数字都在 [1, n] 范围内（包括 1 和 n），可知至少存在一个重复的整数。
- 假设 nums 只有 一个重复的整数 ，返回 这个重复的数 。
- 你设计的解决方案必须 不修改 数组 nums 且只用常量级 O(1) 的额外空间。

```Swift
func findDuplicate(_ nums: [Int]) -> Int {
    var slow = nums[0]
    var fast = nums[0]

    // 利用快慢指针找到相遇点
    repeat {
        slow = nums[slow]
        fast = nums[nums[fast]]
    } while slow != fast

    // 将其中一个指针移回起点，然后两个指针以相同的速度移动，直到它们再次相遇
    slow = nums[0]
    while slow != fast {
        slow = nums[slow]
        fast = nums[fast]
    }

    return slow
}
```



