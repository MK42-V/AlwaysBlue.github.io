# Swift Charts

### 快速搭建一个图表
1. 确定数据结构
```swift
struct ToyShape: Identifiable {
    var type: String
    var count: Double
    var id = UUID()
}
```
2. 初始化数据
```swift
var data: [ToyShape] = [
    .init(type: "Cube", count: 5),
    .init(type: "Sphere", count: 4),
    .init(type: "Pyramid", count: 4)
]
```
3. 创建Chart（使用BarMark）
```swift
import SwiftUI
import Charts

struct BarChart: View {
    var body: some View {
        Chart {
            ForEach(data) { shape in
                BarMark(
                    x: .value("Shape Type", shape.type),
                    y: .value("Total Count", shape.count)
                )
            }
        }
    }
}
```

![](./pics/quick_chart1.png)

4. 使用不同颜色区分
```swift
struct ToyShape: Identifiable {
    var color: String
    var type: String
    var count: Double
    var id = UUID()
}

var stackedBarData: [ToyShape] = [
    .init(color: "Green", type: "Cube", count: 2),
    .init(color: "Green", type: "Sphere", count: 0),
    .init(color: "Green", type: "Pyramid", count: 1),
    .init(color: "Purple", type: "Cube", count: 1),
    .init(color: "Purple", type: "Sphere", count: 1),
    .init(color: "Purple", type: "Pyramid", count: 1),
    .init(color: "Pink", type: "Cube", count: 1),
    .init(color: "Pink", type: "Sphere", count: 2),
    .init(color: "Pink", type: "Pyramid", count: 0),
    .init(color: "Yellow", type: "Cube", count: 1),
    .init(color: "Yellow", type: "Sphere", count: 1),
    .init(color: "Yellow", type: "Pyramid", count: 2)
]

Chart {
    ForEach(stackedBarData) { shape in
        BarMark(
            x: .value("Shape Type", shape.type),
            y: .value("Total Count", shape.count)
        )
        .foregroundStyle(by: .value("Shape Color", shape.color))
    }
}
```

![](./pics/quick_chart2.png)

5. 自定义颜色图示含义
```swift
Chart {
    ForEach(stackedBarData) { shape in
        BarMark(
            x: .value("Shape Type", shape.type),
            y: .value("Total Count", shape.count)
        )
        .foregroundStyle(by: .value("Shape Color", shape.count))
    }  
}
.chartForegroundStyleScale([
    "Green": .green, "Purple": .purple, "Pink": .pink, "Yellow": .yellow
])
```

![](./pics/quick_chart3.png)

### 图表种类

- AreaMark
![](./pics/mark_area.png)

- LineMark
![](./pics/mark_line.png)

- PointMark
![](./pics/mark_point.png)

- RectangleMark
![](./pics/mark_rectangle.png)

- RuleMark
![](./pics/mark_rule.png)

- BarMark
![](./pics/mark_bar.png)

