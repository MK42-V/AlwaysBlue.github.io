# Flutter Widget

使用MainAxisAlignment.spaceBetween和Spacer()可以实现在同一行中，第一个Widget靠左，最后一个Widget靠右的效果

```dart
Row(
    mainAxisAlignment: MainAxisAlignment.spaceBetween,
    children: [
        SizedBox(width: 20),
        SingleProduct(product: leftProduct),
        Spacer(),
        SingleProduct(product: middleProduct),
        Spacer(),
        SingleProduct(product: rightProduct),
        SizedBox(width: 20),
    ],
),
```

Standard StatefullWidget

```dart
class MyWidget extends StatefulWidget {
  const MyWidget({super.key});

  @override
  State<MyWidget> createState() => _MyWidgetState();
}

class _MyWidgetState extends State<MyWidget> {
  @override
  Widget build(BuildContext context) {
    return const Placeholder();
  }
}
```