# Dart基础用法

## 字符串插值

下面的方法接收两个整形变量作为参数，然后让它返回一个包含以空格分隔的整数的字符串。例如，stringify(2, 3) 应该返回 '2 3'。

```dart
String stringify(int x, int y) {
  return '$x $y';
}
```

## 可空的变量

一个可空的 String，名为 name，值为 'Jane'。
一个可空的 String，名为 address，值为 null。

```dart
String? name = 'Jane';
String? address;
```