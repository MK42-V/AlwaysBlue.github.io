# Dart基础用法

### 字符串插值

使用 ${expression}将表达式的值放在字符串中，若表达式为单个标识符，则可以省略 {}。

代码样例
- 下面的方法接收两个整形变量作为参数，然后让它返回一个包含以空格分隔的整数的字符串。例如，stringify(2, 3) 应该返回 '2 3'。

```dart
/// 字符串	-> 	结果
/// '${3 + 2}'	  -> 	'5'
/// '${"word".toUpperCase()}'	   ->  	'WORD'
/// '$myObject' 	 -> 	myObject.toString() 的值

String stringify(int x, int y) {
  return '$x $y';
}
```

### 可空的变量

Dart 2.12 引入了健全的空安全，这意味着除非变量显式声明为可空类型，否则它们将不能为空。换句话说，类型默认是不可为空的。

代码样例
```dart
int a = null; // INVALID in null-safe Dart. ERROR
int? a = null; // Valid in null-safe Dart.
int? a; // The initial value of a is null.

String? name = 'Jane';
String? address;
```

### 避空运算符

Dart 提供了一系列方便的运算符用于处理可能会为空值的变量。其中一个是 ??= 赋值运算符，仅当该变量为空值时才为其赋值：

```dart
int? a; // = null
a ??= 3;
print(a); // <-- Prints 3.

a ??= 5;
print(a); // <-- Still prints 3.
```

另外一个避空运算符是 ??，如果该运算符左边的表达式返回的是空值，则会计算并返回右边的表达式。

```dart
print(1 ?? 3); // <-- Prints 1.
print(null ?? 12); // <-- Prints 12.
```

代码样例
```dart
String? foo = 'a string';
String? bar; // = null

// Substitute an operator that makes 'a string' be assigned to baz.
String? baz = foo ?? bar;

void updateSomeVars() {
  // Substitute an operator that makes 'a string' be assigned to bar.
  bar ??= 'a string';
}
```

### 条件属性访问

要保护可能会为空的属性的正常访问，请在点（.）之前加一个问号（?）。

代码样例
```dart
myObject?.someProperty

/// 上述代码等效于以下内容：
(myObject != null) ? myObject.someProperty : null

/// 可以在一个表达式中连续使用多个 ?.：
myObject?.someProperty?.someMethod()

// This method should return the uppercase version of `str`
// or null if `str` is null.
String? upperCaseIt(String? str) {
  return str?.toUpperCase();
}
```

### 集合字面量 (Collection literals)

Dart 内置了对 list、map 以及 set 的支持，可以通过字面量直接创建它们。Dart 的类型推断可以自动帮你分配这些变量的类型。在这个例子中，推断类型是 List<String>、Set<String>和 Map<String, int>。

```dart
final aListOfStrings = ['one', 'two', 'three'];
final aSetOfStrings = {'one', 'two', 'three'};
final aMapOfStringsToInts = {
  'one': 1,
  'two': 2,
  'three': 3,
};
```

你也可以手动指定类型：

```dart
final aListOfInts = <int>[];
final aSetOfInts = <int>{};
final aMapOfIntToDouble = <int, double>{};
```

在使用子类型的内容初始化列表，但仍希望列表为 List <BaseType> 时，指定其类型很方便：
```dart
final aListOfBaseType = <BaseType>[SubType(), SubType()];
```

代码样例

```dart
// Assign this a list containing 'a', 'b', and 'c' in that order:
final aListOfStrings = ['a', 'b', 'c'];

// Assign this a set containing 3, 4, and 5:
final aSetOfInts = {3, 4, 5};

// Assign this a map of String to int so that aMapOfStringsToInts['myKey'] returns 12:
final aMapOfStringsToInts = {'myKey': 12};

// Assign this an empty List<double>:
final anEmptyListOfDouble = <double>[];

// Assign this an empty Set<String>:
final anEmptySetOfString = <String>{};

// Assign this an empty Map of double to int:
final anEmptyMapOfDoublesToInts = <double, int>{};
```

### 箭头语法

### 级联

### Getters and Setters

### 可选位置参数

### 命名参数

### 异常

### 在构造方法中使用this

Dart 提供了一个方便的快捷方式，用于为构造方法中的属性赋值：在声明构造方法时使用 this.propertyName。

```dart
class MyColor {
  int red;
  int green;
  int blue;

  MyColor(this.red, this.green, this.blue);
}

final color = MyColor(80, 80, 128);
```

此技巧同样也适用于命名参数。属性名为参数的名称：
```dart
class MyColor {
  ...

  MyColor({required this.red, required this.green, required this.blue});
}

final color = MyColor(red: 80, green: 80, blue: 80);
```

在上面的代码中，red、green 和 blue 被标记为 required，因为这些 int 数值不能为空。如果你指定了默认值，你可以忽略 required。

对于可选参数，默认值为期望值：
```dart
MyColor([this.red = 0, this.green = 0, this.blue = 0]);
// or
MyColor({this.red = 0, this.green = 0, this.blue = 0});
```

代码样例
- 使用 this 语法向 MyClass 添加一行构造方法，并接收和分配全部（三个）属性。

```dart
class MyClass {
  final int anInt;
  final String aString;
  final double aDouble;
  
  MyClass(this.anInt, this.aString, this.aDouble);
}
```

### Initializer lists

### 命名构造方法

### 工厂构造方法

Dart 支持工厂构造方法。它能够返回其子类甚至 null 对象。要创建一个工厂构造方法，请使用 factory 关键字。
```dart
class Square extends Shape {}

class Circle extends Shape {}

class Shape {
  Shape();

  factory Shape.fromTypeName(String typeName) {
    if (typeName == 'square') return Square();
    if (typeName == 'circle') return Circle();

    throw ArgumentError('Unrecognized $typeName');
  }
}
```

代码样例
- 填写名为 IntegerHolder.fromList 的工厂构造方法，使其执行以下操作：
- 若列表只有一个值，那么就用它来创建一个 IntegerSingle。
- 如果这个列表有两个值，那么按其顺序创建一个 IntegerDouble。
- 如果这个列表有三个值，那么按其顺序创建一个 IntegerTriple。
- 否则，抛出一个 Error。

```dart
class IntegerHolder {
  IntegerHolder();
  
  factory IntegerHolder.fromList(List<int> list) {
    if (list.length == 1) {
      return IntegerSingle(list[0]);
    } else if (list.length == 2) {
      return IntegerDouble(list[0], list[1]);
    } else if (list.length == 3) {
      return IntegerTriple(list[0], list[1], list[2]);
    } else {
      throw Error();
    } 
  }
}

class IntegerSingle extends IntegerHolder {
  final int a;
  IntegerSingle(this.a); 
}

class IntegerDouble extends IntegerHolder {
  final int a;
  final int b;
  IntegerDouble(this.a, this.b); 
}

class IntegerTriple extends IntegerHolder {
  final int a;
  final int b;
  final int c;
  IntegerTriple(this.a, this.b, this.c); 
}
```

### 重定向构造方法

有时一个构造方法仅仅用来重定向到该类的另一个构造方法。重定向方法没有主体，它在冒号（:）之后调用另一个构造方法。

```dart
class Automobile {
  String make;
  String model;
  int mpg;

  // The main constructor for this class.
  Automobile(this.make, this.model, this.mpg);

  // Delegates to the main constructor.
  Automobile.hybrid(String make, String model) : this(make, model, 60);

  // Delegates to a named constructor
  Automobile.fancyHybrid() : this.hybrid('Futurecar', 'Mark 2');
}
```

代码样例
- 给Color类创建一个叫black的命名构造方法，不要手动分配属性，而是将 0 作为参数，重定向到默认的构造方法。

```dart
class Color {
  int red;
  int green;
  int blue;
  
  Color(this.red, this.green, this.blue);

  Color.black() : this(0, 0, 0);
}
```

### Const 构造方法

如果一个类生成的对象永远都不会更改，可以让这些对象成为编译时的常量。为此需要定义 const 构造方法并确保所有实例变量都是 final 的。

```dart
class ImmutablePoint {
  static const ImmutablePoint origin = ImmutablePoint(0, 0);

  final int x;
  final int y;

  const ImmutablePoint(this.x, this.y);
}
```

代码样例
- 定义 Recipe 类，使其实例成为常量，并创建一个执行以下操作的常量构造方法：
- 该方法有三个参数：ingredients、calories 和 milligramsOfSodium。（按照此顺序）
- 使用 this 语法自动将参数值分配给同名的对象属性。
- 在 Recipe 的构造方法声明之前，用 const 关键字使其成为常量。

```dart
class Recipe {
  final List<String> ingredients;
  final int calories;
  final double milligramsOfSodium;

  const Recipe(this.ingredients, this.calories, this.milligramsOfSodium);
}
```

