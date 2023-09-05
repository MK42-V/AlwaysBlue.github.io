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
箭头语法是一种定义函数的方法，该函数将在其右侧执行表达式并返回其值
```dart
bool hasEmpty = aListOfStrings.any((s) {
  return s.isEmpty;
});

// 使用箭头语法

bool hasEmpty = aListOfStrings.any((s) => s.isEmpty);
```

```dart
  class MyClass {
    int value1 = 2;
    int value2 = 3;
    int value3 = 5;

    // Returns the product of the above values:
    int get product => value1 * value2 * value3;
    
    // Adds 1 to value1:
    void incrementValue1() => value1++; 
    
    // Returns a string containing each item in the
    // list, separated by commas (e.g. 'a,b,c'): 
    String joinWithCommas(List<String> strings) => strings.join(',');
  }
```

### 级联
要对同一对象执行一系列操作，请使用级联（..）
```dart
  // 下方的代码使用了空判断调用符 (?.) 在 button 不为 null 时获取属性：
  var button = querySelector('#confirm');
  button?.text = 'Confirm';
  button?.classes.add('important');
  button?.onClick.listen((e) => window.alert('Confirmed!'));
  button?.scrollIntoView();

  // 现在你可以在第一个级联位置，使用 空判断 级联操作符 (?..)，它可以确保级联操作均在实例不为 null 时执行。使用空判断级联后，你也不再需要 button 变量了：
  querySelector('#confirm')
    ?..text = 'Confirm'
    ..classes.add('important')
    ..onClick.listen((e) => window.alert('Confirmed!'))
    ..scrollIntoView();
```

```dart
  // 使用级联创建一个语句，分别将 BigObject 的 anInt 属性设为 1、aString 属性设为 String!、aList 属性设置为 [3.0] 然后调用 allDone()。
  class BigObject {
    int anInt = 0;
    String aString = '';
    List<double> aList = [];
    bool _done = false;
    
    void allDone() {
      _done = true;
    }
  }

  BigObject fillBigObject(BigObject obj) {
    return obj
      ..anInt = 1
      ..aString = 'String!'
      ..aList.add(3)
      ..allDone();
  }
```

### Getters and Setters
任何需要对属性进行更多控制而不是允许简单字段访问的时候，你都可以自定义 getter 和 setter。
```dart
  // 例如，你可以用来确保属性值合法：
  class MyClass {
    int _aProperty = 0;

    int get aProperty => _aProperty;

    set aProperty(int value) {
      if (value >= 0) {
        _aProperty = value;
      }
    }
  }

  // 你还可以使用 getter 来定义计算属性：
  class MyClass {
    final List<int> _values = [];

    void addValue(int value) {
      _values.add(value);
    }

    // A computed property.
    int get count {
      return _values.length;
    }
  }
```

```dart
// 想象你有一个购物车类，其中有一个私有的 List<double> 类型的 prices 属性。添加以下内容：
// 一个名为 total 的 getter，用于返回总价格。
// 只要新列表不包含任何负价格， setter 就会用新的列表替换列表（在这种情况下，setter 应该抛出 InvalidPriceException）。
  class InvalidPriceException {}

  class ShoppingCart {
    List<double> _prices = [];
    
    double get total => _prices.fold(0, (e, t) => e + t);
    
    set prices(List<double> value) {
      if (value.any((p) => p < 0)) {
        throw InvalidPriceException();
      }
      
      _prices = value;
    }
  }
```

### 可选位置参数
Dart 有两种传参方法：位置参数和命名参数。
```dart
// 位置参数
int sumUp(int a, int b, int c) {
  return a + b + c;
}
// ···
  int total = sumUp(1, 2, 3);
```

在 Dart 里，你可以将这些参数包裹在方括号中，使其变成可选位置参数：
```dart
  int sumUpToFive(int a, [int? b, int? c, int? d, int? e]) {
    int sum = a;
    if (b != null) sum += b;
    if (c != null) sum += c;
    if (d != null) sum += d;
    if (e != null) sum += e;
    return sum;
  }
  // ···
  int total = sumUpToFive(1, 2);
  int otherTotal = sumUpToFive(1, 2, 3, 4, 5);
```

可选位置参数永远放在方法参数列表的最后。除非你给它们提供一个默认值，否则默认为 null:
```dart
  int sumUpToFive(int a, [int b = 2, int c = 3, int d = 4, int e = 5]) {
  // ···
  }
  // ···
  int newTotal = sumUpToFive(1);
  print(newTotal); // <-- prints 15
```

### 命名参数
你可以在参数列表的靠后位置使用花括号 ({}) 来定义命名参数。
除非显式使用 required 进行标记，否则命名参数默认是可选的。
```dart
  void printName(String firstName, String lastName, {String? middleName}) {
    print('$firstName ${middleName ?? ''} $lastName');
  }
// ···
  printName('Dash', 'Dartisan');
  printName('John', 'Smith', middleName: 'Who');
  // Named arguments can be placed anywhere in the argument list
  printName('John', middleName: 'Who', 'Smith');
```

正如你所料，这些参数默认为 null，但你也可以为其提供默认值。

如果一个参数的类型是非空的，那么你必须要提供一个默认值（如下方代码所示），或者将其标记为 required（如 构造部分所示）。
```dart
void printName(String firstName, String lastName, {String middleName = ''}) {
  print('$firstName $middleName $lastName');
}
```
一个方法不能同时使用可选位置参数和可选命名参数。
```dart
  // 向 MyDataObject 类添加一个 copyWith() 实例方法，它应该包含三个可空的命名参数。
  // int? newInt
  // String? newString
  // double? newDouble
  // copyWith 方法应该根据当前实例返回一个新的 MyDataObject 并将前面参数（如果有的话）的数据复制到对象的属性中。例如，如果 newInt 不为空，则将其值复制到 anInt 中。

  class MyDataObject {
    final int anInt;
    final String aString;
    final double aDouble;

    MyDataObject({
      this.anInt = 1,
      this.aString = 'Old!',
      this.aDouble = 2.0,
    });

    MyDataObject copyWith({int? newInt, String? newString, double? newDouble}) {
      return MyDataObject(
        anInt: newInt ?? this.anInt,
        aString: newString ?? this.aString,
        aDouble: newDouble ?? this.aDouble,
      );
    }
  }
```

### 异常
Dart 代码可以抛出和捕获异常。与 Java 相比，Dart 的所有异常都是 unchecked exception。方法不会声明它们可能抛出的异常，你也不需要捕获任何异常。

虽然 Dart 提供了 Exception 和 Error 类型，但是你可以抛出任何非空对象：
```dart
  throw Exception('Something bad happened.');
  throw 'Waaaaaaah!';

  // 使用 try、on 以及 catch 关键字来处理异常：
  try {
    breedMoreLlamas();
  } on OutOfLlamasException {
    // A specific exception
    buyMoreLlamas();
  } on Exception catch (e) {
    // Anything else that is an exception
    print('Unknown exception: $e');
  } catch (e) {
    // No specified type, handles all
    print('Something really unknown: $e');
  }

  // 如果你无法完全处理该异常，请使用 rethrow 关键字再次抛出异常：
  try {
    breedMoreLlamas();
  } catch (e) {
    print('I was just trying to breed llamas!');
    rethrow;
  }

  // 要执行一段无论是否抛出异常都会执行的代码，请使用 finally：
  try {
    breedMoreLlamas();
  } catch (e) {
    // ... handle exception ...
  } finally {
    // Always clean up, even if an exception is thrown.
    cleanLlamaStalls();
  }
```

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
```dart
// 有时，当你在实现构造函数时，您需要在构造函数体执行之前进行一些初始化。例如，final 修饰的字段必须在构造函数体执行之前赋值。在初始化列表中执行此操作，该列表位于构造函数的签名与其函数体之间：
Point.fromJson(Map<String, double> json)
    : x = json['x']!,
      y = json['y']! {
  print('In Point.fromJson(): ($x, $y)');
}

// 初始化列表也是放置断言的便利位置，它仅会在开发期间运行：
NonNegativePoint(this.x, this.y)
    : assert(x >= 0),
      assert(y >= 0) {
  print('I just made a NonNegativePoint: ($x, $y)');
}

//  FirstTwoLetters 的构造函数。使用的初始化列表将 word 的前两个字符分配给 letterOne 和 LetterTwo 属性。要获得额外的信用，请添加一个 断言 以捕获少于两个字符的单词。
class FirstTwoLetters {
  final String letterOne;
  final String letterTwo;

  FirstTwoLetters(String word)
      : assert(word.length >= 2),
        letterOne = word[0],
        letterTwo = word[1];
}
```

### 命名构造方法
```dart
// 为了允许一个类具有多个构造方法， Dart 支持命名构造方法：
class Point {
  double x, y;

  Point(this.x, this.y);

  Point.origin()
      : x = 0,
        y = 0;
}

// 为了使用命名构造方法，请使用全名调用它：
final myPoint = Point.origin();

// 给 Color 类添加一个叫做 Color.black 的方法，它将会把三个属性的值都设为 0。
class Color {
  int red;
  int green;
  int blue;
  
  Color(this.red, this.green, this.blue);

  Color.black()
      : red = 0,
        green = 0,
        blue = 0;
}
```

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

