# Flutter Game: Tank Combat

### Contents

1. Based On FLAME
2. Project Dependencies
3. Project Files
4. Major Content
5. Reference

### Based On FLAME

The goal of the Flame Engine is to provide a complete set of out-of-the-way solutions for common problems that games developed with Flutter might share.

Some of the key features provided are:

- A game loop.
- A component/object system (FCS).
- Effects and particles.
- Collision detection.
- Gesture and input handling.
- Images, animations, sprites, and sprite sheets.
- General utilities to make development easier.

On top of those features, you can augment Flame with bridge packages. Through these libraries, you will be able to access bindings to other packages, including custom Flame components and helpers, in order to make integrations seamless.

Flame officially provides bridge libraries to the following packages:

- flame_audio for AudioPlayers: Play multiple audio files simultaneously.
- flame_bloc for Bloc: A predictable state management library.
- flame_fire_atlas for FireAtlas: Create texture atlases for games.
- flame_forge2d for Forge2D: A Box2D physics engine.
- flame_lint - Our set of linting (analysis_options.yaml) rules.
- flame_oxygen for Oxygen: A lightweight Entity Component System (ECS) framework.
- flame_rive for Rive: Create interactive animations.
- flame_svg for flutter_svg: Draw SVG files in Flutter.
- flame_tiled for Tiled: 2D tile map level editor.

### Project Dependencies

environment:

    sdk: '>=3.0.5 <4.0.0'
dependencies:

    flame: ^1.8.2


### Project Files

- game_screen.dart
- background.dart
- base_component.dart
- bullet.dart
- computer_timer.dart
- control_panel_widget.dart
- controller_listener.dart
- decoration_theater.dart
- extension.dart
- fire_button.dart
- game_action.dart
- game_observer.dart
- joy_stick.dart
- tank_factory.dart
- tank_game.dart
- tank_model.dart

### Major Content
```dart
import 'dart:io';
import 'dart:ui' as ui;
import 'package:flame/flame.dart';
import 'package:flame/game.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'game/tank_game.dart';
import 'game/control_panel_widget.dart';

class GameScreen extends StatefulWidget {
  @override
  
  State<GameScreen> createState() => _GameScreenState();
}

class _GameScreenState extends State<GameScreen> {
  TankGame tankGame = TankGame();

  @override
  void initState() {
    super.initState();
    WidgetsFlutterBinding.ensureInitialized();
    bool isMobile = Platform.isAndroid || Platform.isIOS;
    if (isMobile) {
      ///设置横屏
      SystemChrome.setPreferredOrientations([DeviceOrientation.landscapeRight, DeviceOrientation.landscapeLeft]);
      ///全面屏
      SystemChrome.setEnabledSystemUIMode(SystemUiMode.manual, overlays: []);
    }
  }

  @override
  void dispose() {
    super.dispose();
    SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  }

  @override
  Widget build(BuildContext context) {
    return Directionality(
      textDirection: TextDirection.ltr,
      child: Stack(
        children: [
          FutureBuilder<List<ui.Image>>(
            future: loadAssets(),
            initialData: [],
            builder: (context, snapShot) {
              if (snapShot.data?.isEmpty ?? true) {
                return Center(
                  child: LinearProgressIndicator(
                    color: Colors.green,
                  ),
                );
              }
              return GameWidget(game: tankGame);
            },
          ),
          ControlPanelWidget(tankController: tankGame),
        ],
      ),
    );
  }

  Future<List<ui.Image>> loadAssets() {
    return Flame.images.loadAll([
      'new_map.webp',
      'tank/t_body_blue.webp',
      'tank/t_turret_blue.webp',
      'tank/t_body_green.webp',
      'tank/t_turret_green.webp',
      'tank/t_body_sand.webp',
      'tank/t_turret_sand.webp',
      'tank/bullet_blue.webp',
      'tank/bullet_green.webp',
      'tank/bullet_sand.webp',
      'explosion/explosion1.webp',
      'explosion/explosion2.webp',
      'explosion/explosion3.webp',
      'explosion/explosion4.webp',
      'explosion/explosion5.webp',
    ]);
  }
}
```

```dart
import 'dart:math';
import 'package:flutter/material.dart';

///摇杆
class JoyStick extends StatefulWidget {
  final void Function(Offset) onChange;

  const JoyStick({Key? key, required this.onChange}) : super(key: key);

  @override
  State<StatefulWidget> createState() {
    return JoyStickState();
  }
}

class JoyStickState extends State<JoyStick> {
  ///偏移量
  Offset delta = Offset.zero;

  ///更新位置
  void updateDelta(Offset newD) {
    widget.onChange(newD);
    setState(() {
      delta = newD;
    });
  }

  Offset calculateDelta(Offset offset) {
    Offset newD = offset - Offset(stickSize / 2, stickSize / 2);
    //活动范围控制在stickSize之内
    return Offset.fromDirection(newD.direction, min(stickSize / 4, newD.distance));
  }

  ///遥感尺寸
  /// * 外层大圆的直径，上层圆为1/2直径
  final double stickSize = 120;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: stickSize,
      height: stickSize,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(stickSize / 2),
        color: const Color(0x88ffffff),
      ),
      child: GestureDetector(
        //摇杆背景
        child: Center(
          child: Transform.translate(
            offset: delta,
            //摇杆
            child: SizedBox(
              width: stickSize / 2,
              height: stickSize / 2,
              child: Container(
                decoration: BoxDecoration(
                  color: const Color(0xccffffff),
                  borderRadius: BorderRadius.circular(30),
                ),
              ),
            ),
          ),
        ),
        onPanDown: onDragDown,
        onPanUpdate: onDragUpdate,
        onPanEnd: onDragEnd,
      ),
    );
  }

  void onDragDown(DragDownDetails d) {
    updateDelta(calculateDelta(d.localPosition));
  }

  void onDragUpdate(DragUpdateDetails d) {
    updateDelta(calculateDelta(d.localPosition));
  }

  void onDragEnd(DragEndDetails d) {
    updateDelta(Offset.zero);
  }
}
```

```dart
import 'dart:ui';
import 'tank_model.dart';
import 'tank_game.dart';

///用于生产绿色tank
class GreenTankFlowLine extends ComputerTankFlowLine<ComputerTank> {
  GreenTankFlowLine(Offset depositPosition, Size activeSize) : super(depositPosition, activeSize);

  @override
  ComputerTank spawnTank() {
      final TankModelBuilder greenBuilder = TankModelBuilder(
          id: DateTime.now().millisecondsSinceEpoch,
          bodySpritePath: 'tank/t_body_green.webp',
          turretSpritePath: 'tank/t_turret_green.webp',
          activeSize: activeSize);
      return TankFactory.buildGreenTank(greenBuilder.build(), depositPosition);
  }

}

///用于生产黄色tank
class SandTankFlowLine extends ComputerTankFlowLine<ComputerTank> {
  SandTankFlowLine(Offset depositPosition, Size activeSize) : super(depositPosition, activeSize);

  @override
  ComputerTank spawnTank() {
    final TankModelBuilder sandBuilder = TankModelBuilder(
        id: DateTime.now().millisecondsSinceEpoch,
        bodySpritePath: 'tank/t_body_sand.webp',
        turretSpritePath: 'tank/t_turret_sand.webp',
        activeSize: activeSize);
    return TankFactory.buildSandTank(sandBuilder.build(), depositPosition);
  }
}

///流水线基类
/// * 用于生成电脑tank
/// * 见[ComputerTankSpawner]
abstract class ComputerTankFlowLine<T extends ComputerTank> implements ComputerTankSpawnerTrigger<T> {
  ComputerTankFlowLine(this.depositPosition, this.activeSize);

  ///活动范围
  final Size activeSize;

  ///部署位置
  final Offset depositPosition;
}

abstract class ComputerTankSpawnerTrigger<T extends ComputerTank> {
  T spawnTank();
}

///电脑生成器
/// * [TankTheater]下，tank生成的直接参与者，负责电脑的随机生成。
///
/// * [spawners]为具体[ComputerTank]的生成流水线，见[GreenTankFlowLine]和[SandTankFlowLine]
///   流水线内部的[ComputerTank]产出由[TankFactory]负责。
class ComputerTankSpawner {

  ///流水线
  List<ComputerTankFlowLine> spawners = [];

  ///生成器初始化完成
  bool standby = false;

  ///构建中
  /// * 将会影响是否相应tank生成
  bool building = false;

  ///初始化调用
  /// * 用于配置流水线
  void warmUp(Size bgSize) {
    if (standby) {
      return;
    }
    standby = true;

    spawners.addAll([
      GreenTankFlowLine(const Offset(100, 100), bgSize),
      GreenTankFlowLine(Offset(100, bgSize.height * 0.8), bgSize),
      SandTankFlowLine(Offset(bgSize.width - 100, 100), bgSize),
      SandTankFlowLine(Offset(bgSize.width - 100, bgSize.height * 0.8), bgSize)
    ]);
  }

  ///快速生成tank
  /// * 各生产线生成一辆tank
  /// * [plaza]为接收tank对象
  void fastSpawn(List<ComputerTank> plaza) {
    plaza.addAll(spawners.map<ComputerTank>((e) => e.spawnTank()..deposit()).toList());
  }

  ///随机生成一辆tank
  /// * [plaza]为接收tank对象
  void randomSpan(List<ComputerTank> plaza) {
    if(building) {
      return;
    }
    building = true;
    _startSpawn(() {
      spawners.shuffle();
      plaza.add(spawners.first.spawnTank()..deposit());
      building = false;
    });
  }

  ///用于约束生产速度
  void _startSpawn(Function task) {
    Future.delayed(const Duration(milliseconds: 1500)).then((value) {
      task();
    });
  }

}

///用于构建tank
class TankFactory {
  static PlayerTank buildPlayerTank(TankModel model, Offset birthPosition) {
    return PlayerTank(id: model.id, birthPosition: birthPosition, config: model);
  }

  static ComputerTank buildGreenTank(TankModel model, Offset birthPosition) {
    return ComputerTank.green(id: model.id, birthPosition: birthPosition, config: model);
  }

  static ComputerTank buildSandTank(TankModel model, Offset birthPosition) {
    return ComputerTank.sand(id: model.id, birthPosition: birthPosition, config: model);
  }
}

///[TankModel]构建器
class TankModelBuilder {
  TankModelBuilder({
    required this.id,
    required this.bodySpritePath,
    required this.turretSpritePath,
    required this.activeSize,
  });

  final int id;

  ///车体纹理
  final String bodySpritePath;

  ///炮塔纹理
  final String turretSpritePath;

  ///活动范围
  /// * 一般是地图尺寸
  Size activeSize;

  ///车体宽度
  double bodyWidth = 38;

  ///车体高度
  double bodyHeight = 32;

  ///炮塔宽度(长)
  double turretWidth = 22;

  ///炮塔高度(直径)
  double turretHeight = 6;

  ///坦克尺寸比例
  double ratio = 0.7;

  ///直线速度
  double speed = 80;

  ///转弯速度
  double turnSpeed = 40;

  ///设置活动范围
  void setActiveSize(Size size) {
    activeSize = size;
  }

  ///设置车身尺寸
  void setBodySize(double width, double height) {
    bodyWidth = width;
    bodyHeight = height;
  }

  ///设置炮塔尺寸
  void setTurretSize(double width, double height) {
    turretWidth = width;
    turretHeight = height;
  }

  ///设置tank尺寸比例
  void setTankRatio(double r) {
    ratio = r;
  }

  ///设置直线速度
  void setDirectSpeed(double s) {
    speed = s;
  }

  ///设置转弯速度
  void setTurnSpeed(double s) {
    turnSpeed = s;
  }

  TankModel build() {
    return TankModel(
      id: id,
      bodySpritePath: bodySpritePath,
      turretSpritePath: turretSpritePath,
      ratio: ratio,
      speed: speed,
      turnSpeed: turnSpeed,
      bodyWidth: bodyWidth,
      bodyHeight: bodyHeight,
      turretWidth: turretWidth,
      turretHeight: turretHeight,
      activeSize: activeSize,
    );
  }
}

///坦克基础配置模型
/// * 由[TankModelBuilder]负责构建
class TankModel {
  TankModel(
      {required this.id,
      required this.bodySpritePath,
      required this.turretSpritePath,
      required this.ratio,
      required this.speed,
      required this.turnSpeed,
      required this.bodyWidth,
      required this.bodyHeight,
      required this.turretWidth,
      required this.turretHeight,
      required this.activeSize});

  final int id;

  ///车体宽度
  final double bodyWidth;

  ///车体高度
  final double bodyHeight;

  ///炮塔宽度(长)
  final double turretWidth;

  ///炮塔高度(直径)
  final double turretHeight;

  ///坦克尺寸比例
  final double ratio;

  ///直线速度
  final double speed;

  ///转弯速度
  final double turnSpeed;

  ///车体纹理
  final String bodySpritePath;

  ///炮塔纹理
  final String turretSpritePath;

  ///活动范围
  /// * 一般是地图尺寸
  Size activeSize;
}
```

```dart

import 'dart:ui';

import 'package:flame/components.dart';
import 'package:flame/game.dart';
import 'package:flutter/cupertino.dart';
import 'decoration_theater.dart';
import 'tank_model.dart';
import 'game_observer.dart';
import 'computer_timer.dart';
import 'extension.dart';

import 'bullet.dart';
import 'tank_factory.dart';
import 'controller_listener.dart';
import 'game_action.dart';


///游戏入口
/// * 继承于[FlameGame]
/// * 所混入的类 : [BulletTheater]、[TankTheater]、[ComputerTimer]、[DecorationTheater]、[GameObserver]
/// *           用于拓展[TankGame]的场景内容和[Sprite]交互行为，具体见各自的注释。
class TankGame extends FlameGame with BulletTheater, TankTheater, ComputerTimer, DecorationTheater, GameObserver{

  TankGame() {
    setTimerListener(this);
  }



  @override
  void update(double t) {
    super.update(t);
  }

}

///负责管理玩家和电脑tank
mixin TankTheater on FlameGame, BulletTheater implements TankController, ComputerTimerListener{

  ComputerTankSpawner _computerSpawner = ComputerTankSpawner();

  PlayerTank? player;

  final List<ComputerTank> computers = [];

  void initPlayer(Vector2 canvasSize) {
    final Size bgSize = canvasSize.toSize();

    final TankModelBuilder playerBuilder = TankModelBuilder(
        id: DateTime.now().millisecondsSinceEpoch,
        bodySpritePath: 'tank/t_body_blue.webp',
        turretSpritePath: 'tank/t_turret_blue.webp',
        activeSize: bgSize);

    player ??= TankFactory.buildPlayerTank(playerBuilder.build(), Offset(bgSize.width/2,bgSize.height/2));
    player!.deposit();
  }

  ///初始化敌军
  /// * 一般情况下是在游戏伊始时执行。
  void initEnemyTank() {
    _computerSpawner.fastSpawn(computers);
  }

  void randomSpanTank() {
    _computerSpawner.randomSpan(computers);
  }


  @override
  void onFireTimerTrigger() {
    computers.shuffle();
    computers.forEach(computerTankFire);
  }

  @override
  void onGameResize(Vector2 canvasSize) {
    if(player == null) {
      initPlayer(canvasSize);
    }
    if(computers.isEmpty) {
      _computerSpawner.warmUp(canvasSize.toSize());
      initEnemyTank();
      for (var element in computers) {
        element.deposit();
      }
    }
    player?.onGameResize(canvasSize);
    computers.onGameResize(canvasSize);
    super.onGameResize(canvasSize);
  }

  @override
  void render(Canvas canvas) {
    player?.render(canvas);
    computers.render(canvas);
    super.render(canvas);
  }

  @override
  void update(double dt) {
    player?.update(dt);
    computers.update(dt);
    super.update(dt);
    computers.removeWhere((element) => element.isDead);
  }

  @override
  void fireButtonTriggered() {
    if(player != null){
      playerTankFire(player!);
    }
  }

  @override
  void bodyAngleChanged(Offset newAngle) {
    if(newAngle == Offset.zero){
      player?.targetBodyAngle = null;
    }else{
      player?.targetBodyAngle = newAngle.direction;//范围（pi,-pi）
    }
  }

  @override
  void turretAngleChanged(Offset newAngle) {
    if (newAngle == Offset.zero) {
      player?.targetTurretAngle = null;
    } else {
      player?.targetTurretAngle = newAngle.direction;
    }
  }

}


///负责坦克的开火系统
mixin BulletTheater on FlameGame implements ComputerTankAction{

  ///电脑tank的开火器
  final BulletTrigger trigger = BulletTrigger();

  ///玩家炮弹最大数量
  final int maxPlayerBulletNum = 20;


  List<BaseBullet> computerBullets = [];

  List<BaseBullet> playerBullets = [];

  void playerTankFire(TankFireHelper helper) {
    if(playerBullets.length < maxPlayerBulletNum) {
      playerBullets.add(helper.getBullet());
    }
  }

  @override
  void computerTankFire(TankFireHelper helper) {
    trigger.chargeLoading(() {
      computerBullets.add(helper.getBullet());
    });
  }

  @override
  void onGameResize(Vector2 canvasSize) {
    computerBullets.onGameResize(canvasSize);
    playerBullets.onGameResize(canvasSize);
    super.onGameResize(canvasSize);
  }

  @override
  void render(Canvas canvas) {
    computerBullets.render(canvas);
    playerBullets.render(canvas);
    super.render(canvas);
  }


  @override
  void update(double dt) {
    computerBullets.update(dt);
    playerBullets.update(dt);
    super.update(dt);
    computerBullets.removeWhere((element) => element.dismissible);
    playerBullets.removeWhere((element) => element.dismissible);
  }


}
```

```dart
import 'dart:math';
import 'package:flame/components.dart';
import 'package:flutter/material.dart';
import 'bullet.dart';

import 'base_component.dart';
import 'tank_factory.dart';

///电脑
class ComputerTank extends DefaultTank {
  factory ComputerTank.green({
    required int id,
    required Offset birthPosition,
    required TankModel config,
  }) {
    return ComputerTank(
        id: id, birthPosition: birthPosition, config: config, bullet: ComputerBullet.green(tankId: id, activeSize: config.activeSize));
  }

  factory ComputerTank.sand({
    required int id,
    required Offset birthPosition,
    required TankModel config,
  }) {
    return ComputerTank(id: id, birthPosition: birthPosition, config: config, bullet: ComputerBullet.sand(tankId: id, activeSize: config.activeSize));
  }

  ComputerTank({
    required int id,
    required Offset birthPosition,
    required TankModel config,
    required this.bullet,
  }) : super(id: id, birthPosition: birthPosition, config: config);

  ///用于生成随机路线
  static final Random random = Random();

  ///活动边界
  static final double activeBorderLow = 0.01, activeBorderUp = 1 - activeBorderLow;

  ///最大单向移动距离
  static const double maxMovedDistance = 100;

  BaseBullet bullet;

  ///移动的距离
  double movedDis = 0;

  void generateNewTarget() {
    final double x = random.nextDouble().clamp(activeBorderLow, activeBorderUp) * config.activeSize.width;
    final double y = random.nextDouble().clamp(activeBorderLow, activeBorderUp) * config.activeSize.height;

    targetOffset = Offset(x, y);

    final Offset vector = targetOffset - position;
    targetBodyAngle = vector.direction;
    targetTurretAngle = vector.direction;
  }

  @override
  void deposit() {
    super.deposit();
    generateNewTarget();
  }

  @override
  void move(double t) {
    if (targetBodyAngle != null) {
      movedDis += speed * t;
      if (movedDis < maxMovedDistance) {
        super.move(t);
      } else {
        movedDis = 0;
        generateNewTarget();
      }
    }
  }

  @override
  BaseBullet getBullet() => bullet.copyWith(position: getBulletFirePosition(), angle: getBulletFireAngle());
}

///玩家
class PlayerTank extends DefaultTank {
  PlayerTank({required int id, required Offset birthPosition, required TankModel config})
      : bullet = PlayerBullet(tankId: id, activeSize: config.activeSize),
        super(id: id, birthPosition: birthPosition, config: config);

  final PlayerBullet bullet;

  @override
  BaseBullet getBullet() => bullet.copyWith(position: getBulletFirePosition(), angle: getBulletFireAngle());
}


///可实例化的tank模型
///
/// * [BaseTank]实例化的基准模型，不具备业务区分能力。见[PlayerTank]和[ComputerTank]
abstract class DefaultTank extends BaseTank {
  DefaultTank({
    required int id,
    required Offset birthPosition,
    required TankModel config,
  }) : super(id: id, birthPosition: birthPosition, config: config);

  @override
  void deposit() {
    isDead = false;
  }

  @override
  void onGameResize(Vector2 canvasSize) {
    config.activeSize = canvasSize.toSize();
    super.onGameResize(canvasSize);
  }

  @override
  void render(Canvas canvas) {
    if (!isStandBy || isDead) {
      return;
    }
    //将canvas 原点设置在tank上
    canvas.save();
    canvas.translate(position.dx, position.dy);
    drawBody(canvas);
    drawTurret(canvas);
    canvas.restore();
  }

  @override
  void update(double t) {
    if (!isStandBy || isDead) {
      return;
    }
    rotateBody(t);
    rotateTurret(t);
    move(t);
  }

  @override
  void drawBody(Canvas canvas) {
    canvas.rotate(bodyAngle);
    bodySprite?.renderRect(canvas, bodyRect);
  }

  @override
  void drawTurret(Canvas canvas) {
    //旋转炮台
    canvas.rotate(turretAngle);
    // 绘制炮塔
    turretSprite?.renderRect(canvas, turretRect);
  }

  @override
  void move(double t) {
    if (targetBodyAngle == null) return;
    if (bodyAngle == targetBodyAngle) {
      //tank 直线时 移动速度快
      position += Offset.fromDirection(bodyAngle, speed * t); //100 是像素
    } else {
      //tank旋转时 移动速度要慢
      position += Offset.fromDirection(bodyAngle, turnSpeed * t);
    }
  }

  @override
  void rotateBody(double t) {
    if (targetBodyAngle != null) {
      final double rotationRate = pi * t;
      if (bodyAngle < targetBodyAngle!) {
        //车体角度和目标角度差额
        if ((targetBodyAngle! - bodyAngle).abs() > pi) {
          bodyAngle -= rotationRate;
          if (bodyAngle < -pi) {
            bodyAngle += pi * 2;
          }
        } else {
          bodyAngle += rotationRate;
          if (bodyAngle > targetBodyAngle!) {
            bodyAngle = targetBodyAngle!;
          }
        }
      } else if (bodyAngle > targetBodyAngle!) {
        if ((targetBodyAngle! - bodyAngle).abs() > pi) {
          bodyAngle += rotationRate;
          if (bodyAngle > pi) {
            bodyAngle -= pi * 2;
          }
        } else {
          bodyAngle -= rotationRate;
          if (bodyAngle < targetBodyAngle!) {
            bodyAngle = targetBodyAngle!;
          }
        }
      }
    }
  }

  @override
  void rotateTurret(double t) {
    if (targetTurretAngle != null) {
      final double rotationRate = pi * t;
      //炮塔和车身夹角
      final double localTargetTurretAngle = targetTurretAngle! - bodyAngle;
      if (turretAngle < localTargetTurretAngle) {
        if ((localTargetTurretAngle - turretAngle).abs() > pi) {
          turretAngle -= rotationRate;
          //超出临界值，进行转换 即：小于-pi时，转换成pi之后继续累加，具体参考 笛卡尔坐标，范围是（-pi,pi）
          if (turretAngle < -pi) {
            turretAngle += pi * 2;
          }
        } else {
          turretAngle += rotationRate;
          if (turretAngle > localTargetTurretAngle) {
            turretAngle = localTargetTurretAngle;
          }
        }
      }
      if (turretAngle > localTargetTurretAngle) {
        if ((localTargetTurretAngle - turretAngle).abs() > pi) {
          turretAngle += rotationRate;
          if (turretAngle > pi) {
            turretAngle -= pi * 2;
          }
        } else {
          turretAngle -= rotationRate;
          if (turretAngle < localTargetTurretAngle) {
            turretAngle = localTargetTurretAngle;
          }
        }
      }
    }
  }

  @override
  int getTankId() => id;

  @override
  double getBulletFireAngle() {
    double bulletAngle = bodyAngle + turretAngle;
    while (bulletAngle > pi) {
      bulletAngle -= pi * 2;
    }
    while (bulletAngle < -pi) {
      bulletAngle += pi * 2;
    }
    return bulletAngle;
  }

  @override
  Offset getBulletFirePosition() =>
      position +
      Offset.fromDirection(
        getBulletFireAngle(),
        bulletBornLoc,
      );
}

///tank 开火辅助接口
abstract class TankFireHelper {
  ///隶属于的坦克
  int getTankId();

  ///获取炮弹发射位置
  Offset getBulletFirePosition();

  ///获取炮弹发射角度
  double getBulletFireAngle();

  ///获取tank所装配的炮弹
  BaseBullet getBullet();
}


///tank基础模型
/// * 定义基础行为属性，外观属性依赖于[TankModel]
/// * @see [TankModelBuilder]
abstract class BaseTank extends WindowComponent implements TankFireHelper {
  BaseTank({
    required int id,
    required this.config,
    required Offset birthPosition,
  }) : position = birthPosition {
    bodyRect = Rect.fromCenter(center: Offset.zero, width: bodyWidth * ratio, height: bodyHeight * ratio);
    turretRect = Rect.fromCenter(center: Offset.zero, width: turretWidth * ratio, height: turretHeight * ratio);
    init();
  }

  final TankModel config;

  ///坦克位置
  Offset position;

  ///移动到目标位置
  late Offset targetOffset;

  ///车体角度
  double bodyAngle = 0;

  ///炮塔角度
  double turretAngle = 0;

  ///车体目标角度
  /// * 为空时，说明没有角度变动
  double? targetBodyAngle;

  ///炮塔目标角度
  /// * 为空时，说明没有角度变动
  double? targetTurretAngle;

  ///炮弹出炮口位置
  double get bulletBornLoc => 18;

  ///车体尺寸
  late Rect bodyRect;

  ///炮塔尺寸
  late Rect turretRect;

  ///tank是否存活
  bool isDead = true;

  ///车体
  Sprite? bodySprite;

  ///炮塔
  Sprite? turretSprite;

  ///配置完成
  bool isStandBy = false;

  int get id => config.id;

  ///车体宽度
  double get bodyWidth => config.bodyWidth;

  ///车体高度
  double get bodyHeight => config.bodyHeight;

  ///炮塔宽度(长)
  double get turretWidth => config.turretWidth;

  ///炮塔高度(直径)
  double get turretHeight => config.turretHeight;

  ///坦克尺寸比例
  double get ratio => config.ratio;

  ///直线速度
  double get speed => config.speed;

  ///转弯速度
  double get turnSpeed => config.turnSpeed;

  ///车体纹理
  String get bodySpritePath => config.bodySpritePath;

  ///炮塔纹理
  String get turretSpritePath => config.turretSpritePath;

  Future<bool> init() async {
    bodySprite = await Sprite.load(bodySpritePath);
    turretSprite = await Sprite.load(turretSpritePath);
    isStandBy = true;
    return isStandBy;
  }

  ///部署
  void deposit();

  ///移动
  /// [t] 过渡时间-> 理论值16.66ms
  void move(double t);

  ///绘制车体
  void drawBody(Canvas canvas);

  ///绘制炮塔
  void drawTurret(Canvas canvas);

  ///旋转车体
  /// [t] 过渡时间-> 理论值16.66ms
  void rotateBody(double t);

  ///旋转炮塔
  /// [t] 过渡时间-> 理论值16.66ms
  void rotateTurret(double t);
}
```

```dart
import 'dart:ui';
import 'package:flame/components.dart';
import 'base_component.dart';
import 'dart:async' as async;
import 'dart:collection';

///电脑炮弹
class ComputerBullet extends BaseBullet{

  factory ComputerBullet.green({required int tankId, required Size activeSize}) {
    return ComputerBullet('tank/bullet_green.webp', tankId: tankId, activeSize: activeSize);
  }

  factory ComputerBullet.sand({required int tankId, required Size activeSize}) {
    return ComputerBullet('tank/bullet_green.webp', tankId: tankId, activeSize: activeSize);
  }

  ComputerBullet(this.spritePath, {required int tankId, required Size activeSize})
      : super(tankId: tankId, activeSize: activeSize);

  final String spritePath;

  late Sprite _sprite;

  Rect _rect = Rect.fromLTWH(-4, -2, 6, 4);

  @override
  Rect get bulletRect => _rect;

  @override
  Sprite get bulletSprite => _sprite;

  @override
  Future<void> loadSprite() async {
    _sprite = await Sprite.load(spritePath);
  }

  @override
  ComputerBullet copyWith({int? tankId, Size? activeSize, Offset? position, double? angle, BulletStatus status = BulletStatus.none}) {
    final ComputerBullet pb = ComputerBullet(spritePath, tankId: tankId ?? this.tankId, activeSize: activeSize ?? this.activeSize);
    pb.position = position ?? Offset.zero;
    pb.angle = angle ?? 0;
    pb.status = status;
    return pb;
  }

}

///玩家炮弹
class PlayerBullet extends BaseBullet{

  PlayerBullet({required int tankId, required Size activeSize})
      : super(tankId: tankId, activeSize: activeSize);

  late Sprite _sprite;

  Rect _rect = Rect.fromLTWH(-4, -2, 8, 4);

  @override
  Sprite get bulletSprite => _sprite;

  @override
  Rect get bulletRect => _rect;

  @override
  Future<void> loadSprite() async {
    _sprite = await Sprite.load('tank/bullet_blue.webp');
  }

  @override
  PlayerBullet copyWith({int? tankId, Size? activeSize, Offset? position, double? angle, BulletStatus status = BulletStatus.none}) {
    final PlayerBullet pb = PlayerBullet(tankId: tankId ?? this.tankId, activeSize: activeSize ?? this.activeSize);
    pb.position = position ?? Offset.zero;
    pb.angle = angle ?? 0;
    pb.status = status;
    return pb;
  }

}


///炮弹状态
enum BulletStatus{
  none, //初始状态
  standBy,// 准备状态： 可参与绘制
  hit, //击中状态
  outOfBorder, //飞出边界
}


///炮弹基类
abstract class BaseBullet extends WindowComponent{

  BaseBullet({
    required this.tankId,
    required this.activeSize,
  }) {
    init();
  }

  BaseBullet copyWith({
    int? tankId,
    Size? activeSize,
    Offset? position,
    double? angle,
    BulletStatus status = BulletStatus.none,
  });

  ///隶属于的tank
  final int tankId;

  ///子弹尺寸
  /// * 后期可以加入特效等
  Rect get bulletRect;

  ///子弹皮肤
  Sprite get bulletSprite;

  ///可活动范围
  /// * 超出判定为失效子弹
  Size activeSize;

  ///位置
  Offset position = Offset.zero;

  ///速度
  double speed = 200;

  ///角度
  double angle = 0;

  ///子弹状态
  BulletStatus status = BulletStatus.none;

  ///可移除的子弹
  bool get dismissible => status.index > 1;

  void hit() {
    status = BulletStatus.hit;
  }

  ///加载炮弹纹理
  Future<void> loadSprite();

  ///初始化炮弹
  void init() async {
    await loadSprite();
    status = BulletStatus.standBy;
  }

  @override
  void onGameResize(Vector2 canvasSize) {
    activeSize = canvasSize.toSize();
    super.onGameResize(canvasSize);
  }


  @override
  void render(Canvas canvas) {
    if(dismissible) {
      return;
    }
    canvas.save();
    canvas.translate(position.dx, position.dy);
    canvas.rotate(angle);
    bulletSprite.renderRect(canvas, bulletRect);
    canvas.restore();
  }

  @override
  void update(double t) {
    if(dismissible) {
      return;
    }
    position += Offset.fromDirection(angle,speed * t);

    if(!activeSize.contains(position)) {
      status = BulletStatus.outOfBorder;
    }
  }

}

class BulletTrigger{

  BulletTrigger() {
    trigger = async.Timer.periodic(const Duration(milliseconds: 100), _onTick);
  }

  final Queue<Function> _task = Queue();

  late final async.Timer trigger;

  void _onTick(async.Timer timer) {
    if(_task.isEmpty) {
      return;
    }
    final Function t = _task.removeFirst();
    t.call();
  }

  void chargeLoading(Function b) {
    _task.add(b);
  }

}
```

### Reference

- [FLAME](https://pub.dev/packages/flame)

- [见微知著，Flutter在游戏开发的表现及跨平台带来的优势](https://juejin.cn/post/6856681993418211336)

- [Flutter&Flame——TankCombat游戏开发（一）](https://juejin.cn/post/6857049079000760334)

- [Flutter&Flame——TankCombat游戏开发（二）](https://juejin.cn/post/6857381047723065351)

- [Flutter&Flame——TankCombat游戏开发（三）](https://juejin.cn/post/6857762289362976776)

- [Flutter&Flame——TankCombat游戏开发（四）](https://juejin.cn/post/6858885636313579528)