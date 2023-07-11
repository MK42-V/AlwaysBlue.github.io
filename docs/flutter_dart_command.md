# Flutter/Dart 常用命令行

> flutter doctor 

查看flutter的状态，查看环境配置是否有问题

> flutter doctor -v 

查看flutter 状态的详细信息

> flutter build apk 

打包安卓包

> flutter build ios 

打包苹果ipa

> flutter run 

运行项目 默认--debug

> flutter run --profile 

运行线上测试包

> flutter run --release 

运行线上包

> flutter channel 

查看flutter 的所有分支

> flutter channel stable 

切换到具体的分支

> flutter upgrade 

升级flutter

> flutter upgrade --force 

如果升级flutter出现问题可以尝试 强制更新

> flutter logs 

当链接到某一个设备的时候，通过此命令可以查看到当前设备的log

> flutter screenshot 

可以截取项目当前屏幕展示的图到项目里

> futter clean 

清除缓存

> flutter analyze 

Dart默认的linter配置有点弱, 有很多有问题代码也不报错或警告. 通过此命令可以应用dart的最佳代码实践, 对一些不好的代码风格提出警告或者直接报错, 从而提高代码质量

> flutter attach 

混合开发常用命令，1，首先起项目，运行起整个工程；2，到命令行，打开 flutter_lib 目录（Flutter module工程）；3，输入命令：flutter attach

> flutter test 

当前项目的单元测试

> flutter downgrade 

从flutter当前channel 下降到上一个稳定版本

> flutter install 

直接下载apk到手机上，很方便使用，不用重复build or run

> flutter create 

创建一个flutter 新项目 例如：flutter create --org com.yourdomain your_app_name

> flutter -h 

如果忘记具体的命令行 可以通过此命令查找

---

> dart fix --dry-run 

检查dart fix可修复的语法，但不会直接应用

> dart fix --apply 

应用dart fix的修复

> dart pub get 

获取依赖项

> dart pub upgrade 

忽略掉任何已存在的lockfile文件，获取所有依赖项的最新版本

> dart pub outdated 

查找哪些依赖需要更新版本

> dart pub upgrade --major-versions --dry-run 

查看哪些依赖会被更新

> dart pub upgrade --major-versions 

更新依赖项