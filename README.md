# MMT的Blender插件

本插件从DarkStartSword的3Dfixs仓库fork并做了一些改进来更好的适应我们的需求，主要是为MMT-Community(MigotoModTool)开发的配套插件，
也可以作为通用插件使用。

原仓库地址：https://github.com/DarkStarSword/3d-fixes

# 注意事项
1,如何使用？

克隆项目后，把MMT文件夹打包为MMT.zip然后直接在Blender里安装即可，或者你也可以下载Release里的版本。

2,本项目的代码可能会直接复制或者从其它人的代码中学习，当然是在遵守开源协议的前提下，对应用到的代码会给出具体的来源和署名。

3,本项目仍在不断改进中，任何API都可能随时改变，如果你发现版本更新后代码产生了较大的变动，不要感到惊讶，这是正常的，永远使用最新版本即可。

# 用户使用环境
本插件只能在Blender官方版本使用，其它版本比如Steam版本可能存在兼容性问题。

当前支持的Blender版本：Blender 3.6 LTS (Windows Installer)

https://www.blender.org/download/lts/3-6/

# 开发环境配置
- 操作系统: Windows 11 Pro
- IDE集成开发环境: Pycharm Community 2023.3
- Pycharm插件: Pycharm-Blender-Plugin(https://github.com/BlackStartx/PyCharm-Blender-Plugin) 2023.3
- Blender版本: Blender3.6LTS

额外需要导入的包:
- Fake bpy: https://github.com/nutti/fake-bpy-module (pip install fake-bpy-module-3.6)
- Numpy: pip install numpy

注意如果你要向本仓库提交代码，请确保代码的注释和可读性拉满，Python不是为了让你炫技而是为了方便团队合作所以不要使用过于高级而罕见的语法特性。

# LICENSE开源协议
- GNU GENERAL PUBLIC LICENSE Version 3
- Mozilla Public License 2.0

# 致谢
The original code is mainly forked from @Ian Munsie (darkstarsword@gmail.com),
see https://github.com/DarkStarSword/3d-fixes,
big thanks to his original blender plugin design.

And part of the code is learned or copied from these projects below,
their codes use their project's LICENSE ,you can find LICENSE file in LICENSE folder,
huge thanks for their great code:
- https://github.com/SilentNightSound/GI-Model-Importer
- https://github.com/SilentNightSound/SR-Model-Importer
- https://github.com/leotorrez/LeoTools
- https://github.com/falling-ts/free-model
- https://github.com/SpectrumQT/WWMI-TOOLS