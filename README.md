# MMT Blender Plugin
Special version of 3Dmigoto blender plugin originally forked from https://github.com/DarkStarSword/3d-fixes.
And modify some code to better meets our needs, mainly developed for MMT-Community(MigotoModTool).


# Notice
1,How to use?

Pack MMT folder into MMT.zip and install it from your blender's preferences menu or download latest version from Release.

2,Files in References may directly copied from other's work and credit to their repo and only used as a reference,
and also some tutorials or documents in there, it keep change with development,
so don't be suprised if things disappear suddenly.

# User Environment
This plugin is designed to work on official Blender release: 

Current stable support version: Blender 3.6 LTS (Windows Installer):

https://www.blender.org/download/lts/3-6/

# Develop Environment
- OS: Windows 11 Pro
- IDE: Pycharm Community 2023.3
- Pycharm plugin: Pycharm-Blender-Plugin(https://github.com/BlackStartx/PyCharm-Blender-Plugin) 2023.3
- Blender Version: 4.0 (or at least 3.6.8 LTS)

Extra modules: 
- Fake bpy: https://github.com/nutti/fake-bpy-module (pip install fake-bpy-module-4.0 or pip install fake-bpy-module-3.6)
- Numpy: pip install numpy

Notice: commit code back to this repo need to follow a core rule: code need to be easy to read more than execute speed.
Python is not design to execute fast,but designed to let people work together and make life easier,
not a language for you to show your programming skills or try to use unusual feature to kill other programmer's valuable time.

# LICENSE
This project use GNU3 as main LICENSE,and MPL-2.0 as backup, you can use one of them.
- GNU GENERAL PUBLIC LICENSE Version 3
- Mozilla Public License 2.0

# Ask for help
https://discord.gg/Cz577BcRf5

# Acknowledgements
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
