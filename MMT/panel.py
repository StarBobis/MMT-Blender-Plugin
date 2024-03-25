# Acknowledgements
# The original code is mainly forked from @Ian Munsie (darkstarsword@gmail.com)
# see https://github.com/DarkStarSword/3d-fixes,
# big thanks to his original blender plugin design.
# And part of the code is learned from projects below, huge thanks for their great code:
# - https://github.com/SilentNightSound/GI-Model-Importer
# - https://github.com/SilentNightSound/SR-Model-Importer
# - https://github.com/leotorrez/LeoTools

# This panel.py is only used in common help functions and imports.

import io
import re
from array import array
import struct
import numpy
import itertools
import collections
import os
from glob import glob
import json
import copy
import textwrap
import operator  # to get function names for operators like @, +, -
import bpy
from bpy_extras.io_utils import unpack_list, ImportHelper, ExportHelper, axis_conversion
from bpy.props import BoolProperty, StringProperty, CollectionProperty
from bpy_extras.image_utils import load_image
from mathutils import Matrix, Vector
from bpy_extras.io_utils import orientation_helper

import logging
import json


def save_mmt_path(path):
    # 获取当前脚本文件的路径
    script_path = os.path.abspath(__file__)

    # 获取当前插件的工作目录
    plugin_directory = os.path.dirname(script_path)

    # 构建保存文件的路径
    config_path = os.path.join(plugin_directory, 'Config.json')

    # 创建字典对象
    config = {'mmt_path': bpy.context.scene.mmt_props.path}

    # 将字典对象转换为 JSON 格式的字符串
    json_data = json.dumps(config)

    # 保存到文件
    with open(config_path, 'w') as file:
        file.write(json_data)


def load_path():
    # 获取当前脚本文件的路径
    script_path = os.path.abspath(__file__)

    # 获取当前插件的工作目录
    plugin_directory = os.path.dirname(script_path)

    # 构建配置文件的路径
    config_path = os.path.join(plugin_directory, 'Config.json')

    # 读取文件
    with open(config_path, 'r') as file:
        json_data = file.read()

    # 将 JSON 格式的字符串解析为字典对象
    config = json.loads(json_data)

    # 读取保存的路径
    return config['mmt_path']


class MMTPathProperties(bpy.types.PropertyGroup):
    path: bpy.props.StringProperty(
        name="MMT Path",
        description="Select a folder path of MMT",
        default=load_path(),
        subtype='DIR_PATH'
    )


class MMTPathOperator(bpy.types.Operator):
    bl_idname = "mmt.select_folder"
    bl_label = "Select Folder"

    def execute(self, context):
        # 在这里处理文件夹选择逻辑
        bpy.ops.ui.directory_dialog('INVOKE_DEFAULT', directory=context.scene.mmt_props.path)
        return {'FINISHED'}


# MMT的侧边栏
class MMTPanel(bpy.types.Panel):
    bl_label = "MMT Panel"
    bl_idname = "VIEW3D_PT_MMT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MMT'
    bl_region_width = 600

    def draw(self, context):

        layout = self.layout

        row = layout.row()
        # 在这里添加你的侧边栏内容
        row.label(text="NicoMico's MMT Plugin V1.4.2")

        row.operator("wm.url_open", text="Check Update", icon='URL').url = "https://github.com/StarBobis/3Dmigoto-Blender-MMT/releases"
        props = context.scene.mmt_props
        layout.prop(props, "path")

        # 获取MMT.exe的路径
        mmt_path = os.path.join(context.scene.mmt_props.path, "MMT.exe")
        if os.path.exists(mmt_path):
            layout.label(text="MMT: " + mmt_path)
        else:
            layout.label(text="Error: Invalid MMT path! ", icon='ERROR')

        # 读取MainSetting.json中当前游戏名称
        current_game = ""
        main_setting_path = os.path.join(context.scene.mmt_props.path, "Configs\\wheel_setting\\MainSetting.json")
        if os.path.exists(main_setting_path):
            main_setting_file = open(main_setting_path)
            main_setting_json = json.load(main_setting_file)
            main_setting_file.close()
            current_game = main_setting_json["GameName"]
            layout.label(text="Working game: " + current_game)
        else:
            layout.label(text="Error: Can't find MainSetting.json! ", icon='ERROR')

        # 根据当前游戏名称，读取GameSetting中的OutputFolder路径并设置
        output_folder_path = ""
        game_setting_path = os.path.join(context.scene.mmt_props.path, "Configs\\wheel_setting\\GameSetting.json")
        if os.path.exists(game_setting_path):
            game_setting_file = open(game_setting_path)
            game_setting_json = json.load(game_setting_file)
            game_setting_file.close()
            output_folder_path = game_setting_json[current_game + "_Dev"]["OutputFolder"]
            layout.label(text="OutputFolder: " + output_folder_path)
        else:
            layout.label(text="Error: Can't find GameSetting.json! ", icon='ERROR')

        layout.separator()
        layout.label(text="Seperate import&export in OutputFolder")

        # 快速导入，点这个之后默认路径为OutputFolder，这样直接就能去导入不用翻很久文件夹找路径了
        operator_import_txt = self.layout.operator("import_mesh.migoto_frame_analysis", text="Import .txt models ")
        operator_import_txt.directory = output_folder_path

        # 快速导出同理，点这个之后默认路径为OutputFolder，这样直接就能去导出不用翻很久文件夹找路径了
        operator_export_ibvb = self.layout.operator("export_mesh.migoto", text="Export .ib & .vb files ")
        operator_export_ibvb.filepath = output_folder_path + "1.vb"

        # 添加分隔符
        layout.separator()

        # 一键快速导入所有位于OutputFolder下的.txt模型
        layout.label(text="Fast import&export in OutputFolder")
        operator_fast_import = self.layout.operator("mmt.import_all", text="Import All .txt model in OutputFolder ")


        # 一键快速导出当前选中Collection中的所有model到对应的hash值文件夹中，并直接调用MMT.exe的Mod生成方法，做到导出完即可游戏里F10刷新看效果。
        operator_export_ibvb = self.layout.operator("mmt.export_all", text="Export selected collection's vb model to OutputFolder")


