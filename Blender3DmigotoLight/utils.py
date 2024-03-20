# Acknowledgements
# The original code is mainly forked from @Ian Munsie (darkstarsword@gmail.com)
# see https://github.com/DarkStarSword/3d-fixes,
# big thanks to his original blender plugin design.
# And part of the code is learned from projects below, huge thanks for their great code:
# - https://github.com/SilentNightSound/GI-Model-Importer
# - https://github.com/SilentNightSound/SR-Model-Importer
# - https://github.com/leotorrez/LeoTools

# This utils.py is only used in common help functions and imports.

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


class MMTPathProperties(bpy.types.PropertyGroup):
    path: bpy.props.StringProperty(
        name="MMT Path",
        description="Select a folder path of MMT",
        default="",
        subtype='DIR_PATH'
    )


class MMTPathOperator(bpy.types.Operator):
    bl_idname = "mmt.select_folder"
    bl_label = "Select Folder"

    def execute(self, context):
        bpy.ops.wm.console_toggle()  # 打开控制台面板
        # 在这里处理文件夹选择逻辑
        bpy.ops.ui.directory_dialog('INVOKE_DEFAULT', directory=context.scene.mmt_props.path)

        print(context.scene.mmt_props.path)
        mmt_path = os.path.join(context.scene.mmt_props.path, "MMT.exe")
        # TODO 在这里读取配置文件，并设置outputFolder的路径为当前选择游戏的路径。设置当前游戏的提示，设置MMT path
        if os.path.exists(mmt_path):
            print("MMT.exe exists.")
        else:
            print(mmt_path)

        bpy.ops.wm.console_toggle()  # 关闭控制台面板
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

        # 在这里添加你的侧边栏内容
        layout.label(text="Thanks for using MMT Plugin! Happy Day!")

        props = context.scene.mmt_props
        layout.prop(props, "path")

        layout.separator()
        layout.label(text="Current working game: HI3")

        layout.separator()
        layout.label(text="Seperate import&export in OutputFolder")

        # TODO 快速导入，点这个之后默认路径为OutputFolder，这样直接就能去导入不用翻很久文件夹找路径了
        operator_import_txt = self.layout.operator("import_mesh.migoto_frame_analysis", text="Import .txt models ")
        operator_import_txt.directory = props.path

        # TODO 快速导出同理，点这个之后默认路径为OutputFolder，这样直接就能去导出不用翻很久文件夹找路径了
        operator_export_ibvb = self.layout.operator("export_mesh.migoto", text="Export .ib & .vb files ")
        operator_export_ibvb.filepath = context.scene.mmt_props.path + "1.vb"

        # 添加分隔符
        layout.separator()

        # TODO 一键快速导入所有位于OutputFolder下的.txt模型
        layout.label(text="Fast import&export in OutputFolder")
        operator44 = self.layout.operator("import_mesh.migoto_frame_analysis", text="Import All .txt model in OutputFolder ")
        operator44.filepath = context.scene.mmt_props.path + "1.vb"

        # TODO 一键快速导出当前选中Collection中的所有model到对应的hash值文件夹中，并直接调用MMT.exe的Mod生成方法，做到导出完即可游戏里F10刷新看效果。
        operator_export_ibvb = self.layout.operator("export_mesh.migoto", text="Export selected collection to generated mod folder ")
        operator_export_ibvb.filepath = context.scene.mmt_props.path + "1.vb"





