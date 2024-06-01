# This animation_operator.py is only used in animation mod related options.
import json
import shutil

import bpy.props

from .panel import *
from datetime import datetime


class MMDModIniGenerator(bpy.types.Operator):
    # ini 属性内容
    BufFileSuffix = None
    CategoryDrawCategoryMap = {}
    CategoryHashMap = {}
    CategoryList = []
    CategorySlotMap = {}
    CategoryStrideMap = {}
    DrawIB = None
    DrawNumber = None
    IBFileSuffix = None
    MatchFirstIndexList = []
    PartNameList = []
    TextureMap = {}
    VertexLimitVB = None
    blendElementByteWidth = 0
    categoryUUIDMap = {}
    partNameUUIDMap = {}
    patchBLENDWEIGHTS = False

    # OutputFolder路径
    output_folder: bpy.props.StringProperty(
        name="Output Folder",
        description="The Output Folder of this game"
    )
    # 完整buf文件导出路径  operator_export_mmd_bone_matrix.output_bone_matrix_filename = output_folder_path + "BoneMatrix.buf"
    output_bone_matrix_file_path = None
    # 完整ini文件导出路径
    output_ini_file_path = None

    bl_idname = "mmt.export_mmd_animation_mod"
    bl_label = "Export MMD Based Animation Mod"

    def execute(self, context):
        # 根据当前的姿态
        # 获取当前选中的骨骼名称
        selected_armature = bpy.context.object
        armature_name = ""
        if selected_armature and selected_armature.type == 'ARMATURE':
            armature_name = selected_armature.name
        else:
            armature_name = "No Armature selected"

        # self.report({'INFO'}, "当前选中的骨骼名称:" + armature_name)

        # 获取当前骨骼下面的第一个网格名称
        mesh_name = None
        if selected_armature:
            for obj in selected_armature.children:
                if obj.type == 'MESH':
                    mesh_name = obj.name
                    break

        # self.report({'INFO'}, "当前骨骼下面的第一个网格名称:" + mesh_name)

        draw_ib = str(mesh_name).split("-")[0]

        # 获取当前日期
        current_date = datetime.now().date()
        # 将日期转换为指定格式的字符串
        date_string = current_date.strftime("%Y_%m_%d")
        # 读取IniConfig.json中的数据
        output_draw_ib_path = self.output_folder + date_string + "/" + draw_ib + "/"
        json_file_path = output_draw_ib_path + "IniConfig.json"
        with open(json_file_path) as f:
            json_data = json.load(f)
        self.BufFileSuffix = json_data["BufFileSuffix"]
        self.CategoryDrawCategoryMap = json_data["CategoryDrawCategoryMap"]
        self.CategoryHashMap = json_data["CategoryHashMap"]
        self.CategoryList = json_data["CategoryList"]
        self.CategorySlotMap = json_data["CategorySlotMap"]
        self.CategoryStrideMap = json_data["CategoryStrideMap"]
        self.DrawIB = json_data["DrawIB"]
        self.DrawNumber = json_data["DrawNumber"]
        self.IBFileSuffix = json_data["IBFileSuffix"]
        self.MatchFirstIndexList = json_data["MatchFirstIndexList"]
        self.PartNameList = json_data["PartNameList"]
        self.TextureMap = json_data["TextureMap"]
        self.VertexLimitVB = json_data["VertexLimitVB"]
        self.blendElementByteWidth = json_data["blendElementByteWidth"]
        self.categoryUUIDMap = json_data["categoryUUIDMap"]
        self.partNameUUIDMap = json_data["partNameUUIDMap"]
        self.patchBLENDWEIGHTS = json_data["patchBLENDWEIGHTS"]
        # 这里测试可以正常输出
        # self.report({'INFO'}, "CategoryHashMap:" + str(self.CategoryHashMap))

        # 把旧的ini文件，如果不是Disabled就变为disabled
        for filename in os.listdir(output_draw_ib_path):
            if filename.endswith(".ini") and not filename.startswith("DISABLED"):
                # 构建新的文件名
                new_filename = "DISABLED_" + filename
                # 源文件的完整路径
                src = os.path.join(output_draw_ib_path, filename)
                # 目标文件的完整路径
                dst = os.path.join(output_draw_ib_path, new_filename)
                # 重命名文件
                shutil.move(src, dst)
                # print(f"重命名文件：{filename} -> {new_filename}")

        # 然后导出这个骨骼姿态变换矩阵
        output_bonematrix_file_path = output_draw_ib_path + draw_ib + "PoseMatrix.buf"
        frame_start = context.scene.mmt_mmd_animation_mod_start_frame
        frame_end = context.scene.mmt_mmd_animation_mod_end_frame
        play_speed = context.scene.mmt_mmd_animation_mod_play_speed

        result = bytearray()
        for z in range(frame_start, frame_end):
            context.scene.frame_set(z)
            vst = bytearray()

            bones = {}
            for bone in bpy.data.objects[armature_name].pose.bones:
                bones[bone.name] = bone

            for vg in bpy.data.objects[mesh_name].vertex_groups:
                # print(vg.name)
                bone = bones[vg.name]
                for i in range(3):
                    for j in range(4):
                        vst += struct.pack("f", bone.matrix_channel[i][j])
            result += vst

        with open(output_bonematrix_file_path, "wb") as f:
            f.write(result)

        # 然后需要获取一下顶点组数量
        vertex_group_number = len(bpy.data.objects[mesh_name].vertex_groups)

        # 然后生成我们新的ini文件
        output_ini_content = ""
        output_ini_name = output_draw_ib_path + draw_ib + ".ini"

        generate_switch_key = True
        active_flag_name = "Active_Flag_" + draw_ib
        switch_var_name = "SwitchVar_" + draw_ib
        replace_prefix = "  "

        output_ini_content = output_ini_content + "[Constants]" + "\n"
        output_ini_content = output_ini_content + "global persist $" + switch_var_name + " = 1" + "\n"
        output_ini_content = output_ini_content + "global $" + active_flag_name + " = 1" + "\n"
        output_ini_content = output_ini_content + "\n"

        output_ini_content = output_ini_content + "[Key" + switch_var_name + "]" + "\n"
        output_ini_content = output_ini_content + "condition = $" + active_flag_name + " == 1" + "\n"
        output_ini_content = output_ini_content + "key = i" + "\n"
        output_ini_content = output_ini_content + "type = cycle" + "\n"
        output_ini_content = output_ini_content + "$" + switch_var_name + " = 0,1" + "\n"
        output_ini_content = output_ini_content + "\n"

        output_ini_content = output_ini_content + "[Present]" + "\n"
        output_ini_content = output_ini_content + "post $" + active_flag_name + " = 0" + "\n"
        output_ini_content = output_ini_content + "\n"

        mmd_frame_name = "$Frame_" + draw_ib
        mmd_start_frame_name = "$StartFrame_" + draw_ib
        mmd_end_frame_name = "$EndFrame_" + draw_ib
        mmd_vertex_group_count_name = "$VertexGroupCount_" + draw_ib
        mmd_play_speed_name = "$PlaySpeed_" + draw_ib
        mmd_pose_matrix_resource_name = "Resource_PoseMatrix"

        output_ini_content = output_ini_content + "[Constants]" + "\n"
        output_ini_content = output_ini_content + "global persist " + mmd_frame_name + " = 0" + "\n"
        output_ini_content = output_ini_content + "global persist " + mmd_start_frame_name + " = " + str(frame_start) + "\n"
        output_ini_content = output_ini_content + "global persist " + mmd_end_frame_name + " = " + str(frame_end) + "\n"
        output_ini_content = output_ini_content + "global persist " + mmd_vertex_group_count_name + " = " + str(vertex_group_number) + "\n"
        output_ini_content = output_ini_content + "global persist " + mmd_play_speed_name + " = " + str(play_speed) + "\n"
        output_ini_content = output_ini_content + "\n"

        output_ini_content = output_ini_content + "[Present]" + "\n"
        output_ini_content = output_ini_content + "if " + mmd_frame_name + " > " + mmd_end_frame_name + "\n"
        output_ini_content = output_ini_content + "  " + mmd_frame_name + " = " + mmd_start_frame_name + "\n"
        output_ini_content = output_ini_content + "endif" + "\n"
        output_ini_content = output_ini_content + mmd_frame_name + " = " + mmd_frame_name + " + " + mmd_play_speed_name + "\n"
        if generate_switch_key:
            output_ini_content = output_ini_content + "if $" + switch_var_name + " == 0" + "\n"
        output_ini_content = output_ini_content + replace_prefix + mmd_frame_name + " = " + mmd_start_frame_name + "\n"
        if generate_switch_key:
            output_ini_content = output_ini_content + "endif" + "\n"
        output_ini_content = output_ini_content + "\n"



        output_ini_content = output_ini_content + "[TextureOverride_" + draw_ib + "_IB_SKIP]" + "\n"
        output_ini_content = output_ini_content + "hash = " + draw_ib + "\n"
        if generate_switch_key:
            output_ini_content = output_ini_content + "if $" + switch_var_name + " == 1" + "\n"
        output_ini_content = output_ini_content + replace_prefix + "handling = skip" + "\n"
        if generate_switch_key:
            output_ini_content = output_ini_content + "endif" + "\n"
        output_ini_content = output_ini_content + "\n"

        # TextureOverride IB部分
        for i in range(len(self.PartNameList)):
            ib_first_index = self.MatchFirstIndexList[i]
            part_name = self.PartNameList[i]
            output_ini_content = output_ini_content + "[TextureOverride_IB_" + draw_ib + "_" + part_name + "]" + "\n"
            output_ini_content = output_ini_content + "hash = " + draw_ib + "\n"
            output_ini_content = output_ini_content + "match_first_index = " + ib_first_index + "\n"
            if generate_switch_key:
                output_ini_content = output_ini_content + "if $" + switch_var_name + " == 1" + "\n"
            output_ini_content = output_ini_content + replace_prefix + "ib = Resource_IB_" + draw_ib + "_" + part_name + "\n"
            output_ini_content = output_ini_content + replace_prefix + "drawindexed = auto" + "\n"
            if generate_switch_key:
                output_ini_content = output_ini_content + "endif" + "\n"
            output_ini_content = output_ini_content + "\n"

        # VertexLimitRaise
        output_ini_content = output_ini_content + "[TextureOverride_VB_" + draw_ib + "_" + str(self.CategoryStrideMap["Position"]) + "_" + str(self.DrawNumber) + "_VertexLimitRaise]" + "\n"
        output_ini_content = output_ini_content + "hash = " + self.VertexLimitVB + "\n"
        output_ini_content = output_ini_content + "\n"

        # TextureOverride VB部分
        for i in range(len(self.CategoryList)):
            category_name = self.CategoryList[i]
            category_hash = self.CategoryHashMap[category_name]
            category_slot = self.CategorySlotMap[category_name]
            output_ini_content = output_ini_content + "[TextureOverride_VB_" + draw_ib + "_" + category_name + "]" + "\n"
            output_ini_content = output_ini_content + "hash = " + category_hash + "\n"
            if generate_switch_key:
                output_ini_content = output_ini_content + "if $" + switch_var_name + " == 1" + "\n"

            draw_category_name = self.CategoryDrawCategoryMap[category_name]
            if category_name == draw_category_name:
                category_original_slot = self.CategorySlotMap[category_name]
                output_ini_content = output_ini_content + replace_prefix + category_original_slot + " = " + "Resource_VB_" + category_name + "\n"

                if category_name == "Position":
                    output_ini_content = output_ini_content + replace_prefix + "vs-t10 = " + mmd_pose_matrix_resource_name + "\n"
                    output_ini_content = output_ini_content + replace_prefix + "x140 = " + mmd_vertex_group_count_name + "\n"
                    output_ini_content = output_ini_content + replace_prefix + "x141 = " + mmd_frame_name + "\n"

            if category_name == self.CategoryDrawCategoryMap["Blend"]:
                output_ini_content = output_ini_content + replace_prefix + "handling = skip" + "\n"
                output_ini_content = output_ini_content + replace_prefix + "draw = " + str(self.DrawNumber) + ", 0" + "\n"

            if generate_switch_key:
                output_ini_content = output_ini_content + "endif" + "\n"

            if category_name == self.CategoryDrawCategoryMap["Position"]:
                output_ini_content = output_ini_content + "$" + active_flag_name + " = 1" + "\n"
            output_ini_content = output_ini_content + "\n"

        output_ini_content = output_ini_content + "\n"
        output_ini_content = output_ini_content + ";------------------------------------------------------------------------------------------"
        output_ini_content = output_ini_content + ";-------------------------------------Resource Section-------------------------------------"
        output_ini_content = output_ini_content + ";------------------------------------------------------------------------------------------"
        output_ini_content = output_ini_content + "\n"


        # VB Resource部分
        for category_name in self.CategoryList:
            output_ini_content = output_ini_content + "[Resource_VB_" + category_name + "]" + "\n"
            output_ini_content = output_ini_content + "type = Buffer" + "\n"

            if category_name == "Blend" and self.patchBLENDWEIGHTS:
                final_blend_stride = self.CategoryStrideMap[category_name] - self.blendElementByteWidth
                output_ini_content = output_ini_content + "stride = " + str(final_blend_stride) + "\n"
            else:
                output_ini_content = output_ini_content + "stride = " + str(self.CategoryStrideMap[category_name]) + "\n"

            output_ini_content = output_ini_content + "filename = " + self.categoryUUIDMap[category_name] + "." + self.BufFileSuffix + "\n"
            output_ini_content = output_ini_content + "\n"

        # IB Resource部分
        for part_name in self.PartNameList:
            output_ini_content = output_ini_content + "[Resource_IB_" + draw_ib + "_" + part_name + "]" + "\n"
            output_ini_content = output_ini_content + "type = Buffer" + "\n"
            output_ini_content = output_ini_content + "format = DXGI_FORMAT_R32_UINT" + "\n"
            output_ini_content = output_ini_content + "filename = " + self.partNameUUIDMap[part_name] + "." + self.IBFileSuffix + "\n"
            output_ini_content = output_ini_content + "\n"

        # PoseMatrix Resource部分
        output_ini_content = output_ini_content + "[" + mmd_pose_matrix_resource_name + "]" + "\n"
        output_ini_content = output_ini_content + "type = Buffer" + "\n"
        output_ini_content = output_ini_content + "format = DXGI_FORMAT_R32G32B32A32_FLOAT" + "\n"
        output_ini_content = output_ini_content + "filename = " + draw_ib + "PoseMatrix.buf" + "\n"
        output_ini_content = output_ini_content + "\n"

        # 最终写出到ini文件
        with open(output_ini_name, "w") as f:
            f.write(output_ini_content)
        return {'FINISHED'}



