# This mesh_operator.py is only used in right click options.
import math

from .panel import *
from .migoto_format import *


def remove_unused_vertex_group(self, context):
    for obj in bpy.context.selected_objects:
        if obj.type == "MESH":
            # obj = bpy.context.active_object
            obj.update_from_editmode()
            vgroup_used = {i: False for i, k in enumerate(obj.vertex_groups)}

            for v in obj.data.vertices:
                for g in v.groups:
                    if g.weight > 0.0:
                        vgroup_used[g.group] = True

            for i, used in sorted(vgroup_used.items(), reverse=True):
                if not used:
                    obj.vertex_groups.remove(obj.vertex_groups[i])

    return {'FINISHED'}


class RemoveUnusedVertexGroupOperator(bpy.types.Operator):
    bl_idname = "object.remove_unused_vertex_group"
    bl_label = "Remove Unused Vertex Group"

    def execute(self, context):
        return remove_unused_vertex_group(self, context)


def merge_vertex_group_with_same_number(self, context):
    # Author: SilentNightSound#7430
    # Combines vertex groups with the same prefix into one, a fast alternative to the Vertex Weight Mix that works for multiple groups
    # You will likely want to use blender_fill_vg_gaps.txt after this to fill in any gaps caused by merging groups together
    # Nico: we only need mode 3 here.

    import bpy
    import itertools
    class Fatal(Exception):
        pass

    selected_obj = [obj for obj in bpy.context.selected_objects]
    vgroup_names = []

    ##### USAGE INSTRUCTIONS
    # MODE 1: Runs the merge on a specific list of vertex groups in the selected object(s). Can add more names or fewer to the list - change the names to what you need
    # MODE 2: Runs the merge on a range of vertex groups in the selected object(s). Replace smallest_group_number with the lower bound, and largest_group_number with the upper bound
    # MODE 3 (DEFAULT): Runs the merge on ALL vertex groups in the selected object(s)

    # Select the mode you want to run:
    mode = 3

    # Required data for MODE 1:
    vertex_groups = ["replace_with_first_vertex_group_name", "second_vertex_group_name", "third_name_etc"]

    # Required data for MODE 2:
    smallest_group_number = 000
    largest_group_number = 999

    ######

    if mode == 1:
        vgroup_names = [vertex_groups]
    elif mode == 2:
        vgroup_names = [[f"{i}" for i in range(smallest_group_number, largest_group_number + 1)]]
    elif mode == 3:
        vgroup_names = [[x.name.split(".")[0] for x in y.vertex_groups] for y in selected_obj]
    else:
        raise Fatal("Mode not recognized, exiting")

    if not vgroup_names:
        raise Fatal(
            "No vertex groups found, please double check an object is selected and required data has been entered")

    for cur_obj, cur_vgroup in zip(selected_obj, itertools.cycle(vgroup_names)):
        for vname in cur_vgroup:
            relevant = [x.name for x in cur_obj.vertex_groups if x.name.split(".")[0] == f"{vname}"]

            if relevant:

                vgroup = cur_obj.vertex_groups.new(name=f"x{vname}")

                for vert_id, vert in enumerate(cur_obj.data.vertices):
                    available_groups = [v_group_elem.group for v_group_elem in vert.groups]

                    combined = 0
                    for v in relevant:
                        if cur_obj.vertex_groups[v].index in available_groups:
                            combined += cur_obj.vertex_groups[v].weight(vert_id)

                    if combined > 0:
                        vgroup.add([vert_id], combined, 'ADD')

                for vg in [x for x in cur_obj.vertex_groups if x.name.split(".")[0] == f"{vname}"]:
                    cur_obj.vertex_groups.remove(vg)

                for vg in cur_obj.vertex_groups:
                    if vg.name[0].lower() == "x":
                        vg.name = vg.name[1:]

        bpy.context.view_layer.objects.active = cur_obj
        bpy.ops.object.vertex_group_sort()
    return {'FINISHED'}


class MergeVertexGroupsWithSameNumber(bpy.types.Operator):
    bl_idname = "object.merge_vertex_group_with_same_number"
    bl_label = "Merge Vertex Groups"

    def execute(self, context):
        return merge_vertex_group_with_same_number(self, context)


def fill_vertex_group_gaps(self, context):
    # Author: SilentNightSound#7430
    # Fills in missing vertex groups for a model so there are no gaps, and sorts to make sure everything is in order
    # Works on the currently selected object
    # e.g. if the selected model has groups 0 1 4 5 7 2 it adds an empty group for 3 and 6 and sorts to make it 0 1 2 3 4 5 6 7
    # Very useful to make sure there are no gaps or out-of-order vertex groups

    import bpy

    # Can change this to another number in order to generate missing groups up to that number
    # e.g. setting this to 130 will create 0,1,2...130 even if the active selected object only has 90
    # Otherwise, it will use the largest found group number and generate everything up to that number
    largest = 0

    ob = bpy.context.active_object
    ob.update_from_editmode()

    for vg in ob.vertex_groups:
        try:
            if int(vg.name.split(".")[0]) > largest:
                largest = int(vg.name.split(".")[0])
        except ValueError:
            print("Vertex group not named as integer, skipping")

    missing = set([f"{i}" for i in range(largest + 1)]) - set([x.name.split(".")[0] for x in ob.vertex_groups])
    for number in missing:
        ob.vertex_groups.new(name=f"{number}")

    bpy.ops.object.vertex_group_sort()
    return {'FINISHED'}


class FillVertexGroupGaps(bpy.types.Operator):
    bl_idname = "object.fill_vertex_group_gaps"
    bl_label = "Fill Vertex Group Gaps"

    def execute(self, context):
        return fill_vertex_group_gaps(self, context)


def add_bone_from_vertex_group(self, context):
    # 这玩意实际上没啥用，但是好像又有点用，反正鸡肋，加上吧。
    # 获取当前选中的物体
    selected_object = bpy.context.object

    # 创建骨骼
    bpy.ops.object.armature_add()
    armature_object = bpy.context.object
    armature = armature_object.data

    # 切换到编辑模式
    bpy.ops.object.mode_set(mode='EDIT')

    # 遍历所有的顶点组
    for vertex_group in selected_object.vertex_groups:
        # 获取顶点组的名称
        vertex_group_name = vertex_group.name

        # 创建骨骼
        bone = armature.edit_bones.new(vertex_group_name)

        # 根据顶点组位置生成骨骼
        for vertex in selected_object.data.vertices:
            for group_element in vertex.groups:
                if group_element.group == vertex_group.index:
                    # 获取顶点位置
                    vertex_position = selected_object.matrix_world @ vertex.co

                    # 设置骨骼位置
                    bone.head = vertex_position
                    bone.tail = Vector(vertex_position) + Vector((0, 0, 0.1))  # 设置骨骼长度

                    # 分配顶点到骨骼
                    bone_vertex_group = selected_object.vertex_groups[vertex_group_name]
                    bone_vertex_group.add([vertex.index], 0, 'ADD')

    # 刷新场景
    bpy.context.view_layer.update()

    # 切换回对象模式
    bpy.ops.object.mode_set(mode='OBJECT')
    return {'FINISHED'}


class AddBoneFromVertexGroup(bpy.types.Operator):
    bl_idname = "object.add_bone_from_vertex_group"
    bl_label = "Add Bone From Vertex Group"

    def execute(self, context):
        return add_bone_from_vertex_group(self, context)


def remove_not_number_vertex_group(self, context):
    for obj in bpy.context.selected_objects:
        for vg in reversed(obj.vertex_groups):
            if vg.name.isdecimal():
                continue
            # print('Removing vertex group', vg.name)
            obj.vertex_groups.remove(vg)
    return {'FINISHED'}


class RemoveNotNumberVertexGroup(bpy.types.Operator):
    bl_idname = "object.remove_not_number_vertex_group"
    bl_label = "Remove Not Number Vertex Group"

    def execute(self, context):
        return remove_not_number_vertex_group(self, context)


def convert_to_fragment(self, context):
    # 获取当前选中的对象
    selected_objects = bpy.context.selected_objects

    # 检查是否选中了一个Mesh对象
    if len(selected_objects) != 1 or selected_objects[0].type != 'MESH':
        raise ValueError("请选中一个Mesh对象")

    # 获取选中的网格对象
    mesh_obj = selected_objects[0]
    mesh = mesh_obj.data

    # 遍历所有面
    selected_face_index = -1
    for i, face in enumerate(mesh.polygons):
        # 检查当前面是否已经是一个三角形
        if len(face.vertices) == 3:
            selected_face_index = i
            break

    if selected_face_index == -1:
        raise ValueError("没有选中的三角形面")

    # 选择指定索引的面
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')

    # 选择指定面的所有顶点
    bpy.context.tool_settings.mesh_select_mode[0] = True
    bpy.context.tool_settings.mesh_select_mode[1] = False
    bpy.context.tool_settings.mesh_select_mode[2] = False

    bpy.ops.object.mode_set(mode='OBJECT')

    # 获取选中面的所有顶点索引
    selected_face = mesh.polygons[selected_face_index]
    selected_vertices = [v for v in selected_face.vertices]

    # 删除非选定面的顶点
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')

    bpy.context.tool_settings.mesh_select_mode[0] = True
    bpy.context.tool_settings.mesh_select_mode[1] = False
    bpy.context.tool_settings.mesh_select_mode[2] = False

    bpy.ops.object.mode_set(mode='OBJECT')

    for vertex in mesh.vertices:
        if vertex.index not in selected_vertices:
            vertex.select = True

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.delete(type='VERT')

    # 切换回对象模式
    bpy.ops.object.mode_set(mode='OBJECT')

    return {'FINISHED'}


class ConvertToFragmentOperator(bpy.types.Operator):
    bl_idname = "object.convert_to_fragment"
    bl_label = "Convert To Fragment"

    def execute(self, context):
        return convert_to_fragment(self, context)


def delete_loose(self, context):
    # 获取当前选中的对象
    selected_objects = bpy.context.selected_objects
    # 检查是否选中了一个Mesh对象
    for obj in selected_objects:
        if obj.type == 'MESH':
            # 获取选中的网格对象
            bpy.ops.object.mode_set(mode='EDIT')
            # 选择所有的顶点
            bpy.ops.mesh.select_all(action='SELECT')
            # 执行删除孤立顶点操作
            bpy.ops.mesh.delete_loose()
            # 切换回对象模式
            bpy.ops.object.mode_set(mode='OBJECT')
    return {'FINISHED'}


class MMTDeleteLoose(bpy.types.Operator):
    bl_idname = "object.mmt_delete_loose"
    bl_label = "Delete Mesh's Loose Vertex"

    def execute(self, context):
        return delete_loose(self, context)


def mmt_reset_rotation(self, context):
    for obj in bpy.context.selected_objects:
        if obj.type == "MESH":
            # 将旋转角度归零
            obj.rotation_euler[0] = 0.0  # X轴
            obj.rotation_euler[1] = 0.0  # Y轴
            obj.rotation_euler[2] = 0.0  # Z轴

            # 应用旋转变换
            # bpy.context.view_layer.objects.active = obj
            # bpy.ops.object.transform_apply(rotation=True)
    return {'FINISHED'}


class MMTResetRotation(bpy.types.Operator):
    bl_idname = "object.mmt_reset_rotation"
    bl_label = "Reset x,y,z rotation number to 0 (UE Model)"

    def execute(self, context):
        return mmt_reset_rotation(self, context)


def mmt_cancel_auto_smooth(self, context):
    for obj in bpy.context.selected_objects:
        if obj.type == "MESH":
            # 取消勾选"Auto Smooth"
            obj.data.use_auto_smooth = False
    return {'FINISHED'}


class MMTCancelAutoSmooth(bpy.types.Operator):
    bl_idname = "object.mmt_cancel_auto_smooth"
    bl_label = "Cancel Auto Smooth for NORMAL (UE Model)"

    def execute(self, context):
        return mmt_cancel_auto_smooth(self, context)


def mmt_set_auto_smooth_89(self, context):
    for obj in bpy.context.selected_objects:
        if obj.type == "MESH":
            # 取消勾选"Auto Smooth"
            obj.data.use_auto_smooth = True
            obj.data.auto_smooth_angle = math.radians(89)
    return {'FINISHED'}


class MMTSetAutoSmooth89(bpy.types.Operator):
    bl_idname = "object.mmt_set_auto_smooth_89"
    bl_label = "Set Auto Smooth to 89° (Unity)"

    def execute(self, context):
        return mmt_set_auto_smooth_89(self, context)


# -----------------------------------这个属于右键菜单注册，单独的函数要往上面放---------------------------------------
class MigotoRightClickMenu(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_object_3Dmigoto"
    bl_label = "3Dmigoto"

    def draw(self, context):
        layout = self.layout
        layout.operator("object.remove_unused_vertex_group")
        layout.operator("object.merge_vertex_group_with_same_number")
        layout.operator("object.fill_vertex_group_gaps")
        layout.operator("object.add_bone_from_vertex_group")
        layout.operator("object.remove_not_number_vertex_group")
        layout.operator("object.convert_to_fragment")
        layout.operator("object.mmt_delete_loose")
        layout.operator("object.mmt_reset_rotation")
        layout.operator("object.mmt_cancel_auto_smooth")
        layout.operator("object.mmt_set_auto_smooth_89")


# 定义菜单项的注册函数
def menu_func_migoto_right_click(self, context):
    self.layout.menu(MigotoRightClickMenu.bl_idname)


# -----------------------------------下面这两个不属于右键菜单，属于MMT面板，所以放到最下面---------------------------------------
class MMTImportAllTextModel(bpy.types.Operator):
    bl_idname = "mmt.import_all"
    bl_label = "Import all txt model from current OutputFolder"

    def execute(self, context):
        # 首先根据MMT路径，获取
        mmt_path = bpy.context.scene.mmt_props.path
        current_game = ""
        main_setting_path = os.path.join(context.scene.mmt_props.path, "Configs\\wheel_setting\\MainSetting.json")
        if os.path.exists(main_setting_path):
            main_setting_file = open(main_setting_path)
            main_setting_json = json.load(main_setting_file)
            main_setting_file.close()
            current_game = main_setting_json["GameName"]

        game_config_path = os.path.join(context.scene.mmt_props.path, "Configs\\game_config\\" + current_game + "Config.json")
        game_config_file = open(game_config_path)
        game_config_json = json.load(game_config_file)
        game_config_file.close()

        output_folder_path = ""
        game_setting_path = os.path.join(context.scene.mmt_props.path, "Configs\\wheel_setting\\GameSetting.json")
        if os.path.exists(game_setting_path):
            game_setting_file = open(game_setting_path)
            game_setting_json = json.load(game_setting_file)
            game_setting_file.close()
            output_folder_path = str(game_setting_json[current_game + "_Dev"]["OutputFolder"]).replace("/","\\")

        import_folder_path_list = []
        for ib_config in game_config_json:
            draw_ib = ib_config["DrawIB"]
            # print("DrawIB:", draw_ib)
            import_folder_path_list.append(os.path.join(output_folder_path, draw_ib))

        # self.report({'INFO'}, "读取到的drawIB文件夹总数量：" + str(len(import_folder_path_list)))

        for import_folder_path in import_folder_path_list:
            prefix_set = set()
            # (1) 获取所有txt文件列表
            # self.report({'INFO'}, "Folder Name：" + import_folder_path)
            # 构造需要匹配的文件路径模式
            file_pattern = os.path.join(import_folder_path, "*.txt")
            # 使用 glob.glob 获取匹配的文件列表
            txt_file_list = glob(file_pattern)
            for txt_file_path in txt_file_list:
                # self.report({'INFO'}, "txt file: " + txt_file_path)
                txt_file_splits = os.path.basename(txt_file_path).split("-")
                prefix_set.add(txt_file_splits[0] + "-" + txt_file_splits[1])

            # (2)
            special_path4_set = set()
            for prefix in prefix_set:
                self.report({'INFO'}, "txt file prefix ib: " + import_folder_path + "\\" + prefix + "-ib.txt")
                self.report({'INFO'}, "txt file prefix vb0: " + import_folder_path + "\\" + prefix + "-vb0.txt")
                special_path4_set.add((import_folder_path + "\\" + prefix + "-vb0.txt", import_folder_path + "\\" + prefix + "-ib.txt", False,None))
            import_3dmigoto(self, context, special_path4_set)
        return {'FINISHED'}


class MMTExportAllIBVBModel(bpy.types.Operator):
    bl_idname = "mmt.export_all"
    bl_label = "Export all .ib and .vb model to current OutputFolder"

    def execute(self, context):
        # 首先根据MMT路径，获取
        mmt_path = bpy.context.scene.mmt_props.path
        current_game = ""
        main_setting_path = os.path.join(context.scene.mmt_props.path, "Configs\\Main.json")
        if os.path.exists(main_setting_path):
            main_setting_file = open(main_setting_path)
            main_setting_json = json.load(main_setting_file)
            main_setting_file.close()
            current_game = main_setting_json["GameName"]

        output_folder_path = mmt_path + "Games\\" + current_game + "\\3Dmigoto\\Mods\\output\\"
        # 创建 Export3DMigoto 类的实例对象


        # 遍历当前选中列表的所有mesh，根据名称导出到对应的文件夹中
        # 获取当前选中的对象列表
        selected_collection = bpy.context.collection

        # 遍历选中的对象
        for obj in selected_collection.objects:
            # 判断对象是否为网格对象
            if obj.type == 'MESH':
                bpy.context.view_layer.objects.active = obj
                mesh = obj.data  # 获取网格数据

                self.report({'INFO'}, "export name: " + mesh.name)

                # 处理当前网格对象
                # 例如，打印网格名称

                name_splits = str(mesh.name).split("-")
                draw_ib = name_splits[0]
                draw_index = name_splits[1]
                draw_index = draw_index[0:len(draw_index) - 3]
                if draw_index.endswith(".vb."):
                    draw_index = draw_index[0:len(draw_index) - 4]

                # 设置类属性的值
                vb_path = output_folder_path + draw_ib + "\\" + draw_index + ".vb"
                self.report({'INFO'}, "export path: " + vb_path)

                ib_path = os.path.splitext(vb_path)[0] + '.ib'
                fmt_path = os.path.splitext(vb_path)[0] + '.fmt'

                # FIXME: ExportHelper will check for overwriting vb_path, but not ib_path

                # Nico: now we use falling-ts 's solution to not change vertex group.
                # export_3dmigoto_without_position_increase(self, context, vb_path, ib_path, fmt_path)
                export_3dmigoto(self, context, vb_path, ib_path, fmt_path)

        self.report({'INFO'}, "Export Success!")
        return {'FINISHED'}

