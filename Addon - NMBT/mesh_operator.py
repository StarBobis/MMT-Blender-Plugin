# Acknowledgements
# The original code is mainly forked from @Ian Munsie (darkstarsword@gmail.com)
# see https://github.com/DarkStarSword/3d-fixes,
# big thanks to his original blender plugin design.
# And part of the code is learned from projects below, huge thanks for their great code:
# - https://github.com/SilentNightSound/GI-Model-Importer
# - https://github.com/SilentNightSound/SR-Model-Importer
# - https://github.com/leotorrez/LeoTools

from .utils import *








def remove_unused_vertex_group(self, context):
    obj = bpy.context.active_object
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


class MigotoRightClickMenu(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_object_3Dmigoto"
    bl_label = "3Dmigoto"

    def draw(self, context):
        layout = self.layout
        layout.operator("object.remove_unused_vertex_group")


# 定义菜单项的注册函数
def menu_func_remove_unused_vgs(self, context):
    self.layout.menu(MigotoRightClickMenu.bl_idname)





