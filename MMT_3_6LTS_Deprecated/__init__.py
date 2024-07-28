import bpy.props

# The __init__.py only designed to register and unregister ,so as a simple control for the whole plugin,
# keep it clean and don't add too many code,code should be in other files and import it here.
# we use .utils instead of utils because blender can't locate where utils is
# Blender can only locate panel.py  when you add a . before it.
from .panel import *
from .migoto_format import *
from .mesh_operator import *
from .animation_operator import *


bl_info = {
    "name": "MMT-Community's Blender Plugin",
    "fork author": "NicoMico",
    "original author": "DarkStarSword",
    "description": "Special fork version of DarkStarSword's blender_3dmigoto.py",
    "blender": (3, 6, 0),
    "version": (1, 5, 5),
    "location": "View3D",
    "warning": "",
    "category": "Generic"
}


register_classes = (
    # migoto_format
    MMTPathProperties,
    MMTPathOperator,
    MMTPanel,

    Import3DMigotoFrameAnalysis,
    Import3DMigotoRaw,
    Import3DMigotoReferenceInputFormat,
    Export3DMigoto,

    # mesh_operator 右键菜单栏
    RemoveUnusedVertexGroupOperator,
    MergeVertexGroupsWithSameNumber,
    FillVertexGroupGaps,
    AddBoneFromVertexGroup,
    RemoveNotNumberVertexGroup,
    ConvertToFragmentOperator,
    MMTDeleteLoose,
    MMTResetRotation,
    MigotoRightClickMenu,
    MMTCancelAutoSmooth,
    MMTShowIndexedVertices,
    MMTSetAutoSmooth89,
    SplitMeshByCommonVertexGroup,

    # MMT的一键导入导出
    MMTImportAllTextModel,
    MMTExportAllIBVBModel,

    # MMD类型动画Mod支持
    MMDModIniGenerator
)


def register():
    for cls in register_classes:
        # make_annotations(cls)
        bpy.utils.register_class(cls)

    # 新建一个属性用来专门装MMT的路径
    bpy.types.Scene.mmt_props = bpy.props.PointerProperty(type=MMTPathProperties)

    # mesh_operator
    bpy.types.VIEW3D_MT_object_context_menu.append(menu_func_migoto_right_click)

    # 在Blender退出前保存选择的MMT的路径
    bpy.app.handlers.depsgraph_update_post.append(save_mmt_path)

    # MMT数值保存的变量
    bpy.types.Scene.mmt_mmd_animation_mod_start_frame = bpy.props.IntProperty(name="Start Frame")
    bpy.types.Scene.mmt_mmd_animation_mod_end_frame = bpy.props.IntProperty(name="End Frame")
    bpy.types.Scene.mmt_mmd_animation_mod_play_speed = bpy.props.FloatProperty(name="Play Speed")

    
def unregister():
    for cls in reversed(register_classes):
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.mmt_props

    # mesh_operator
    bpy.types.VIEW3D_MT_object_context_menu.remove(menu_func_migoto_right_click)

    # 退出注册时删除MMT的MMD变量
    del bpy.types.Scene.mmt_mmd_animation_mod_start_frame
    del bpy.types.Scene.mmt_mmd_animation_mod_end_frame
    del bpy.types.Scene.mmt_mmd_animation_mod_play_speed

