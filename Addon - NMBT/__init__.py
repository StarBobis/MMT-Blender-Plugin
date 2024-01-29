# Acknowledgements
# The original code is mainly forked from @Ian Munsie (darkstarsword@gmail.com)
# see https://github.com/DarkStarSword/3d-fixes,
# big thanks to his original blender plugin design.
# And part of the code is learned from projects below, huge thanks for their great code:
# - https://github.com/SilentNightSound/GI-Model-Importer
# - https://github.com/SilentNightSound/SR-Model-Importer
# - https://github.com/leotorrez/LeoTools


from .utils import *
from .migoto_format import *
from .mesh_operator import *

bl_info = {
    "name": "3Dmigoto Plugin",
    "author": "NicoMico",
    "description": "Special version blender plugin for 3Dmigoto Mod.",
    "blender": (4, 0, 0),
    "version": (1, 2),
    "location": "View3D",
    "warning": "",
    "category": "Generic"
}


def menu_func_import_fa(self, context):
    self.layout.operator(Import3DMigotoFrameAnalysis.bl_idname, text="3DMigoto FrameAnalysis dump (ib.txt + vb0.txt)")


def menu_func_import_raw(self, context):
    self.layout.operator(Import3DMigotoRaw.bl_idname, text="3DMigoto raw buffers (.vb + .ib)")


def menu_func_export(self, context):
    self.layout.operator(Export3DMigoto.bl_idname, text="3DMigoto raw buffers (.vb + .ib)")


register_classes = (
    Import3DMigotoFrameAnalysis,
    Import3DMigotoRaw,
    Import3DMigotoReferenceInputFormat,
    Export3DMigoto
)

# TODO we don't need any version compatible, remove this later.
# https://theduckcow.com/2019/update-addons-both-blender-28-and-27-support/
def make_annotations(cls):
    """Converts class fields to annotations"""
    bl_props = {k: v for k, v in cls.__dict__.items() if isinstance(v, tuple)}
    if bl_props:
        if '__annotations__' not in cls.__dict__:
            setattr(cls, '__annotations__', {})
        annotations = cls.__dict__['__annotations__']
        for k, v in bl_props.items():
            annotations[k] = v
            delattr(cls, k)
    return cls


def register():
    for cls in register_classes:
        make_annotations(cls)
        bpy.utils.register_class(cls)

    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_fa)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_raw)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    for cls in reversed(register_classes):
        bpy.utils.unregister_class(cls)

    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_fa)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_raw)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)


