# This migoto_format.py is only used in 3dmigoto import and export options.
import bpy

from .panel import *

# 这里使用type关键字创建了一个类，类名是DummyIOOBJOrientationHelper，(object,)表示继承自object对象，{}表示没定义属性和方法
IOOBJOrientationHelper = type('DummyIOOBJOrientationHelper', (object,), {})

# Constants
vertex_color_layer_channels = 4

# TODO we don't need any old version compatible, remove these in later version.
def set_active_object(context, obj):
    context.view_layer.objects.active = obj  # the 2.8 way

def get_active_object(context):
    return context.view_layer.objects.active


def link_object_to_scene(context, obj):
    context.scene.collection.objects.link(obj)


def unlink_object(context, obj):
    context.scene.collection.objects.unlink(obj)


def matmul(a, b):
    return operator.matmul(a, b)  # the same as writing a @ b


def keys_to_ints(d):
    return {k.isdecimal() and int(k) or k: v for k, v in d.items()}


def keys_to_strings(d):
    return {str(k): v for k, v in d.items()}


# This is used to catch any exception in run time and raise it to blender output console.
class Fatal(Exception):
    pass


'''
DXGI Formats but why use this unreadable format ?
'''
f32_pattern = re.compile(r'''(?:DXGI_FORMAT_)?(?:[RGBAD]32)+_FLOAT''')
f16_pattern = re.compile(r'''(?:DXGI_FORMAT_)?(?:[RGBAD]16)+_FLOAT''')
u32_pattern = re.compile(r'''(?:DXGI_FORMAT_)?(?:[RGBAD]32)+_UINT''')
u16_pattern = re.compile(r'''(?:DXGI_FORMAT_)?(?:[RGBAD]16)+_UINT''')
u8_pattern = re.compile(r'''(?:DXGI_FORMAT_)?(?:[RGBAD]8)+_UINT''')
s32_pattern = re.compile(r'''(?:DXGI_FORMAT_)?(?:[RGBAD]32)+_SINT''')
s16_pattern = re.compile(r'''(?:DXGI_FORMAT_)?(?:[RGBAD]16)+_SINT''')
s8_pattern = re.compile(r'''(?:DXGI_FORMAT_)?(?:[RGBAD]8)+_SINT''')
unorm16_pattern = re.compile(r'''(?:DXGI_FORMAT_)?(?:[RGBAD]16)+_UNORM''')
unorm8_pattern = re.compile(r'''(?:DXGI_FORMAT_)?(?:[RGBAD]8)+_UNORM''')
snorm16_pattern = re.compile(r'''(?:DXGI_FORMAT_)?(?:[RGBAD]16)+_SNORM''')
snorm8_pattern = re.compile(r'''(?:DXGI_FORMAT_)?(?:[RGBAD]8)+_SNORM''')

misc_float_pattern = re.compile(r'''(?:DXGI_FORMAT_)?(?:[RGBAD][0-9]+)+_(?:FLOAT|UNORM|SNORM)''')
misc_int_pattern = re.compile(r'''(?:DXGI_FORMAT_)?(?:[RGBAD][0-9]+)+_[SU]INT''')


# TODO strange design and hard to read and understand for mod author, make this easier and more readable later.
def EncoderDecoder(fmt):
    if f32_pattern.match(fmt):
        return (lambda data: b''.join(struct.pack('<f', x) for x in data),
                lambda data: numpy.frombuffer(data, numpy.float32).tolist())
    if f16_pattern.match(fmt):
        return (lambda data: numpy.fromiter(data, numpy.float16).tobytes(),
                lambda data: numpy.frombuffer(data, numpy.float16).tolist())
    if u32_pattern.match(fmt):
        return (lambda data: numpy.fromiter(data, numpy.uint32).tobytes(),
                lambda data: numpy.frombuffer(data, numpy.uint32).tolist())
    if u16_pattern.match(fmt):
        return (lambda data: numpy.fromiter(data, numpy.uint16).tobytes(),
                lambda data: numpy.frombuffer(data, numpy.uint16).tolist())
    if u8_pattern.match(fmt):
        return (lambda data: numpy.fromiter(data, numpy.uint8).tobytes(),
                lambda data: numpy.frombuffer(data, numpy.uint8).tolist())
    if s32_pattern.match(fmt):
        return (lambda data: numpy.fromiter(data, numpy.int32).tobytes(),
                lambda data: numpy.frombuffer(data, numpy.int32).tolist())
    if s16_pattern.match(fmt):
        return (lambda data: numpy.fromiter(data, numpy.int16).tobytes(),
                lambda data: numpy.frombuffer(data, numpy.int16).tolist())
    if s8_pattern.match(fmt):
        return (lambda data: numpy.fromiter(data, numpy.int8).tobytes(),
                lambda data: numpy.frombuffer(data, numpy.int8).tolist())

    if unorm16_pattern.match(fmt):
        return (
        lambda data: numpy.around((numpy.fromiter(data, numpy.float32) * 65535.0)).astype(numpy.uint16).tobytes(),
        lambda data: (numpy.frombuffer(data, numpy.uint16) / 65535.0).tolist())
    if unorm8_pattern.match(fmt):
        return (lambda data: numpy.around((numpy.fromiter(data, numpy.float32) * 255.0)).astype(numpy.uint8).tobytes(),
                lambda data: (numpy.frombuffer(data, numpy.uint8) / 255.0).tolist())
    if snorm16_pattern.match(fmt):
        return (
        lambda data: numpy.around((numpy.fromiter(data, numpy.float32) * 32767.0)).astype(numpy.int16).tobytes(),
        lambda data: (numpy.frombuffer(data, numpy.int16) / 32767.0).tolist())
    if snorm8_pattern.match(fmt):
        return (lambda data: numpy.around((numpy.fromiter(data, numpy.float32) * 127.0)).astype(numpy.int8).tobytes(),
                lambda data: (numpy.frombuffer(data, numpy.int8) / 127.0).tolist())

    raise Fatal('File uses an unsupported DXGI Format: %s' % fmt)


# TODO strange design again ,why don't use a class to wrap all encode and decode??
components_pattern = re.compile(r'''(?<![0-9])[0-9]+(?![0-9])''')


def format_components(fmt):
    return len(components_pattern.findall(fmt))


def format_size(fmt):
    matches = components_pattern.findall(fmt)
    return sum(map(int, matches)) // 8


# TODO why the parameter is an object type instead of bpy.types.Operator or specific type??
class InputLayoutElement(object):
    SemanticName = ""
    SemanticIndex = ""
    Format = ""
    InputSlot = ""
    AlignedByteOffset = ""
    InputSlotClass = ""
    InstanceDataStepRate = ""

    def __init__(self, arg):
        if isinstance(arg, io.IOBase):
            self.from_file(arg)
        else:
            self.from_dict(arg)

        self.encoder, self.decoder = EncoderDecoder(self.Format)

    def from_file(self, f):
        self.SemanticName = self.next_validate(f, 'SemanticName')
        self.SemanticIndex = int(self.next_validate(f, 'SemanticIndex'))
        self.Format = self.next_validate(f, 'Format')
        self.InputSlot = int(self.next_validate(f, 'InputSlot'))
        self.AlignedByteOffset = self.next_validate(f, 'AlignedByteOffset')
        if self.AlignedByteOffset == 'append':
            raise Fatal('Input layouts using "AlignedByteOffset=append" are not yet supported')
        self.AlignedByteOffset = int(self.AlignedByteOffset)
        self.InputSlotClass = self.next_validate(f, 'InputSlotClass')
        self.InstanceDataStepRate = int(self.next_validate(f, 'InstanceDataStepRate'))

    def to_dict(self):
        d = {'SemanticName': self.SemanticName, 'SemanticIndex': self.SemanticIndex, 'Format': self.Format,
             'InputSlot': self.InputSlot, 'AlignedByteOffset': self.AlignedByteOffset,
             'InputSlotClass': self.InputSlotClass, 'InstanceDataStepRate': self.InstanceDataStepRate}
        return d

    def to_string(self, indent=2):
        return textwrap.indent(textwrap.dedent('''
            SemanticName: %s
            SemanticIndex: %i
            Format: %s
            InputSlot: %i
            AlignedByteOffset: %i
            InputSlotClass: %s
            InstanceDataStepRate: %i
        ''').lstrip() % (
            self.SemanticName,
            self.SemanticIndex,
            self.Format,
            self.InputSlot,
            self.AlignedByteOffset,
            self.InputSlotClass,
            self.InstanceDataStepRate,
        ), ' ' * indent)

    def from_dict(self, d):
        self.SemanticName = d['SemanticName']
        self.SemanticIndex = d['SemanticIndex']
        self.Format = d['Format']
        self.InputSlot = d['InputSlot']
        self.AlignedByteOffset = d['AlignedByteOffset']
        self.InputSlotClass = d['InputSlotClass']
        self.InstanceDataStepRate = d['InstanceDataStepRate']

    @staticmethod
    def next_validate(f, field):
        line = next(f).strip()
        assert (line.startswith(field + ': '))
        return line[len(field) + 2:]

    @property
    def name(self):
        if self.SemanticIndex:
            return '%s%i' % (self.SemanticName, self.SemanticIndex)
        return self.SemanticName

    def pad(self, data, val):

        padding = format_components(self.Format) - len(data)
        assert (padding >= 0)
        return data + [val] * padding

        # 这里SinsOfSeven做了改进：https://github.com/DarkStarSword/3d-fixes/issues/32
        # 但是3.6.8LTS测试无法使用
        # padding = self.format_len - len(data)
        # assert (padding >= 0)
        # data.extend([val] * padding)
        # return data


    def clip(self, data):
        return data[:format_components(self.Format)]

    def size(self):
        return format_size(self.Format)

    def is_float(self):
        return misc_float_pattern.match(self.Format)

    def is_int(self):
        return misc_int_pattern.match(self.Format)

    def encode(self, data):
        # print(self.Format, data)
        return self.encoder(data)

    def decode(self, data):
        return self.decoder(data)

    def __eq__(self, other):
        return \
                self.SemanticName == other.SemanticName and \
                self.SemanticIndex == other.SemanticIndex and \
                self.Format == other.Format and \
                self.InputSlot == other.InputSlot and \
                self.AlignedByteOffset == other.AlignedByteOffset and \
                self.InputSlotClass == other.InputSlotClass and \
                self.InstanceDataStepRate == other.InstanceDataStepRate


class InputLayout(object):
    def __init__(self, custom_prop=[], stride=0):
        self.elems = collections.OrderedDict()
        self.stride = stride
        for item in custom_prop:
            elem = InputLayoutElement(item)
            self.elems[elem.name] = elem

    def serialise(self):
        return [x.to_dict() for x in self.elems.values()]

    def to_string(self):
        ret = ''
        for i, elem in enumerate(self.elems.values()):
            ret += 'element[%i]:\n' % i
            ret += elem.to_string()
        return ret

    def parse_element(self, f):
        elem = InputLayoutElement(f)
        self.elems[elem.name] = elem

    def __iter__(self):
        return iter(self.elems.values())

    def __getitem__(self, semantic):
        return self.elems[semantic]

    def encode(self, vertex):
        buf = bytearray(self.stride)

        for semantic, data in vertex.items():
            if semantic.startswith('~'):
                continue
            elem = self.elems[semantic]
            data = elem.encode(data)
            buf[elem.AlignedByteOffset:elem.AlignedByteOffset + len(data)] = data

        assert (len(buf) == self.stride)
        return buf

    def decode(self, buf):
        vertex = {}
        for elem in self.elems.values():
            data = buf[elem.AlignedByteOffset:elem.AlignedByteOffset + elem.size()]
            vertex[elem.name] = elem.decode(data)
        return vertex

    def __eq__(self, other):
        return self.elems == other.elems


class HashableVertex(dict):
    # 旧的代码注释掉了，不过不删，留着用于参考防止忘记原本的设计
    # def __hash__(self):
    #     # Convert keys and values into immutable types that can be hashed
    #     immutable = tuple((k, tuple(v)) for k, v in sorted(self.items()))
    #     return hash(immutable)

    def __hash__(self):
        # 这里将步骤拆分开来，更易于理解
        immutable_items = []
        for k, v in self.items():
            tuple_v = tuple(v)
            pair = (k, tuple_v)
            immutable_items.append(pair)
        sorted_items = sorted(immutable_items)
        immutable = tuple(sorted_items)
        return hash(immutable)


class VertexBuffer(object):
    vb_elem_pattern = re.compile(r'''vb\d+\[\d*\]\+\d+ (?P<semantic>[^:]+): (?P<data>.*)$''')

    # Python gotcha - do not set layout=InputLayout() in the default function
    # parameters, as they would all share the *same* InputLayout since the
    # default values are only evaluated once on file load
    def __init__(self, f=None, layout=None, load_vertices=True):
        self.vertices = []
        self.layout = layout and layout or InputLayout()
        self.first = 0
        self.vertex_count = 0
        self.offset = 0
        self.topology = 'trianglelist'

        if f is not None:
            self.parse_vb_txt(f, load_vertices)

    def parse_vb_txt(self, f, load_vertices):
        for line in map(str.strip, f):
            # print(line)
            if line.startswith('byte offset:'):
                self.offset = int(line[13:])
            if line.startswith('first vertex:'):
                self.first = int(line[14:])
            if line.startswith('vertex count:'):
                self.vertex_count = int(line[14:])
            if line.startswith('stride:'):
                self.layout.stride = int(line[7:])
            if line.startswith('element['):
                self.layout.parse_element(f)
            if line.startswith('topology:'):
                self.topology = line[10:]
                if line != 'topology: trianglelist':
                    raise Fatal('"%s" is not yet supported' % line)
            if line.startswith('vertex-data:'):
                if not load_vertices:
                    return
                self.parse_vertex_data(f)
        assert (len(self.vertices) == self.vertex_count)

    def parse_vb_bin(self, f):
        f.seek(self.offset)
        # XXX: Should we respect the first/base vertex?
        # f.seek(self.first * self.layout.stride, whence=1)
        self.first = 0
        while True:
            vertex = f.read(self.layout.stride)
            if not vertex:
                break
            self.vertices.append(self.layout.decode(vertex))
        # We intentionally disregard the vertex count when loading from a
        # binary file, as we assume frame analysis might have only dumped a
        # partial buffer to the .txt files (e.g. if this was from a dump where
        # the draw call index count was overridden it may be cut short, or
        # where the .txt files contain only sub-meshes from each draw call and
        # we are loading the .buf file because it contains the entire mesh):
        self.vertex_count = len(self.vertices)

    def append(self, vertex):
        self.vertices.append(vertex)
        self.vertex_count += 1

    def parse_vertex_data(self, f):
        vertex = {}
        for line in map(str.strip, f):
            # print(line)
            if line.startswith('instance-data:'):
                break

            match = self.vb_elem_pattern.match(line)
            if match:
                vertex[match.group('semantic')] = self.parse_vertex_element(match)
            elif line == '' and vertex:
                self.vertices.append(vertex)
                vertex = {}
        if vertex:
            self.vertices.append(vertex)

    def parse_vertex_element(self, match):
        fields = match.group('data').split(',')

        if self.layout[match.group('semantic')].Format.endswith('INT'):
            return tuple(map(int, fields))

        return tuple(map(float, fields))

    def remap_blendindices(self, obj, mapping):
        def lookup_vgmap(x):
            vgname = obj.vertex_groups[x].name
            return mapping.get(vgname, mapping.get(x, x))

        for vertex in self.vertices:
            for semantic in list(vertex):
                if semantic.startswith('BLENDINDICES'):
                    vertex['~' + semantic] = vertex[semantic]
                    vertex[semantic] = tuple(lookup_vgmap(x) for x in vertex[semantic])

    def revert_blendindices_remap(self):
        # Significantly faster than doing a deep copy
        for vertex in self.vertices:
            for semantic in list(vertex):
                if semantic.startswith('BLENDINDICES'):
                    vertex[semantic] = vertex['~' + semantic]
                    del vertex['~' + semantic]

    def disable_blendweights(self):
        for vertex in self.vertices:
            for semantic in list(vertex):
                if semantic.startswith('BLENDINDICES'):
                    vertex[semantic] = (0, 0, 0, 0)

    def write(self, output, operator=None):
        for vertex in self.vertices:
            output.write(self.layout.encode(vertex))

        msg = 'Wrote %i vertices to %s' % (len(self), output.name)
        if operator:
            operator.report({'INFO'}, msg)
        else:
            print(msg)

    def __len__(self):
        return len(self.vertices)

    def merge(self, other):
        if self.layout != other.layout:
            raise Fatal(
                'Vertex buffers have different input layouts - ensure you are only trying to merge the same vertex buffer split across multiple draw calls')
        if self.first != other.first:
            # FIXME: Future 3DMigoto might automatically set first from the
            # index buffer and chop off unreferenced vertices to save space
            raise Fatal(
                'Cannot merge multiple vertex buffers - please check for updates of the 3DMigoto import script, or import each buffer separately')
        self.vertices.extend(other.vertices[self.vertex_count:])
        self.vertex_count = max(self.vertex_count, other.vertex_count)
        assert (len(self.vertices) == self.vertex_count)

    def wipe_semantic_for_testing(self, semantic, val=0):
        print('WARNING: WIPING %s FOR TESTING PURPOSES!!!' % semantic)
        semantic, _, components = semantic.partition('.')
        if components:
            components = [{'x': 0, 'y': 1, 'z': 2, 'w': 3}[c] for c in components]
        else:
            components = range(4)
        for vertex in self.vertices:
            for s in list(vertex):
                if s == semantic:
                    v = list(vertex[semantic])
                    for component in components:
                        if component < len(v):
                            v[component] = val
                    vertex[semantic] = v


class IndexBuffer(object):
    def __init__(self, *args, load_indices=True):
        self.faces = []
        self.first = 0
        self.index_count = 0
        self.format = 'DXGI_FORMAT_UNKNOWN'
        self.offset = 0
        self.topology = 'trianglelist'

        if isinstance(args[0], io.IOBase):
            assert (len(args) == 1)
            self.parse_ib_txt(args[0], load_indices)
        else:
            self.format, = args

        self.encoder, self.decoder = EncoderDecoder(self.format)

    def append(self, face):
        self.faces.append(face)
        self.index_count += len(face)

    def parse_ib_txt(self, f, load_indices):
        for line in map(str.strip, f):
            if line.startswith('byte offset:'):
                self.offset = int(line[13:])
            if line.startswith('first index:'):
                self.first = int(line[13:])
            elif line.startswith('index count:'):
                self.index_count = int(line[13:])
            elif line.startswith('topology:'):
                self.topology = line[10:]
                if line != 'topology: trianglelist':
                    raise Fatal('"%s" is not yet supported' % line)
            elif line.startswith('format:'):
                self.format = line[8:]
            elif line == '':
                if not load_indices:
                    return
                self.parse_index_data(f)
        assert (len(self.faces) * 3 == self.index_count)

    def parse_ib_bin(self, f):
        f.seek(self.offset)
        stride = format_size(self.format)
        # XXX: Should we respect the first index?
        # f.seek(self.first * stride, whence=1)
        self.first = 0

        face = []
        while True:
            index = f.read(stride)
            if not index:
                break
            face.append(*self.decoder(index))
            if len(face) == 3:
                self.faces.append(tuple(face))
                face = []
        assert (len(face) == 0)

        # We intentionally disregard the index count when loading from a
        # binary file, as we assume frame analysis might have only dumped a
        # partial buffer to the .txt files (e.g. if this was from a dump where
        # the draw call index count was overridden it may be cut short, or
        # where the .txt files contain only sub-meshes from each draw call and
        # we are loading the .buf file because it contains the entire mesh):
        self.index_count = len(self.faces) * 3

    def parse_index_data(self, f):
        for line in map(str.strip, f):
            face = tuple(map(int, line.split()))
            assert (len(face) == 3)
            self.faces.append(face)

    def merge(self, other):
        if self.format != other.format:
            raise Fatal(
                'Index buffers have different formats - ensure you are only trying to merge the same index buffer split across multiple draw calls')
        self.first = min(self.first, other.first)
        self.index_count += other.index_count
        self.faces.extend(other.faces)

    def write(self, output, operator=None):
        for face in self.faces:
            output.write(self.encoder(face))

        msg = 'Wrote %i indices to %s' % (len(self), output.name)
        if operator:
            operator.report({'INFO'}, msg)
        else:
            print(msg)

    def __len__(self):
        return len(self.faces) * 3


def load_3dmigoto_mesh_bin(operator, vb_paths, ib_paths, pose_path):
    if len(vb_paths) != 1 or len(ib_paths) > 1:
        raise Fatal('Cannot merge meshes loaded from binary files')

    # Loading from binary files, but still need to use the .txt files as a
    # reference for the format:
    vb_bin_path, vb_txt_path = vb_paths[0]
    ib_bin_path, ib_txt_path = ib_paths[0]

    vb = VertexBuffer(open(vb_txt_path, 'r'), load_vertices=False)
    vb.parse_vb_bin(open(vb_bin_path, 'rb'))

    ib = None
    if ib_paths:
        ib = IndexBuffer(open(ib_txt_path, 'r'), load_indices=False)
        ib.parse_ib_bin(open(ib_bin_path, 'rb'))

    return vb, ib, os.path.basename(vb_bin_path), pose_path


def load_3dmigoto_mesh(operator, paths):
    vb_paths, ib_paths, use_bin, pose_path = zip(*paths)
    pose_path = pose_path[0]

    if use_bin[0]:
        return load_3dmigoto_mesh_bin(operator, vb_paths, ib_paths, pose_path)

    vb = VertexBuffer(open(vb_paths[0], 'r'))
    # Merge additional vertex buffers for meshes split over multiple draw calls:
    for vb_path in vb_paths[1:]:
        tmp = VertexBuffer(open(vb_path, 'r'))
        vb.merge(tmp)

    # For quickly testing how importent any unsupported semantics may be:
    # vb.wipe_semantic_for_testing('POSITION.w', 1.0)
    # vb.wipe_semantic_for_testing('TEXCOORD.w', 0.0)
    # vb.wipe_semantic_for_testing('TEXCOORD5', 0)
    # vb.wipe_semantic_for_testing('BINORMAL')
    # vb.wipe_semantic_for_testing('TANGENT')
    # vb.write(open(os.path.join(os.path.dirname(vb_paths[0]), 'TEST.vb'), 'wb'), operator=operator)

    ib = None
    if ib_paths:
        ib = IndexBuffer(open(ib_paths[0], 'r'))
        # Merge additional vertex buffers for meshes split over multiple draw calls:
        for ib_path in ib_paths[1:]:
            tmp = IndexBuffer(open(ib_path, 'r'))
            ib.merge(tmp)

    return vb, ib, os.path.basename(vb_paths[0]), pose_path


def import_normals_step1(mesh, data):
    # Nico:
    # Blender不支持4D normal，而UE4 Normal的的第四个分量一般情况下是1，可以忽略后导入
    # 而BINORMAL第四个分量不是1就是-1，这时第四个分量代表手性信息，需要根据是否为-1进行向量翻转。
    # 不过暂时没有发现BINORMAL，现代引擎一般都用不上BINORMAL了，所以我们这里不再考虑兼容
    # 这里直接忽略第四个值，无需多余判断，如果真有游戏是4D的Position那到时候再研究
    # if len(data[0]) == 4:
        # if [x[3] for x in data] != [0.0] * len(data):
        #     raise Fatal('Normals are 4D')
    normals = [(x[0], x[1], x[2]) for x in data]

    # To make sure the normals don't get lost by Blender's edit mode,
    # or mesh.update() we need to set custom normals in the loops, not
    # vertices.
    #
    # For testing, to make sure our normals are preserved let's use
    # garbage ones:
    # import random
    # normals = [(random.random() * 2 - 1,random.random() * 2 - 1,random.random() * 2 - 1) for x in normals]
    #
    # Comment from other import scripts:
    # Note: we store 'temp' normals in loops, since validate() may alter final mesh,
    #       we can only set custom lnors *after* calling it.

    mesh.create_normals_split()
    for l in mesh.loops:
        l.normal[:] = normals[l.vertex_index]


def import_normals_step2(mesh):
    # Taken from import_obj/import_fbx
    clnors = array('f', [0.0] * (len(mesh.loops) * 3))
    mesh.loops.foreach_get("normal", clnors)

    # Not sure this is still required with use_auto_smooth, but the other
    # importers do it, and at the very least it shouldn't hurt...
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    mesh.normals_split_custom_set(tuple(zip(*(iter(clnors),) * 3)))
    mesh.use_auto_smooth = True

    # Nico: 这里我看过了，我们并不需要在导入时默认显示锐边，如果确实需要用到的话自己在选项里开启，而不是作为通用步骤执行。
    # This has a double meaning, one of which is to use the custom normals
    # XXX CHECKME: show_edge_sharp moved in 2.80, but I can't actually
    # recall what it does and have a feeling it was unimportant:
    # mesh.show_edge_sharp = True


def import_vertex_groups(mesh, obj, blend_indices, blend_weights):
    assert (len(blend_indices) == len(blend_weights))
    if blend_indices:
        # We will need to make sure we re-export the same blend indices later -
        # that they haven't been renumbered. Not positive whether it is better
        # to use the vertex group index, vertex group name or attach some extra
        # data. Make sure the indices and names match:
        num_vertex_groups = max(itertools.chain(*itertools.chain(*blend_indices.values()))) + 1
        for i in range(num_vertex_groups):
            obj.vertex_groups.new(name=str(i))
        for vertex in mesh.vertices:
            for semantic_index in sorted(blend_indices.keys()):
                for i, w in zip(blend_indices[semantic_index][vertex.index],
                                blend_weights[semantic_index][vertex.index]):
                    if w == 0.0:
                        continue
                    obj.vertex_groups[i].add((vertex.index,), w, 'REPLACE')


def import_uv_layers(mesh, obj, texcoords, flip_texcoord_v):
    for (texcoord, data) in sorted(texcoords.items()):
        # TEXCOORDS can have up to four components, but UVs can only have two
        # dimensions. Not positive of the best way to handle this in general,
        # but for now I'm thinking that splitting the TEXCOORD into two sets of
        # UV coordinates might work:
        dim = len(data[0])
        if dim == 4:
            components_list = ('xy', 'zw')
        elif dim == 2:
            components_list = ('xy',)
        else:
            raise Fatal('Unhandled TEXCOORD dimension: %i' % dim)
        cmap = {'x': 0, 'y': 1, 'z': 2, 'w': 3}

        for components in components_list:
            uv_name = 'TEXCOORD%s.%s' % (texcoord and texcoord or '', components)
            if hasattr(mesh, 'uv_textures'):  # 2.79
                mesh.uv_textures.new(uv_name)
            else:  # 2.80
                mesh.uv_layers.new(name=uv_name)
            blender_uvs = mesh.uv_layers[uv_name]

            # This will assign a texture to the UV layer, which works fine but
            # working out which texture maps to which UV layer is guesswork
            # before the import and the artist may as well just assign it
            # themselves in the UV editor pane when they can see the unwrapped
            # mesh to compare it with the dumped textures:
            #
            # path = textures.get(uv_layer, None)
            # if path is not None:
            #    image = load_image(path)
            #    for i in range(len(mesh.polygons)):
            #        mesh.uv_textures[uv_layer].data[i].image = image

            # Can't find an easy way to flip the display of V in Blender, so
            # add an option to flip it on import & export:
            # TODO why use lambda? it's hard to understand.
            if flip_texcoord_v:
                flip_uv = lambda uv: (uv[0], 1.0 - uv[1])
                # Record that V was flipped so we know to undo it when exporting:
                obj['3DMigoto:' + uv_name] = {'flip_v': True}
            else:
                flip_uv = lambda uv: uv

            # TODO WTF? why merge them in one line, too hard to understand.
            uvs = [[d[cmap[c]] for c in components] for d in data]

            for l in mesh.loops:
                blender_uvs.data[l.index].uv = flip_uv(uvs[l.vertex_index])


# TODO VertexLayer的设计应该被去除
'''
Nico:
在游戏Mod制作中，所有提取的属性和生成的属性都是应该提前规划好的，
而不是在这里做额外的无用步骤来传递一些垃圾属性。
'''
# This loads unknown data from the vertex buffers as vertex layers
def import_vertex_layers(mesh, obj, vertex_layers):
    for (element_name, data) in sorted(vertex_layers.items()):
        dim = len(data[0])
        cmap = {0: 'x', 1: 'y', 2: 'z', 3: 'w'}
        for component in range(dim):

            if dim != 1 or element_name.find('.') == -1:
                layer_name = '%s.%s' % (element_name, cmap[component])
            else:
                layer_name = element_name

            if type(data[0][0]) == int:
                layer = mesh.vertex_layers.new(name=layer_name, type='INT')

                for v in mesh.vertices:
                    val = data[v.index][component]
                    # Blender integer layers are 32bit signed and will throw an
                    # exception if we are assigning an unsigned value that
                    # can't fit in that range. Reinterpret as signed if necessary:
                    if val < 0x80000000:
                        layer.data[v.index].value = val
                    else:
                        layer.data[v.index].value = struct.unpack('i', struct.pack('I', val))[0]
            elif type(data[0][0]) == float:
                layer = mesh.vertex_layers.new(name=layer_name, type='FLOAT')
                for v in mesh.vertices:
                    layer.data[v.index].value = data[v.index][component]
            else:
                raise Fatal('BUG: Bad layer type %s' % type(data[0][0]))


def import_faces_from_ib(mesh, ib):
    mesh.loops.add(len(ib.faces) * 3)
    mesh.polygons.add(len(ib.faces))
    mesh.loops.foreach_set('vertex_index', unpack_list(ib.faces))
    mesh.polygons.foreach_set('loop_start', [x * 3 for x in range(len(ib.faces))])
    mesh.polygons.foreach_set('loop_total', [3] * len(ib.faces))


# Nico: 这玩意基本上用不到吧，没有IB的情况下要怎么做到自动生成顶点索引呢？这样生成出来真的和游戏里替换所需要的格式一样吗？
def import_faces_from_vb(mesh, vb):
    # Only lightly tested
    num_faces = len(vb.vertices) // 3
    mesh.loops.add(num_faces * 3)
    mesh.polygons.add(num_faces)
    mesh.loops.foreach_set('vertex_index', [x for x in range(num_faces * 3)])
    mesh.polygons.foreach_set('loop_start', [x * 3 for x in range(num_faces)])
    mesh.polygons.foreach_set('loop_total', [3] * num_faces)


def import_vertices(mesh, vb):
    mesh.vertices.add(len(vb.vertices))

    seen_offsets = set()
    blend_indices = {}
    blend_weights = {}
    texcoords = {}
    vertex_layers = {}
    use_normals = False

    for elem in vb.layout:
        if elem.InputSlotClass != 'per-vertex':
            continue

        # Discard elements that reuse offsets in the vertex buffer, e.g. COLOR
        # and some TEXCOORDs may be aliases of POSITION:
        if (elem.InputSlot, elem.AlignedByteOffset) in seen_offsets:
            assert (elem.name != 'POSITION')
            continue
        seen_offsets.add((elem.InputSlot, elem.AlignedByteOffset))

        data = tuple(x[elem.name] for x in vb.vertices)
        if elem.name == 'POSITION':
            # Ensure positions are 3-dimensional:
            if len(data[0]) == 4:
                if ([x[3] for x in data] != [1.0] * len(data)):
                    # XXX: Leaving this fatal error in for now, as the meshes
                    # it triggers on in DOA6 (skirts) lie about almost every
                    # semantic and we cannot import them with this version of
                    # the script regardless. Comment it out if you want to try
                    # importing anyway and preserving the W coordinate in a
                    # vertex group. It might also be possible to project this
                    # back into 3D if we assume the coordinates are homogeneous
                    # (i.e. divide XYZ by W), but that might be assuming too
                    # much for a generic script.
                    raise Fatal('Positions are 4D')

                    # Nico: Blender暂时不支持4D索引，加了也没用，直接不行就报错，转人工处理。
                    # Occurs in some meshes in DOA6, such as skirts.
                    # W coordinate must be preserved in these cases.
                    # print('Positions are 4D, storing W coordinate in POSITION.w vertex layer')
                    # vertex_layers['POSITION.w'] = [[x[3]] for x in data]
            positions = [(x[0], x[1], x[2]) for x in data]
            mesh.vertices.foreach_set('co', unpack_list(positions))
        elif elem.name.startswith('COLOR'):
            if len(data[0]) <= 3 or vertex_color_layer_channels == 4:
                # Either a monochrome/RGB layer, or Blender 2.80 which uses 4
                # channel layers
                mesh.vertex_colors.new(name=elem.name)
                color_layer = mesh.vertex_colors[elem.name].data
                c = vertex_color_layer_channels
                for l in mesh.loops:
                    color_layer[l.index].color = list(data[l.vertex_index]) + [0] * (c - len(data[l.vertex_index]))
            else:
                mesh.vertex_colors.new(name=elem.name + '.RGB')
                mesh.vertex_colors.new(name=elem.name + '.A')
                color_layer = mesh.vertex_colors[elem.name + '.RGB'].data
                alpha_layer = mesh.vertex_colors[elem.name + '.A'].data
                for l in mesh.loops:
                    color_layer[l.index].color = data[l.vertex_index][:3]
                    alpha_layer[l.index].color = [data[l.vertex_index][3], 0, 0]
        elif elem.name == 'NORMAL':
            use_normals = True
            import_normals_step1(mesh, data)
        elif elem.name in ('TANGENT', 'BINORMAL'):
            #    # XXX: loops.tangent is read only. Not positive how to handle
            #    # this, or if we should just calculate it when re-exporting.
            #    for l in mesh.loops:
            #        assert(data[l.vertex_index][3] in (1.0, -1.0))
            #        l.tangent[:] = data[l.vertex_index][0:3]
            print('NOTICE: Skipping import of %s in favour of recalculating on export' % elem.name)
        elif elem.name.startswith('BLENDINDICES'):
            blend_indices[elem.SemanticIndex] = data
        elif elem.name.startswith('BLENDWEIGHT'):
            blend_weights[elem.SemanticIndex] = data
        elif elem.name.startswith('TEXCOORD') and elem.is_float():
            texcoords[elem.SemanticIndex] = data
        else:
            print('NOTICE: Storing unhandled semantic %s %s as vertex layer' % (elem.name, elem.Format))
            vertex_layers[elem.name] = data

    return (blend_indices, blend_weights, texcoords, vertex_layers, use_normals)


def import_3dmigoto(operator, context, paths, merge_meshes=False, **kwargs):
    if merge_meshes:
        return import_3dmigoto_vb_ib(operator, context, paths, **kwargs)
    else:
        obj = []
        for p in paths:
            try:
                obj.append(import_3dmigoto_vb_ib(operator, context, [p], **kwargs))
            except Fatal as e:
                operator.report({'ERROR'}, str(e) + ': ' + str(p[:2]))
        # FIXME: Group objects together  (Nico:这里他的意思应该是导入后自动放入一个集合里)
        return obj


def import_3dmigoto_vb_ib(operator, context, paths, flip_texcoord_v=True, axis_forward='-Z', axis_up='Y',
                          pose_cb_off=[0, 0], pose_cb_step=1):
    vb, ib, name, pose_path = load_3dmigoto_mesh(operator, paths)

    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(mesh.name, mesh)

    global_matrix = axis_conversion(from_forward=axis_forward, from_up=axis_up).to_4x4()
    obj.matrix_world = global_matrix

    # Attach the vertex buffer layout to the object for later exporting. Can't
    # seem to retrieve this if attached to the mesh - to_mesh() doesn't copy it:
    obj['3DMigoto:VBLayout'] = vb.layout.serialise()
    obj['3DMigoto:VBStride'] = vb.layout.stride  # FIXME: Strides of multiple vertex buffers
    obj['3DMigoto:FirstVertex'] = vb.first

    if ib is not None:
        import_faces_from_ib(mesh, ib)
        # Attach the index buffer layout to the object for later exporting.
        obj['3DMigoto:IBFormat'] = ib.format
        obj['3DMigoto:FirstIndex'] = ib.first
    else:
        import_faces_from_vb(mesh, vb)

    (blend_indices, blend_weights, texcoords, vertex_layers, use_normals) = import_vertices(mesh, vb)

    import_uv_layers(mesh, obj, texcoords, flip_texcoord_v)

    import_vertex_layers(mesh, obj, vertex_layers)

    import_vertex_groups(mesh, obj, blend_indices, blend_weights)

    # Validate closes the loops so they don't disappear after edit mode and probably other important things:
    mesh.validate(verbose=False, clean_customdata=False)  # *Very* important to not remove lnors here!
    # Not actually sure update is necessary. It seems to update the vertex normals, not sure what else:
    mesh.update()

    # Must be done after validate step:
    if use_normals:
        import_normals_step2(mesh)
    else:
        mesh.calc_normals()

    link_object_to_scene(context, obj)
    obj.select_set(True)
    set_active_object(context, obj)

    # if pose_path is not None:
    #     import_pose(operator, context, pose_path, limit_bones_to_vertex_groups=True,
    #                 axis_forward=axis_forward, axis_up=axis_up,
    #                 pose_cb_off=pose_cb_off, pose_cb_step=pose_cb_step)
    #     set_active_object(context, obj)
    operator.report({'INFO'}, "Import Into 3Dmigoto")

    return obj


# from export_obj:
def mesh_triangulate(me):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()



def blender_vertex_to_3dmigoto_vertex(mesh, obj, blender_loop_vertex, layout, texcoords):

    # 根据循环顶点中的顶点索引来从总的顶点中获取对应的顶点
    blender_vertex = mesh.vertices[blender_loop_vertex.vertex_index]
    vertex = {}
    seen_offsets = set()

    # TODO: Warn if vertex is in too many vertex groups for this layout,
    # ignoring groups with weight=0.0
    vertex_groups = sorted(blender_vertex.groups, key=lambda x: x.weight, reverse=True)

    for elem in layout:
        # 只处理per-vertex的
        if elem.InputSlotClass != 'per-vertex':
            continue

        # 用于跳过在同一个顶点上重复元素的处理，这个代码真的会被执行到吗？看起来永远不会触发。
        if (elem.InputSlot, elem.AlignedByteOffset) in seen_offsets:
            continue
        seen_offsets.add((elem.InputSlot, elem.AlignedByteOffset))

        if elem.name == 'POSITION':
            vertex[elem.name] = elem.pad(list(blender_vertex.undeformed_co), 1.0)
        elif elem.name.startswith('COLOR'):
            if elem.name in mesh.vertex_colors:
                vertex[elem.name] = elem.clip(list(mesh.vertex_colors[elem.name].data[blender_loop_vertex.index].color))
            else:
                vertex[elem.name] = list(mesh.vertex_colors[elem.name + '.RGB'].data[blender_loop_vertex.index].color)[
                                    :3] + \
                                    [mesh.vertex_colors[elem.name + '.A'].data[blender_loop_vertex.index].color[0]]
        elif elem.name == 'NORMAL':
            vertex[elem.name] = elem.pad(list(blender_loop_vertex.normal), 0.0)
        elif elem.name.startswith('TANGENT'):
            # DOAXVV has +1/-1 in the 4th component. Not positive what this is,
            # but guessing maybe the bitangent sign? Not even sure it is used...
            # FIXME: Other games
            # vertex[elem.name] = elem.pad(list(blender_loop_vertex.tangent), blender_loop_vertex.bitangent_sign)
            vertex[elem.name] = elem.pad(list(blender_loop_vertex.tangent), blender_loop_vertex.bitangent_sign)
        elif elem.name.startswith('BLENDINDICES'):
            i = elem.SemanticIndex * 4
            vertex[elem.name] = elem.pad([x.group for x in vertex_groups[i:i + 4]], 0)
        elif elem.name.startswith('BLENDWEIGHT'):
            # TODO: Warn if vertex is in too many vertex groups for this layout
            i = elem.SemanticIndex * 4
            vertex[elem.name] = elem.pad([x.weight for x in vertex_groups[i:i + 4]], 0.0)
        elif elem.name.startswith('TEXCOORD') and elem.is_float():
            # FIXME: Handle texcoords of other dimensions
            uvs = []
            for uv_name in ('%s.xy' % elem.name, '%s.zw' % elem.name):
                if uv_name in texcoords:
                    uvs += list(texcoords[uv_name][blender_loop_vertex.index])
            vertex[elem.name] = uvs

        # Nico: 不需要考虑BINORMAL，现代游戏的渲染基本上不会使用BINORMAL这种过时的渲染方案
        # elif elem.name.startswith('BINORMAL'):
            # Some DOA6 meshes (skirts) use BINORMAL, but I'm not certain it is
            # actually the binormal. These meshes are weird though, since they
            # use 4 dimensional positions and normals, so they aren't something
            # we can really deal with at all. Therefore, the below is untested,
            # FIXME: So find a mesh where this is actually the binormal,
            # uncomment the below code and test.
            # normal = blender_loop_vertex.normal
            # tangent = blender_loop_vertex.tangent
            # binormal = numpy.cross(normal, tangent)
            # XXX: Does the binormal need to be normalised to a unit vector?
            # binormal = binormal / numpy.linalg.norm(binormal)
            # vertex[elem.name] = elem.pad(list(binormal), 0.0)
            # pass

        else:
            # Unhandled semantics are saved in vertex layers
            data = []
            for component in 'xyzw':
                layer_name = '%s.%s' % (elem.name, component)
                if layer_name in mesh.vertex_layers_int:
                    data.append(mesh.vertex_layers_int[layer_name].data[blender_loop_vertex.vertex_index].value)
                elif layer_name in mesh.vertex_layers_float:
                    data.append(mesh.vertex_layers_float[layer_name].data[blender_loop_vertex.vertex_index].value)
            if data:
                # print('Retrieved unhandled semantic %s %s from vertex layer' % (elem.name, elem.Format), data)
                vertex[elem.name] = data

        if elem.name not in vertex:
            print('NOTICE: Unhandled vertex element: %s' % elem.name)
        # else:
        #    print('%s: %s' % (elem.name, repr(vertex[elem.name])))

    return vertex


def write_fmt_file(f, vb, ib):
    f.write('stride: %i\n' % vb.layout.stride)
    f.write('topology: %s\n' % vb.topology)
    if ib is not None:
        f.write('format: %s\n' % ib.format)
    f.write(vb.layout.to_string())


def export_3dmigoto(operator, context, vb_path, ib_path, fmt_path):
    # 获取当前场景中的obj对象
    obj = context.object

    # 为空时不导出
    if obj is None:
        raise Fatal('No object selected')

    stride = obj['3DMigoto:VBStride']
    layout = InputLayout(obj['3DMigoto:VBLayout'], stride=stride)

    # 获取Mesh
    if hasattr(context, "evaluated_depsgraph_get"):  # 2.80
        mesh = obj.evaluated_get(context.evaluated_depsgraph_get()).to_mesh()
    else:  # 2.79
        mesh = obj.to_mesh(context.scene, True, 'PREVIEW', calc_tessface=False)


    # 使用bmesh复制出一个新mesh并三角化
    mesh_triangulate(mesh)

    # 获取所有索引
    indices = [l.vertex_index for l in mesh.loops]
    # 获取所有顶点
    faces = [indices[i:i + 3] for i in range(0, len(indices), 3)]

    try:
        if obj['3DMigoto:IBFormat'] == "DXGI_FORMAT_R16_UINT":
            ib_format = "DXGI_FORMAT_R32_UINT"
        else:
            ib_format = obj['3DMigoto:IBFormat']
    except KeyError:
        ib = None
        raise Fatal('FIXME: Add capability to export without an index buffer')
    else:
        ib = IndexBuffer(ib_format)

    # Calculates tangents and makes loop normals valid (still with our
    # custom normal data from import time):
    # 这一步如果存在TANGENT属性则会导致顶点数量增加
    mesh.calc_tangents()

    # 拼凑texcoord层级，有几个UVMap就拼出几个来
    texcoord_layers = {}
    for uv_layer in mesh.uv_layers:
        texcoords = {}

        try:
            flip_texcoord_v = obj['3DMigoto:' + uv_layer.name]['flip_v']
            if flip_texcoord_v:
                flip_uv = lambda uv: (uv[0], 1.0 - uv[1])
            else:
                flip_uv = lambda uv: uv
        except KeyError:
            flip_uv = lambda uv: uv

        for l in mesh.loops:
            uv = flip_uv(uv_layer.data[l.index].uv)
            texcoords[l.index] = uv
        texcoord_layers[uv_layer.name] = texcoords

    # Blender's vertices have unique positions, but may have multiple
    # normals, tangents, UV coordinates, etc - these are stored in the
    # loops. To export back to DX we need these combined together such that
    # a vertex is a unique set of all attributes, but we don't want to
    # completely blow this out - we still want to reuse identical vertices
    # via the index buffer. There might be a convenience function in
    # Blender to do this, but it's easy enough to do this ourselves
    indexed_vertices = collections.OrderedDict()

    unique_position_vertices = {}
    # 先输出看一下这里的顶点数量为多少，经过测试确实是原本的顶点数量
    operator.report({'INFO'}, "mesh.vertices: " + str(mesh.vertices))

    '''
    Nico:
    顶点转换为3dmigoto类型的顶点再经过hashable后，如果存在TANGENT则会导致数量变多，不存在则不会导致数量变多。
    '''
    calcNumber = 0
    for poly in mesh.polygons:
        face = []
        for blender_lvertex in mesh.loops[poly.loop_start:poly.loop_start + poly.loop_total]:
            # Nico: 初始的Vertex即使是经过TANGENT计算，数量也是和原来一样的
            # 但是这里使用了blender_lvertex导致了生成的HashableVertex不一样，因为其它都是固定的只有这个blender_lvertex会改变
            # 也许应该参考Blender官方文档来进一步学习blender_lvertex具体作用以及继续研究blender_vertex_to_3dmigoto_vertex函数
            # 搞清楚为什么这个会导致Hash计算不重复，需要注意的是如果不计算TANGENT或者没有TANGENT属性时不会额外生成顶点
            vertex = blender_vertex_to_3dmigoto_vertex(mesh, obj, blender_lvertex, layout, texcoord_layers)
            # 首先将当前顶点计算为Hash后的顶点然后如果该计算后的Hash顶点不存在，则插入到indexed_vertices里
            # 随后将该顶点添加到face[]里，索引为该顶点在字典里的索引
            if tuple(vertex["POSITION"]) in unique_position_vertices:
                tangent_var = unique_position_vertices[tuple(vertex["POSITION"])]
                vertex["TANGENT"] = tangent_var
            else:
                tangent_var = vertex["TANGENT"]
                unique_position_vertices[tuple(vertex["POSITION"])] = tangent_var
                vertex["TANGENT"] = tangent_var

            # TODO 这里我们把获取到的vertex的切线加到一个vertex:切线值的字典中
            #   如果vertex的顶点在字典中出现了，则返回字典中对应列表和当前值的平均值，否则不进行更新
            #   这样就能得到每个Position对应的平均切线，在切线值相同的情况下，就不会产生额外的多余顶点了。

            indexed_vertex = indexed_vertices.setdefault(HashableVertex(vertex), len(indexed_vertices))
            # indexed_vertex = indexed_vertices[HashableVertex(vertex)] = len(indexed_vertices)
            face.append(indexed_vertex)
            calcNumber = calcNumber + 1

        if ib is not None:
            ib.append(face)

    operator.report({'INFO'}, "IB Append Number: " + str(calcNumber))
    operator.report({'INFO'}, "indexed_vertices: " + str(len(indexed_vertices)))
    operator.report({'INFO'}, "IB number : " + str(len(ib)))

    vb = VertexBuffer(layout=layout)
    vbAppendNumber = 0
    # for vertex in indexed_vertices:
    for vertex in indexed_vertices:
        vb.append(vertex)
        vbAppendNumber = vbAppendNumber + 1
    operator.report({'INFO'}, "Export Vertex Number: " + str(vbAppendNumber))

    vgmaps = {k[15:]: keys_to_ints(v) for k, v in obj.items() if k.startswith('3DMigoto:VGMap:')}

    if '' not in vgmaps:
        vb.write(open(vb_path, 'wb'), operator=operator)

    base, ext = os.path.splitext(vb_path)
    for (suffix, vgmap) in vgmaps.items():
        path = vb_path
        if suffix:
            path = '%s-%s%s' % (base, suffix, ext)
        vgmap_path = os.path.splitext(path)[0] + '.vgmap'
        print('Exporting %s...' % path)
        vb.remap_blendindices(obj, vgmap)
        vb.write(open(path, 'wb'), operator=operator)
        vb.revert_blendindices_remap()
        sorted_vgmap = collections.OrderedDict(sorted(vgmap.items(), key=lambda x: x[1]))
        json.dump(sorted_vgmap, open(vgmap_path, 'w'), indent=2)

    if ib is not None:
        ib.write(open(ib_path, 'wb'), operator=operator)

    # Write format reference file
    write_fmt_file(open(fmt_path, 'w'), vb, ib)


@orientation_helper(axis_forward='-Z', axis_up='Y')
class Import3DMigotoFrameAnalysis(bpy.types.Operator, ImportHelper, IOOBJOrientationHelper):
    """Import a mesh dumped with 3DMigoto's frame analysis"""
    bl_idname = "import_mesh.migoto_frame_analysis_mmt"
    bl_label = "Import 3DMigoto Frame Analysis Dump  (MMT)"
    bl_options = {'PRESET', 'UNDO'}

    filename_ext = '.txt'

    directory: StringProperty(
        name="Directory",
        subtype='DIR_PATH',
        default= "",
    )

    filter_glob: StringProperty(
        default='*.txt',
        options={'HIDDEN'},
    )

    files: CollectionProperty(
        name="File Path",
        type=bpy.types.OperatorFileListElement,
    )

    flip_texcoord_v: BoolProperty(
        name="Flip TEXCOORD V",
        description="Flip TEXCOORD V asix during importing",
        default=True,
    )

    load_related: BoolProperty(
        name="Auto-load related meshes",
        description="Automatically load related meshes found in the frame analysis dump",
        default=True,
    )

    pose_cb_step: bpy.props.IntProperty(
        name="Vertex group step",
        description='If used vertex groups are 0,1,2,3,etc specify 1. If they are 0,3,6,9,12,etc specify 3',
        default=1,
        min=1,
    )

    def get_vb_ib_paths(self):
        buffer_pattern = re.compile(r'''-(?:ib|vb[0-9]+)(?P<hash>=[0-9a-f]+)?(?=[^0-9a-f=])''')

        dirname = os.path.dirname(self.filepath)
        ret = set()

        files = []
        if self.load_related:
            for filename in self.files:
                match = buffer_pattern.search(filename.name)
                if match is None or not match.group('hash'):
                    continue
                paths = glob(os.path.join(dirname, '*%s*.txt' % filename.name[match.start():match.end()]))
                files.extend([os.path.basename(x) for x in paths])
        if not files:
            files = [x.name for x in self.files]

        for filename in files:
            match = buffer_pattern.search(filename)
            if match is None:
                raise Fatal(
                    'Unable to find corresponding buffers from filename - ensure you are loading a dump from a timestamped Frame Analysis directory (not a deduped directory)')

            use_bin = False
            if not match.group('hash') and not use_bin:
                self.report({'INFO'},
                            'Filename did not contain hash - if Frame Analysis dumped a custom resource the .txt file may be incomplete, Using .buf files instead')
                use_bin = True  # FIXME: Ask

            ib_pattern = filename[:match.start()] + '-ib*' + filename[match.end():]
            vb_pattern = filename[:match.start()] + '-vb*' + filename[match.end():]
            ib_paths = glob(os.path.join(dirname, ib_pattern))
            vb_paths = glob(os.path.join(dirname, vb_pattern))

            if vb_paths and use_bin:
                vb_bin_paths = [os.path.splitext(x)[0] + '.buf' for x in vb_paths]
                ib_bin_paths = [os.path.splitext(x)[0] + '.buf' for x in ib_paths]
                if all([os.path.exists(x) for x in itertools.chain(vb_bin_paths, ib_bin_paths)]):
                    # When loading the binary files, we still need to process
                    # the .txt files as well, as they indicate the format:
                    ib_paths = list(zip(ib_bin_paths, ib_paths))
                    vb_paths = list(zip(vb_bin_paths, vb_paths))
                else:
                    self.report({'WARNING'}, 'Corresponding .buf files not found - using .txt files')
                    use_bin = False

            pose_path = None
            # if self.pose_cb:
            #     pose_pattern = filename[:match.start()] + '*-' + self.pose_cb + '=*.txt'
            #     try:
            #         pose_path = glob(os.path.join(dirname, pose_pattern))[0]
            #     except IndexError:
            #         pass

            if len(ib_paths) != 1 or len(vb_paths) != 1:
                raise Fatal(
                    'Only draw calls using a single vertex buffer and a single index buffer are supported for now')

            ret.add((vb_paths[0], ib_paths[0], use_bin, pose_path))
        return ret

    def execute(self, context):
        # if self.load_buf:
        #     # Is there a way to have the mutual exclusivity reflected in
        #     # the UI? Grey out options or use radio buttons or whatever?
        #     if self.merge_meshes or self.load_related:
        #         self.report({'INFO'}, 'Loading .buf files selected: Disabled incompatible options')
        #     self.merge_meshes = False
        #     self.load_related = False

        try:
            keywords = self.as_keywords(
                ignore=('filepath', 'files', 'filter_glob', 'load_related', 'load_buf', 'pose_cb','directory'))
            paths = self.get_vb_ib_paths()
            self.report({'INFO'}, "test：" + str(paths))
            import_3dmigoto(self, context, paths, **keywords)
        except Fatal as e:
            self.report({'ERROR'}, str(e))
        return {'FINISHED'}


def import_3dmigoto_raw_buffers(operator, context, vb_fmt_path, ib_fmt_path, vb_path=None, ib_path=None,
                                vgmap_path=None, **kwargs):
    paths = (((vb_path, vb_fmt_path), (ib_path, ib_fmt_path), True, None),)
    obj = import_3dmigoto(operator, context, paths, merge_meshes=False, **kwargs)
    if obj and vgmap_path:
        apply_vgmap(operator, context, targets=obj, filepath=vgmap_path, rename=True, cleanup=True)


@orientation_helper(axis_forward='-Z', axis_up='Y')
class Import3DMigotoRaw(bpy.types.Operator, ImportHelper, IOOBJOrientationHelper):
    """Import raw 3DMigoto vertex and index buffers"""
    bl_idname = "import_mesh.migoto_raw_buffers_mmt"
    bl_label = "Import 3DMigoto Raw Buffers (MMT)"
    # bl_options = {'PRESET', 'UNDO'}
    bl_options = {'UNDO'}

    filename_ext = '.vb;.ib'
    filter_glob: StringProperty(
        default='*.vb;*.ib',
        options={'HIDDEN'},
    )

    files: CollectionProperty(
        name="File Path",
        type=bpy.types.OperatorFileListElement,
    )

    flip_texcoord_v: BoolProperty(
        name="Flip TEXCOORD V",
        description="Flip TEXCOORD V asix during importing",
        default=True,
    )

    def get_vb_ib_paths(self, filename):
        vb_bin_path = os.path.splitext(filename)[0] + '.vb'
        ib_bin_path = os.path.splitext(filename)[0] + '.ib'
        fmt_path = os.path.splitext(filename)[0] + '.fmt'
        vgmap_path = os.path.splitext(filename)[0] + '.vgmap'
        if not os.path.exists(vb_bin_path):
            raise Fatal('Unable to find matching .vb file for %s' % filename)
        if not os.path.exists(ib_bin_path):
            raise Fatal('Unable to find matching .ib file for %s' % filename)
        if not os.path.exists(fmt_path):
            fmt_path = None
        if not os.path.exists(vgmap_path):
            vgmap_path = None
        return (vb_bin_path, ib_bin_path, fmt_path, vgmap_path)

    def execute(self, context):
        # I'm not sure how to find the Import3DMigotoReferenceInputFormat
        # instance that Blender instantiated to pass the values from one
        # import dialog to another, but since everything is modal we can
        # just use globals:
        global migoto_raw_import_options
        migoto_raw_import_options = self.as_keywords(ignore=('filepath', 'files', 'filter_glob'))

        done = set()
        dirname = os.path.dirname(self.filepath)
        for filename in self.files:
            try:
                (vb_path, ib_path, fmt_path, vgmap_path) = self.get_vb_ib_paths(os.path.join(dirname, filename.name))
                if os.path.normcase(vb_path) in done:
                    continue
                done.add(os.path.normcase(vb_path))

                if fmt_path is not None:
                    import_3dmigoto_raw_buffers(self, context, fmt_path, fmt_path, vb_path=vb_path, ib_path=ib_path,
                                                vgmap_path=vgmap_path, **migoto_raw_import_options)
                else:
                    migoto_raw_import_options['vb_path'] = vb_path
                    migoto_raw_import_options['ib_path'] = ib_path
                    bpy.ops.import_mesh.migoto_input_format('INVOKE_DEFAULT')
            except Fatal as e:
                self.report({'ERROR'}, str(e))
        return {'FINISHED'}


# used to import .fmt file.
class Import3DMigotoReferenceInputFormat(bpy.types.Operator, ImportHelper):
    bl_idname = "import_mesh.migoto_input_format"
    bl_label = "Select a .txt file with matching format"
    bl_options = {'UNDO', 'INTERNAL'}

    filename_ext = '.txt;.fmt'
    filter_glob: StringProperty(
        default='*.txt;*.fmt',
        options={'HIDDEN'},
    )

    def get_vb_ib_paths(self):
        if os.path.splitext(self.filepath)[1].lower() == '.fmt':
            return (self.filepath, self.filepath)

        buffer_pattern = re.compile(r'''-(?:ib|vb[0-9]+)(?P<hash>=[0-9a-f]+)?(?=[^0-9a-f=])''')

        dirname = os.path.dirname(self.filepath)
        filename = os.path.basename(self.filepath)

        match = buffer_pattern.search(filename)
        if match is None:
            raise Fatal('Reference .txt filename does not look like a 3DMigoto timestamped Frame Analysis Dump')
        ib_pattern = filename[:match.start()] + '-ib*' + filename[match.end():]
        vb_pattern = filename[:match.start()] + '-vb*' + filename[match.end():]
        ib_paths = glob(os.path.join(dirname, ib_pattern))
        vb_paths = glob(os.path.join(dirname, vb_pattern))
        if len(ib_paths) < 1 or len(vb_paths) < 1:
            raise Fatal('Unable to locate reference files for both vertex buffer and index buffer format descriptions')
        return (vb_paths[0], ib_paths[0])

    def execute(self, context):
        global migoto_raw_import_options

        try:
            vb_fmt_path, ib_fmt_path = self.get_vb_ib_paths()
            import_3dmigoto_raw_buffers(self, context, vb_fmt_path, ib_fmt_path, **migoto_raw_import_options)
        except Fatal as e:
            self.report({'ERROR'}, str(e))
        return {'FINISHED'}


class Export3DMigoto(bpy.types.Operator, ExportHelper):
    """Export a mesh for re-injection into a game with 3DMigoto"""
    bl_idname = "export_mesh.migoto_mmt"
    bl_label = "Export 3DMigoto Vertex & Index Buffers (MMT)"

    # file extension
    filename_ext = '.vb'

    # file type filter
    filter_glob: StringProperty(
        default='*.vb',
        options={'HIDDEN'},
    )

    # 默认选择文件路径
    filepath: bpy.props.StringProperty(
        name="File Path",
        description="Filepath used for exporting",
        subtype='FILE_PATH',
        default="//output.vb",  # 设置默认路径
    )

    # where you do export logic
    def execute(self, context):
        try:
            vb_path = self.filepath
            ib_path = os.path.splitext(vb_path)[0] + '.ib'
            fmt_path = os.path.splitext(vb_path)[0] + '.fmt'

            # FIXME: ExportHelper will check for overwriting vb_path, but not ib_path

            export_3dmigoto(self, context, vb_path, ib_path, fmt_path)
        except Fatal as e:
            self.report({'ERROR'}, str(e))
        return {'FINISHED'}


def apply_vgmap(operator, context, targets=None, filepath='', commit=False, reverse=False, suffix='', rename=False,
                cleanup=False):
    if not targets:
        targets = context.selected_objects

    if not targets:
        raise Fatal('No object selected')

    vgmap = json.load(open(filepath, 'r'))

    if reverse:
        vgmap = {int(v): int(k) for k, v in vgmap.items()}
    else:
        vgmap = {k: int(v) for k, v in vgmap.items()}

    for obj in targets:
        if commit:
            raise Fatal('commit not yet implemented')

        prop_name = '3DMigoto:VGMap:' + suffix
        obj[prop_name] = keys_to_strings(vgmap)

        if rename:
            for k, v in vgmap.items():
                if str(k) in obj.vertex_groups.keys():
                    continue
                if str(v) in obj.vertex_groups.keys():
                    obj.vertex_groups[str(v)].name = k
                else:
                    obj.vertex_groups.new(name=str(k))
        if cleanup:
            for vg in obj.vertex_groups:
                if vg.name not in vgmap:
                    obj.vertex_groups.remove(vg)

        if '3DMigoto:VBLayout' not in obj:
            operator.report({'WARNING'},
                            '%s is not a 3DMigoto mesh. Vertex Group Map custom property applied anyway' % obj.name)
        else:
            operator.report({'INFO'}, 'Applied vgmap to %s' % obj.name)


# -----------------------------Menu Function define-------------------------------------
def menu_func_import_fa(self, context):
    self.layout.operator(Import3DMigotoFrameAnalysis.bl_idname, text="3DMigoto FrameAnalysis dump (ib.txt + vb0.txt) (MMT)")


def menu_func_import_raw(self, context):
    self.layout.operator(Import3DMigotoRaw.bl_idname, text="3DMigoto raw buffers (.vb + .ib) (MMT)")


def menu_func_export(self, context):
    self.layout.operator(Export3DMigoto.bl_idname, text="3DMigoto raw buffers (.vb + .ib) (MMT)")
