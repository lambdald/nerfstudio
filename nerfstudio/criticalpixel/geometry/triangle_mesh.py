import open3d as o3d
import numpy as np
import copy
import torch
import trimesh

from .scene_box import SceneBox


class TriangleMesh:
    """
    Triangular grid, the data can be numpy or torch.
    In the training or rendering phase, the data is of torch type. In data processing stages, the data is of numpy type.
    For convenience, a variety of mesh conversions have been added, such as open3d.geometry.TriangleMesh, trimesh.Trimesh, etc.
    """

    def __init__(self, vertices=None, triangles=None, triangle_normals=None, vertex_normals=None) -> None:
        self.vertices = vertices
        self.triangles = triangles
        self.triangle_normals = triangle_normals
        self.vertex_normals = vertex_normals
        self.check_type()
        self.check_device()

    @staticmethod
    def from_o3d(o3d_trimesh: o3d.geometry.TriangleMesh) -> 'TriangleMesh':

        if not o3d_trimesh.has_vertex_normals():
            o3d_trimesh = o3d_trimesh.compute_vertex_normals()

        if not o3d_trimesh.has_triangle_normals():
            o3d_trimesh = o3d_trimesh.compute_triangle_normals()

        trimesh = TriangleMesh(vertices=np.asarray(o3d_trimesh.vertices).astype(np.float32),
                               triangles=np.asarray(o3d_trimesh.triangles).astype(np.int64),
                               triangle_normals=np.asarray(o3d_trimesh.triangle_normals).astype(np.float32),
                               vertex_normals=np.asarray(o3d_trimesh.vertex_normals).astype(np.float32))
        trimesh.check_type()
        return trimesh

    def has_vertex_normals(self):
        return self.vertex_normals is not None

    def has_triangle_normals(self):
        return self.triangle_normals is not None

    def torch(self, device='cpu'):
        if self.type == 'torch':
            return self.to(device)

        vertices = torch.from_numpy(self.vertices.astype(np.float32))
        triangles = torch.from_numpy(self.triangles.astype(np.long))
        triangle_normals = torch.from_numpy(self.triangle_normals.astype(np.float32)) if self.has_triangle_normals() else None
        vertex_normals = torch.from_numpy(self.vertex_normals.astype(np.float32)) if self.has_vertex_normals() else None

        mesh = TriangleMesh(vertices=vertices,
                            triangles=triangles,
                            triangle_normals=triangle_normals,
                            vertex_normals=vertex_normals).to(device)
        mesh.check_type()
        return mesh

    def to(self, device):
        assert self.type == 'torch'
        vertices = self.vertices.to(device)
        triangles = self.triangles.to(device)
        triangle_normals = self.triangle_normals.to(device) if self.has_triangle_normals() else None
        vertex_normals = self.vertex_normals.to(device) if self.has_vertex_normals() else None
        mesh = TriangleMesh(vertices=vertices,
                            triangles=triangles,
                            triangle_normals=triangle_normals,
                            vertex_normals=vertex_normals)
        mesh.check_type()
        return mesh

    def numpy(self) -> 'TriangleMesh':
        if self.type == 'numpy':
            return copy.deepcopy(self)

        assert self.type == 'torch'
        vertices = self.vertices.cpu().numpy()
        triangles = self.triangles.cpu().numpy()
        triangle_normals = self.triangle_normals.cpu().numpy() if self.has_triangle_normals() else None
        vertex_normals = self.vertex_normals.cpu().numpy() if self.has_vertex_normals() else None

        mesh = TriangleMesh(vertices=vertices,
                            triangles=triangles,
                            triangle_normals=triangle_normals,
                            vertex_normals=vertex_normals)
        mesh.check_type()
        return mesh

    def o3d_mesh(self) -> o3d.geometry.TriangleMesh:
        np_mesh = self.numpy()
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(np_mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(np_mesh.triangles)
        if np_mesh.has_vertex_normals():
            o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(np_mesh.vertex_normals)
        if np_mesh.has_triangle_normals():
            o3d_mesh.triangle_normals = o3d.utility.Vector3dVector(np_mesh.triangle_normals)
        return o3d_mesh

    def trimesh(self) -> trimesh.Trimesh:

        mesh_np = self.numpy()

        mesh_tm = trimesh.Trimesh(vertices=mesh_np.vertices,
                                  faces=mesh_np.triangles,
                                  face_normals=mesh_np.triangle_normals,
                                  vertex_normals=mesh_np.vertex_normals)
        return mesh_tm

    def transfrom(self, transform):
        self.vertices = self.vertices @ transform[:3, :3].T + transform[:3, 3:4].T

    @property
    def device(self):
        assert self.type == 'torch'
        self.check_device()
        return self.vertices.device

    @property
    def type(self):
        if isinstance(self.vertices, torch.Tensor):
            return 'torch'
        elif isinstance(self.vertices, np.ndarray):
            return 'numpy'
        else:
            raise NotImplementedError(f'unknown dtype of elements:{type(self.vertices)}')

    def check_type(self):
        if self.type == 'torch':
            data_lib = torch
        elif self.type == 'numpy':
            data_lib = np
        assert self.vertices.dtype == data_lib.float32
        assert self.triangles.dtype == data_lib.int64
        if self.has_triangle_normals():
            assert self.triangle_normals.dtype == data_lib.float32
        if self.has_vertex_normals():
            assert self.vertex_normals.dtype == data_lib.float32

    def check_device(self):
        if self.type == 'torch':
            assert self.vertices.device == self.triangles.device
            if self.vertex_normals is not None:
                assert self.vertices.device == self.vertex_normals.device
            if self.triangle_normals is not None:
                assert self.vertices.device == self.triangle_normals.device

    @staticmethod
    def load(mesh_filepath: str, out_o3d=False):
        o3d_mesh = o3d.io.read_triangle_mesh(mesh_filepath)
        if out_o3d:
            return o3d_mesh
        else:
            return TriangleMesh.from_o3d(o3d_mesh)

    def save(self, filepath):
        mesh_o3d = self.o3d_mesh()
        o3d.io.write_triangle_mesh(filepath, mesh_o3d, compressed=True, write_ascii=False)

    def __str__(self) -> str:
        s = f'TriangleMesh with {len(self.vertices)} points and {len(self.triangles)} triangles of '
        if self.type == 'torch':
            s += f'torch({self.device})'
        elif self.type == 'numpy':
            s += f'numpy'
        return s

    def get_mesh_within_box(self, bbox: SceneBox):

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        mesh_torch = self.torch(device=device)
        bbox = bbox.to(device)

        is_vertices_in_bbox = bbox.get_point_indices_within_box(mesh_torch.vertices)
        is_triangles_vertices_in_bbox = is_vertices_in_bbox[mesh_torch.triangles.view(-1)].view(-1, 3)
        is_triangles_in_bbox = torch.any(is_triangles_vertices_in_bbox, dim=-1)
        triangles_in_bbox = mesh_torch.triangles[is_triangles_in_bbox].to(torch.int32)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(mesh_torch.vertices.cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(triangles_in_bbox.cpu().numpy())
        mesh_filtered = mesh.remove_unreferenced_vertices()
        return TriangleMesh.from_o3d(mesh_filtered)

    def get_mesh_outside_box(self, bbox: SceneBox):

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        mesh_torch = self.torch(device=device)
        bbox = bbox.to(device)

        is_vertices_in_bbox = bbox.get_point_indices_within_box(mesh_torch.vertices)
        is_vertices_not_in_bbox = ~is_vertices_in_bbox

        is_triangles_vertices_not_in_bbox = is_vertices_not_in_bbox[mesh_torch.triangles.view(-1)].view(-1, 3)
        is_triangles_not_in_bbox = torch.any(is_triangles_vertices_not_in_bbox, dim=-1)
        triangles_not_in_bbox = mesh_torch.triangles[is_triangles_not_in_bbox].to(torch.int32)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(mesh_torch.vertices.cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(triangles_not_in_bbox.cpu().numpy())
        mesh_filtered = mesh.remove_unreferenced_vertices()
        return TriangleMesh.from_o3d(mesh_filtered)

    def draw(self):
        o3d_mesh = self.o3d_mesh()
        o3d.visualization.draw([o3d_mesh])

    def bbox(self):
        mesh_torch = self.torch('cuda')
        min_pt, _ = torch.min(mesh_torch.vertices, dim=0)
        max_pt, _ = torch.max(mesh_torch.vertices, dim=0)
        bbox = torch.stack([min_pt, max_pt], dim=0)
        return SceneBox(bbox)
