import open3d as o3d

# 파일 경로
point_cloud_files = ["./output/points_3.ply", "./output/points_4.ply", "./output/points_5.ply", "./output/points_6.ply"]
mesh_files = ["./output/cameras_3.ply", "./output/cameras_4.ply", "./output/cameras_5.ply", "./output/cameras_6.ply"]

# 포인트 클라우드 및 메시 데이터 로드
point_clouds = []
meshes = []

# 포인트 클라우드 로드
for file in point_cloud_files:
    pcd = o3d.io.read_point_cloud(file)
    if pcd.is_empty():
        print(f"Failed to load point cloud from {file}")
    else:
        point_clouds.append(pcd)

# 메시 데이터 로드
for file in mesh_files:
    mesh = o3d.io.read_triangle_mesh(file)
    if mesh.is_empty():
        print(f"Failed to load mesh from {file}")
    else:
        meshes.append(mesh)

# 시각화
o3d.visualization.draw_geometries(point_clouds + meshes)