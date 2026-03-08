import numpy as np
import open3d as o3d

EVAL_THRESHOLD = 0.02


def register(source: o3d.geometry.PointCloud,
             target: o3d.geometry.PointCloud) -> np.ndarray:

    diagonal   = np.linalg.norm(source.get_axis_aligned_bounding_box().get_extent())
    voxel_size = max(diagonal * 0.02, 0.001)   # for FPFH features
    icp_voxel  = max(diagonal * 0.01, 0.001)   # finer, for ICP quality

    # Downsample for ICP — normals estimated here, not on 198k full cloud
    src_icp = source.voxel_down_sample(icp_voxel)
    tgt_icp = target.voxel_down_sample(icp_voxel)
    for pcd in [src_icp, tgt_icp]:
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=icp_voxel * 2, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=15)

    # Coarser downsample for FPFH features
    src_down = src_icp.voxel_down_sample(voxel_size)
    tgt_down = tgt_icp.voxel_down_sample(voxel_size)
    for pcd in [src_down, tgt_down]:
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=15)

    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        tgt_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))

    print(f"  RANSAC pts: {len(src_down.points)}/{len(tgt_down.points)}  "
          f"ICP pts: {len(src_icp.points)}/{len(tgt_icp.points)}")

    # RANSAC global alignment
    dist = voxel_size * 1.5
    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        mutual_filter=False,
        max_correspondence_distance=dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(500_000, 0.9999))

    ev = o3d.pipelines.registration.evaluate_registration(
        src_icp, tgt_icp, EVAL_THRESHOLD, ransac.transformation)
    print(f"  RANSAC → fitness={ev.fitness:.4f}")

    # Point-to-plane ICP on downsampled clouds — same transform, much faster
    r = o3d.pipelines.registration.registration_icp(
        src_icp, tgt_icp,
        max_correspondence_distance=0.10,
        init=ransac.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(k=0.05)),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=200))

    ev = o3d.pipelines.registration.evaluate_registration(
        src_icp, tgt_icp, EVAL_THRESHOLD, r.transformation)
    print(f"  ICP   → fitness={ev.fitness:.4f}  rmse={ev.inlier_rmse:.5f}")

    return r.transformation