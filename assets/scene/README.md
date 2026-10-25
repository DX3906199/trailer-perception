# Trailer scene assets

All distances are in metres. In the fixed asset frame, the zero-angle hitch centre
is the origin, +X points towards the cargo body, and +Z points upwards. The world
places this origin at Z=0.21 m. The hinge is at `[1.39, 0, 0.03]` in the asset
frame, with rotation about +Z. `articulation.yaml` records the split convention.

| File | Contents |
| --- | --- |
| `trailer_visual_3dgs_local.ply` | Entire 48,041-Gaussian zero-angle asset |
| `trailer_body_3dgs_local.ply` | Fixed 41,890-Gaussian cargo/body part |
| `trailer_movable_assembly_hinge_local.ply` | 6,151 Gaussians for the triangular drawbar, front axle and front wheels, in hinge-local coordinates |
| `trailer_full_reconstructed_multires_local.obj` | Complete static reconstructed mesh for MuJoCo preview, without texture |
| `clean_ground_texture.png` | Periodic clean ground texture, 0.02 m per texel |

The PLY files contain Gaussian scales, orientations, opacity, and degree-zero
spherical-harmonic colour. They are not ordinary coloured point clouds. MuJoCo
loads the OBJ; the optional `gsplat` backend renders the PLY files separately.

The static whole OBJ is not split into movable meshes. Use Gaussian mode when
changing the articulation angle. Both offline backends use six-face rendering
and the calibrated equidistant mapping to keep the full body in the fisheye image.
The interactive MuJoCo viewer is an orbit-view preview of the static mesh.

`provenance.json` gives hashes, dimensions, and source information. The source
files were copied without editing. `configs/geometry.yaml` is the runtime camera
and geometric-prior configuration used by both rendering and pose estimation.

For custom calibrated image size, the renderer rescales intrinsics and stores
the resulting full configuration in its image JSON sidecar. Keep that calibration
with the resized image when solving pose. The sidecar also contains simulator
truth for evaluation; it is not an input to the image-based pose initializer.
