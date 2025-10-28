This document defines the coordinate system conventions used in the POW4R paper (DOI: 10.1109/LRA.2025.3606356) and provides a recommended simulation design to correctly use the solve_full_velocity function.

All coordinate systems are right-handed.

1. Camera Coordinate System (Primary)
This is the system in which the motion vector (t_x, t_y, t_z) from Equation 16a is solved.

Origin: The camera's optical center at time t (position 'A' in Figure 2).

Axes:

+Z: Points forward, along the camera's principal axis (out of the lens).

+X: Points rightward from the camera's perspective.

+Y: Points downward from the camera's perspective.

Reference: This convention is confirmed by the projection equation (Eq. 8), P = (u_p*d, v_p*d, d), where the depth d maps directly to the Z-coordinate.

2. Radar Coordinate System
This system is used for the radar's measurements and the final output vector v_f.

Origin: The 4D radar sensor's phase center (position 'R' in Figure 2).

Axes: The paper uses the same Z-forward, X-right, Y-down convention, as visualized in Figure 3.

Vectors:

The radial velocity vector v_r is measured in this system.

The final full-velocity vector v_f is also expressed in this system.

Transformation: The matrix T_A_to_R (and its rotation part R_cam_to_radar) transforms points and vectors from the Camera System to this Radar System.

3. Normalized Image Plane Coordinates
These are the 2D coordinates (u, v) used in the optical flow calculations.

Origin: The principal point (center of the image, where the +Z camera axis intersects the image plane).

Axes:

+u: Points rightward.

+v: Points downward.

Relation: This convention directly matches the +X and +Y axes of the Camera Coordinate System.

4. Simulation Design & Workflow
To make the simulation as easy as possible, you should base your entire virtual world on the paper's convention.

4.1. The World Coordinate System (Recommended)
Define your single, global "World" coordinate system to be identical to the paper's convention:

Origin: (0, 0, 0)

+X: Right

+Z: Forward (into the scene)

+Y: Down

4.2. Object & Sensor Poses
Every item in your world (the target object, camera, radar) will have its own pose, defined as a 4x4 homogeneous transformation matrix R | T.

M_world_to_local: This matrix (e.g., CamPose_A) transforms a point from World coordinates into the object's Local coordinates.

M_local_to_world = np.linalg.inv(M_world_to_local): This matrix transforms a point from Local coordinates into World coordinates.

4.3. Simulation Workflow (Getting Function Inputs)
At any two time steps, t (A) and t+delta_t (B), you can get all 10 inputs for solve_full_velocity from your simulation's "ground truth" data.

Given (from your simulation):

CamPose_A: 4x4 "world-to-camera" pose at time A.

CamPose_B: 4x4 "world-to-camera" pose at time B.

RadarPose: 4x4 "world-to-radar" pose.

TargetVel_world: The target's 3D velocity vector (vx, vy, vz) in world coordinates.

P_target_world_A: The 4x1 target point [x,y,z,1] in world coordinates at time A.

P_target_world_B: The 4x1 target point [x,y,z,1] in world coordinates at time B.

delta_t: The simulation time step (e.g., 1/30.0 seconds).

Calculate the Inputs:

T_A_to_B (Camera Motion): This matrix transforms from camera "A" space to camera "B" space.

T_A_to_B = CamPose_B @ np.linalg.inv(CamPose_A)

T_A_to_R (Extrinsics): This matrix transforms from camera "A" space to radar space.

T_A_to_R = RadarPose @ np.linalg.inv(CamPose_A)

d, (up, vp) (Coords at A): Transform the world point into camera "A" space.

P_in_CamA = CamPose_A @ P_target_world_A

d = P_in_CamA[2]

up = P_in_CamA[0] / d

vp = P_in_CamA[1] / d

(uq, vq) (Coords at B): Transform the new world point into camera "B" space.

P_in_CamB = CamPose_B @ P_target_world_B

d_B = P_in_CamB[2]

uq = P_in_CamB[0] / d_B

vq = P_in_CamB[1] / d_B

v_r (Radial Velocity): This requires projecting the target's world velocity onto the radar's line of sight.

RadarPose_inv = np.linalg.inv(RadarPose)

RadarOrigin_world = RadarPose_inv @ [0, 0, 0, 1] (Get radar's position in the world)

RadarToTarget_vec = P_target_world_A[0:3] - RadarOrigin_world[0:3] (Vector from radar to target)

u_vec = RadarToTarget_vec / np.linalg.norm(RadarToTarget_vec) (Unit direction vector)

radial_speed = np.dot(TargetVel_world, u_vec) (Project velocity)

v_r_world = radial_speed * u_vec (3D radial velocity in world coordinates)

Final Step: Convert this world vector to radar coordinates for the function.

R_world_to_radar = RadarPose[0:3, 0:3] (Get the 3x3 rotation part)

v_r = R_world_to_radar @ v_r_world

delta_t: This is simply your simulation time step.

5. ⚠️ CRITICAL: Simulation (OpenGL) vs. Paper Convention
Your simulation logic should run in the (Z-fwd, Y-down) system, but your OpenGL renderer expects a (Z-back, Y-up) system.

Convention	X-Axis	Y-Axis	Z-Axis	Handedness
Paper / Sim Logic	Right	Down	Forward	Right
OpenGL Renderer	Right	Up	Backward	Right

Solution: Do not change your simulation logic. Instead, "flip" your world at the very last second for the renderer.

When you set your OpenGL "View" matrix, multiply it by a conversion matrix that flips the Y and Z axes.

ConversionMatrix = np.diag([1, -1, -1, 1])

Your final view matrix for rendering will be:

Final_GL_ViewMatrix = Your_GL_Viewer_Pose @ ConversionMatrix

This keeps your simulation math clean and identical to the paper, while satisfying the renderer.