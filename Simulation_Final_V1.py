import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
import matplotlib.animation as animation
import itertools
from scipy.optimize import differential_evolution
import os
import datetime

# --- Centralized Configuration for Easy Tuning ---
SIMULATION_CONFIG = {
    # Lander Physical Properties
    # --- GEOMETRY SWAP FOR DEBUGGING ---
    "mass": 730, "height": 0.6, "length": 2.7, "width": 1.7, # Swapped length and width back to original
    "leg_length": 1.8, "leg_angle_deg": 20, "g_moon": 1.62,

    # Ground Contact Model
    "ground_stiffness": 5e5, "ground_damping": 2e4,
    "mu_kinetic": 0.6, "mu_static": 0.75,
    "stiction_velocity_threshold": 1e-3,

    # Actuator Properties
    "internal_mass_actuator_force": 4000.0, "internal_mass_speed": 2.0,
    "max_torque": 2500.0,

    # PID Controller & State Machine
    "wait_timer": 0.5, "braking_angle_deg": 20.0, "stabilize_angle_deg": 5.0,
    "stabilize_vel_dps": 5.0, "finish_angle_deg": 0.5, "finish_vel_dps": 0.5,
    "initial_push_duration": 0.75,

    # --- Analysis Parameters ---
    "proximity_analysis_start_time": 0.5, # seconds

    # --- Environmental Simulation Parameters ---
    "solar_panel_area": 2.5,
    "solar_panel_efficiency": 0.30,
    "solar_constant_moon": 1361,   # W/m^2
    "sun_elevation_deg": 1.5,
    "surface_solar_absorptivity": 0.8, # Typical for MLI blankets
}

# --- Face Proximity Analysis Configuration ---
FACE_IMPORTANCE = {
    'TOP_DECK':     1.0, 'BOTTOM_HULL':    5.0, 'FRONT_FACE':     10.0,
    'BACK_FACE':    8.0, 'LEFT_FACE':      2.0, 'RIGHT_FACE':     2.0,
}

# --- Helper function ---
def is_point_in_polygon(polygon_vertices, point):
    """Checks if a 2D point is inside a 2D polygon."""
    if len(polygon_vertices) < 3: return False
    from matplotlib.path import Path
    return Path(polygon_vertices).contains_point(point)

class LunarLander:
    """
    Represents the physical properties and state of the lunar lander.
    Handles the geometry, mass properties, and the core physics integration.
    """
    def __init__(self, config):
        """Initializes the lander's physical properties based on the config."""
        self.config = config
        self.mass, self.g_moon = config['mass'], config['g_moon']
        self.gravity_vector_world = np.array([0, 0, -self.g_moon])
        self.height, self.length, self.width = config['height'], config['length'], config['width']
        self.leg_length, self.leg_angle = config['leg_length'], np.deg2rad(config['leg_angle_deg'])
        self.ground_stiffness, self.ground_damping = config['ground_stiffness'], config['ground_damping']
        self.mu_kinetic, self.mu_static = config['mu_kinetic'], config['mu_static']
        self.stiction_vel_thresh = config['stiction_velocity_threshold']
        self.internal_mass_actuator_force, self.internal_mass_speed = config['internal_mass_actuator_force'], config['internal_mass_speed']

        w, l, h = self.width, self.length, self.height
        self.body_corners_body = np.array([
            [-w/2,-l/2,-h/2], [w/2,-l/2,-h/2], [w/2,l/2,-h/2], [-w/2,l/2,-h/2], # Bottom 0-3
            [-w/2,-l/2, h/2], [w/2,-l/2, h/2], [w/2,l/2, h/2], [-w/2,l/2, h/2]  # Top 4-7
        ])
        self.body_faces = {
            'TOP_DECK': [4,5,6,7], 'BOTTOM_HULL': [0,1,2,3], 'FRONT_FACE': [3,2,6,7],
            'BACK_FACE': [0,1,5,4], 'LEFT_FACE': [0,3,7,4], 'RIGHT_FACE': [1,2,6,5],
        }

        self.face_geometry = {
            'TOP_DECK':     {'area': w*l, 'normal': np.array([ 0, 0, 1])},
            'BOTTOM_HULL': {'area': w*l, 'normal': np.array([ 0, 0,-1])},
            'FRONT_FACE':   {'area': w*h, 'normal': np.array([ 0, 1, 0])},
            'BACK_FACE':   {'area': w*h, 'normal': np.array([ 0,-1, 0])},
            'LEFT_FACE':   {'area': l*h, 'normal': np.array([-1, 0, 0])},
            'RIGHT_FACE':   {'area': l*h, 'normal': np.array([ 1, 0, 0])},
        }

        self.leg_mount_positions_body = np.array([
            [ w/2,  l/2, -h/2], [-w/2,  l/2, -h/2],
            [-w/2, -l/2, -h/2], [ w/2, -l/2, -h/2]
        ])

        self.footpads_body = self._calculate_footpad_positions_body()
        self.all_hard_points_body = np.vstack([self.footpads_body, self.body_corners_body])

        self.com_offset_body = np.zeros(3)
        self.calculate_inertia_tensor()
        self.position, self.velocity, self.angular_velocity = np.zeros(3), np.zeros(3), np.zeros(3)
        self.orientation_q = R.from_matrix(np.eye(3))

        self.energy_consumed = 0.0
        self.energy_dissipated = 0.0

    @property
    def orientation(self):
        return self.orientation_q.as_matrix()

    @orientation.setter
    def orientation(self, matrix):
        self.orientation_q = R.from_matrix(matrix)

    def _calculate_footpad_positions_body(self):
        footpads = []
        for mount_pos in self.leg_mount_positions_body:
            dx = self.leg_length * np.sin(self.leg_angle) * np.sign(mount_pos[0])
            dy = self.leg_length * np.sin(self.leg_angle) * np.sign(mount_pos[1])
            dz = -self.leg_length * np.cos(self.leg_angle)
            footpads.append(mount_pos + np.array([dx, dy, dz]))
        return np.array(footpads)

    def calculate_inertia_tensor(self):
        m, w, l, h = self.mass, self.width, self.length, self.height
        Ixx = (m/12) * (l**2 + h**2)
        Iyy = (m/12) * (w**2 + h**2)
        Izz = (m/12) * (w**2 + l**2)
        self.inertia_tensor = np.diag([Ixx, Iyy, Izz])
        self.inv_inertia_tensor = np.linalg.inv(self.inertia_tensor)

    def get_geometric_center_world(self, p, o_mat):
        return p - (o_mat @ self.com_offset_body)

    def get_hard_points_world(self, p, o_mat):
        return self.get_geometric_center_world(p, o_mat) + (o_mat @ self.all_hard_points_body.T).T

    def _get_state_vector(self):
        return np.concatenate([self.position, self.velocity, self.orientation_q.as_quat(), self.angular_velocity])

    def _set_state_from_vector(self, y):
        self.position, self.velocity = y[0:3], y[3:6]
        quat = y[6:10]
        norm = np.linalg.norm(quat)
        if np.any(np.isnan(quat)) or norm < 1e-9:
            return
        
        if norm > 1e-9:
            self.orientation_q = R.from_quat(quat / norm)
        self.angular_velocity = y[10:13]

    def _calculate_derivatives_from_vector(self, y, actuator_force_body):
        position, velocity, angular_velocity = y[0:3], y[3:6], y[10:13]
        orientation = R.from_quat(y[6:10]).as_matrix()

        np.clip(angular_velocity, -1000, 1000, out=angular_velocity)

        hard_points_world = self.get_hard_points_world(position, orientation)
        contact_indices = np.where(hard_points_world[:, 2] < 0)[0]

        force_ground = np.zeros(3)
        torque_body = np.zeros(3)

        if contact_indices.size > 0:
            contact_points = hard_points_world[contact_indices]
            levers_world = contact_points - position
            velocities_world = velocity + np.cross(angular_velocity, levers_world)

            f_norm_mag_unclipped = -self.ground_stiffness * contact_points[:, 2] - self.ground_damping * velocities_world[:, 2]
            f_norm_mag = np.clip(f_norm_mag_unclipped, 0, 50 * self.mass * self.g_moon)
            f_norm_vec = np.zeros_like(contact_points); f_norm_vec[:, 2] = f_norm_mag

            f_fric_vec = np.zeros_like(contact_points)
            speed_tan = np.linalg.norm(velocities_world[:, :2], axis=1)

            static_mask = speed_tan < self.stiction_vel_thresh
            if np.any(static_mask):
                f_brake_needed = -velocities_world[static_mask, :2] * self.mass / 0.01
                f_brake_mag = np.linalg.norm(f_brake_needed, axis=1)
                max_static_force = self.mu_static * f_norm_mag[static_mask]
                clipped_brake_mag = np.minimum(f_brake_mag, max_static_force)
                f_fric_vec[static_mask, :2] = (f_brake_needed / (f_brake_mag[:, np.newaxis] + 1e-9)) * clipped_brake_mag[:, np.newaxis]

            kinetic_mask = ~static_mask
            if np.any(kinetic_mask):
                moving_vels = velocities_world[kinetic_mask][:, :2]
                moving_speeds = speed_tan[kinetic_mask]
                dirs = -moving_vels / (moving_speeds[:, np.newaxis] + 1e-9)
                fric_mag = self.mu_kinetic * f_norm_mag[kinetic_mask]
                f_fric_vec[kinetic_mask, :2] = dirs * fric_mag[:, np.newaxis]

            force_ground = np.sum(f_norm_vec + f_fric_vec, axis=0)

            levers_body = (orientation.T @ levers_world.T).T
            f_contact_body = (orientation.T @ (f_norm_vec + f_fric_vec).T).T
            torque_body = np.sum(np.cross(levers_body, f_contact_body), axis=0)

        force_total = (self.mass * self.gravity_vector_world) + force_ground - (orientation @ actuator_force_body)
        torque_body += np.cross(self.com_offset_body, -actuator_force_body)

        accel = force_total / self.mass
        ang_accel = self.inv_inertia_tensor @ (torque_body - np.cross(angular_velocity, self.inertia_tensor @ angular_velocity))

        q = y[6:10]; qw, qx, qy, qz = q[3],q[0],q[1],q[2]
        wx, wy, wz = angular_velocity
        dw, dx, dy, dz = -0.5*(qx*wx+qy*wy+qz*wz), 0.5*(qw*wx+qy*wz-qz*wy), 0.5*(qw*wy-qx*wz+qz*wx), 0.5*(qw*wz+qx*wy-qy*wx)

        return np.concatenate([velocity, accel, [dx,dy,dz,dw], ang_accel])

    def _update_energy_tracking(self, dt):
        hard_points_world = self.get_hard_points_world(self.position, self.orientation)
        contact_indices = np.where(hard_points_world[:, 2] < 0)[0]

        if contact_indices.size > 0:
            contact_points = hard_points_world[contact_indices]
            levers_world = contact_points - self.position
            velocities_world = self.velocity + np.cross(self.angular_velocity, levers_world)
            power_dissipated_per_point = self.ground_damping * velocities_world[:, 2]**2
            self.energy_dissipated += np.sum(power_dissipated_per_point) * dt

    def simulation_step(self, dt, actuator_force_body):
        y0 = self._get_state_vector()
        k1 = self._calculate_derivatives_from_vector(y0, actuator_force_body)
        if np.any(np.isnan(k1)): return
        k2 = self._calculate_derivatives_from_vector(y0 + 0.5*dt*k1, actuator_force_body)
        if np.any(np.isnan(k2)): return
        k3 = self._calculate_derivatives_from_vector(y0 + 0.5*dt*k2, actuator_force_body)
        if np.any(np.isnan(k3)): return
        k4 = self._calculate_derivatives_from_vector(y0 + dt*k3, actuator_force_body)
        if np.any(np.isnan(k4)): return

        y_final = y0 + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self._set_state_from_vector(y_final)
        self._update_energy_tracking(dt)

class PrecisionPIDController:
    """
    PID controller with a state machine for robust lander reorientation.
    """
    def __init__(self, lander, target_orientation, Kp, Ki, Kd, config):
        self.lander = lander
        self.target_orientation_q = R.from_matrix(target_orientation)
        self.target_euler_deg = R.from_matrix(target_orientation).as_euler('xyz', degrees=True)
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral_error, self.state = np.zeros(3), "ACCELERATING"
        self.config = config
        self.wait_timer = config['wait_timer']
        self.push_timer = config.get('initial_push_duration', 0)
        self.zero_crossing_count, self.last_roll_error_sign = 0, 0
        self.braking_angle_deg = config['braking_angle_deg']
        self.stabilize_angle_deg = config['stabilize_angle_deg']
        self.stabilize_vel_dps = config['stabilize_vel_dps']
        self.finish_angle_deg = config['finish_angle_deg']
        self.finish_vel_dps = config['finish_vel_dps']
        self.max_shift = min(self.lander.width, self.lander.length) / 2.1

    def update(self, dt):
        if self.wait_timer > 0:
            self.wait_timer -= dt
            # Return zero for both actuator force and commanded torque during wait
            return np.zeros(3), np.zeros(3)

        q_err = self.target_orientation_q * self.lander.orientation_q.inv()
        err_vec = q_err.as_rotvec()
        err_ang_deg = np.rad2deg(np.linalg.norm(err_vec))
        ang_vel_dps = np.rad2deg(np.linalg.norm(self.lander.angular_velocity))

        is_finished = (err_ang_deg < self.finish_angle_deg and ang_vel_dps < self.finish_vel_dps) or \
                      (self.state == "STABILIZING" and self.zero_crossing_count >= 10)

        if self.state != "FINISHED":
            if is_finished:
                self.state = "FINISHED"; self.integral_error.fill(0)
            elif err_ang_deg < self.stabilize_angle_deg and ang_vel_dps < self.stabilize_vel_dps:
                self.state = "STABILIZING"
            elif err_ang_deg < self.braking_angle_deg and self.state == "ACCELERATING":
                self.state = "DECELERATING"

        commanded_torque, target_pos = np.zeros(3), np.zeros(3)
        if self.state == "FINISHED":
            pass # Zero torque
        elif self.state == "ACCELERATING" and self.push_timer > 0:
            torque_direction = err_vec / (np.linalg.norm(err_vec) + 1e-9)
            commanded_torque = torque_direction * self.config['max_torque'] * 0.85
            self.push_timer -= dt
        else:
            if self.state == "STABILIZING":
                roll_err_sign = np.sign(self.lander.orientation_q.as_euler('xyz',degrees=True)[0] - self.target_euler_deg[0])
                if self.last_roll_error_sign != 0 and roll_err_sign != self.last_roll_error_sign:
                    self.zero_crossing_count += 1
                self.last_roll_error_sign = roll_err_sign
            p = self.Kp * err_vec
            d = self.Kd * self.lander.angular_velocity
            self.integral_error = np.clip(self.integral_error + err_vec * dt, -10., 10.) if self.state == "STABILIZING" else np.zeros(3)
            commanded_torque = p - d + self.Ki * self.integral_error
            
        # Don't clip commanded torque here, we want to see what the PID is asking for.
        # The physical limits will be handled by the actuator model.

        if self.state != "FINISHED":
            g_body = self.lander.orientation.T @ (self.lander.gravity_vector_world * self.lander.mass)
            # Calculate the required CoM shift to achieve the COMMANDED torque
            com_req = np.cross(g_body, commanded_torque) / (np.linalg.norm(g_body)**2 + 1e-9)
            mag = np.linalg.norm(com_req)
            if mag > 1e-6:
                # But physically limit the shift
                target_pos = (com_req / mag) * min(mag, self.max_shift)

        err_pos = target_pos - self.lander.com_offset_body
        actuator_force = np.zeros(3)
        if np.linalg.norm(err_pos) > 0.01:
            direction = err_pos / np.linalg.norm(err_pos)
            self.lander.com_offset_body += direction * self.lander.internal_mass_speed * dt
            actuator_force = direction * self.lander.internal_mass_actuator_force
            self.lander.energy_consumed += np.linalg.norm(actuator_force) * self.lander.internal_mass_speed * dt
        
        return actuator_force, commanded_torque

def define_stable_poses(lander):
    poses = {'UPSIDE_DOWN': R.from_euler('x', 180, degrees=True).as_matrix()}
    defs = {
        'UPRIGHT': lander.footpads_body,
        'SIDE_LEFT': np.vstack([lander.footpads_body[[1,2]], lander.body_corners_body[[7,4]]]),
        'SIDE_FRONT': np.vstack([lander.footpads_body[[0,1]], lander.body_corners_body[[6,7]]]),
    }
    for name, points in defs.items():
        normal = np.cross(points[1]-points[0], points[2]-points[0])
        normal /= np.linalg.norm(normal)
        if np.dot(normal, np.mean(points, axis=0)) < 0: normal = -normal
        rot, _ = R.align_vectors([[0,0,-1]], [normal])
        poses[name] = rot.as_matrix()
    return poses

def plot_lander_model(ax, lander, orientation=np.eye(3), position=np.zeros(3), alpha=0.25, face_color='cyan', edge_color='b'):
    all_pts = (orientation @ lander.all_hard_points_body.T).T + position
    body_pts, foot_pts = all_pts[4:], all_pts[:4]
    faces = [[body_pts[j] for j in i] for i in [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=face_color, linewidths=1, edgecolors=edge_color, alpha=alpha))
    mounts = (orientation @ lander.leg_mount_positions_body.T).T + position
    for j in range(4): ax.plot(*zip(mounts[j], foot_pts[j]), 'k-')
    max_r = max(lander.width, lander.length, lander.height) * 1.5

def define_all_poses(lander):
    """
    Defines both stable and off-nominal resting poses for the lander.
    Returns two dictionaries: stable_poses, off_nominal_poses
    """
    stable_poses = {'UPSIDE_DOWN': R.from_euler('x', 180, degrees=True).as_matrix()}
    
    # --- Define Stable Poses based on multiple contact points ---
    stable_defs = {
        'UPRIGHT': lander.footpads_body,
        'SIDE_LEFT': np.vstack([lander.footpads_body[[1,2]], lander.body_corners_body[[7,4]]]),
        'SIDE_FRONT': np.vstack([lander.footpads_body[[0,1]], lander.body_corners_body[[6,7]]]),
    }
    for name, points in stable_defs.items():
        # Calculate the normal of the plane defined by the first three points
        normal = np.cross(points[1]-points[0], points[2]-points[0])
        normal /= np.linalg.norm(normal)
        # Ensure normal points away from the CoM (outward from the lander)
        if np.dot(normal, np.mean(points, axis=0)) < 0: normal = -normal
        # Align this normal with the world's up vector (so the face is on the ground)
        rot, _ = R.align_vectors([[0,0,-1]], [normal])
        stable_poses[name] = rot.as_matrix()

    # --- Define Off-Nominal Poses by mirroring the stable side poses ---
    off_nominal_poses = {}
    # Naming convention: footpads [0,1,2,3] -> [+x+y, -x+y, -x-y, +x-y]
    # Corners [4,5,6,7] -> [-x-y, +x-y, +x+y, -x+y] on the top deck
    off_nominal_defs = {
        # Mirrored version of SIDE_FRONT
        'SIDE_BACK': np.vstack([lander.footpads_body[[3,2]], lander.body_corners_body[[5,4]]]),
        # Mirrored version of SIDE_LEFT
        'SIDE_RIGHT': np.vstack([lander.footpads_body[[0,3]], lander.body_corners_body[[6,5]]]),
    }
    
    for name, points in off_nominal_defs.items():
        # Use the same robust method to calculate orientation
        normal = np.cross(points[1]-points[0], points[2]-points[0])
        normal /= np.linalg.norm(normal)
        if np.dot(normal, np.mean(points, axis=0)) < 0: normal = -normal
        rot, _ = R.align_vectors([[0,0,-1]], [normal])
        off_nominal_poses[name] = rot.as_matrix()

    return stable_poses, off_nominal_poses

def plot_lander_model(ax, lander, orientation=np.eye(3), position=np.zeros(3), alpha=0.25, face_color='cyan', edge_color='b'):
    all_pts = (orientation @ lander.all_hard_points_body.T).T + position
    body_pts, foot_pts = all_pts[4:], all_pts[:4]
    faces = [[body_pts[j] for j in i] for i in [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=face_color, linewidths=1, edgecolors=edge_color, alpha=alpha))
    mounts = (orientation @ lander.leg_mount_positions_body.T).T + position
    for j in range(4): ax.plot(*zip(mounts[j], foot_pts[j]), 'k-')
    max_r = max(lander.width, lander.length, lander.height) * 1.5
    ax.set_xlim(-max_r, max_r); ax.set_ylim(-max_r, max_r); ax.set_zlim(0, max_r*1.5)
    ax.set_xlabel('X-axis (m)'); ax.set_ylabel('Y-axis (m)'); ax.set_zlabel('Z-axis (m)')

def run_static_analysis_for_pose(lander, start_pose_name, start_pose, output_dir, step_size=0.05):
    print(f"\n--- Running Static Analysis for Pose: {start_pose_name} ---")
    pts_rot = (start_pose @ lander.all_hard_points_body.T).T
    v_shift = np.array([0,0,-np.min(pts_rot[:,2])])
    pts_world = pts_rot + v_shift
    contact_idx = np.where(np.isclose(pts_world[:,2], 0))[0]
    support_2d = pts_world[contact_idx][:,:2]
    hull = ConvexHull(support_2d) if len(np.unique(support_2d,axis=0)) >= 3 else None
    poly_verts = hull.points[hull.vertices] if hull else support_2d
    shifts = [np.arange(-d/2*0.9, d/2*0.9+s, s) for d,s in zip([lander.width, lander.length, lander.height], [step_size]*3)]
    all_shifts = np.array(list(itertools.product(*shifts)))
    g_body = start_pose.T @ lander.gravity_vector_world
    torques = np.cross(all_shifts, g_body*lander.mass)
    accels = (lander.inv_inertia_tensor @ torques.T).T
    com_world = (start_pose @ all_shifts.T).T + v_shift
    zmps = com_world[:,:2]
    unstable_mask = [not is_point_in_polygon(poly_verts, z) for z in zmps]
    fig1 = plt.figure(figsize=(10,8)); ax1 = fig1.add_subplot(111, projection='3d')
    plot_lander_model(ax1, lander, start_pose, v_shift, alpha=0.1)
    p1 = ax1.scatter(com_world[:,0],com_world[:,1],com_world[:,2], c=np.linalg.norm(accels,axis=1), cmap='viridis', s=15, alpha=0.7, label='CoM Positions')
    fig1.colorbar(p1, ax=ax1, label="Angular Acceleration (rad/s²)"); ax1.set_title(f"Achievable Angular Accelerations for {start_pose_name} Pose", fontsize=16)
    ax1.legend()
    plt.savefig(os.path.join(output_dir, f"static_accelerations_{start_pose_name}.png")); plt.close(fig1)
    
    fig3 = plt.figure(figsize=(8,8)); ax3 = fig3.add_subplot(111)
    ax3.add_patch(plt.Polygon(poly_verts, closed=True, color='lightblue', label='Base of Support (BoS)'))
    if np.any(unstable_mask):
        ax3.scatter(zmps[unstable_mask,0], zmps[unstable_mask,1], c='red', s=10, label='Unstable ZMPs', marker='x')
    stable_mask = ~np.array(unstable_mask)
    if np.any(stable_mask):
        ax3.scatter(zmps[stable_mask,0], zmps[stable_mask,1], c='green', s=10, label='Stable ZMPs', marker='o')

    ax3.set_title(f"Zero Moment Point (ZMP) Stability for {start_pose_name} Pose", fontsize=16)
    ax3.set_xlabel("Ground X-axis (m)"); ax3.set_ylabel("Ground Y-axis (m)")
    ax3.set_aspect('equal', adjustable='box'); ax3.grid(True); ax3.legend()
    plt.savefig(os.path.join(output_dir, f"static_zmp_{start_pose_name}.png")); plt.close(fig3)

def run_dynamic_simulation(start_pose, end_pose, Kp, Ki, Kd, config):
    lander = LunarLander(config)
    controller = PrecisionPIDController(lander, end_pose, Kp, Ki, Kd, config)
    min_z = np.min((start_pose @ lander.all_hard_points_body.T).T[:, 2])
    lander.orientation = start_pose
    lander.position = np.array([0, 0, -min_z + 0.01])

    sim_duration, dt = 300.0, 0.01
    steps = int(sim_duration / dt)
    history = {
        'time':[], 'roll':[], 'pitch':[], 'yaw':[], 'frames':[], 'min_z':[], 'orientations':[],
        'commanded_torque': [], 'applied_torque': [], # DEBUGGING: Added torque tracking
        'ke': [], 'pe': [], 'energy_consumed': [], 'energy_dissipated': []
    }
    face_proximity_tracker = {name: float('inf') for name in lander.body_faces}
    is_com_centered = False

    for i in range(steps):
        # Get commanded torque from the controller
        actuator_force, commanded_torque = controller.update(dt)
        
        # Calculate the actual torque that will be applied in the next physics step
        g_body = lander.orientation.T @ (lander.gravity_vector_world * lander.mass)
        applied_torque = np.cross(lander.com_offset_body, g_body)

        # Run the physics simulation
        lander.simulation_step(dt, actuator_force)
        
        # Log everything
        time_s = i * dt
        history['time'].append(time_s)
        history['orientations'].append(lander.orientation_q)
        rpy = lander.orientation_q.as_euler('xyz', degrees=True)
        history['roll'].append(rpy[0]); history['pitch'].append(rpy[1]); history['yaw'].append(rpy[2])
        
        # Store magnitudes for easier plotting
        history['commanded_torque'].append(np.linalg.norm(commanded_torque))
        history['applied_torque'].append(np.linalg.norm(applied_torque))

        min_z_val = np.min(lander.get_hard_points_world(lander.position, lander.orientation)[:, 2])
        history['min_z'].append(min_z_val)
        
        if time_s > config['proximity_analysis_start_time']:
            run_proximity_analysis(lander, face_proximity_tracker)
        
        ke_lin = 0.5 * lander.mass * np.dot(lander.velocity, lander.velocity)
        ke_rot = 0.5 * np.dot(lander.angular_velocity, lander.inertia_tensor @ lander.angular_velocity)
        pe = lander.mass * lander.g_moon * lander.position[2]
        history['ke'].append(ke_lin + ke_rot); history['pe'].append(pe)
        history['energy_consumed'].append(lander.energy_consumed)
        history['energy_dissipated'].append(lander.energy_dissipated)

        if i % 10 == 0:
            history['frames'].append({'o':lander.orientation, 'p':lander.position, 'c':lander.com_offset_body.copy()})
            
        if controller.state == "FINISHED":
            if np.linalg.norm(lander.com_offset_body) < 0.01: is_com_centered = True
            if is_com_centered: break
            
    return history, face_proximity_tracker, lander.energy_consumed, lander.energy_dissipated

def objective_function(params, start_pose, end_pose, target_euler, config):
    Kp, Ki, Kd = params
    history, _ , _, _ = run_dynamic_simulation(start_pose, end_pose, Kp, Ki, Kd, config)
    if not history['time']: return 1e9
    lift_off_penalty = 10000 if np.any(np.array(history['min_z']) > 0.1) else 0
    errors = np.abs(np.array([history['roll'], history['pitch'], history['yaw']]).T - target_euler)
    iae = np.sum(errors) * 0.01
    settling_time = history['time'][-1]
    for i in range(len(history['time'])-1, -1, -1):
        if np.all(errors[i] < 5.0): settling_time = history['time'][i]
        else: break
    final_error = np.linalg.norm(errors[-1])
    
    # Use applied torque for cost function as it reflects real effort
    total_torque_effort = np.sum(history['applied_torque']) * 0.001
    
    return (10.0 * final_error) + (0.5 * settling_time) + (0.01 * iae) + lift_off_penalty + total_torque_effort

def plot_all_landing_poses(lander, output_dir):
    stable_poses, off_nominal_poses = define_all_poses(lander)
    all_poses = {**stable_poses, **off_nominal_poses}
    
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("Nominal and Off-Nominal Landing Pose Analysis", fontsize=20)
    
    # Create a 2x3 grid for 6 poses
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.1)
    axes = gs.subplots(subplot_kw={'projection': '3d'})
    axes = axes.flatten()

    for i, (name, orientation) in enumerate(all_poses.items()):
        ax = axes[i]
        # Place the lowest point of the lander on the z=0 plane
        min_z = np.min((orientation @ lander.all_hard_points_body.T).T[:, 2])
        position = np.array([0, 0, -min_z])
        
        is_stable = name in stable_poses
        
        # Customize appearance based on stability
        if is_stable:
            plot_lander_model(ax, lander, orientation, position, alpha=0.25)
            ax.set_title(name, fontsize=14, color='black')
        else: # Off-nominal
            plot_lander_model(ax, lander, orientation, position, alpha=0.4, face_color='orangered', edge_color='r')
            ax.set_title(f"{name}\n(Off-Nominal)", fontsize=14, color='red')
            
    # Turn off any unused subplots if there are fewer than 6
    for i in range(len(all_poses), len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "all_landing_poses.png"))
    plt.close(fig)

def plot_force_diagram(lander, output_dir):
    """
    Creates a 2D side-view plot showing the forces acting on the lander in an upright, stable position.
    """
    # Use the upright orientation
    orientation = np.eye(3) 
    # Calculate vertical shift to place footpads on the ground (z=0)
    min_z = np.min((orientation @ lander.all_hard_points_body.T).T[:, 2])
    position = np.array([0, 0, -min_z])

    # Get points in world coordinates
    all_pts_world = (orientation @ lander.all_hard_points_body.T).T + position
    com_world = position + (orientation @ lander.com_offset_body)
    footpads_world = all_pts_world[:4]

    # --- Create the plot ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # --- Draw Lander Body (as a 2D projection) ---
    body_corners_world = all_pts_world[4:]
    profile_points = np.vstack([body_corners_world[:, 0], body_corners_world[:, 2]]).T
    hull = ConvexHull(profile_points)
    
    # Draw the lander body outline
    ax.fill(profile_points[hull.vertices, 0], profile_points[hull.vertices, 1], color='cyan', alpha=0.3, label='Lander Body (Profile)')
    ax.plot(np.append(profile_points[hull.vertices, 0], profile_points[hull.vertices, 0][0]), 
            np.append(profile_points[hull.vertices, 1], profile_points[hull.vertices, 1][0]), 'b-')

    # --- Draw Legs ---
    leg_mounts_world = (orientation @ lander.leg_mount_positions_body.T).T + position
    for i in range(4):
        ax.plot([leg_mounts_world[i, 0], footpads_world[i, 0]], [leg_mounts_world[i, 2], footpads_world[i, 2]], 'k-', lw=2)

    # --- Plot Hardpoints and CoM ---
    ax.plot(all_pts_world[:, 0], all_pts_world[:, 2], 'ko', markersize=5, label='Hard Points')
    ax.plot(com_world[0], com_world[2], 'ro', markersize=10, label='Center of Mass (CoM)')

    # --- Draw Force Vectors ---
    gravity_force_magnitude = lander.mass * lander.g_moon
    force_scale_factor = 1.5 # Adjust to make arrows a reasonable length
    ax.quiver(com_world[0], com_world[2], 0, -gravity_force_magnitude, 
              angles='xy', scale_units='xy', scale=gravity_force_magnitude/force_scale_factor, color='red', width=0.005,
              label=f'Gravity ({gravity_force_magnitude:.0f} N)')

    # Group footpads by their x-position to draw combined force vectors
    footpad_groups = {}
    for fp in footpads_world:
        x_key = round(fp[0], 4)
        if x_key not in footpad_groups:
            footpad_groups[x_key] = []
        footpad_groups[x_key].append(fp)

    total_normal_force = gravity_force_magnitude
    num_legs = len(footpads_world)
    force_per_leg = total_normal_force / num_legs if num_legs > 0 else 0

    has_labeled_grf = False
    for x_pos, pads in footpad_groups.items():
        num_pads_at_pos = len(pads)
        combined_force = num_pads_at_pos * force_per_leg
        # Draw from the lowest point at that x-position
        draw_pad = min(pads, key=lambda p: p[2])
        label = f'Ground Reaction ({num_pads_at_pos} legs)' if not has_labeled_grf else ""
        ax.quiver(draw_pad[0], draw_pad[2], 0, combined_force,
                  angles='xy', scale_units='xy', scale=gravity_force_magnitude/force_scale_factor, color='green', width=0.005,
                  label=label)
        has_labeled_grf = True

    # --- Plotting Configuration ---
    ax.axhline(0, color='saddlebrown', linestyle='-', linewidth=2, label='Lunar Surface')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Force Diagram of Lander in Upright Position (2D Side View)', fontsize=16)
    ax.set_xlabel('X-axis (m)', fontsize=12)
    ax.set_ylabel('Z-axis (m)', fontsize=12)
    ax.grid(True, linestyle='--')
    ax.legend(fontsize=10)
    ax.set_ylim(-1, lander.height + lander.leg_length)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "force_diagram.png"))
    plt.close(fig)

def run_proximity_analysis(lander, tracker):
    gc = lander.get_geometric_center_world(lander.position, lander.orientation)
    body_corners_world = gc + (lander.orientation @ lander.body_corners_body.T).T
    for face_name, indices in lander.body_faces.items():
        min_z = np.min(body_corners_world[indices][:, 2])
        tracker[face_name] = min(tracker[face_name], min_z)

def print_report(title, data_dict, unit=""):
    print(f"\n--- {title} ---"); print("="*60)
    max_key_len = max(len(k) for k in data_dict.keys())
    for key, value in data_dict.items():
        print(f"{key:<{max_key_len}} | {value:.3f} {unit}")
    print("="*60)

def plot_bar_report(title, data_dict, y_label, output_path, color='skyblue', log_scale=False):
    labels, values = list(data_dict.keys()), list(data_dict.values())
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(labels, values, color=color, edgecolor='black')
    ax.set_ylabel(y_label, fontsize=12); ax.set_title(title, fontsize=16, pad=20)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    if log_scale: ax.set_yscale('log')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center') # va: vertical alignment

    plt.tight_layout(); plt.savefig(output_path); plt.close(fig)

def analyze_thermal_power(history, lander, config):
    """Calculates the average absorbed solar POWER for each face across all sun azimuths."""
    duration = history['time'][-1] if history['time'] else 1.0
    if duration < 1e-6: duration = 1.0 # Avoid division by zero for very short simulations
    
    thermal_power_tracker = { name: {'min_power': float('inf'), 'min_azimuth': -1, 'max_power': 0.0, 'max_azimuth': -1} for name in lander.face_geometry }
    dt = history['time'][1] - history['time'][0] if len(history['time']) > 1 else 0.01
    sc, alpha, elevation_rad = config['solar_constant_moon'], config['surface_solar_absorptivity'], np.deg2rad(config['sun_elevation_deg'])
    world_normals_history = { name: np.array([q.as_matrix() @ geo['normal'] for q in history['orientations']]) for name, geo in lander.face_geometry.items() }
    
    for azimuth_deg in range(360):
        azimuth_rad = np.deg2rad(azimuth_deg)
        sun_vector = np.array([ np.cos(elevation_rad) * np.cos(azimuth_rad), np.cos(elevation_rad) * np.sin(azimuth_rad), np.sin(elevation_rad) ])
        for name, geo in lander.face_geometry.items():
            cos_theta = np.dot(world_normals_history[name], sun_vector)
            cos_theta[cos_theta < 0] = 0
            total_energy_for_azimuth = np.sum(sc * geo['area'] * alpha * cos_theta * dt)
            average_power_for_azimuth = total_energy_for_azimuth / duration

            if average_power_for_azimuth < thermal_power_tracker[name]['min_power']:
                thermal_power_tracker[name]['min_power'] = average_power_for_azimuth
                thermal_power_tracker[name]['min_azimuth'] = azimuth_deg
            if average_power_for_azimuth > thermal_power_tracker[name]['max_power']:
                thermal_power_tracker[name]['max_power'] = average_power_for_azimuth
                thermal_power_tracker[name]['max_azimuth'] = azimuth_deg
                
    return thermal_power_tracker

def print_thermal_power_report(thermal_results):
    print("\n--- Thermal Analysis Min/Max Power Report ---"); print("=" * 80)
    print(f"{'Lander Face':<15} | {'Max Power (W)':<20} | {'Max Azimuth (deg)':<20} | {'Min Power (W)':<20}")
    print("-" * 80)
    for face, results in thermal_results.items():
        print(f"{face:<15} | {results['max_power']:<20.3f} | {results['max_azimuth']:<20} | {results['min_power']:<20.3f}")
    print("=" * 80)

def plot_thermal_power_reports(title_prefix, thermal_results, output_dir):
    """
    Creates two separate plots for the hot and cold thermal cases, showing average power.
    """
    labels = list(thermal_results.keys())
    
    # --- Hot Case Plot ---
    max_power_w = [res['max_power'] for res in thermal_results.values()]
    fig_hot, ax_hot = plt.subplots(figsize=(12, 8))
    bars_hot = ax_hot.bar(labels, max_power_w, color='orangered', alpha=0.8, label='Max Power', edgecolor='black')
    ax_hot.set_ylabel('Average Absorbed Solar Power (W)', fontsize=12)
    ax_hot.set_title(f'{title_prefix}: Maximum Solar Load (Hot Case)', fontsize=16, pad=20)
    ax_hot.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax_hot.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, bar in enumerate(bars_hot):
        power_val = bar.get_height()
        azimuth = thermal_results[labels[i]]['max_azimuth']
        ax_hot.text(bar.get_x() + bar.get_width() / 2, power_val, f"{power_val:.1f} W\n(@{azimuth}° Sun)", ha='center', va='bottom', fontsize=9, color='darkred')
        
    fig_hot.tight_layout()
    fig_hot.savefig(os.path.join(output_dir, "thermal_hot_case_report.png"))
    plt.close(fig_hot)

    # --- Cold Case Plot ---
    min_power_w = [res['min_power'] for res in thermal_results.values()]
    fig_cold, ax_cold = plt.subplots(figsize=(12, 8))
    bars_cold = ax_cold.bar(labels, min_power_w, color='deepskyblue', alpha=0.8, label='Min Power', edgecolor='black')
    ax_cold.set_ylabel('Average Absorbed Solar Power (W)', fontsize=12)
    ax_cold.set_title(f'{title_prefix}: Minimum Solar Load (Cold Case)', fontsize=16, pad=20)
    ax_cold.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax_cold.grid(axis='y', linestyle='--', alpha=0.7)

    for i, bar in enumerate(bars_cold):
        power_val = bar.get_height()
        azimuth = thermal_results[labels[i]]['min_azimuth']
        # Only add text if power is non-negligible
        if power_val > 0.01:
            ax_cold.text(bar.get_x() + bar.get_width() / 2, power_val, f"{power_val:.1f} W\n(@{azimuth}° Sun)", ha='center', va='bottom', fontsize=9, color='navy')
            
    fig_cold.tight_layout()
    fig_cold.savefig(os.path.join(output_dir, "thermal_cold_case_report.png"))
    plt.close(fig_cold)

def plot_reworked_polar(history, output_dir, start_pose_name, end_pose_name):
    from matplotlib.collections import LineCollection
    angles = {'Roll': history['roll'], 'Pitch': history['pitch'], 'Yaw': history['yaw']}
    time = np.array(history['time'])
    for name, data in angles.items():
        fig = plt.figure(figsize=(8, 8)); ax = fig.add_subplot(111, projection='polar')
        theta, r = np.deg2rad(data), time
        points = np.array([theta, r]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=plt.get_cmap('viridis_r'), norm=plt.Normalize(r.min(), r.max()))
        lc.set_array(r); lc.set_linewidth(2.5); line = ax.add_collection(lc)
        ax.plot(theta[0], r[0], 'go', markersize=12, label='Start', zorder=10, markeredgecolor='k')
        ax.plot(theta[-1], r[-1], 'rX', markersize=12, label='End', zorder=10, markeredgecolor='k')
        ax.set_title(f'{name} Trajectory: {start_pose_name} to {end_pose_name}', va='bottom', fontsize=16, pad=20)
        ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
        ax.set_xlabel('Angle (degrees)', labelpad=15, fontsize=12)
        ax.set_ylabel('Time (seconds)', labelpad=-60, fontsize=12)
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.15))
        cbar = fig.colorbar(line, ax=ax, orientation='vertical', fraction=0.046, pad=0.1)
        cbar.set_label('Time (s)', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"orientation_polar_{name.lower()}.png")); plt.close(fig)

def plot_torque_comparison(history, output_dir, start_pose, end_pose):
    """Creates a plot comparing commanded vs applied torque."""
    fig, ax = plt.subplots(figsize=(12, 7))
    time = history['time']
    
    ax.plot(time, history['commanded_torque'], label='Commanded Torque (from PID)', color='red', linestyle='--', linewidth=2.5)
    ax.plot(time, history['applied_torque'], label='Applied Torque (Physical Actuator)', color='blue', linewidth=2.5, alpha=0.8)
    
    ax.set_title(f'Torque Analysis: {start_pose} to {end_pose}', fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Torque Magnitude (Nm)', fontsize=12)
    ax.grid(True, which='both', linestyle=':', linewidth=0.5)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "torque_comparison.png"))
    plt.close(fig)

def analyze_solar_generation_scenarios(history, energy_consumed, config):
    duration = history['time'][-1] if history['time'] else 1.0
    if duration < 1e-6: duration = 1.0

    p_max = config['solar_constant_moon'] * config['solar_panel_area'] * config['solar_panel_efficiency']
    power_consumed_avg = energy_consumed / duration
    panel_normal_body = np.array([0., 0., 1.])
    dt = history['time'][1] - history['time'][0] if len(history['time']) > 1 else 0.01
    panel_normals_world = np.array([q.as_matrix() @ panel_normal_body for q in history['orientations']])
    results = []
    elevation_rad = np.deg2rad(config['sun_elevation_deg'])
    for azimuth_deg in range(360):
        azimuth_rad = np.deg2rad(azimuth_deg)
        sun_vector = np.array([np.cos(elevation_rad)*np.cos(azimuth_rad), np.cos(elevation_rad)*np.sin(azimuth_rad), np.sin(elevation_rad)])
        cos_theta = np.dot(panel_normals_world, sun_vector)
        cos_theta[cos_theta < 0] = 0
        energy_produced = np.sum(p_max * cos_theta) * dt
        power_produced_avg = energy_produced / duration
        results.append({'azimuth': azimuth_deg, 'produced_w': power_produced_avg, 'net_w': power_produced_avg - power_consumed_avg})
    return results, power_consumed_avg

def plot_solar_power_analysis(solar_results, power_consumed_avg, output_dir, start_pose_name, end_pose_name):
    azimuths, produced_w = [r['azimuth'] for r in solar_results], [r['produced_w'] for r in solar_results]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(azimuths, produced_w, label='Average Power Produced by Solar Panel', color='gold', linewidth=2.5)
    ax.axhline(y=power_consumed_avg, color='red', linestyle='--', linewidth=2, label=f'Average Power Consumed ({power_consumed_avg:.2f} W)')
    ax.fill_between(azimuths, produced_w, power_consumed_avg, where=(np.array(produced_w) >= power_consumed_avg), color='green', alpha=0.3, interpolate=True, label='Power Surplus')
    ax.fill_between(azimuths, produced_w, power_consumed_avg, where=(np.array(produced_w) < power_consumed_avg), color='red', alpha=0.3, interpolate=True, label='Power Deficit')
    ax.set_title(f'Solar Power Balance vs. Sun Azimuth ({start_pose_name} to {end_pose_name})', fontsize=16)
    ax.set_xlabel('Sun Azimuth (Degrees around Lander)', fontsize=12)
    ax.set_ylabel('Average Power (W)', fontsize=12)
    ax.set_xlim(0, 359); ax.set_xticks(np.arange(0, 361, 45)); ax.grid(True, linestyle=':'); ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "solar_power_balance.png")); plt.close(fig)

def plot_energy_balance(history, output_dir, start_pose_name, end_pose_name):
    fig, ax = plt.subplots(figsize=(12, 7)); time = history['time']
    ke, pe, total_mech = np.array(history['ke']), np.array(history['pe']), np.array(history['ke']) + np.array(history['pe'])
    consumed, dissipated = np.array(history['energy_consumed']), np.array(history['energy_dissipated'])
    net_system_energy = total_mech[0] + consumed - dissipated
    ax.plot(time, total_mech, label='Total Mechanical Energy (KE+PE)', color='blue', linewidth=2.5)
    ax.plot(time, net_system_energy, label='Expected System Energy (Initial + Consumed - Dissipated)', color='red', linestyle='--', linewidth=2.5)
    ax.plot(time, consumed, label='Cumulative Energy Consumed (Actuator)', color='green', linestyle=':', linewidth=2)
    ax.plot(time, dissipated, label='Cumulative Energy Dissipated (Ground Contact)', color='orange', linestyle=':', linewidth=2)
    ax.set_title(f'System Energy Balance: {start_pose_name} to {end_pose_name}', fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=12); ax.set_ylabel('Energy (Joules)', fontsize=12)
    ax.grid(True); ax.legend(fontsize=12); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "energy_balance.png")); plt.close(fig)

if __name__ == "__main__":
    # --- IMPORTANT: Set this to the failing configuration ---
    start_pose_name = 'UPSIDE_DOWN' 
    end_pose_name = 'UPRIGHT'
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    test_run_name = f"Run_Final_SwitchGeo_{start_pose_name}_to_{end_pose_name}_{timestamp}"
    base_save_path = './lander_simulation_results'
    output_dir = os.path.join(base_save_path, test_run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving figures and animation to: {output_dir}")

    lander_setup = LunarLander(SIMULATION_CONFIG)
    
    # Generate new analysis plots
    plot_force_diagram(lander_setup, output_dir)
    plot_all_landing_poses(lander_setup, output_dir)

    stable_poses, _ = define_all_poses(lander_setup) # Use this to get poses for simulation
    start_orientation, end_orientation = stable_poses[start_pose_name], stable_poses[end_pose_name]
    
    run_static_analysis_for_pose(lander_setup, start_pose_name, start_orientation, output_dir)

    print("\n--- Starting Automatic PID Optimization ---")
    bounds = [(0, 5000), (0, 1000), (0, 15000)]
    target_euler = R.from_matrix(end_orientation).as_euler('xyz', degrees=True)
    result = differential_evolution(
        objective_function, bounds, args=(start_orientation, end_orientation, target_euler, SIMULATION_CONFIG),
        strategy='best1bin', maxiter=2, popsize=20, tol=0.01,
        mutation=(0.5, 1), recombination=0.7, disp=True, workers=-1, polish=False
    )

    best_params = result.x
    print("\n--- Optimization Complete ---")
    print(f"Best Cost: {result.fun:.4f}")
    print(f"Optimal Parameters: Kp={best_params[0]:.2f}, Ki={best_params[1]:.2f}, Kd={best_params[2]:.2f}")

    print("\nRunning final simulation with optimal parameters...")
    final_history, proximity_results, energy_consumed, energy_dissipated = run_dynamic_simulation(
        start_orientation, end_orientation, *best_params, SIMULATION_CONFIG
    )

    # --- Post-simulation Analysis & Reporting ---
    thermal_results = analyze_thermal_power(final_history, lander_setup, SIMULATION_CONFIG)
    
    print_report("Maneuver Proximity Report (m)", proximity_results, unit="m")
    print_report("Energy Analysis (J)", {'Consumed': energy_consumed, 'Dissipated': energy_dissipated}, unit="J")
    print_thermal_power_report(thermal_results)
    
    # --- Plotting and Saving Results ---
    plot_bar_report(
        f"Closest Approach During Maneuver ({start_pose_name} to {end_pose_name})", 
        proximity_results, 
        "Minimum Distance to Ground (m)", 
        os.path.join(output_dir, "proximity_report.png")
    )
    plot_thermal_power_reports(
        f"Thermal Power Analysis ({start_pose_name} to {end_pose_name})", 
        thermal_results, 
        output_dir
    )
    
    plot_torque_comparison(final_history, output_dir, start_pose_name, end_pose_name)

    solar_results, power_consumed_avg = analyze_solar_generation_scenarios(final_history, energy_consumed, SIMULATION_CONFIG)
    plot_solar_power_analysis(solar_results, power_consumed_avg, output_dir, start_pose_name, end_pose_name)
    plot_reworked_polar(final_history, output_dir, start_pose_name, end_pose_name)
    plot_energy_balance(final_history, output_dir, start_pose_name, end_pose_name)

    fig_pid, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig_pid.suptitle(f'PID Controller Performance: {start_pose_name} to {end_pose_name}', fontsize=16)
    
    target_angles = R.from_matrix(end_orientation).as_euler('xyz', degrees=True)
    
    data = [final_history['roll'], final_history['pitch'], final_history['yaw']]
    labels = ['Roll','Pitch','Yaw']
    colors = ['#1f77b4','#ff7f0e','#2ca02c'] # Professional color palette
    target_lines = ['--', '-.', ':']

    for i, (d, l, c) in enumerate(zip(data, labels, colors)):
        axes[i].plot(final_history['time'], d, label=f'Actual {l}', color=c, linewidth=2)
        axes[i].axhline(y=target_angles[i], color='k', linestyle=target_lines[i], linewidth=1.5, label=f'Target {l} ({target_angles[i]:.1f}°)')
        axes[i].set_ylabel(f'{l} Angle (deg)', fontsize=12)
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].legend(fontsize=10)
    
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "pid_performance.png")); plt.close(fig_pid)

    # --- Animation ---
    print("\nGenerating and saving animation (this may take a moment)...")
    fig_anim = plt.figure(figsize=(10, 8))
    ax_anim = fig_anim.add_subplot(111, projection='3d')
    def update_anim(frame):
        ax_anim.cla()
        geometric_center = frame['p'] - (frame['o'] @ frame['c'])
        plot_lander_model(ax_anim, lander_setup, frame['o'], geometric_center, alpha=0.25)
        ax_anim.scatter(*frame['p'], c='red', s=60, label='Center of Mass (CoM)', depthshade=False)
        ax_anim.set_title(f'Optimized Reorientation: {start_pose_name} to {end_pose_name}', fontsize=16)
        ax_anim.legend()
    
    ani = animation.FuncAnimation(fig_anim, update_anim, frames=final_history['frames'], interval=50, blit=False, repeat=False)
    gif_path = os.path.join(output_dir, "reorientation_animation.gif")
    ani.save(gif_path, writer='pillow', fps=20)
    plt.close(fig_anim)
    print(f"Animation saved successfully to {gif_path}")