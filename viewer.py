# Basic OpenCV viewer with sliders for rotation and translation.
# Can be easily customizable to different use cases.
from dataclasses import dataclass
from typing import Literal, Optional
import torch
from gsplat import rasterization
import cv2
from scipy.spatial.transform import Rotation as scipyR
import pycolmap_scene_manager as pycolmap
import warnings

import numpy as np
import json
import tyro

from utils import (
    get_rpy_matrix,
    get_viewmat_from_colmap_image,
    prune_by_gradients,
    torch_to_cv,
    load_checkpoint,
)

# Check if CUDA is available. Else raise an error.
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. Please install the correct version of PyTorch with CUDA support."
    )

device = torch.device("cuda")
torch.set_default_device("cuda")


@dataclass
class Args:
    checkpoint: str  # Path to the 3DGS checkpoint file (.pth/.pt) to be visualized.
    data_dir: (
        str  # Path to the COLMAP project directory containing sparse reconstruction.
    ) = None
    format: Optional[Literal["inria", "gsplat", "ply"]] = (
        "ply"  # Format of the checkpoint: 'inria' (original 3DGS), 'gsplat', or 'ply'.
    )
    rasterizer: Optional[Literal["inria", "gsplat"]] = (
        None  # [Deprecated] Use --format instead. Provided for backward compatibility.
    )
    data_factor: int = 4  # Downscaling factor for the renderings.
    turntable: bool = False  # Whether to use a turntable mode for the viewer.


@dataclass
class ViewerArgs:
    turntable: bool = False


class Viewer:
    def __init__(self, splats, viewer_args):
        self.splats = None
        self.camera_matrix = None
        self.width = None
        self.height = None
        self.viewmat = None
        self.window_name = "GSplat Explorer"
        self._init_sliders()
        self._load_splats(splats)
        self.turntable = viewer_args.turntable
        self.mouse_down = False
        self.mouse_x = 0
        self.mouse_y = 0
        cv2.setMouseCallback(self.window_name, self.handle_mouse_event)

    def _load_splats(self, splats):
        K = splats["camera_matrix"].cuda()
        width = int(K[0, 2] * 2)
        height = int(K[1, 2] * 2)

        means = splats["means"].float()
        opacities = splats["opacity"]
        quats = splats["rotation"]
        scales = splats["scaling"].float()

        opacities = torch.sigmoid(opacities)
        scales = torch.exp(scales)
        colors = torch.cat([splats["features_dc"], splats["features_rest"]], 1)

        self.splats = splats
        self.camera_matrix = K
        self.width = width
        self.height = height
        self.means = means
        self.opacities = opacities
        self.quats = quats
        self.scales = scales
        self.colors = colors

        self.opacities_backup = opacities.clone()

    def _init_sliders(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        trackbars = {
            "Roll": (-180, 180),
            "Pitch": (-180, 180),
            "Yaw": (-180, 180),
            "X": (-1000, 1000),
            "Y": (-1000, 1000),
            "Z": (-1000, 1000),
            "Scaling": (0, 100),
        }

        for name, (min_val, max_val) in trackbars.items():
            cv2.createTrackbar(name, self.window_name, 0, max_val, lambda x: None)
            cv2.setTrackbarMin(name, self.window_name, min_val)
            cv2.setTrackbarMax(name, self.window_name, max_val)

        cv2.setTrackbarPos(
            "Scaling", self.window_name, 100
        )  # Default value for scaling is 100

    def update_trackbars_from_viewmat(self, world_to_camera):
        # if torch tensor is passed, convert to numpy
        if isinstance(world_to_camera, torch.Tensor):
            world_to_camera = world_to_camera.cpu().numpy()
        r = scipyR.from_matrix(world_to_camera[:3, :3])
        roll, pitch, yaw = r.as_euler("xyz")
        cv2.setTrackbarPos("Roll", self.window_name, np.rad2deg(roll).astype(int))
        cv2.setTrackbarPos("Pitch", self.window_name, np.rad2deg(pitch).astype(int))
        cv2.setTrackbarPos("Yaw", self.window_name, np.rad2deg(yaw).astype(int))
        cv2.setTrackbarPos("X", self.window_name, int(world_to_camera[0, 3] * 100))
        cv2.setTrackbarPos("Y", self.window_name, int(world_to_camera[1, 3] * 100))
        cv2.setTrackbarPos("Z", self.window_name, int(world_to_camera[2, 3] * 100))

    def get_special_viewmat(self, viewmat, side="top"):
        if isinstance(viewmat, torch.Tensor):
            viewmat = viewmat.cpu().numpy()
        if not self.turntable:
            warnings.warn("Top view is only available in turntable mode.")
            return viewmat
        world_to_pcd = np.eye(4)

        # Trick: just put the new axes in columns, done!
        # world_to_pcd[:3, :3] = np.array(
        #     [
        #         self.view_direction,
        #         np.cross(self.upvector, self.view_direction),
        #         self.upvector,
        #     ]
        # ).T
        world_to_pcd[:3, 3] = self.center_point
        pcd_to_world = np.linalg.inv(world_to_pcd)

        world_to_camera = np.eye(4)
        if side == "top":
            world_to_camera[:3, :3] = np.array(
                [
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1],
                ]
            ).T
        elif side == "front":
            world_to_camera[:3, :3] = np.array(
                [
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0],
                ]
            ).T
        elif side == "right":
            world_to_camera[:3, :3] = np.array(
                [
                    [0, 0, -1],
                    [1, 0, 0],
                    [0, -1, 0],
                ]
            ).T
        else:
            warnings.warn(f"Unknown view type: {side}.")

        world_to_camera_before = viewmat @ world_to_pcd
        dist = np.linalg.norm(world_to_camera_before[:3, 3])
        world_to_camera[:3, 3] = np.array([0, 0, dist])

        # cam_point = world_to_camera @ pcd_to_world @ pcd_coord
        # cam_point = viewmat @ pcd_coord
        # viewmat = world_to_camera @ pcd_to_world
        viewmat = world_to_camera @ pcd_to_world
        viewmat = torch.tensor(viewmat).float().to(device)
        return viewmat

    def _get_viewmat_from_trackbars(self):
        roll = cv2.getTrackbarPos("Roll", self.window_name)
        pitch = cv2.getTrackbarPos("Pitch", self.window_name)
        yaw = cv2.getTrackbarPos("Yaw", self.window_name)

        roll_rad = np.deg2rad(roll)
        pitch_rad = np.deg2rad(pitch)
        yaw_rad = np.deg2rad(yaw)

        viewmat = (
            torch.tensor(get_rpy_matrix(roll_rad, pitch_rad, yaw_rad))
            .float()
            .to(device)
        )

        viewmat[0, 3] = cv2.getTrackbarPos("X", self.window_name) / 100.0
        viewmat[1, 3] = cv2.getTrackbarPos("Y", self.window_name) / 100.0
        viewmat[2, 3] = cv2.getTrackbarPos("Z", self.window_name) / 100.0

        return viewmat

    def render_gaussians(self, viewmat, scaling, anaglyph=False):
        # print(self.colors.shape, self.means.shape, self.quats.shape, self.scales.shape)
        output, _, _ = rasterization(
            self.means,
            self.quats,
            self.scales * scaling,
            self.opacities,
            self.colors,
            viewmat[None],
            self.camera_matrix[None],
            width=self.width,
            height=self.height,
            sh_degree=3,
        )
        if not anaglyph:
            return np.ascontiguousarray(torch_to_cv(output[0]))
        left = torch_to_cv(output[0])
        viewmat_right_eye = viewmat.clone()
        viewmat_right_eye[0, 3] -= 0.05  # Offset for the right eye
        output, _, _ = rasterization(
            self.means,
            self.quats,
            self.scales * scaling,
            self.opacities,
            self.colors,
            viewmat_right_eye[None],
            self.camera_matrix[None],
            width=self.width,
            height=self.height,
            sh_degree=3,
        )
        right = torch_to_cv(output[0])
        left_copy = left.copy()
        right_copy = right.copy()
        left_copy[..., :2] = 0  # Set left eye's red and green channels to zero
        right_copy[..., -1] = 0  # Set right eye's blue channel to zero
        return (
            left_copy + right_copy,
            np.ascontiguousarray(left_copy),
            np.ascontiguousarray(right_copy),
        )

    def compute_world_frame(self):
        return np.eye(4, dtype=np.float32)  # Placeholder for the world frame computation
        """
        Compute the new world frame (center_point, upvector, view_direction, ortho_direction)
        based on the average camera positions and orientations.
        """
        # Initialize vectors
        center_point = np.zeros(3, dtype=np.float32)
        upvector_sum = np.zeros(3, dtype=np.float32)
        view_direction_sum = np.zeros(3, dtype=np.float32)

        # Iterate over camera images to compute average position and orientation
        for image in self.splats["colmap_project"].images.values():
            viewmat = get_viewmat_from_colmap_image(image)
            viewmat_np = viewmat.cpu().numpy()
            c2w = np.linalg.inv(viewmat_np)
            center_point += c2w[:3, 3].squeeze()  # camera position
            upvector_sum += -c2w[:3, 1].squeeze()  # up direction
            view_direction_sum += c2w[:3, 2].squeeze()  # viewing direction

        # Average position and orientation vectors
        num_images = len(self.splats["colmap_project"].images)
        center_point /= num_images
        upvector = upvector_sum / np.linalg.norm(upvector_sum)
        view_direction = view_direction_sum / np.linalg.norm(view_direction_sum)

        # Make view_direction orthogonal to upvector
        view_direction -= upvector * np.dot(view_direction, upvector)
        view_direction /= np.linalg.norm(view_direction)

        # Compute the orthogonal direction (right vector)
        ortho_direction = np.cross(upvector, view_direction)
        ortho_direction /= np.linalg.norm(ortho_direction)

        # Optionally override center_point with the mean of your 3D data
        center_point = torch.mean(self.means, dim=0).cpu().numpy()

        # Save the computed frame vectors as attributes
        self.center_point = center_point
        self.upvector = upvector
        self.view_direction = view_direction
        self.ortho_direction = ortho_direction

    def visualize_world_frame(self, output_cv, viewmat):
        viewmat_np = viewmat.cpu().numpy()
        T = np.eye(4)
        z_axis = self.upvector
        x_axis = self.view_direction
        y_axis = np.cross(z_axis, x_axis)
        T[:3, :3] = np.array([x_axis, y_axis, z_axis]).T
        T[:3, 3] = self.center_point
        T = viewmat_np @ T
        rvec = cv2.Rodrigues(T[:3, :3])[0]
        tvec = T[:3, 3]
        cv2.drawFrameAxes(
            output_cv,
            self.camera_matrix.cpu().numpy(),
            None,
            rvec,
            tvec,
            length=1,
            thickness=2,
        )

    def run(self):
        """Run the interactive Gaussian Splat viewer loop once until exit."""
        self.show_anaglyph = False
        self.compute_world_frame()

        while True:
            scaling = cv2.getTrackbarPos("Scaling", self.window_name) / 100.0
            viewmat = self._get_viewmat_from_trackbars()

            if self.show_anaglyph:
                output_cv, _, _ = self.render_gaussians(viewmat, scaling, anaglyph=True)
            else:
                output_cv = self.render_gaussians(viewmat, scaling)

            if self.turntable:
                self.visualize_world_frame(output_cv, viewmat)

            cv2.imshow(self.window_name, output_cv)
            full_key = cv2.waitKeyEx(1)
            key = full_key & 0xFF

            should_continue = self.handle_key_press(key, {"viewmat": viewmat})
            if not should_continue:
                break

        cv2.destroyAllWindows()

    def handle_key_press(self, key, data):
        viewmat = data["viewmat"]
        if key == ord("q") or key == 27:
            return False  # Exit the viewer
        if key == ord("3"):
            self.show_anaglyph = not self.show_anaglyph
        if key in [ord("w"), ord("a"), ord("s"), ord("d")]:
            # Modify viewmat and sync UI
            delta = 0.1
            if key == ord("w"):
                viewmat[2, 3] -= delta
            elif key == ord("s"):
                viewmat[2, 3] += delta
            elif key == ord("a"):
                viewmat[0, 3] += delta
            elif key == ord("d"):
                viewmat[0, 3] -= delta
            self.update_trackbars_from_viewmat(viewmat)
        if key in [ord("7")]:
            viewmat = self.get_special_viewmat(viewmat, side="top")
            self.update_trackbars_from_viewmat(viewmat)
        elif key in [ord("8")]:
            viewmat = self.get_special_viewmat(viewmat, side="front")
            self.update_trackbars_from_viewmat(viewmat)
        elif key in [ord("9")]:
            viewmat = self.get_special_viewmat(viewmat, side="right")
            self.update_trackbars_from_viewmat(viewmat)
        return True  # Continue the viewer loop

    def handle_mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            screen_dummy = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # Draw x, y coordinates on the screen
            cv2.circle(screen_dummy, (x, y), 5, (0, 255, 0), -1)
            screen_dummy = torch.tensor(screen_dummy).float().to(device) / 255.0
            dummy_colors = torch.zeros((len(self.means), 3), device=device)
            dummy_colors.requires_grad = True
            self.viewmat = self._get_viewmat_from_trackbars()
            output, _, _ = rasterization(
                self.means,
                self.quats,
                self.scales,
                self.opacities,
                dummy_colors,
                self.viewmat[None],
                self.camera_matrix[None],
                width=self.width,
                height=self.height,
                # sh_degree=3,
            )
            pseudo_loss = output[0] * screen_dummy
            pseudo_loss = pseudo_loss.sum()
            pseudo_loss.backward()
            dummy_colors_grad = dummy_colors.grad
            dummy_colors_grad = dummy_colors_grad.reshape(-1, 2048*8,3)
            dummy_colors_grad = dummy_colors_grad.sum(dim=(1,2))
            print(f"Dummy colors grad shape: {dummy_colors_grad.shape}")
            selected_object = torch.argmax(dummy_colors_grad)
            max_value = torch.max(dummy_colors_grad)#.values
            print(f"Max value: {max_value}")
            print(f"Selected object index: {selected_object.item()}"), 
            if max_value == 0:
                print("No object selected.")
                self.opacities = self.opacities_backup.clone()
            else:
                self.opacities = self.opacities_backup.clone()
                self.opacities[:selected_object*2048*8] = 0.0
                self.opacities[(selected_object+1)*2048*8:] = 0.0
            print(dummy_colors_grad.shape)
        return
        if not self.turntable:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_down = True
            self.mouse_x = x
            self.mouse_y = y
            self.view_mat_progress = self._get_viewmat_from_trackbars()
            self.is_alt_pressed = flags & cv2.EVENT_FLAG_ALTKEY
            self.is_shift_pressed = flags & cv2.EVENT_FLAG_SHIFTKEY
            self.is_ctrl_pressed = flags & cv2.EVENT_FLAG_CTRLKEY
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_down = False
        elif event == cv2.EVENT_MOUSEMOVE and self.mouse_down:
            dx = x - self.mouse_x
            dy = y - self.mouse_y

            if self.is_ctrl_pressed:
                viewmat = self._get_viewmat_from_trackbars()
                viewmat[2, 3] += dy / self.height * 10  # Move camera forward/backward
                self.update_trackbars_from_viewmat(viewmat)
                self.mouse_x = x
                self.mouse_y = y
                return

            # viewmat = self._get_viewmat_from_trackbars()
            viewmat = self.view_mat_progress.clone()
            viewmat_np = viewmat.cpu().numpy()  # w2c
            world_to_pcd = np.eye(4)
            world_to_pcd[:3, :3] = np.array(
                [
                    self.view_direction,
                    np.cross(self.upvector, self.view_direction),
                    self.upvector,
                ]
            ).T
            world_to_pcd[:3, 3] = self.center_point
            pcd_to_world = np.linalg.inv(world_to_pcd)
            # camera_coordinates = viewmat @ world_to_pcd @ transform @ pcd_to_world @ pcd_coods
            # camera_coordinates = viewmat_new @ pcd_coords
            # ie. viewmat_new = viewmat @ world_to_pcd @ transform @ pcd_to_world
            transform = np.eye(4)
            height, width = self.height, self.width
            if self.is_shift_pressed:
                viewmat_np[0, 3] += dx / width * 10
                viewmat_np[1, 3] += dy / height * 10
            else:
                # Rotation of the world
                c2pcd = np.linalg.inv(viewmat_np)
                c2w = pcd_to_world @ c2pcd
                direction_with_respect_to_world = -c2w[:3, 2]
                lambda_ = -c2w[2, 3] / direction_with_respect_to_world[2]
                intersection_point = (
                    c2w[:3, 3] + lambda_ * direction_with_respect_to_world
                )

                world_to_intersection = np.eye(4)
                world_to_intersection[:3, 3] = -intersection_point
                intersection_to_world = np.linalg.inv(world_to_intersection)
                transform = get_rpy_matrix(0, 0, dx / width * 10)
                if self.is_alt_pressed:
                    world_to_intersection = np.eye(4)
                    intersection_to_world = np.eye(4)

                # rotation of camera
                viewmat_np[:3, :3] = (
                    get_rpy_matrix(dy / height * 10, 0, 0)[:3, :3] @ viewmat_np[:3, :3]
                )

                # rotation of the world
                viewmat_np = (
                    viewmat_np
                    @ world_to_pcd
                    @ intersection_to_world
                    @ transform
                    @ world_to_intersection
                    @ pcd_to_world
                )


            self.update_trackbars_from_viewmat(
                torch.tensor(viewmat_np).float().to(device)
            )


def create_sphere(radius, color=np.array([1, 0, 0]), offset=np.array([0, 0, 0])):
    # Sample 1000 points on a sphere
    phi = np.linspace(0, np.pi, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    phi, theta = np.meshgrid(phi, theta)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    means = torch.tensor(np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)).float().to(device)
    means += torch.tensor(offset, device=device).float()  # Apply offset
    opacities = torch.ones(means.shape[0], device=device)
    quats = torch.tensor(np.random.rand(means.shape[0], 4)).float().to(device)
    quats /= quats.norm(dim=1, keepdim=True)  # Normalize quaternions
    surface_area = 4 * np.pi * radius**2
    area_per_point = surface_area / means.shape[0]
    scales = torch.ones(means.shape[0],3, device=device) * (area_per_point ** (1 / 3))
    colors = torch.tensor(np.tile(color, (means.shape[0], 1)), device=device).float()
    fox = 90 # Field of view in degrees
    fy = 90
    height, width = 512, 512
    focal_length = 0.5 * width / np.tan(np.deg2rad(fox) / 2)
    camera_matrix = torch.tensor(
        [
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1],
        ],
        device=device,
    ).float()
    splats = {
        "means": means,
        "opacity": opacities,
        "rotation": quats,
        "scaling": scales,
        "features_dc": colors[:, :3],
        "features_rest": colors[:, 3:],
        "camera_matrix": camera_matrix
        # "colmap_project": pycolmap.Project(),
    }
    return splats

def create_cube(size=0.2, color=np.array([1, 0, 0]), offset=np.array([0, 0, 0])):
    # Create a cube centered at the origin with the specified size
    half_size = size / 2
    means = torch.tensor(
        [
            [-half_size, -half_size, -half_size],
            [half_size, -half_size, -half_size],
            [half_size, half_size, -half_size],
            [-half_size, half_size, -half_size],
            [-half_size, -half_size, half_size],
            [half_size, -half_size, half_size],
            [half_size, half_size, half_size],
            [-half_size, half_size, half_size],
        ],
        device=device,
    ).float() + torch.tensor(offset, device=device).float()

    # Sample 1000 points on the cube surface
    faces = torch.tensor(
        [
            [0, 1, 2, 3],  # Bottom face
            [4, 5, 6, 7],  # Top face
            [0, 1, 5, 4],  # Front face
            [2, 3, 7, 6],  # Back face
            [0, 3, 7, 4],  # Left face
            [1, 2, 6, 5],  # Right face
        ],
        device=device,
    )
    means = means[faces].view(-1, 3)  # Flatten the faces into a single list of points
    means += torch.tensor(offset, device=device).float()  # Apply offset
    print(f"Means shape: {means.shape}")
    exit()

    opacities = torch.ones(means.shape[0], device=device)
    quats = torch.tensor(np.random.rand(means.shape[0], 4)).float().to(device)
    quats /= quats.norm(dim=1, keepdim=True)  # Normalize quaternions
    scales = torch.ones(means.shape[0], 3, device=device) * (size ** (1 / 3))
    colors = torch.tensor(np.tile(color, (means.shape[0], 1)), device=device).float()
    fox = 90  # Field of view in degrees
    fy = 90
    height, width = 512, 512
    focal_length = 0.5 * width / np.tan(np.deg2rad(fox) / 2)
    camera_matrix = torch.tensor(
        [
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1],
        ],
        device=device,
    ).float()
    
    splats = {
        "means": means,
        "opacity": opacities,
        "rotation": quats,
        "scaling": scales,
        "features_dc": colors[:, :3],
        "features_rest": colors[:, 3:],
        "camera_matrix": camera_matrix
        # "colmap_project": pycolmap.Project(),
    }
    return splats

def main(args: Args):
    format = args.format or args.rasterizer
    if args.rasterizer:
        warnings.warn(
            "`rasterizer` is deprecated. Use `format` instead.", DeprecationWarning
        )
    if not format:
        raise ValueError("Must specify --format or the deprecated --rasterizer")

    splats = load_checkpoint(args.checkpoint, args.data_dir, format, args.data_factor)
    fov = 90  # Field of view in degrees
    fy = 90
    height, width = 512, 512
    focal_length = 0.5 * width / np.tan(np.deg2rad(fov) / 2)
    camera_matrix = torch.tensor(
        [
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1],
        ],
        device=device,
    ).float()
    from scipy.spatial.transform import Rotation as scipyR
    # Make params of zero length but with the right shape
    for key in splats.keys():
        if key not in ["camera_matrix", "colmap_project", "active_sh_degree"]:
            print(f"Key: {key}")
            splats[key] = torch.zeros(
                (0, *splats[key].shape[1:]), device=device
            ).float()
    from tqdm import tqdm
    for i in tqdm(range(1,13)):
        asset_path = f"data/{i}/3dgs/seg_rgba.ply"
        splats_i = load_checkpoint(
            asset_path, args.data_dir, format, args.data_factor
        )
        transform = scipyR.from_euler("xyz", [90, -180, 0], degrees=True).as_matrix()
        splats_i["means"] = splats_i["means"] @ torch.tensor(transform, device=device).float() + torch.tensor([[0,0,2]], device=device).float()
        quats = splats_i["rotation"]
        quats_to_R = torch.tensor(
            scipyR.from_quat(quats.cpu().numpy(), scalar_first=False).as_matrix(), device=device
        ).float()
        R = quats_to_R @ torch.tensor(transform, device=device).float()
        quats = scipyR.from_matrix(R.cpu().numpy()).as_quat(scalar_first=False)
        splats_i["rotation"] = torch.tensor(quats, device=device).float()
        print(f"R shape: {R.shape}")
        transform2 = scipyR.from_euler("xyz", [0, i*360/12, 0], degrees=True).as_matrix()
        splats_i["means"] = splats_i["means"] @ torch.tensor(transform2, device=device).float()

        quats = splats_i["rotation"]
        quats_to_R = torch.tensor(
            scipyR.from_quat(quats.cpu().numpy()).as_matrix(), device=device
        ).float()
        R = quats_to_R @ torch.tensor(transform2, device=device).float()
        quats = scipyR.from_matrix(R.cpu().numpy()).as_quat()
        splats_i["rotation"] = torch.tensor(quats, device=device).float()
        # Concatenate the splats
        for key in splats_i.keys():
            if key not in ["camera_matrix", "colmap_project", "active_sh_degree"]:
                print(key)
                if key in splats:
                    splats[key] = torch.cat([splats[key], splats_i[key]], dim=0)
                else:
                    splats[key] = splats_i[key]
        print(f"Transform shape: {transform.shape}")
    # exit()
    # splats = splats_i
    splats["camera_matrix"] = camera_matrix
    # # splats = prune_by_gradients(splats)
    # splats = create_sphere(0.2, color=np.array([1, 0, 0]), offset=np.array([0, 0, 0]))
    # splats = create_cube(0.2, color=np.array([1, 0, 0]), offset=np.array([0, 0, 0]))

    if args.turntable:
        viewer_args = ViewerArgs(turntable=True)
    else:
        viewer_args = ViewerArgs(turntable=False)

    viewer = Viewer(splats, viewer_args=viewer_args)
    viewer.run()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
