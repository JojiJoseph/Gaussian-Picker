# Basic OpenCV viewer with sliders for rotation and translation.
# Can be easily customizable to different use cases.
from dataclasses import dataclass
from typing import Literal, Optional
import torch
from gsplat import rasterization
import cv2
from scipy.spatial.transform import Rotation as scipyR
import warnings

from tqdm import tqdm

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




class Viewer:
    def __init__(self, splats):
        self.splats = None
        self.camera_matrix = None
        self.width = None
        self.height = None
        self.viewmat = None
        self.window_name = "Gaussian Picker"
        self._init_sliders()
        self._load_splats(splats)
        self.mouse_down = False
        self.mouse_x = 0
        self.mouse_y = 0
        self.contour = None
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
        world_to_pcd = np.eye(4)
        # world_to_pcd[:3, 3] = self.center_point
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
        return np.eye(
            4, dtype=np.float32
        )  # Placeholder for the world frame computation

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

            if self.contour is not None:
                cv2.drawContours(output_cv, [self.contour], -1, (0, 0, 255), 2)

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
        # if True:
        self.contour = None
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
        dummy_colors_grad = dummy_colors_grad.reshape(-1, 2048 * 8, 3)
        dummy_colors_grad = dummy_colors_grad.sum(dim=(1, 2))

        selected_object = torch.argmax(dummy_colors_grad)
        max_value = torch.max(dummy_colors_grad)  # .values
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Max value: {max_value}")
            print(f"Selected object index: {selected_object.item()}"),
            if max_value == 0:
                print("No object selected.")
                self.opacities = self.opacities_backup.clone()
            else:
                self.opacities = self.opacities_backup.clone()
                self.opacities[: selected_object * 2048 * 8] = 0.0
                self.opacities[(selected_object + 1) * 2048 * 8 :] = 0.0
        elif event == cv2.EVENT_MOUSEMOVE:
            if max_value == 0:
                self.countour = None
            else:
                opacities_copy = self.opacities.clone()
                opacities_copy[: selected_object * 2048 * 8] = 0.0
                opacities_copy[(selected_object + 1) * 2048 * 8 :] = 0.0
                colors_copy = self.colors.clone()
                colors_copy[:, :, :3] = 1.0
                output, _, _ = rasterization(
                    self.means,
                    self.quats,
                    self.scales,
                    opacities_copy,
                    colors_copy[:, 0, :],
                    self.viewmat[None],
                    self.camera_matrix[None],
                    width=self.width,
                    height=self.height,
                    # sh_degree=3,
                )

                del colors_copy, opacities_copy

                # Get contour of the rendered image
                output_cv = torch_to_cv(output[0])
                gray = cv2.cvtColor(output_cv, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    self.contour = max(contours, key=cv2.contourArea)
                    self.contour = self.contour.reshape(-1, 2)


def main():

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

    splats = {}
    splats["camera_matrix"] = camera_matrix
    splats["colmap_project"] = None
    splats["active_sh_degree"] = 3  # Default SH degree
    splats["means"] = torch.zeros((0, 3), device=device).float()
    splats["opacity"] = torch.zeros((0,), device=device).float()
    splats["rotation"] = torch.zeros((0, 4), device=device).float()
    splats["scaling"] = torch.zeros((0, 3), device=device).float()
    splats["features_dc"] = torch.zeros((0, 1, 3), device=device).float()
    splats["features_rest"] = torch.zeros((0, 15, 3), device=device).float()
    for i in tqdm(range(1, 13), desc="Loading assets"):
        asset_path = f"data/{i}/3dgs/seg_rgba.ply"
        splats_i = load_checkpoint(asset_path, None, "ply", 4)
        transform = scipyR.from_euler("xyz", [90, -180, 0], degrees=True).as_matrix()
        splats_i["means"] = (
            splats_i["means"] @ torch.tensor(transform, device=device).float()
            + torch.tensor([[0, 0, 2]], device=device).float()
        )
        quats = splats_i["rotation"]
        quats_to_R = torch.tensor(
            scipyR.from_quat(quats.cpu().numpy(), scalar_first=False).as_matrix(),
            device=device,
        ).float()
        R = quats_to_R @ torch.tensor(transform, device=device).float()
        quats = scipyR.from_matrix(R.cpu().numpy()).as_quat(scalar_first=False)
        splats_i["rotation"] = torch.tensor(quats, device=device).float()

        transform2 = scipyR.from_euler(
            "xyz", [0, i * 360 / 12, 0], degrees=True
        ).as_matrix()
        splats_i["means"] = (
            splats_i["means"] @ torch.tensor(transform2, device=device).float()
        )

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
                if key in splats:
                    splats[key] = torch.cat([splats[key], splats_i[key]], dim=0)
                else:
                    splats[key] = splats_i[key]
    splats["camera_matrix"] = camera_matrix



    viewer = Viewer(splats)
    viewer.run()


if __name__ == "__main__":
    main()
