from typing import Any, List, Optional
import cv2
import threading
import os
import torch
import numpy as np

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.typing import Frame, Face
from modules.utilities import conditional_download, resolve_relative_path, is_image, is_video
from modules.face_analyser import get_one_face, get_many_faces, get_one_face_left, get_one_face_right, get_face_analyser
from modules.processors.frame.face_swapper import crop_face_region, create_adjusted_face, reset_face_tracking

# Model instances
GFPGAN_ENHANCER: Any = None
CODEFORMER_ENHANCER: Any = None
GPEN_ENHANCER: Any = None
THREAD_LOCK = threading.Lock()
NAME = 'DLC.FACE-ENHANCER'

# Model URLs
MODEL_URLS = {
    'gfpgan': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth',
    'codeformer': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
    'gpen': 'https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-512.pth'
}


def pre_check() -> bool:
    """Download required model weights based on selected enhancer."""
    download_directory_path = resolve_relative_path('..\models') if os.name == 'nt' else resolve_relative_path('../models')

    # Always download GFPGAN (default)
    conditional_download(download_directory_path, [MODEL_URLS['gfpgan']])

    # Download other models if selected
    model = modules.globals.face_enhancer_model
    if model == 'codeformer':
        conditional_download(download_directory_path, [MODEL_URLS['codeformer']])
    elif model == 'gpen':
        conditional_download(download_directory_path, [MODEL_URLS['gpen']])

    return True


def pre_start() -> bool:
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def get_model_path(model_name: str) -> str:
    """Get the path to a model file."""
    if os.name == 'nt':
        base_path = resolve_relative_path('..\models')
    else:
        base_path = resolve_relative_path('../models')

    model_files = {
        'gfpgan': 'GFPGANv1.4.pth',
        'codeformer': 'codeformer.pth',
        'gpen': 'GPEN-BFR-512.pth'
    }
    return os.path.join(base_path, model_files.get(model_name, 'GFPGANv1.4.pth'))


def get_gfpgan_enhancer() -> Any:
    """Load GFPGAN model."""
    global GFPGAN_ENHANCER

    with THREAD_LOCK:
        if GFPGAN_ENHANCER is None:
            import gfpgan
            model_path = get_model_path('gfpgan')
            GFPGAN_ENHANCER = gfpgan.GFPGANer(model_path=model_path, upscale=1)
    return GFPGAN_ENHANCER


def get_codeformer_enhancer() -> Any:
    """Load CodeFormer model."""
    global CODEFORMER_ENHANCER

    with THREAD_LOCK:
        if CODEFORMER_ENHANCER is None:
            try:
                from basicsr.archs.codeformer_arch import CodeFormer

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model_path = get_model_path('codeformer')

                net = CodeFormer(
                    dim_embd=512,
                    codebook_size=1024,
                    n_head=8,
                    n_layers=9,
                    connect_list=['32', '64', '128', '256']
                ).to(device)

                checkpoint = torch.load(model_path, map_location=device)
                net.load_state_dict(checkpoint['params_ema'])
                net.eval()

                CODEFORMER_ENHANCER = {'net': net, 'device': device}
            except ImportError:
                print("CodeFormer not available. Install with: pip install basicsr")
                return None
    return CODEFORMER_ENHANCER


def get_gpen_enhancer() -> Any:
    """Load GPEN model."""
    global GPEN_ENHANCER

    with THREAD_LOCK:
        if GPEN_ENHANCER is None:
            try:
                # GPEN requires custom integration - using a simplified approach
                # For full GPEN support, clone the GPEN repository
                from basicsr.archs.rrdbnet_arch import RRDBNet

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model_path = get_model_path('gpen')

                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=device)
                    GPEN_ENHANCER = {'checkpoint': checkpoint, 'device': device}
                else:
                    print(f"GPEN model not found at {model_path}")
                    return None
            except Exception as e:
                print(f"GPEN loading error: {e}")
                return None
    return GPEN_ENHANCER


def get_face_enhancer() -> Any:
    """Get the appropriate face enhancer based on global setting."""
    model = modules.globals.face_enhancer_model

    if model == 'codeformer':
        enhancer = get_codeformer_enhancer()
        if enhancer is not None:
            return enhancer
        # Fallback to GFPGAN
        print("Falling back to GFPGAN")
        modules.globals.face_enhancer_model = 'gfpgan'

    elif model == 'gpen':
        enhancer = get_gpen_enhancer()
        if enhancer is not None:
            return enhancer
        # Fallback to GFPGAN
        print("Falling back to GFPGAN")
        modules.globals.face_enhancer_model = 'gfpgan'

    return get_gfpgan_enhancer()


def enhance_face_gfpgan(temp_frame: Frame) -> Frame:
    """Enhance face using GFPGAN."""
    enhancer = get_gfpgan_enhancer()
    _, restored_faces, _ = enhancer.enhance(temp_frame, paste_back=False)
    if restored_faces is not None and len(restored_faces) > 0:
        return restored_faces[0]
    return temp_frame


def enhance_face_codeformer(temp_frame: Frame, w: float = 0.5) -> Frame:
    """Enhance face using CodeFormer.

    Args:
        temp_frame: Input face image
        w: Fidelity weight (0=quality, 1=fidelity). Default 0.5
    """
    enhancer = get_codeformer_enhancer()
    if enhancer is None:
        return enhance_face_gfpgan(temp_frame)

    net = enhancer['net']
    device = enhancer['device']

    try:
        # Prepare input
        h, w_orig = temp_frame.shape[:2]

        # Resize to 512x512 for CodeFormer
        face_input = cv2.resize(temp_frame, (512, 512), interpolation=cv2.INTER_LINEAR)
        face_input = face_input.astype(np.float32) / 255.0
        face_input = (face_input - 0.5) / 0.5  # Normalize to [-1, 1]
        face_input = face_input.transpose(2, 0, 1)  # HWC to CHW
        face_input = torch.from_numpy(face_input).unsqueeze(0).to(device)

        with torch.no_grad():
            output = net(face_input, w=0.5, adain=True)[0]

        # Post-process
        output = output.squeeze(0).cpu().numpy()
        output = output.transpose(1, 2, 0)  # CHW to HWC
        output = (output * 0.5 + 0.5) * 255.0
        output = output.clip(0, 255).astype(np.uint8)

        # Resize back to original size
        if output.shape[0] != h or output.shape[1] != w_orig:
            output = cv2.resize(output, (w_orig, h), interpolation=cv2.INTER_LINEAR)

        return output
    except Exception as e:
        print(f"CodeFormer error: {e}")
        return enhance_face_gfpgan(temp_frame)


def enhance_face_gpen(temp_frame: Frame) -> Frame:
    """Enhance face using GPEN.

    Note: Full GPEN requires the official repository.
    This is a simplified fallback implementation.
    """
    enhancer = get_gpen_enhancer()
    if enhancer is None:
        return enhance_face_gfpgan(temp_frame)

    # GPEN requires more complex setup - fallback to GFPGAN for now
    # For full GPEN support, integrate the official GPEN repository
    print("GPEN full support requires official repository. Using GFPGAN.")
    return enhance_face_gfpgan(temp_frame)


def enhance_face(temp_frame: Frame) -> Frame:
    """Enhance face using the selected model."""
    model = modules.globals.face_enhancer_model

    if model == 'codeformer':
        return enhance_face_codeformer(temp_frame)
    elif model == 'gpen':
        return enhance_face_gpen(temp_frame)
    else:
        return enhance_face_gfpgan(temp_frame)


def process_frame(source_face: Optional[Face], temp_frame: Frame, detected_faces: Optional[list] = None) -> Frame:
    # Use pre-detected faces if provided (live mode optimization)
    if detected_faces is not None:
        all_faces = detected_faces
    else:
        face_analyser = get_face_analyser()
        try:
            all_faces = face_analyser.get(temp_frame)
        except Exception as e:
            # If face detection fails, return the original frame without processing
            return temp_frame

    # Determine which faces to process based on user settings
    if modules.globals.many_faces:
        # If 'many_faces' is enabled, process all detected faces
        # Sort faces from left to right based on their bounding box x-coordinate
        target_faces = sorted(all_faces, key=lambda face: face.bbox[0])
    elif modules.globals.both_faces:
        # If 'both_faces' is enabled, process two faces
        if modules.globals.detect_face_right:
            # If 'detect_face_right' is enabled, sort faces from right to left and take the two rightmost faces
            target_faces = sorted(all_faces, key=lambda face: -face.bbox[0])[:2]
        else:
            # Otherwise, sort faces from left to right and take the two leftmost faces
            target_faces = sorted(all_faces, key=lambda face: face.bbox[0])[:2]
    else:
        if modules.globals.detect_face_right:
            # Select the rightmost face if 'detect_face_right' is enabled
            target_faces = [max(all_faces, key=lambda face: face.bbox[0])] if all_faces else []
        else:
            # Otherwise, select the leftmost face
            target_faces = [min(all_faces, key=lambda face: face.bbox[0])] if all_faces else []

    # Limit the number of faces to process if not in 'many_faces' mode
    if modules.globals.many_faces is False:
        # Limit to max two faces if both_faces is True, otherwise just one face
        max_faces = 2 if modules.globals.both_faces else 1
        target_faces = target_faces[:max_faces]

    for i, target_face in enumerate(target_faces):
        # Crop the face region
        cropped_frame, crop_info = crop_face_region(temp_frame, target_face, 0.2)

        enhanced_frame = enhance_face(cropped_frame)

        # Paste the swapped region back into the original frame
        x, y, w, h = crop_info

        # Ensure dimensions match before pasting
        if enhanced_frame.shape[0] != h or enhanced_frame.shape[1] != w:
            enhanced_frame = cv2.resize(enhanced_frame, (w, h))

        temp_frame[y:y+h, x:x+w] = enhanced_frame

    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], progress: Any = None) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(None, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if progress:
            progress.update(1)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    result = process_frame(None, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    modules.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
