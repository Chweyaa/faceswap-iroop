from typing import Any, List, Optional, Dict, Tuple
import cv2
import threading
import gfpgan
import os

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face
from modules.typing import Frame, Face
from modules.utilities import conditional_download, resolve_relative_path, is_image, is_video
from modules.face_analyser import get_one_face, get_many_faces, get_one_face_left, get_one_face_right, get_face_analyser
from modules.processors.frame.face_swapper import crop_face_region, create_adjusted_face, create_edge_blur_mask, blend_with_mask, reset_face_tracking
from modules.profiler import profile_section

FACE_ENHANCER = None
THREAD_LOCK = threading.Lock()
NAME = 'DLC.FACE-ENHANCER'

# Cache for edge blur masks to avoid recreating them for same-sized faces
_MASK_CACHE: Dict[Tuple[int, int, int], Any] = {}
_MASK_CACHE_MAX_SIZE = 50

# dowload GFPAN
def pre_check() -> bool:
    download_directory_path = resolve_relative_path('..\models')
    conditional_download(download_directory_path, ['https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'])
    return True


def pre_start() -> bool:
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def get_face_enhancer() -> Any:
    global FACE_ENHANCER

    # Double-check locking pattern: avoid lock overhead when model already loaded
    if FACE_ENHANCER is not None:
        return FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            if os.name == 'nt':
                model_path = resolve_relative_path('..\models\GFPGANv1.4.pth')
            else:
                model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            with profile_section('load_face_enhancer_model'):
                FACE_ENHANCER = gfpgan.GFPGANer(model_path=model_path, upscale=1)  # type: ignore[attr-defined]
            print("GFPGAN model loaded")
    return FACE_ENHANCER


def get_cached_edge_blur_mask(shape: Tuple[int, ...], blur_amount: int = 30) -> Any:
    """Get or create a cached edge blur mask for the given shape."""
    global _MASK_CACHE

    cache_key = (shape[0], shape[1], blur_amount)

    if cache_key in _MASK_CACHE:
        return _MASK_CACHE[cache_key]

    # Create new mask
    mask = create_edge_blur_mask(shape, blur_amount=blur_amount)

    # Cache management: remove oldest entries if cache is too large
    if len(_MASK_CACHE) >= _MASK_CACHE_MAX_SIZE:
        # Remove first (oldest) entry
        first_key = next(iter(_MASK_CACHE))
        del _MASK_CACHE[first_key]

    _MASK_CACHE[cache_key] = mask
    return mask


def enhance_face(temp_frame: Frame) -> Frame:
    # Remove THREAD_SEMAPHORE to allow parallel GPU processing
    # Set paste_back=False to skip redundant face detection inside GFPGAN
    with profile_section('face_enhance_gfpgan'):
        _, restored_faces, _ = get_face_enhancer().enhance(
            temp_frame,
            paste_back=False
        )
    # Return the first (and only) restored face from the cropped input
    if restored_faces is not None and len(restored_faces) > 0:
        return restored_faces[0]
    return temp_frame


def process_frame(source_face: Face, temp_frame: Frame, pre_detected_faces: Optional[List[Face]] = None) -> Frame:
    """
    Process a frame with face enhancement.

    Args:
        source_face: Source face (unused, kept for API compatibility)
        temp_frame: The frame to process
        pre_detected_faces: Optional list of pre-detected faces to skip redundant detection
    """
    # Use pre-detected faces if provided, otherwise detect faces
    if pre_detected_faces is not None:
        all_faces = pre_detected_faces
    else:
        face_analyser = get_face_analyser()
        try:
            all_faces = face_analyser.get(temp_frame)
        except Exception as e:
            return temp_frame

    if not all_faces:
        return temp_frame

    # Determine which faces to process based on user settings
    if modules.globals.many_faces:
        target_faces = sorted(all_faces, key=lambda face: face.bbox[0])
    elif modules.globals.both_faces:
        if modules.globals.detect_face_right:
            target_faces = sorted(all_faces, key=lambda face: -face.bbox[0])[:2]
        else:
            target_faces = sorted(all_faces, key=lambda face: face.bbox[0])[:2]
    else:
        if modules.globals.detect_face_right:
            target_faces = [max(all_faces, key=lambda face: face.bbox[0])] if all_faces else []
        else:
            target_faces = [min(all_faces, key=lambda face: face.bbox[0])] if all_faces else []

    # Limit the number of faces to process if not in 'many_faces' mode
    if not modules.globals.many_faces:
        max_faces = 2 if modules.globals.both_faces else 1
        target_faces = target_faces[:max_faces]

    for target_face in target_faces:
        # Crop the face region
        cropped_frame, crop_info = crop_face_region(temp_frame, target_face, 0.2)

        # Store original cropped region for blending
        original_cropped = cropped_frame.copy()
        original_size = (cropped_frame.shape[1], cropped_frame.shape[0])  # (width, height)

        # Enhance the face
        enhanced_frame = enhance_face(cropped_frame)

        # GFPGAN outputs fixed 512x512, resize back to original crop size
        if enhanced_frame.shape[:2] != original_cropped.shape[:2]:
            enhanced_frame = cv2.resize(enhanced_frame, original_size, interpolation=cv2.INTER_LANCZOS4)

        # Use cached mask for better performance
        mask = get_cached_edge_blur_mask(enhanced_frame.shape, blur_amount=30)

        # Blend enhanced face with ORIGINAL cropped region (not itself)
        blended_region = blend_with_mask(enhanced_frame, original_cropped, mask)

        # Paste the enhanced region back into the original frame
        x, y, w, h = crop_info
        temp_frame[y:y+h, x:x+w] = blended_region

    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], progress: Any = None) -> None:

    for temp_frame_path in temp_frame_paths:
        with profile_section('frame_read'):
            temp_frame = cv2.imread(temp_frame_path)
        with profile_section('frame_process_enhance'):
            result = process_frame(None, temp_frame)
        with profile_section('frame_write'):
            cv2.imwrite(temp_frame_path, result)
        if progress:
            progress.update(1)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    result = process_frame(None, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    modules.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
