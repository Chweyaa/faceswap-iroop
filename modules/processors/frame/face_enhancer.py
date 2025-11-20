from typing import Any, List, Optional
import cv2
import threading
import gfpgan
import os

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.typing import Frame, Face
from modules.utilities import conditional_download, resolve_relative_path, is_image, is_video
from modules.face_analyser import get_one_face, get_many_faces, get_one_face_left, get_one_face_right, get_face_analyser
from modules.processors.frame.face_swapper import crop_face_region,create_adjusted_face,reset_face_tracking

FACE_ENHANCER: Any = None
THREAD_LOCK = threading.Lock()
NAME = 'DLC.FACE-ENHANCER'

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

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            if os.name == 'nt':
                model_path = resolve_relative_path('..\models\GFPGANv1.4.pth')
                
            else:
                model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            FACE_ENHANCER = gfpgan.GFPGANer(model_path=model_path, upscale=1) # type: ignore[attr-defined]
    return FACE_ENHANCER


def enhance_face(temp_frame: Frame) -> Frame:
    # Remove THREAD_SEMAPHORE to allow parallel GPU processing
    # Set paste_back=False to skip redundant face detection inside GFPGAN
    _, restored_faces, _ = get_face_enhancer().enhance(
        temp_frame,
        paste_back=False
    )
    # Return the first (and only) restored face from the cropped input
    if restored_faces is not None and len(restored_faces) > 0:
        return restored_faces[0]
    return temp_frame


def process_frame(source_face: Optional[Face], temp_frame: Frame) -> Frame:

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

    # target_face = get_one_face(temp_frame)
    for i, target_face in enumerate(target_faces):
        
        # Crop the face region
        cropped_frame, crop_info = crop_face_region(temp_frame, target_face,0.2)
        # Create an adjusted face for the cropped region

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
