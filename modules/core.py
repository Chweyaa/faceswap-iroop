import os
import sys
import time
import subprocess
import shutil
import argparse
import signal
import platform
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import List

# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
import torch
import onnxruntime
import tensorflow
import cv2
import numpy as np
from tqdm import tqdm

import modules.globals
import modules.metadata
import modules.ui as ui
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path
from modules.face_analyser import initialize_face_analyser, get_many_faces
# Import rotation estimator
from modules.processors.frame.face_swapper import estimate_head_rotation


if 'ROCMExecutionProvider' in modules.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser()
    program.add_argument('-s', '--source', help='select an source image', dest='source_path')
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--frame-processor', help='pipeline of frame processors', dest='frame_processor', default=['face_swapper'], choices=['face_swapper', 'face_enhancer'], nargs='+')
    program.add_argument('--keep-fps', help='keep original fps', dest='keep_fps', action='store_true', default=True)
    program.add_argument('--keep-audio', help='keep original audio', dest='keep_audio', action='store_true', default=True)
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true', default=False)
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true', default=False)
    program.add_argument('--nsfw-filter', help='filter the NSFW image or video', dest='nsfw_filter', action='store_true', default=False)
    program.add_argument('--video-encoder', help='adjust output video encoder', dest='video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9'])
    program.add_argument('--video-quality', help='adjust output video quality', dest='video_quality', type=int, default=18, choices=range(52), metavar='[0-51]')
    program.add_argument('--flip-x', help='The live camera display flipped on x axis', dest='flip_x', action='store_true', default=False)
    program.add_argument('--flip-y', help='The live camera display flipped on y axis', dest='flip_y', action='store_true', default=False)
    program.add_argument('--live-resizable', help='The live camera frame is resizable', dest='live_resizable', action='store_true', default=True)
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int, default=suggest_max_memory())
    program.add_argument('--execution-provider', help='execution provider', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('-v', '--version', action='version', version=f'{modules.metadata.name} {modules.metadata.version}')

    program.add_argument('--both-faces', help='use two faces in source image', dest='both_faces', action='store_true', default=False)
    program.add_argument('--flip-faces', help='flip two faces in source image from right to left', dest='flip_faces', action='store_true', default=False)
    program.add_argument('--detect-face-right', help='detect target face from right of frame', dest='detect_face_right', action='store_true', default=False)
    program.add_argument('--show-target-face-box', help='show target face box', dest='show_target_face_box', action='store_true', default=False)
    program.add_argument('--mouth-mask', help='show target mouth using mask', dest='mouth_mask', action='store_true', default=False)
    program.add_argument('--show-mouth-mask-box', help='show mouth mask box', dest='show_mouth_mask_box', action='store_true', default=False)
    program.add_argument('--face-tracking', help='track one or two faces when two people are in frame. Max two faces tracked', dest='face_tracking', action='store_true', default=False)

    # register deprecated args
    program.add_argument('-f', '--face', help=argparse.SUPPRESS, dest='source_path_deprecated')
    program.add_argument('--cpu-cores', help=argparse.SUPPRESS, dest='cpu_cores_deprecated', type=int)
    program.add_argument('--gpu-vendor', help=argparse.SUPPRESS, dest='gpu_vendor_deprecated')
    program.add_argument('--gpu-threads', help=argparse.SUPPRESS, dest='gpu_threads_deprecated', type=int)

    args = program.parse_args()

    modules.globals.source_path = args.source_path
    modules.globals.target_path = args.target_path
    modules.globals.output_path = normalize_output_path(modules.globals.source_path, modules.globals.target_path, args.output_path)
    modules.globals.frame_processors = args.frame_processor
    modules.globals.headless = args.source_path or args.target_path or args.output_path
    modules.globals.keep_fps = args.keep_fps
    modules.globals.keep_audio = args.keep_audio
    modules.globals.keep_frames = args.keep_frames
    modules.globals.many_faces = args.many_faces
    modules.globals.nsfw_filter = args.nsfw_filter
    modules.globals.video_encoder = args.video_encoder
    modules.globals.video_quality = args.video_quality
    modules.globals.flip_x = args.flip_x
    modules.globals.flip_y = args.flip_y
    modules.globals.live_resizable = args.live_resizable
    modules.globals.max_memory = args.max_memory
    modules.globals.execution_providers = decode_execution_providers(args.execution_provider)
    modules.globals.execution_threads = args.execution_threads

    modules.globals.both_faces = args.both_faces
    modules.globals.flip_faces = args.flip_faces
    modules.globals.detect_face_right = args.detect_face_right
    modules.globals.show_target_face_box = args.show_target_face_box
    modules.globals.mouth_mask = args.mouth_mask
    modules.globals.show_mouth_mask_box = args.show_mouth_mask_box
    modules.globals.face_tracking = args.face_tracking

    #for ENHANCER tumbler:
    if 'face_enhancer' in args.frame_processor:
        modules.globals.fp_ui['face_enhancer'] = True
    else:
        modules.globals.fp_ui['face_enhancer'] = False

    # translate deprecated args
    if args.source_path_deprecated:
        print('\033[33mArgument -f and --face are deprecated. Use -s and --source instead.\033[0m')
        modules.globals.source_path = args.source_path_deprecated
        modules.globals.output_path = normalize_output_path(args.source_path_deprecated, modules.globals.target_path, args.output_path)
    if args.cpu_cores_deprecated:
        print('\033[33mArgument --cpu-cores is deprecated. Use --execution-threads instead.\033[0m')
        modules.globals.execution_threads = args.cpu_cores_deprecated
    if args.gpu_vendor_deprecated == 'apple':
        print('\033[33mArgument --gpu-vendor apple is deprecated. Use --execution-provider coreml instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['coreml'])
    if args.gpu_vendor_deprecated == 'nvidia':
        print('\033[33mArgument --gpu-vendor nvidia is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['cuda'])
    if args.gpu_vendor_deprecated == 'amd':
        print('\033[33mArgument --gpu-vendor amd is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['rocm'])
    if args.gpu_threads_deprecated:
        print('\033[33mArgument --gpu-threads is deprecated. Use --execution-threads instead.\033[0m')
        modules.globals.execution_threads = args.gpu_threads_deprecated


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 16


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'DmlExecutionProvider' in modules.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in modules.globals.execution_providers:
        return 1
    return 8


def limit_resources() -> None:
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)
    # limit memory usage
    if modules.globals.max_memory:
        memory = modules.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = modules.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def release_resources() -> None:
    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
        torch.cuda.empty_cache()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    return True


def update_status(message: str, scope: str = 'DLC.CORE') -> None:
    print(f'[{scope}] {message}')
    if not modules.globals.headless:
        ui.update_status(message)

# --- THREAD-SAFE FRAME PROCESSING ---
# This function handles the rotation locally to avoid race conditions
def process_frame_stream(processors, source_images, frame, rotation_angle):
    """
    Helper for the thread pool to process a single frame without global overhead.
    """
    # 1. Apply Thread-Local Rotation
    if rotation_angle != 0:
        if rotation_angle == -90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation_angle == 180 or rotation_angle == -180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

    # 2. Apply Global Flips (if needed)
    if modules.globals.flip_x:
        frame = cv2.flip(frame, 1)
    if modules.globals.flip_y:
        frame = cv2.flip(frame, 0)
        
    try:
        # 3. Run Processors (Global face_rot_range is forced to 0, so they won't rotate)
        for processor in processors:
            frame = processor.process_frame(source_images, frame)
    except Exception:
        pass
        
    # 4. Re-apply Global Flips
    if modules.globals.flip_x:
        frame = cv2.flip(frame, 1)
    if modules.globals.flip_y:
        frame = cv2.flip(frame, 0)

    # 5. Un-Rotate
    if rotation_angle != 0:
        if rotation_angle == -90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180 or rotation_angle == -180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            
    return frame

def start() -> None:
    # 1. Initialize Processors
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_start():
            return
    
    # 2. Prepare Source Faces
    source_images = []
    if modules.globals.source_path:
        source_image = cv2.imread(modules.globals.source_path)
        faces = get_many_faces(source_image)
        if faces:
            source_images = sorted(faces, key=lambda face: face.bbox[0])[:20]

    update_status('Processing...')
    
    # --- IMAGE PROCESSING ---
    if has_image_extension(modules.globals.target_path):
        if modules.globals.nsfw_filter and ui.check_and_ignore_nsfw(modules.globals.target_path, destroy):
            return
        try:
            target_frame = cv2.imread(modules.globals.target_path)
            for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
                target_frame = frame_processor.process_frame(source_images, target_frame)
            cv2.imwrite(modules.globals.output_path, target_frame)
            release_resources()
        except Exception as e:
             print(f"Error processing image: {e}")
        
        if is_image(modules.globals.output_path):
            update_status('Processing to image succeed!')
        else:
            update_status('Processing to image failed!')
        return

    # --- VIDEO PROCESSING (FAST STREAMING) ---
    if modules.globals.nsfw_filter and ui.check_and_ignore_nsfw(modules.globals.target_path, destroy):
        return

    # 1. Get Video Info
    cap = cv2.VideoCapture(modules.globals.target_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # We write to a temp video file first (no audio yet)
    temp_video_path = modules.globals.output_path + ".temp.mp4"
    
    # 2. Setup FFmpeg Pipe
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-', 
        '-c:v', modules.globals.video_encoder,
        '-crf', str(modules.globals.video_quality),
        '-pix_fmt', 'yuv420p',
        temp_video_path
    ]
    
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    processors = get_frame_processors_modules(modules.globals.frame_processors)
    
    # 3. Process Loop with ThreadPool
    update_status(f'Processing video with {modules.globals.execution_threads} threads...')
    futures = deque()
    
    # Rotation check counter
    rotation_check_counter = 0
    
    # CRITICAL: Store original manual rotation user preference
    user_manual_rotation = modules.globals.face_rot_range
    # DISABLE Global Rotation during processing so threads don't get confused
    modules.globals.face_rot_range = 0
    
    try:
        with tqdm(total=total_frames, unit='frame') as pbar:
            with ThreadPoolExecutor(max_workers=modules.globals.execution_threads) as executor:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # --- DETERMINE ROTATION FOR THIS FRAME ---
                    current_rotation = 0
                    
                    if modules.globals.auto_rotate_value:
                        # Logic: Use cached value or check every interval
                        # Ideally, we check occasionally. For simplicity/speed in threading, 
                        # we can either check every frame (slow) or use the user's manual value if auto is off.
                        # Since we are single-threaded reading, we can check.
                        
                        if rotation_check_counter % modules.globals.rotation_check_interval == 0:
                            rotation_check_faces = get_many_faces(frame)
                            found_rotation = False
                            
                            # 1. Check Normal
                            if rotation_check_faces:
                                main_face = max(rotation_check_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                                angle = estimate_head_rotation(main_face)
                                if angle > 60: current_rotation = -90 
                                elif angle < -60: current_rotation = 90
                                else: current_rotation = 0
                                found_rotation = True
                            
                            # 2. Check 180
                            if not found_rotation:
                                frame_180 = cv2.rotate(frame, cv2.ROTATE_180)
                                faces_180 = get_many_faces(frame_180)
                                if faces_180:
                                    current_rotation = 180
                                else:
                                    current_rotation = 0
                            
                            # Store this detected rotation for the next few frames? 
                            # For safety in this implementation, we simply use this value for this frame.
                            # To apply it to next frames, we would need a 'sticky_rotation' variable.
                            # Let's use a sticky variable for smoother processing.
                            modules.globals.temp_sticky_rotation = current_rotation
                        
                        # Use sticky rotation
                        if hasattr(modules.globals, 'temp_sticky_rotation'):
                            current_rotation = modules.globals.temp_sticky_rotation
                        else:
                            current_rotation = 0
                            
                    else:
                        # Auto is OFF, use the manual setting
                        current_rotation = user_manual_rotation
                    
                    rotation_check_counter += 1
                    # -----------------------------------------------
                    
                    # Submit task with SPECIFIC ROTATION
                    future = executor.submit(process_frame_stream, processors, source_images, frame, current_rotation)
                    futures.append(future)
                    
                    # Maintain buffer size
                    while len(futures) > modules.globals.execution_threads * 2:
                        result_frame = futures.popleft().result()
                        if result_frame is not None:
                            try:
                                # FIX: Ensure Contiguous Memory
                                result_frame = np.ascontiguousarray(result_frame)
                                process.stdin.write(result_frame.tobytes())
                            except Exception:
                                pass
                        pbar.update(1)
                
                # Flush remaining frames
                while futures:
                    result_frame = futures.popleft().result()
                    if result_frame is not None:
                        try:
                            # FIX: Ensure Contiguous Memory
                            result_frame = np.ascontiguousarray(result_frame)
                            process.stdin.write(result_frame.tobytes())
                        except Exception:
                            pass
                    pbar.update(1)
    finally:
        # RESTORE Global Rotation preference
        modules.globals.face_rot_range = user_manual_rotation
        if hasattr(modules.globals, 'temp_sticky_rotation'):
            del modules.globals.temp_sticky_rotation

    cap.release()
    process.stdin.close()
    process.wait()

    # 5. Restore Audio
    if modules.globals.keep_audio:
        update_status('Restoring audio...')
        cmd_audio = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', temp_video_path,
            '-i', modules.globals.target_path,
            '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0',
            '-shortest', modules.globals.output_path
        ]
        subprocess.run(cmd_audio)
        if os.path.exists(modules.globals.output_path):
            os.remove(temp_video_path)
    else:
        if os.path.exists(modules.globals.output_path):
            os.remove(modules.globals.output_path)
        os.rename(temp_video_path, modules.globals.output_path)

    clean_temp(modules.globals.target_path)
    
    if os.path.exists(modules.globals.output_path):
        update_status('Processing to video succeed!')
        

    else:
        update_status('Processing to video failed!')


def destroy(to_quit=True) -> None:
    if modules.globals.target_path:
        clean_temp(modules.globals.target_path)
    if to_quit: quit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_check():
            return
    limit_resources()
        # Add the initialization here
    # Initialize face analyser
    from modules.face_analyser import initialize_face_analyser
    initialize_face_analyser()

    if modules.globals.headless:
        start()
    else:
        window = ui.init(start, destroy)
        window.mainloop()