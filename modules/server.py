import cv2
import numpy as np
import base64
import threading
import requests
import logging
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve
import modules.globals
# Import both processors
from modules.processors.frame import face_swapper, face_enhancer
from modules.face_analyser import get_many_faces

app = Flask(__name__)
CORS(app)

# Lock to ensure global settings don't clash during parallel processing
PROCESS_LOCK = threading.Lock()

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def img_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def base64_to_img(base64_string):
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    img_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

@app.route('/status', methods=['GET'])
def check_status():
    return jsonify({
        "status": "running", 
        "source_selected": bool(modules.globals.source_path),
        "enhancer_enabled": modules.globals.fp_ui.get('face_enhancer', False)
    })

@app.route('/swap', methods=['POST'])
def swap_image():
    if not modules.globals.source_path:
        return jsonify({"error": "No source face selected"}), 400

    data = request.json
    target_frame = None

    # --- 1. GET IMAGE ---
    if 'url' in data:
        img_url = data['url']
        if img_url.startswith("data:image"):
             target_frame = base64_to_img(img_url)
        else:
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                resp = requests.get(img_url, stream=True, timeout=5, headers=headers)
                if resp.status_code == 200:
                    arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
                    target_frame = cv2.imdecode(arr, -1)
            except:
                return jsonify({"error": "Download failed"}), 400
    elif 'image' in data:
        target_frame = base64_to_img(data['image'])

    if target_frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    if len(target_frame.shape) == 3 and target_frame.shape[2] == 4:
        target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGRA2BGR)

    try:
        source_image = cv2.imread(modules.globals.source_path)
        faces = get_many_faces(source_image)
        source_images = sorted(faces, key=lambda face: face.bbox[0])[:20]
        
        if not source_images:
            return jsonify({"error": "No faces in source"}), 400

        # --- 2. PRE-PROCESS FLIPS ---
        if modules.globals.flip_x: target_frame = cv2.flip(target_frame, 1)
        if modules.globals.flip_y: target_frame = cv2.flip(target_frame, 0)

        # --- 3. APPLY SETTINGS & PROCESS (THREAD SAFE) ---
        requested_index = int(data.get('face_index', -1))
        
        # We use a lock because we are modifying Global Variables that face_swapper reads
        with PROCESS_LOCK:
            # A. Save current state
            old_many_faces = modules.globals.many_faces
            old_index_range = modules.globals.face_index_range
            
            try:
                # B. Apply Web Request Settings
                if requested_index == -1:
                    modules.globals.many_faces = True
                    modules.globals.face_index_range = -1
                else:
                    modules.globals.many_faces = False
                    modules.globals.face_index_range = requested_index

                # C. Run Swap
                result_frame = face_swapper.process_frame(source_images, target_frame)
                
                # D. Run Enhancer (If enabled in UI)
                if modules.globals.fp_ui.get('face_enhancer', False):
                    result_frame = face_enhancer.process_frame(None, result_frame)

            finally:
                # E. Restore Original State (Crucial)
                modules.globals.many_faces = old_many_faces
                modules.globals.face_index_range = old_index_range

        # --- 4. POST-PROCESS FLIPS ---
        if modules.globals.flip_x: result_frame = cv2.flip(result_frame, 1)
        if modules.globals.flip_y: result_frame = cv2.flip(result_frame, 0)

        result_b64 = img_to_base64(result_frame)
        return jsonify({"image": "data:image/jpeg;base64," + result_b64})

    except Exception as e:
        print(f"Swap Error: {e}")
        return jsonify({"error": str(e)}), 500

def run_server():
    serve(app, host='127.0.0.1', port=5000, threads=6)

def start():
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    
    # Improved Console Output
    print("\n" + "="*60)
    print(" iRoop Local Server Running on http://127.0.0.1:5000")
    print(" This server powers the Firefox Face Swap Extension.")
    print("-" * 60)
    print(" HOW TO INSTALL THE EXTENSION:")
    print(" 1. Open Firefox and type in the address bar: about:debugging")
    print(" 2. Click 'This Firefox' on the left menu.")
    print(" 3. Click 'Load Temporary Add-on...'.")
    print(" 4. Browse to the folder: 'FireFox_Face_Swap_Ext'")
    print(" 5. Select the 'manifest.json' file.")
    print("-" * 60)
    print(" PRIVACY NOTE:")
    print(" Webpage images are processed entirely in memory (RAM).")
    print(" No images from websites are saved to your hard drive.")
    print("-" * 60)
    print(" Then visit any website, click the extension icon, and Swap!")
    print("-" * 60)
    print(" HOTKEY GUIDE (Main App & Preview Window):")
    print(" 'a'   : Toggle Auto Face Tracking")
    print(" 't'   : Reset Face Tracking Data")
    print(" 'm'   : Toggle Mouth Mask (Global)")
    print(" 1-9,0 : Toggle Mouth Mask for Faces 1-10")
    print("="*60 + "\n")