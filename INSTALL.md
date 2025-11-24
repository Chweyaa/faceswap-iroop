# Installation Guide - faceswap-iroop

## Prerequisites

- Python 3.9-3.11 (3.11 recommended)
- [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) (for GPU acceleration)
- [uv](https://github.com/astral-sh/uv) package manager
- Git

## Quick Install (Windows with GPU)

### 1. Install uv
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone Repository
```bash
git clone https://github.com/iVideoGameBoss/iRoopDeepFaceCam.git
cd faceswap-iroop
```

### 3. Create Virtual Environment and Install Dependencies

**Option A: Using uv pip (Recommended)**
```bash
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
uv pip install tensorflow==2.12.1
uv pip install typing-extensions==4.15.0
```

**Option B: Using uv sync**
```bash
uv venv
.venv\Scripts\activate
uv sync
uv pip install tensorflow==2.12.1
uv pip install typing-extensions==4.15.0
```

> **Note:** TensorFlow and typing-extensions have version conflicts and must be installed separately after the main requirements.

### 5. Download Required Models
Download and place in the `models` folder:
1. [GFPGANv1.4.pth](https://huggingface.co/ivideogameboss/iroopdeepfacecam/blob/main/GFPGANv1.4.pth)
2. [inswapper_128_fp16.onnx](https://huggingface.co/ivideogameboss/iroopdeepfacecam/blob/main/inswapper_128_fp16.onnx)

### 6. Run the Application
With GPU:
```bash
python run.py --execution-provider cuda --execution-threads 5
```

Without GPU (CPU only):
```bash
python run.py --execution-provider cpu
```

## Alternative: Using pip and requirements.txt

If you prefer using pip instead of uv:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

All dependencies including TensorFlow are listed in `requirements.txt`.

## Troubleshooting

### ModuleNotFoundError: No module named 'tensorflow'
Run:
```bash
uv pip install tensorflow==2.12.1
uv pip install typing-extensions==4.15.0
```

### ImportError: cannot import name 'Sentinel' from 'typing_extensions'
Run:
```bash
uv pip install typing-extensions==4.15.0
```

### GPU not detected
Ensure CUDA Toolkit 11.8 is properly installed and accessible.

## Notes

- The first run will download AI models (~300MB)
- TensorFlow is required for GPU memory management
- numpy is locked to 1.24.3 for TensorFlow compatibility
- For macOS/Linux installations, refer to the main README.md
