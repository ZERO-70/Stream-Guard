## Commands Reference

This document provides ready-to-run commands to set up, configure, run, test, and maintain the project.


### Quick start commands 

### first start the sender

source enviroment/bin/activate && python sender.py

### then start the receiver

## with no GUI (no video, see audio_transcript.txt for live results) :
  
source enviroment/bin/activate && python receiver.py --headless --enable-nsfw --enable-gun --enable-transcription --enable-profanity --whisper-model small

source enviroment/bin/activate && python receiver.py --no-nsfw --no-gun --no-transcription --no-profanity 

## with GUI (with video) :
    
source enviroment/bin/activate && python receiver.py --enable-nsfw --enable-gun --enable-transcription --enable-profanity --whisper-model small

add --no-audio at the end if you want no audio

### Environment Setup

```bash
# Create and activate virtual environment (kept as 'enviroment' for compatibility)
python3 -m venv enviroment
source enviroment/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install FFmpeg (for audio extraction)
sudo apt-get update && sudo apt-get install -y ffmpeg

# Optional (GPU users): Install PyTorch with CUDA per your system
# See: https://pytorch.org/get-started/locally/
# Example (CUDA 12.x):
# pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Optional: YOLO (weapon detection)
pip install ultralytics
```

### Configuration File Management

```bash
# Generate default config file `stream_config.json`
python config.py

# Save current CLI overrides back into the config file
python receiver.py --whisper-model base --enable-gun --save-config

# Use an alternate config file
python receiver.py --config custom_config.json
```

### Environment Variables (secrets, tokens)

```bash
# Hugging Face token for NSFW test script (do NOT hardcode in code)
export HF_TOKEN="hf_xxx_your_token_here"
```

### Start Streaming (Typical Flow)

```bash
# 1) Start the sender (terminal A)
source enviroment/bin/activate && python sender.py

# 2) Start the receiver (terminal B) - GUI mode
source enviroment/bin/activate && python receiver.py

# 2b) Receiver in headless mode (no window; see logs and transcript)
source enviroment/bin/activate && python receiver.py --headless
```

### Feature Toggles (Enable/Disable)

```bash
# Enable all major features explicitly
python receiver.py --enable-nsfw --enable-gun --enable-transcription --enable-profanity

# Disable specific features
python receiver.py --no-nsfw
python receiver.py --no-gun
python receiver.py --no-transcription
python receiver.py --no-profanity
```

### Model, Device, and Performance

```bash
# Choose Whisper model size for transcription
python receiver.py --whisper-model tiny
python receiver.py --whisper-model base
python receiver.py --whisper-model small
python receiver.py --whisper-model medium
python receiver.py --whisper-model large

# Device selection for models (auto picks best available)
python receiver.py --device auto
python receiver.py --device cpu
python receiver.py --device cuda      # NVIDIA GPU
python receiver.py --device mps       # Apple Silicon

# Compute type for Whisper (if supported on your device)
python receiver.py --compute-type auto
python receiver.py --compute-type int8
python receiver.py --compute-type float16
python receiver.py --compute-type float32

# Audio processing chunk duration (seconds) – smaller is faster/more responsive
python receiver.py --chunk-duration 1.0

# Disable audio-video synchronization if timing issues occur
python receiver.py --no-sync
```

### Video/Display Quality

```bash
# Headless mode (no GUI)
python receiver.py --headless

# Target FPS and JPEG quality (1-100)
python receiver.py --fps 24 --jpeg-quality 80
```

### Network Settings

```bash
# Bind sender to all interfaces (for remote receivers)
python sender.py --host 0.0.0.0 --video-port 9999 --audio-port 9998

# Receiver connecting to remote host/ports
python receiver.py --host 192.168.1.50 --video-port 9999 --audio-port 9998
```

### Tests and Diagnostics

```bash
# Verify environment and create a sample video
python test_setup.py

# GPU diagnostics (PyTorch/Transformers/Whisper)
python test_gpu.py

# NSFW detection API test (requires HF_TOKEN)
export HF_TOKEN="hf_xxx_your_token_here" && python test_nsfw.py

# Profanity filter test
python test_profanity_filter.py

# Headless mode test (if provided)
python test_headless.py

# Weapon detection test (ensure ultralytics installed; model at weapon_detection.pt)
python test_weapon_detection.py

# Config validation test (if applicable)
python test_config.py
```

### Maintenance and Utilities

```bash
# Clear logs and temp files
rm -f nsfw_detection.log http_streamer.log temp_audio.raw audio_transcript.txt

# Free occupied ports (Linux)
sudo fuser -k 9999/tcp || true
sudo fuser -k 9998/tcp || true

# Recreate default config
python config.py
```

### Quick Reference: CLI Flags

```text
Feature toggles (disable):
  --no-nsfw --no-gun --no-transcription --no-profanity

Feature toggles (enable overrides):
  --enable-nsfw --enable-gun --enable-transcription --enable-profanity

Network:
  --host HOST --video-port PORT --audio-port PORT

Models/Performance:
  --whisper-model {tiny,base,small,medium,large}
  --device {auto,cpu,cuda,mps}
  --compute-type {auto,int8,float16,float32}
  --chunk-duration SECONDS

Display/Video:
  --headless --fps INT --jpeg-quality INT

Config file:
  --config FILE --save-config
```

### Notes

- Always activate your venv first: `source enviroment/bin/activate`
- Ensure a file named `video.mp4` is present for the sender.
- For remote streaming, set the sender to host `0.0.0.0` and point the receiver to the sender's IP.
- If you encounter YOLO import issues, ensure `pip install ultralytics` and that `weapon_detection.pt` exists.
