# Local Video Streaming with Content Detection

This project implements a comprehensive video streaming system using Python sockets with configurable content detection features including NSFW detection, gun detection, audio transcription, and profanity detection. 

This project was created with the mutual contributions of ChatGPT, GitHub Copilot, and the developer himmself.

## Features

- **Real-time video streaming** using TCP sockets
- **JPEG encoding** for efficient frame transmission
- **Automatic video looping** when the video ends
- **FPS monitoring** and display
- **🔍 NSFW detection** with automatic blurring (configurable)
- **🔫 Gun/weapon detection** with bounding boxes (configurable)
- **🎤 Audio transcription** using Whisper (configurable)
- **🚫 Profanity detection** for transcriptions (configurable)
- **⚙️ Flexible configuration** via config files, command line, or interactive tool
- **Graceful connection handling** and cleanup

## Quick Start

### Please See Commands.md for configurable commands.

### Basic Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run with default configuration (all features enabled)
python sender.py    # In one terminal
python receiver.py  # In another terminal
```

### Custom Configuration
```bash
# Interactive configuration tool (recommended for first-time setup)
python configure.py

# Command line options
python receiver.py --no-nsfw --no-gun  # Disable content detection
python receiver.py --whisper-model base  # Better transcription quality
python receiver.py --help  # See all options
```

## Files

- `sender.py` - Streams video and audio over TCP
- `receiver.py` - Receives and displays stream with content detection  
- `configure.py` - Interactive configuration tool
- `config.py` - Configuration management system
- `stream_config.json` - Configuration file (auto-generated)
- `requirements.txt` - Python dependencies
- `CONFIGURATION.md` - Detailed configuration guide
- `README.md` - This file

## Prerequisites

1. **Python 3.7+** installed on your system
2. **A video file** named `video.mp4` in the same directory as the scripts
3. **Dependencies** (installed via requirements.txt)

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your `video.mp4` file in the same directory as the scripts.

3. **(Optional)** Configure the application:
   ```bash
   python configure.py  # Interactive configuration tool
   ```

## Configuration Options

### Content Detection Features

| Feature | Description | Default | Disable With |
|---------|-------------|---------|--------------|
| **NSFW Detection** | Automatically blurs inappropriate content | ✅ Enabled | `--no-nsfw` |
| **Gun Detection** | Detects weapons and draws bounding boxes | ✅ Enabled | `--no-gun` |
| **Audio Transcription** | Converts speech to text using Whisper | ✅ Enabled | `--no-transcription` |
| **Profanity Filter** | Filters inappropriate language from transcripts | ✅ Enabled | `--no-profanity` |

### Configuration Methods

1. **Interactive Tool** (Recommended):
   ```bash
   python configure.py
   ```

2. **Command Line**:
   ```bash
   # Disable specific features
   python receiver.py --no-nsfw --no-gun
   
   # Change Whisper model for better/faster transcription
   python receiver.py --whisper-model tiny    # Faster
   python receiver.py --whisper-model base    # Better quality
   
   # Custom network settings
   python receiver.py --host 192.168.1.100 --video-port 8080
   ```

3. **Configuration File** (`stream_config.json`):
   ```json
   {
     "nsfw_detection": {"enabled": false},
     "gun_detection": {"enabled": true},
     "transcription": {"enabled": true, "whisper_model": "base"}
   }
   ```

## Usage

### Step 1: Start the Sender

Open a terminal/command prompt and run (make sure that your virtual enviroment is active or you have installed the depencencies globally):
```bash
python sender.py
```

The sender will:
- Start a TCP server on `localhost:9999` (configurable)
- Wait for a client connection
- Begin streaming video frames once connected

### Step 2: Start the Receiver

Open another terminal/command prompt and run (read confi_commands.txt for customized config commands):
```bash
source enviroment/bin/activate && python receiver.py --enable-nsfw --enable-gun --enable-transcription --enable-profanity --whisper-model small
```

The receiver will:
- Connect to the sender  
- Display the video stream in a window
- Show FPS and frame count overlay
- **Perform content detection** (if enabled):
  - NSFW detection with automatic blurring
  - Gun/weapon detection with bounding boxes  
  - Audio transcription to `audio_transcript.txt`
  - Profanity filtering of transcriptions
- **Display status information** for all enabled features

### Example Output

When running with all features enabled, you'll see:
- 🔴 **NSFW: 0.95** (red text when inappropriate content detected)
- 🟢 **SFW: 0.98** (green text when content is safe)
- 🔫 **WEAPON: 0.87** (red text when weapons detected)
- 🎤 **Live transcription** displayed on video
- 📝 **Transcript file** updated in real-time

### Output Files

The application generates these files:
- `audio_transcript.txt` - Real-time speech transcription with timestamps
- `nsfw_detection.log` - System logs and detection events  
- `stream_config.json` - Current configuration settings

### Controls

- **Press 'q'** in the video window to quit the receiver
- **Ctrl+C** in either terminal to stop the respective script

## NSFW Detection Features

The receiver includes advanced NSFW detection:

- **Real-time Analysis**: Checks every frame every second
- **Automatic Blurring**: Heavily blurs NSFW content with dark overlay
- **Status Display**: Shows NSFW/SFW status with confidence scores
- **Model Integration**: Uses Hugging Face's NSFW detection model
- **Error Handling**: Gracefully handles model loading failures

### NSFW Detection Status

- **Green text**: SFW content detected
- **Red text**: NSFW content detected (frame will be blurred)
- **Gray text**: NSFW detection disabled (model unavailable)

## How It Works

### Sender (`sender.py`)
1. Creates a TCP server socket on localhost:9999
2. Waits for client connection
3. Reads video frames using OpenCV
4. Encodes each frame as JPEG (80% quality)
5. Sends frame size followed by frame data
6. Maintains real-time playback speed
7. Loops video when it ends

### Receiver (`receiver.py`)
1. Connects to the sender's TCP server
2. Receives frame size and data
3. Decodes JPEG frames back to OpenCV format
4. **Checks for NSFW content every second**
5. **Applies blur to NSFW frames**
6. Displays frames in real-time with status overlay
7. Shows FPS, frame count, and NSFW status
8. Handles connection errors gracefully

## Technical Details

- **Protocol**: TCP sockets for reliable frame delivery
- **Encoding**: JPEG compression (80% quality) to reduce bandwidth
- **Framing**: Each frame is prefixed with its size (4 bytes)
- **Serialization**: Pickle for Python object serialization
- **Display**: OpenCV window with real-time FPS counter
- **NSFW Detection**: Transformers pipeline with Falconsai/nsfw_image_detection model
- **Blurring**: Gaussian blur + dark overlay for NSFW content

## Troubleshooting

### Common Issues

1. **"Video file not found"**
   - Make sure `video.mp4` exists in the same directory as `sender.py`

2. **"Connection refused"**
   - Start `sender.py` before `receiver.py`
   - Check if port 9999 is already in use

3. **"Module not found"**
   - Install dependencies: `pip install -r requirements.txt`

4. **"NSFW detection not available"**
   - Check your internet connection (first run downloads the model)
   - The system will work without NSFW detection
   - Model download may take time on first run

5. **Poor performance**
   - Reduce video resolution or frame rate
   - Lower JPEG quality in `sender.py` (line with `IMWRITE_JPEG_QUALITY`)
   - NSFW detection adds some latency (checks every second)

### Performance Tips

- Use smaller video files for better performance
- Ensure both scripts run on the same machine
- Close other applications to free up system resources
- Consider reducing video resolution if streaming is slow
- NSFW detection requires model download on first run

## Customization

### Change Port
Edit the port number in both scripts:
```python
# In sender.py and receiver.py
self.port = 9999  # Change to any available port
```

### Change Video Quality
In `sender.py`, modify the JPEG quality:
```python
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # 80% quality
```

### Change Video File
In `sender.py`, change the video path:
```python
sender.stream_video('your_video.mp4')  # Change filename
```

### Adjust NSFW Detection
In `receiver.py`, modify detection settings:
```python
self.nsfw_check_interval = 1.0  # Check every 1 second
# or
self.nsfw_check_interval = 0.5  # Check every 0.5 seconds
```

### Change Blur Strength
In `receiver.py`, modify the blur function:
```python
def blur_frame(self, frame, blur_strength=50):  # Increase for more blur
```

## System Requirements

- **OS**: prefered Linux Ubunutu gnome wayland , otherwise make changes accordingly.
- **Python**: 3.11 or higher
- **hardware**: At least 8GB RAM | 16GB recommended ,you do need a dedicated GPU.
- **Storage**: Enough space for your video file + 7 GB for dependencies.
- **Internet**: Nope , unless you modify to take an actuall stream from internet other than the local stream.

## License

This project is open source and available under the MIT License.


### Contributing

I would love for other developers to contribute to this project. If you have ideas, improvements, or bug fixes, please open an issue or submit a pull request. You can also check `Commands.md` for quick setup and commonly used commands.
