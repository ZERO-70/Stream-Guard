import cv2
import socket
import struct
import pickle
import numpy as np
import sys
import time
import logging
import io
import os
import base64
import threading
import queue
from threading import Lock
from transformers import pipeline
from PIL import Image
# YOLOv8 for local weapon detection - import after logger is set up
def import_yolo():
    """Import YOLO after logger is available"""
    try:
        # Add the correct Python path for the environment
        import sys
        env_path = '/home/zair/Documents/robo/enviroment/lib/python3.12/site-packages'
        if env_path not in sys.path:
            sys.path.insert(0, env_path)
        
        from ultralytics import YOLO
        return True, YOLO
    except ImportError as e:
        return False, e
import subprocess
from faster_whisper import WhisperModel
import wave
import tempfile
import re
from config import Config, parse_command_line_args, apply_cli_overrides

# Import GPU utilities
try:
    from gpu_utils import (
        detect_best_device, 
        log_gpu_info, 
        clear_gpu_cache, 
        monitor_gpu_usage,
        optimize_transformers_device
    )
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False

# Set up logging with UTF-8 encoding for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Use stdout for better encoding
        logging.FileHandler('nsfw_detection.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class VideoReceiver:
    def __init__(self, config: Config = None):
        # Use provided config or create default
        self.config = config if config else Config()
        
        # Video/Display settings
        self.headless_mode = self.config.get('video', 'headless_mode')
        if self.headless_mode:
            pass  # Running in headless mode - no video display window
        
        # Network settings from config
        self.host = self.config.get('network', 'host')
        self.port = self.config.get('network', 'video_port')
        self.audio_port = self.config.get('network', 'audio_port')
        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Audio socket and settings
        self.audio_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.audio_playing = False
        self.audio_process = None  # FFplay process for audio playback
        
        # Audio-Video Synchronization from config
        self.sync_enabled = self.config.get('audio_sync', 'enabled') if self.config.get('audio_sync') else True
        buffer_size = self.config.get('audio_sync', 'buffer_size') if self.config.get('audio_sync') else 10
        self.audio_sync_queue = queue.Queue(maxsize=buffer_size)  # Buffer for audio chunks with timestamps
        self.video_sync_queue = queue.Queue(maxsize=buffer_size)  # Buffer for video frames with timestamps
        self.base_timestamp = None  # Reference timestamp for synchronization
        self.sync_lock = Lock()
        self.max_sync_drift = self.config.get('audio_sync', 'max_drift') if self.config.get('audio_sync') else 0.1
        self.audio_delay_compensation = self.config.get('audio_sync', 'delay_compensation') if self.config.get('audio_sync') else 0.0
        self.max_compensation = 0.1  # Maximum compensation allowed (100ms)
        
        # Video receive/display buffering
        self.video_frame_queue = queue.Queue(maxsize=60)  # buffer of most recent frames
        self.video_running = False
        self.video_reader_thread = None
        
        # Initialize NSFW detection based on config
        self.nsfw_detection_available = False
        if self.config.is_enabled('nsfw_detection'):
            #logger.info("Initializing NSFW detection model...")
            try:
                #logger.info("Loading Falconsai/nsfw_image_detection model...")
                #logger.info("This may take a few minutes on first run as the model will be downloaded.")

                # Enable verbose logging for transformers
                import transformers
                transformers.logging.set_verbosity_info()
                
                # Get device configuration for NSFW model
                nsfw_device = self.config.get('nsfw_detection', 'device')
                if nsfw_device is None:
                    nsfw_device = 'cpu'  # Fallback
                
                # Auto-detect device if GPU utils are available
                if GPU_UTILS_AVAILABLE and nsfw_device == 'auto':
                    device_info = detect_best_device()
                    nsfw_device = optimize_transformers_device(device_info)
                    #logger.info(f"Auto-detected device for NSFW model: {nsfw_device}")
                
                # Validate device
                if nsfw_device == 'cuda':
                    import torch
                    if not torch.cuda.is_available():
                        #logger.warning("CUDA requested but not available, falling back to CPU")
                        nsfw_device = 'cpu'
                elif nsfw_device == 'mps':
                    import torch
                    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                        #logger.warning("MPS requested but not available, falling back to CPU")
                        nsfw_device = 'cpu'
                
                #logger.info(f"Loading NSFW model on device: {nsfw_device}")
                
                # Create pipeline with device specification
                if nsfw_device != 'cpu':
                    # For GPU inference, use device parameter
                    self.nsfw_pipeline = pipeline(
                        "image-classification", 
                        model="Falconsai/nsfw_image_detection",
                        device=nsfw_device
                    )
                else:
                    # For CPU inference
                    self.nsfw_pipeline = pipeline(
                        "image-classification", 
                        model="Falconsai/nsfw_image_detection"
                    )
                
                #logger.info("NSFW detection model loaded successfully!")
                self.nsfw_detection_available = True
                
                # Log model info
                #logger.info(f"Model loaded: {self.nsfw_pipeline.model.name_or_path}")
                #logger.info(f"Pipeline type: {type(self.nsfw_pipeline).__name__}")
                #logger.info(f"Model device: {nsfw_device}")

                # Log GPU info if available
                if GPU_UTILS_AVAILABLE:
                    log_gpu_info()
                
                # Store device for monitoring
                self.nsfw_device = nsfw_device
                
            except Exception as e:
                #logger.error(f"NSFW detection not available: {e}")
                #logger.error("The system will continue without NSFW detection.")
                self.nsfw_detection_available = False
        else:
            #logger.info("NSFW detection disabled by configuration")
            pass

        # Initialize Local Weapon detection based on config
        self.gun_detection_available = False
        self.weapon_model = None
        if self.config.is_enabled('gun_detection'):
            logger.info("Initializing local weapon detection model...")
            try:
                # Try to import YOLO
                yolo_available, yolo_result = import_yolo()
                if not yolo_available:
                    raise ImportError(f"ultralytics package not available: {yolo_result}. Install with: pip install ultralytics")
                
                # Use the imported YOLO class
                YOLO = yolo_result
                logger.info("✅ Ultralytics YOLO imported successfully")
                
                # Get weapon detection settings from config (with safe fallbacks)
                weapon_device = self.config.get('gun_detection', 'device') or 'cpu'
                model_path = self.config.get('gun_detection', 'model_path') or 'weapon_detection.pt'
                
                # Auto-detect device if GPU utils are available
                if GPU_UTILS_AVAILABLE and weapon_device == 'auto':
                    device_info = detect_best_device()
                    weapon_device = device_info['device']
                    logger.info(f"Auto-detected device for weapon model: {weapon_device}")
                
                # Try to load custom weapon model, fallback to general YOLO model
                try:
                    logger.info(f"Loading custom weapon detection model from: {model_path}")
                    self.weapon_model = YOLO(model_path)
                    self.weapon_model_is_custom = True
                except Exception as model_error:
                    logger.warning(f"Custom weapon model not found: {model_error}")
                    logger.info("Falling back to YOLOv8n general object detection model...")
                    # Use general YOLOv8 model and filter for weapon-like objects
                    self.weapon_model = YOLO('yolov8n.pt')  # This will auto-download if not present
                    self.weapon_model_is_custom = False
                
                # Set device for model
                if weapon_device != 'cpu':
                    import torch
                    if weapon_device == 'cuda' and torch.cuda.is_available():
                        logger.info(f"Using CUDA device for weapon detection")
                    elif weapon_device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        logger.info(f"Using MPS device for weapon detection")
                    else:
                        logger.warning(f"Requested device {weapon_device} not available, using CPU")
                        weapon_device = 'cpu'
                
                self.weapon_device = weapon_device
                
                # Compute class names from model metadata
                available_names = set()
                try:
                    if hasattr(self.weapon_model, 'names') and isinstance(self.weapon_model.names, (dict, list)):
                        if isinstance(self.weapon_model.names, dict):
                            available_names = {str(n).lower() for n in self.weapon_model.names.values()}
                        else:
                            available_names = {str(n).lower() for n in self.weapon_model.names}
                    elif hasattr(self.weapon_model, 'model') and hasattr(self.weapon_model.model, 'names'):
                        names_obj = self.weapon_model.model.names
                        if isinstance(names_obj, dict):
                            available_names = {str(n).lower() for n in names_obj.values()}
                        else:
                            available_names = {str(n).lower() for n in names_obj}
                except Exception:
                    available_names = set()

                # For custom models: treat any detection as weapon (single/multiple custom classes)
                # For fallback model: restrict to exact known weapon classes present in model
                if self.weapon_model_is_custom:
                    self.weapon_classes = available_names  # informational only
                    logger.info(f"Custom weapon model loaded. Model classes: {sorted(self.weapon_classes) if self.weapon_classes else '[unknown]'}")
                else:
                    desired_weapon_classes = {
                        'knife', 'gun', 'pistol', 'rifle', 'sword', 'blade'
                    }
                    self.weapon_classes = desired_weapon_classes.intersection(available_names)
                    if self.weapon_classes:
                        logger.info(f"Weapon classes enabled (fallback model): {sorted(self.weapon_classes)}")
                    else:
                        logger.warning("No explicit weapon classes found in fallback model; only 'knife'/'scissors' may exist in COCO")
                
                logger.info("Local weapon detection model loaded successfully!")
                self.gun_detection_available = True
                
            except Exception as e:
                logger.error(f"Local weapon detection not available: {e}")
                logger.error("The system will continue without weapon detection.")
                logger.error("To enable weapon detection, install ultralytics: pip install ultralytics")
                self.gun_detection_available = False
        else:
            logger.info("Weapon detection disabled by configuration")

        # NSFW detection settings from config
        self.last_nsfw_check_time = time.time()
        self.nsfw_time_interval = self.config.get('nsfw_detection', 'time_interval')
        self.nsfw_model_interval = self.config.get('nsfw_detection', 'time_interval')
        self.last_nsfw_model_time = 0
        self.current_nsfw_status = False
        self.nsfw_confidence = 0.0
        self.detection_count = 0
        self.frame_count_since_last_nsfw_check = 0
        
        # Weapon detection settings from config
        self.gun_detection_count = 0
        self.current_gun_status = False
        self.gun_confidence = 0.0
        self.gun_objects = []
        self.last_gun_detection_time = 0  # Track time of last local detection
        self.gun_detection_interval = self.config.get('gun_detection', 'time_interval')
        self.weapon_confidence_threshold = self.config.get('gun_detection', 'confidence_threshold')
        
        # Temperature tracking system
        self.nsfw_temperature = 0.0  # Range 0-100
        self.arm_temperature = 0.0   # Range 0-100
        self.abusive_language_temperature = 0.0  # Range 0-100 for abusive words
        self.last_temperature_update = time.time()
        self.temperature_increase_rate = 5.0  # Points per second when detected
        self.temperature_decrease_rate = 0.5  # Division factor (divide by 2 every second)
        self.abusive_decay_factor = 1.3  # Division factor for abusive language cooling
        
        # Initialize Whisper for audio transcription based on config
        self.transcription_available = False
        if self.config.is_enabled('transcription'):
            #logger.info("Initializing Whisper model for audio transcription...")
            try:
                # Get transcription settings from config
                whisper_model = self.config.get('transcription', 'whisper_model')
                device = self.config.get('transcription', 'device')
                compute_type = self.config.get('transcription', 'compute_type')
                
                self.whisper_model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
                #logger.info(f"Whisper model '{whisper_model}' loaded successfully!")
                self.transcription_available = True
                
                # Transcription settings from config
                self.transcript_file = "audio_transcript.txt"
                self.current_subtitle = ""
                self.subtitle_display_time = self.config.get('transcription', 'subtitle_duration') or 3.0
                self.last_subtitle_time = 0
                self.audio_buffer = []
                self.audio_buffer_duration = self.config.get('transcription', 'chunk_duration') or 2.0  # Smaller chunks for faster processing
                self.last_transcription_time = time.time()
                
                # Initialize transcript file
                with open(self.transcript_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== Audio Transcript Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")

                #logger.info(f"Transcript will be saved to: {self.transcript_file}")

                # Initialize profanity filter
                self._init_profanity_filter()
                
            except Exception as e:
                #logger.error(f"Whisper transcription not available: {e}")
                logger.error("The system will continue without audio transcription.")
                self.transcription_available = False
        else:
            logger.info("Audio transcription disabled by configuration")
        
        # Threading support for NSFW detection
        self.nsfw_detection_queue = queue.Queue(maxsize=1)  # Queue for frames to process
        self.nsfw_results_lock = Lock()  # Lock for thread-safe access to detection results
        self.nsfw_detection_thread = None
        self.nsfw_detection_running = False
        
        # Threading support for gun detection
        self.gun_detection_queue = queue.Queue(maxsize=1)  # Queue for frames to process
        self.gun_results_lock = Lock()  # Lock for thread-safe access to detection results
        self.gun_detection_thread = None
        self.gun_detection_running = False
        
        # Threading support for audio transcription
        if self.transcription_available:
            self.transcription_queue = queue.Queue()  # Queue for audio chunks to transcribe
            self.transcription_lock = Lock()  # Lock for thread-safe access to transcription results
            self.transcription_thread = None
            self.transcription_running = False
        
        # Start the NSFW detection thread if detection is available
        if self.nsfw_detection_available:
            self.nsfw_detection_running = True
            self.nsfw_detection_thread = threading.Thread(target=self._nsfw_detection_worker, daemon=True)
            self.nsfw_detection_thread.start()
            #logger.info("Started NSFW detection in background thread")
            
        # Start the gun detection thread if detection is available
        if self.gun_detection_available:
            self.gun_detection_running = True
            self.gun_detection_thread = threading.Thread(target=self._gun_detection_worker, daemon=True)
            self.gun_detection_thread.start()
            #logger.info("Started gun detection in background thread")
            
        # Start the audio transcription thread if available
        if self.transcription_available:
            self.transcription_running = True
            self.transcription_thread = threading.Thread(target=self._transcription_worker, daemon=True)
            self.transcription_thread.start()
            #logger.info("Started audio transcription in background thread")

    def _recv_exact(self, sock: socket.socket, num_bytes: int) -> bytes:
        """Receive exactly num_bytes from the socket or return None on failure."""
        try:
            data = b""
            while len(data) < num_bytes:
                packet = sock.recv(num_bytes - len(data))
                if not packet:
                    return None
                data += packet
            return data
        except Exception:
            return None
        
    def _init_profanity_filter(self):
        """Initialize profanity filter with external dataset and real-time detection"""
        # Load profanity words from external dataset
        self.profanity_words = []
        
        # Real-time abusive word detection (always initialize these attributes)
        self.abusive_words_detected = []  # Store recently detected words
        self.last_abusive_check_time = time.time()
        
        # Only proceed with profanity filter setup if enabled
        if not self.config.is_enabled('profanity_filter'):
            #logger.info("Profanity filtering disabled by configuration")
            return
        
        try:
            # Get wordlist file from config
            wordlist_file = self.config.get('profanity_filter', 'wordlist_file')
            
            # Load from downloaded profanity wordlist
            with open(wordlist_file, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word and len(word) > 1:  # Skip empty lines and single chars
                        self.profanity_words.append(word)
            
            #logger.info(f"Loaded {len(self.profanity_words)} profanity words from {wordlist_file}")
            
        except FileNotFoundError:
            #logger.warning(f"External profanity wordlist '{wordlist_file}' not found, using fallback list")
            # Fallback list if external file not available
            self.profanity_words = [
                'fuck', 'fucking', 'fucked', 'fucker', 'fucks',
                'shit', 'shitting', 'shits', 'bullshit',
                'bitch', 'bitches', 'bitching',
                'damn', 'damned', 'dammit',
                'ass', 'asshole', 'asses',
                'bastard', 'bastards',
                'crap', 'crappy',
                'piss', 'pissed', 'pissing',
                'hell', 'retard', 'retarded',
                'idiot', 'idiotic', 'moron', 'stupid',
            ]
        
        # Create regex pattern for case-insensitive matching
        # Use word boundaries to avoid partial matches
        pattern = r'\b(' + '|'.join(re.escape(word) for word in self.profanity_words) + r')\b'
        self.profanity_pattern = re.compile(pattern, re.IGNORECASE)
        
        #logger.info(f"Profanity filter initialized with {len(self.profanity_words)} words")
        logger.info("Real-time abusive language detection enabled")
    
    def _filter_profanity(self, text):
        """Filter profanity from text, highlighting with *word* format"""
        # Check if profanity filtering is enabled
        if not self.config.is_enabled('profanity_filter'):
            return text
            
        if not hasattr(self, 'profanity_pattern') or not text:
            return text
        
        def highlight_profanity(match):
            word = match.group(0)
            replacement_char = self.config.get('profanity_filter', 'replacement_char')
            
            # Highlight the entire word with asterisks around it
            return f"{replacement_char}{word}{replacement_char}"
        
        filtered_text = self.profanity_pattern.sub(highlight_profanity, text)
        
        # Log if profanity was detected and filtered
        if filtered_text != text:
            pass  # Profanity was highlighted in transcription
            # logger.info(f"Profanity highlighted in transcription")
            # logger.debug(f"Original: '{text}' -> Filtered: '{filtered_text}'")
        
        return filtered_text
    
    def _detect_abusive_words_realtime(self, text):
        """Real-time detection of abusive words for temperature tracking"""
        if not hasattr(self, 'profanity_pattern') or not text:
            return 0
        
        # Before each detection, cool down the temperature
        current_time = time.time()
        time_elapsed = current_time - self.last_abusive_check_time
        if time_elapsed > 0:
            self.abusive_language_temperature /= (self.abusive_decay_factor ** time_elapsed)
            if self.abusive_language_temperature < 0.1:
                self.abusive_language_temperature = 0.0
        
        self.last_abusive_check_time = current_time
        
        # Find all abusive words in the text
        matches = self.profanity_pattern.findall(text.lower())
        abusive_count = len(matches)
        
        if abusive_count > 0:
            # Increase temperature by 1 for each abusive word detected
            self.abusive_language_temperature += abusive_count*2
            self.abusive_language_temperature = min(100.0, self.abusive_language_temperature)  # Cap at 100
            
            # Store detected words for logging
            unique_words = list(set(matches))
            self.abusive_words_detected.extend(unique_words)
            
            #logger.warning(f"Detected {abusive_count} abusive words: {unique_words}")
            logger.info(f"Abusive language temperature increased to {self.abusive_language_temperature:.1f}")
            
        return abusive_count
    
    def connect_to_server(self):
        """Connect to the video and audio streaming servers"""
        try:
            # Connect to video stream
            logger.info(f"Connecting to video server at {self.host}:{self.port}...")
            self.socket.connect((self.host, self.port))
            logger.info("Connected to video server!")
            
            # Connect to audio stream
            logger.info(f"Connecting to audio server at {self.host}:{self.audio_port}...")
            self.audio_socket.connect((self.host, self.audio_port))
            logger.info("Connected to audio server!")
            
            # Start audio playback thread
            self.audio_playing = True
            self.audio_thread = threading.Thread(target=self._audio_playback_worker, daemon=True)
            self.audio_thread.start()
            logger.info("Started audio playback thread")
            
            return True
        except Exception as e:
            logger.error(f"Error connecting to server: {e}")
            logger.error("Make sure the sender.py is running first!")
            return False
            
    def _nsfw_detection_worker(self):
        """Background worker thread for NSFW detection processing"""
        logger.info("NSFW detection worker thread started")
        
        while self.nsfw_detection_running:
            try:
                # Get a frame from the queue with a timeout
                # This allows the thread to check nsfw_detection_running periodically
                try:
                    frame = self.nsfw_detection_queue.get(timeout=1.0)
                except queue.Empty:
                    # No frame to process, continue waiting
                    continue
                
                # Process the frame (NSFW detection)
                # The time check is now done in check_nsfw, so we only get frames when we're ready to process
                #logger.info("Processing NSFW detection")
                self.last_nsfw_model_time = time.time()
                nsfw_status, confidence = self._process_nsfw_detection(frame)
                
                # Update the results with thread safety
                with self.nsfw_results_lock:
                    self.current_nsfw_status = nsfw_status
                    self.nsfw_confidence = confidence
                
                #logger.info(f"NSFW detection completed: detected={nsfw_status}, confidence={confidence:.2f}")
                
                # Mark task as done
                self.nsfw_detection_queue.task_done()
                
            except Exception as e:
                pass  # Error in NSFW detection thread
                # logger.error(f"Error in NSFW detection thread: {e}")
    
    def _audio_playback_worker(self):
        """Background worker thread for audio playback using FFplay with synchronization"""
        logger.info("Audio playback thread started")
        
        try:
            # Receive audio parameters
            header_len = struct.calcsize("L")
            audio_params_size_bytes = self._recv_exact(self.audio_socket, header_len)
            if not audio_params_size_bytes:
                raise RuntimeError("Failed to read audio params header")
            audio_params_size = struct.unpack("L", audio_params_size_bytes)[0]
            audio_params_data = b""
            while len(audio_params_data) < audio_params_size:
                audio_params_data += self.audio_socket.recv(4096)
            
            audio_params = pickle.loads(audio_params_data)
            logger.info(f"Received audio parameters: {audio_params}")
            
            # Calculate audio timing parameters for synchronization
            sample_rate = audio_params['rate']
            channels = audio_params['channels']
            bytes_per_sample = 2  # 16-bit audio
            bytes_per_second = sample_rate * channels * bytes_per_sample
            
            # Start FFplay process for real-time audio playback
            ffplay_cmd = [
                'ffplay', '-nodisp', '-autoexit',
                '-f', 's16le',  # 16-bit signed little-endian
                '-ac', str(channels),  # channels
                '-ar', str(sample_rate),      # sample rate
                '-bufsize', '512k',  # Reduce buffer size for lower latency
                '-'  # read from stdin
            ]
            
            try:
                self.audio_process = subprocess.Popen(
                    ffplay_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                logger.info("Audio playback started with synchronization")
                
                # Playback loop with synchronization
                while self.audio_playing and self.audio_process.poll() is None:
                    try:
                        # Get audio chunk size
                        chunk_header = self._recv_exact(self.audio_socket, header_len)
                        if not chunk_header:
                            logger.error("Lost connection to audio server")
                            break
                        chunk_size = struct.unpack("L", chunk_header)[0]
                        
                        # Receive audio data
                        audio_data = b""
                        while len(audio_data) < chunk_size:
                            packet = self.audio_socket.recv(min(chunk_size - len(audio_data), 4096))
                            if not packet:
                                break
                            audio_data += packet
                        
                        if len(audio_data) == chunk_size:
                            # Calculate timing for this audio chunk
                            current_time = time.time()
                            
                            # Initialize base timestamp on first chunk
                            with self.sync_lock:
                                if self.base_timestamp is None:
                                    self.base_timestamp = current_time
                                    logger.info(f"Initialized synchronization base timestamp: {self.base_timestamp}")
                            
                            # Calculate audio chunk duration
                            chunk_duration = len(audio_data) / bytes_per_second
                            
                            # Store audio data with timing info for sync
                            if self.sync_enabled:
                                try:
                                    self.audio_sync_queue.put_nowait({
                                        'data': audio_data,
                                        'timestamp': current_time,
                                        'duration': chunk_duration
                                    })
                                except queue.Full:
                                    # Remove oldest audio chunk if queue is full
                                    try:
                                        self.audio_sync_queue.get_nowait()
                                        self.audio_sync_queue.put_nowait({
                                            'data': audio_data,
                                            'timestamp': current_time,
                                            'duration': chunk_duration
                                        })
                                    except queue.Empty:
                                        pass
                            
                            # Send audio data to FFplay (with limited sync compensation)
                            if self.audio_process.stdin:
                                # Apply very limited sync compensation to avoid audio disruption
                                if abs(self.audio_delay_compensation) > 0.005:  # More than 5ms
                                    if self.audio_delay_compensation > 0:
                                        # Audio is ahead, add very small delay (max 50ms)
                                        delay = min(self.audio_delay_compensation, 0.05)
                                        if delay > 0.01:  # Only delay if significant
                                            time.sleep(delay)
                                    # If audio is behind, we don't add extra delays (let it catch up naturally)
                                
                                self.audio_process.stdin.write(audio_data)
                                self.audio_process.stdin.flush()
                                
                                # Buffer audio for transcription if available (always buffer regardless of sync)
                                if self.transcription_available:
                                    self.audio_buffer.append(audio_data)
                                    
                                    # Check if we have enough audio for transcription
                                    if current_time - self.last_transcription_time >= self.audio_buffer_duration:
                                        # Combine audio chunks
                                        combined_audio = b''.join(self.audio_buffer)
                                        
                                        # Queue for transcription without blocking
                                        try:
                                            self.transcription_queue.put_nowait((
                                                combined_audio,
                                                sample_rate,
                                                channels
                                            ))
                                            logger.info(f"Queued {len(combined_audio)} bytes of audio for transcription (sync_enabled={self.sync_enabled})")
                                        except queue.Full:
                                            logger.warning("Transcription queue full, skipping audio chunk")
                                        
                                        # Reset buffer
                                        self.audio_buffer = []
                                        self.last_transcription_time = current_time
                        
                    except Exception as e:
                        logger.error(f"Error during audio playback: {e}")
                        break
                
            except FileNotFoundError:
                logger.warning("FFplay not found. Audio playback disabled. Install FFmpeg for audio support.")
                # Fallback: just consume audio data without playing but maintain sync
                while self.audio_playing:
                    try:
                        chunk_header = self._recv_exact(self.audio_socket, header_len)
                        if not chunk_header:
                            break
                        chunk_size = struct.unpack("L", chunk_header)[0]
                        audio_data = b""
                        while len(audio_data) < chunk_size:
                            packet = self.audio_socket.recv(min(chunk_size - len(audio_data), 4096))
                            if not packet:
                                break
                            audio_data += packet
                        
                        # Still maintain timing for sync even without playback
                        current_time = time.time()
                        with self.sync_lock:
                            if self.base_timestamp is None:
                                self.base_timestamp = current_time
                        
                        # Queue audio for transcription even without FFplay (always queue regardless of sync issues)
                        if self.transcription_available:
                            self.audio_buffer.append(audio_data)
                            if current_time - self.last_transcription_time >= self.audio_buffer_duration:
                                combined_audio = b''.join(self.audio_buffer)
                                try:
                                    self.transcription_queue.put_nowait((
                                        combined_audio,
                                        sample_rate,
                                        channels
                                    ))
                                    logger.info(f"Queued {len(combined_audio)} bytes of audio for transcription (no-ffplay, sync_enabled={self.sync_enabled})")
                                except queue.Full:
                                    logger.warning("Transcription queue full, skipping audio chunk")
                                # Reset buffer and timestamp
                                self.audio_buffer = []
                                self.last_transcription_time = current_time
                    except:
                        break
            
            # Cleanup
            if self.audio_process and self.audio_process.stdin:
                self.audio_process.stdin.close()
            if self.audio_process:
                self.audio_process.terminate()
            
        except Exception as e:
            logger.error(f"Error in audio playback thread: {e}")
        
        logger.info("Audio playback thread stopped")
    
    def _gun_detection_worker(self):
        """Background worker thread for local weapon detection processing"""
        logger.info("Local weapon detection worker thread started")
        
        while self.gun_detection_running:
            try:
                # Get a frame from the queue with a timeout
                # This allows the thread to check gun_detection_running periodically
                try:
                    frame = self.gun_detection_queue.get(timeout=1.0)
                except queue.Empty:
                    # No frame to process, continue waiting
                    continue
                
                # Check if we need to skip detection due to rate limiting
                current_time = time.time()
                time_since_last_call = current_time - self.last_gun_detection_time
                
                if time_since_last_call < self.gun_detection_interval:
                    self.gun_detection_queue.task_done()
                    continue
                
                # Process the frame (local weapon detection)
                logger.info("Background thread processing local weapon detection")
                gun_status, confidence, objects = self._process_local_weapon_detection(frame)
                
                # Update the last detection time for rate limiting
                self.last_gun_detection_time = time.time()
                
                # Update the results with thread safety
                with self.gun_results_lock:
                    self.current_gun_status = gun_status
                    self.gun_confidence = confidence
                    self.gun_objects = objects

                logger.info(f"Local weapon detection completed: detected={gun_status}, confidence={confidence:.2f}, objects={len(objects)}")

                # Mark task as done
                self.gun_detection_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in local weapon detection thread: {e}")
        
        logger.info("Local weapon detection worker thread stopped")
    
    def _transcription_worker(self):
        """Background worker thread for audio transcription processing"""
        logger.info("Audio transcription worker thread started")
        
        while self.transcription_running:
            try:
                # Get audio data from the queue with a timeout
                try:
                    audio_data, sample_rate, channels = self.transcription_queue.get(timeout=1.0)
                except queue.Empty:
                    # No audio to process, continue waiting
                    continue
                
                # Process the audio chunk (transcription)
                logger.info("Processing audio transcription")
                transcription_text = self._process_audio_transcription(audio_data, sample_rate, channels)
                
                if transcription_text.strip():
                    # Update the current subtitle with thread safety
                    with self.transcription_lock:
                        self.current_subtitle = transcription_text
                        self.last_subtitle_time = time.time()
                    
                    # Save to transcript file
                    self._save_transcription(transcription_text)
                    
                    logger.info(f"Transcription completed: '{transcription_text}'")
                    #print(f"[TRANSCRIPT] {transcription_text}")
                
                # Mark task as done
                self.transcription_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in transcription thread: {e}")
        
        logger.info("Audio transcription worker thread stopped")
    
    def _process_audio_transcription(self, audio_data, sample_rate, channels):
        """Internal method to process audio transcription - called from worker thread"""
        try:
            logger.debug(f"Transcribing {len(audio_data)} bytes, sr={sample_rate}, ch={channels}")
            # Create a temporary WAV file for Whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Write audio data to WAV file
                with wave.open(temp_path, 'wb') as wav_file:
                    wav_file.setnchannels(channels)
                    wav_file.setsampwidth(2)  # 16-bit audio
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data)
            
            # Transcribe using Whisper
            logger.debug("Calling Whisper transcribe()...")
            segments, info = self.whisper_model.transcribe(
                temp_path,
                beam_size=5,
                language="en",  # You can change this or set to None for auto-detection
                vad_filter=True,  # Voice activity detection
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            logger.debug(f"Whisper returned: language={getattr(info, 'language', None)}, duration={getattr(info, 'duration', None)}")
            
            # Collect all text from segments
            transcription_text = ""
            seg_count = 0
            for segment in segments:
                seg_text = getattr(segment, 'text', '')
                if seg_text:
                    transcription_text += seg_text + " "
                    seg_count += 1
            logger.debug(f"Collected {seg_count} segments; chars={len(transcription_text)}")
            
            # Apply profanity filter to the transcription
            transcription_text = transcription_text.strip()
            
            # Detect abusive words for temperature tracking (before filtering)
            abusive_count = self._detect_abusive_words_realtime(transcription_text)
            
            # Apply filtering for display and saving
            filtered_text = self._filter_profanity(transcription_text)
            
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
            
            return filtered_text
            
        except Exception as e:
            logger.error(f"Audio transcription error: {e}")
            return ""
    
    def _save_transcription(self, text):
        """Save transcription to file with timestamp and temperature data"""
        try:
            if not text:
                logger.debug("Empty transcription text; skipping file write")
                return
            timestamp = time.strftime('%H:%M:%S')
            
            # Collect temperature data for enabled features
            temp_data = []
            
            if self.config.is_enabled('nsfw_detection'):
                temp_data.append(f"NSFW: {self.nsfw_temperature:.1f}°")
                
            if self.config.is_enabled('gun_detection'):
                temp_data.append(f"Weapon: {self.arm_temperature:.1f}°")
                
            if self.config.is_enabled('profanity_filter'):
                temp_data.append(f"Abusive: {self.abusive_language_temperature:.1f}°")
            
            # Format temperature string
            temp_str = " | ".join(temp_data) if temp_data else "No detection enabled"
            
            # Create the transcript entry
            transcript_entry = f"[{timestamp}] {text}\n"
            if temp_data:  # Only add temperature line if we have temperature data
                transcript_entry += f"[{timestamp}] TEMPS: {temp_str}\n"
            
            # Write to file
            with open(self.transcript_file, 'a', encoding='utf-8') as f:
                f.write(transcript_entry)
            logger.debug(f"Wrote transcript entry ({len(transcript_entry)} bytes)")
            
            # Also log to console for headless mode
            if self.headless_mode:
                print(f"[TRANSCRIPT] {text}")
                if temp_data:
                    print(f"[TEMPS] {temp_str}")
                    
        except Exception as e:
            logger.error(f"Error saving transcription: {e}")
        
        logger.info("Gun detection worker thread stopped")
    
    def _process_nsfw_detection(self, frame):
        """Internal method to process NSFW detection - called from worker thread"""
        self.detection_count += 1
        #logger.info(f"Running NSFW detection in thread (attempt #{self.detection_count})...")
        
        try:
            # Monitor GPU usage if available
            if GPU_UTILS_AVAILABLE and hasattr(self, 'nsfw_device') and self.nsfw_device != 'cpu':
                usage = monitor_gpu_usage()
                if usage > 90:
                    logger.warning("Very high GPU memory usage detected, considering clearing cache")
                    clear_gpu_cache()
            
            # Convert OpenCV BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            #logger.debug(f"Frame converted to PIL Image: {pil_image.size}")
            
            # Run NSFW detection with timing
            start_time = time.time()
            results = self.nsfw_pipeline(pil_image)
            inference_time = time.time() - start_time
            
            device_info = f" on {self.nsfw_device}" if hasattr(self, 'nsfw_device') else ""
            #logger.info(f"NSFW inference completed in {inference_time:.3f} seconds{device_info} (in thread)")
            #logger.debug(f"Raw results: {results}")

            # Parse results - look for the highest confidence result
            nsfw_confidence = 0.0
            sfw_confidence = 0.0
            
            for result in results:
                #logger.info(f"NSFW detection result (in thread): {result['label']} - Confidence: {result['score']:.3f}")
                if result['label'].lower() == 'nsfw':
                    nsfw_confidence = result['score']
                elif result['label'].lower() in ['normal', 'sfw']:
                    sfw_confidence = result['score']
            
            # Determine if NSFW based on confidence comparison
            if nsfw_confidence > sfw_confidence:
                #logger.warning(f"NSFW content detected in thread! Confidence: {nsfw_confidence:.3f}")
                return True, nsfw_confidence
            else:
                #logger.info(f"SFW content detected in thread. Confidence: {sfw_confidence:.3f}")
                return False, sfw_confidence
            
        except Exception as e:
            logger.error(f"NSFW detection error in thread: {e}")
            
            # Clear GPU cache on error if using GPU
            if GPU_UTILS_AVAILABLE and hasattr(self, 'nsfw_device') and self.nsfw_device != 'cpu':
                clear_gpu_cache()
            
            return False, 0.0
            
    def _process_local_weapon_detection(self, frame):
        """Internal method to process local weapon detection using YOLO - called from worker thread"""
        self.gun_detection_count += 1
        logger.info(f"Running local weapon detection in thread (attempt #{self.gun_detection_count})...")

        try:
            if not self.weapon_model:
                logger.error("Weapon model not initialized")
                return False, 0.0, []
            
            # Convert OpenCV BGR to RGB for YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run local weapon detection with timing
            start_time = time.time()
            
            # Use YOLO model for detection
            results = self.weapon_model.predict(
                frame_rgb,
                device=self.weapon_device,
                conf=self.weapon_confidence_threshold * 0.5,  # Lower threshold for detection, we'll filter later
                verbose=False,
                save=False,
                show=False
            )
                
            inference_time = time.time() - start_time
            logger.info(f"Local weapon detection completed in {inference_time:.3f} seconds (in thread)")

            # Parse YOLO results
            detected_objects = []
            highest_confidence = 0.0
            
            if results and len(results) > 0:
                result = results[0]  # Get first result
                
                # Check if we have any detections
                if result.boxes is not None and len(result.boxes) > 0:
                    # Get the class names
                    class_names = result.names if hasattr(result, 'names') else {}
                    
                    for box in result.boxes:
                        # Get confidence and class
                        confidence = float(box.conf[0]) if box.conf is not None else 0.0
                        class_id = int(box.cls[0]) if box.cls is not None else -1
                        class_name = class_names.get(class_id, f"class_{class_id}")
                        
                        # Decide if this is a weapon
                        class_name_lower = class_name.lower()
                        if getattr(self, 'weapon_model_is_custom', False):
                            # Custom weapon model: any detection is considered a weapon
                            is_weapon = True
                        else:
                            # Fallback general model: only exact allowed classes
                            is_weapon = class_name_lower in self.weapon_classes
                        
                        # Only process weapon-related detections or very high confidence objects
                        if is_weapon and confidence >= self.weapon_confidence_threshold:
                            # Get bounding box coordinates (x1, y1, x2, y2)
                            xyxy = box.xyxy[0].cpu().numpy() if box.xyxy is not None else [0, 0, 0, 0]
                            x1, y1, x2, y2 = xyxy
                            
                            # Convert to center x, y, width, height format
                            width = x2 - x1
                            height = y2 - y1
                            center_x = x1 + width / 2
                            center_y = y1 + height / 2
                            
                            detected_objects.append({
                                'class': class_name,
                                'confidence': confidence,
                                'x': center_x,
                                'y': center_y,
                                'width': width,
                                'height': height
                            })
                            
                            if confidence > highest_confidence:
                                highest_confidence = confidence
                            
                            logger.info(f"Weapon detection result (in thread): {class_name} - Confidence: {confidence:.3f}")
            
            # Determine if weapons are detected (only if at least one valid weapon class was found)
            weapon_detected = highest_confidence >= self.weapon_confidence_threshold and len(detected_objects) > 0
            
            if weapon_detected:
                logger.warning(f"Weapon detected in thread! Highest confidence: {highest_confidence:.3f}")
            else:
                logger.info(f"No weapons detected in thread. Highest confidence: {highest_confidence:.3f}")
            
            return weapon_detected, highest_confidence, detected_objects
                
        except Exception as e:
            logger.error(f"Local weapon detection error in thread: {e}")
            return False, 0.0, []
    
    def _calculate_sync_compensation(self):
        """Calculate audio-video synchronization compensation"""
        if not self.sync_enabled or self.base_timestamp is None:
            return 0.0
        
        try:
            # Get current timing info from both queues
            audio_info = None
            video_info = None
            
            # Peek at latest audio timing (don't remove from queue)
            try:
                audio_info = self.audio_sync_queue.queue[-1] if self.audio_sync_queue.qsize() > 0 else None
            except:
                pass
            
            # Peek at latest video timing
            try:
                video_info = self.video_sync_queue.queue[-1] if self.video_sync_queue.qsize() > 0 else None
            except:
                pass
            
            if audio_info and video_info:
                # Calculate the time difference between audio and video
                audio_relative_time = audio_info['timestamp'] - self.base_timestamp
                video_relative_time = video_info['timestamp'] - self.base_timestamp
                
                sync_drift = audio_relative_time - video_relative_time
                
                # If sync drift is extremely large, reset synchronization instead of disabling
                if abs(sync_drift) > 5.0:  # Major timing problem, e.g., start of new video loop
                    self._reset_sync(reason=f"Extreme drift {sync_drift:.3f}s")
                    return 0.0
                
                # Only apply compensation if drift is significant but reasonable
                if abs(sync_drift) > self.max_sync_drift:
                    compensation = sync_drift * 0.05  # Reduced correction factor
                    # Clamp compensation to prevent audio disruption
                    compensation = max(-self.max_compensation, min(self.max_compensation, compensation))
                    
                    if abs(compensation) > 0.01:  # Only log significant compensations
                        logger.info(f"Sync drift: {sync_drift:.3f}s, applying limited compensation: {compensation:.3f}s")
                    
                    return compensation
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating sync compensation: {e}")
            return 0.0

    def _reset_sync(self, reason: str = ""):
        """Soft-reset synchronization baseline and queues to recover from large drift or loop restarts"""
        try:
            with self.sync_lock:
                now = time.time()
                # Debounce resets to avoid rapid loops
                if hasattr(self, '_last_sync_reset_time') and now - getattr(self, '_last_sync_reset_time') < 1.0:
                    return
                self._last_sync_reset_time = now

                logger.warning(f"Resetting A/V sync baseline due to: {reason}")
                self.base_timestamp = now
                self.audio_delay_compensation = 0.0

                # Clear timing queues safely
                try:
                    while not self.audio_sync_queue.empty():
                        self.audio_sync_queue.get_nowait()
                except Exception:
                    pass
                try:
                    while not self.video_sync_queue.empty():
                        self.video_sync_queue.get_nowait()
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Error during sync reset: {e}")
    
    def update_temperatures(self):
        """Update temperature values based on current detection status and time elapsed"""
        current_time = time.time()
        time_elapsed = current_time - self.last_temperature_update
        self.last_temperature_update = current_time
        
        # Update NSFW temperature
        if self.current_nsfw_status:
            # Increase scaled by confidence and intensity factor 2
            nsfw_conf = max(0.0, min(1.0, float(self.nsfw_confidence)))
            increase = (self.temperature_increase_rate * time_elapsed) * 100.0 * nsfw_conf
            self.nsfw_temperature = min(100.0, self.nsfw_temperature + increase)
            # logger.info(f"NSFW detected - +{increase:.2f} (conf={nsfw_conf:.2f}) -> {self.nsfw_temperature:.1f}")
        else:
            # Decrease temperature when no NSFW content (divide by 2 over time)
            # Using exponential decay: new_temp = old_temp * (0.5 ^ time_elapsed)
            decay_factor = 0.5 ** time_elapsed
            self.nsfw_temperature *= decay_factor
            if self.nsfw_temperature < 0.1:  # Threshold to avoid very small numbers
                self.nsfw_temperature = 0.0
            #logger.info(f"No NSFW - Temperature cooled by factor {decay_factor:.3f} to {self.nsfw_temperature:.1f}")
        
        # Update weapon temperature
        if self.current_gun_status:
            # Increase scaled by confidence and intensity factor 2
            gun_conf = max(0.0, min(1.0, float(self.gun_confidence)))
            increase = (self.temperature_increase_rate * time_elapsed) * 100.0 * gun_conf
            self.arm_temperature = min(100.0, self.arm_temperature + increase)
            # logger.info(f"Weapon detected - +{increase:.2f} (conf={gun_conf:.2f}) -> {self.arm_temperature:.1f}")
        else:
            # Decrease temperature when no weapons (divide by 2 over time)
            decay_factor = 0.5 ** time_elapsed
            self.arm_temperature *= decay_factor
            if self.arm_temperature < 0.1:  # Threshold to avoid very small numbers
                self.arm_temperature = 0.0
            #logger.info(f"No weapons - Temperature cooled by factor {decay_factor:.3f} to {self.arm_temperature:.1f}")
    
    def check_nsfw(self, frame):
        """Check if we should process a frame for NSFW detection in the background thread"""
        if not self.nsfw_detection_available:
            return False, 0.0
            
        # Check if it's time to process a new detection (every 0.5 seconds)
        current_time = time.time()
        time_since_last_check = current_time - self.last_nsfw_model_time
        
        if time_since_last_check >= self.nsfw_model_interval:
            self.last_nsfw_check_time = current_time
            self.frame_count_since_last_nsfw_check = 0
            
            # We're ready to detect, so process the frame
            try:
                # Make a deep copy of the frame to avoid issues with shared memory
                frame_copy = frame.copy()
                
                # Put the frame in the queue without blocking - only if we're ready to detect
                if self.nsfw_detection_queue.qsize() == 0:
                    self.nsfw_detection_queue.put_nowait(frame_copy)
                    # logger.info(f"Processing frame for NSFW detection (time since last: {time_since_last_check:.2f}s)")
                else:
                    pass  # Skipped processing - previous NSFW detection still in progress
                    # logger.info(f"Skipped processing - previous NSFW detection still in progress")
            except queue.Full:
                pass  # Skipped processing - NSFW queue is full
                # logger.info(f"Skipped processing - NSFW queue is full")
                
        else:
            # Not enough time has passed since last detection
            self.frame_count_since_last_nsfw_check += 1
            # logger.info(f"Skipping NSFW detection - next check in {self.nsfw_model_interval - time_since_last_check:.2f}s")
        
        # Return the current results (from the thread)
        with self.nsfw_results_lock:
            return self.current_nsfw_status, self.nsfw_confidence
    
    def check_guns(self, frame):
        """Queue a frame for gun detection in the background thread"""
        if not self.gun_detection_available:
            return False, 0.0, []
            
        # Don't block on the queue - if it's full, skip this frame
        try:
            # Make a deep copy of the frame to avoid issues with shared memory
            frame_copy = frame.copy()
            
            # Try to put the frame in the queue without blocking
            if self.gun_detection_queue.qsize() == 0:
                self.gun_detection_queue.put_nowait(frame_copy)
                # logger.info(f"Queued frame for background gun detection")
            else:
                pass  # Skipped queueing frame - background detection still in progress
                # logger.info(f"Skipped queueing frame - background detection still in progress")
        except queue.Full:
            pass  # Skipped queueing frame - queue is full
            # logger.info(f"Skipped queueing frame - queue is full")

        # Return the current results (from the thread)
        with self.gun_results_lock:
            return self.current_gun_status, self.gun_confidence, self.gun_objects
    
    def blur_frame(self, frame, blur_strength=None):
        """Apply heavy blur to NSFW frame"""
        # Use config value if not specified
        if blur_strength is None:
            blur_strength = self.config.get('nsfw_detection', 'blur_strength')
            
        #logger.info(f"Applying blur to NSFW frame (strength: {blur_strength})")
        
        # Make sure blur_strength is an odd number (required by GaussianBlur)
        if blur_strength % 2 == 0:
            blur_strength += 1
            #logger.info(f"Adjusted blur strength to odd number: {blur_strength}")
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
        
        # Add dark overlay for additional obscuring
        overlay = np.zeros_like(frame)
        overlay[:] = (0, 0, 0)  # Black overlay
        
        # Blend blurred frame with dark overlay
        cv2.addWeighted(blurred, 0.3, overlay, 0.7, 0, blurred)
        
        #logger.info("Frame blurred successfully")
        return blurred
        
    def draw_gun_boxes(self, frame, detected_objects):
        """Draw bounding boxes around detected guns/weapons"""
        #logger.info(f"Drawing bounding boxes for {len(detected_objects)} detected objects")
        
        try:
            # Create a copy of the frame to avoid modifying the original
            result_frame = frame.copy()
            
            # Draw each bounding box
            for obj in detected_objects:
                try:
                    # Extract coordinates - handle potential type issues
                    x = float(obj.get('x', 0))
                    y = float(obj.get('y', 0))
                    w = float(obj.get('width', 0))
                    h = float(obj.get('height', 0))
                    
                    # Skip if width or height is 0
                    if w <= 0 or h <= 0:
                        #logger.warning(f"Invalid box dimensions: w={w}, h={h}")
                        continue
                    
                    # Calculate bounding box coordinates
                    x1 = int(x - w/2)
                    y1 = int(y - h/2)
                    x2 = int(x + w/2)
                    y2 = int(y + h/2)
                    
                    # Ensure coordinates are within frame boundaries
                    height, width = frame.shape[:2]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width-1, x2)
                    y2 = min(height-1, y2)
                    
                    # Draw red rectangle around the object
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Add label with confidence
                    confidence = float(obj.get('confidence', 0.0))
                    class_name = str(obj.get('class', 'unknown'))
                    label = f"{class_name}: {confidence:.2f}"
                    
                    # Draw label background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(result_frame, (x1, y1-20), (x1+label_size[0], y1), (0, 0, 255), -1)
                    
                    # Draw label text
                    cv2.putText(result_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                except Exception as e:
                    #logger.error(f"Error drawing box for object {obj}: {e}")
                    continue

            #logger.info("Bounding boxes drawn successfully")
            return result_frame
            
        except Exception as e:
            #logger.error(f"Error in draw_gun_boxes: {e}")
            return frame  # Return original frame if there's an error
    
    def draw_icon(self, frame, x, y, icon_type, color, size=20):
        """Draw simple icons for different detection types"""
        if icon_type == "nsfw":
            # Draw a simple eye icon (circle with inner circle)
            cv2.circle(frame, (x + size//2, y + size//2), size//2, color, 2)
            cv2.circle(frame, (x + size//2, y + size//2), size//4, color, -1)
        elif icon_type == "weapon":
            # Draw a simple crosshair icon
            cv2.line(frame, (x, y + size//2), (x + size, y + size//2), color, 2)
            cv2.line(frame, (x + size//2, y), (x + size//2, y + size), color, 2)
            cv2.circle(frame, (x + size//2, y + size//2), size//3, color, 2)
        elif icon_type == "abusive":
            # Draw a simple speech bubble icon
            cv2.ellipse(frame, (x + size//2, y + size//2), (size//2, size//3), 0, 0, 360, color, 2)
            cv2.line(frame, (x + size//2, y + size), (x + size//2 + size//4, y + size + size//4), color, 2)
        elif icon_type == "fps":
            # Draw a simple speedometer icon
            cv2.circle(frame, (x + size//2, y + size//2), size//2, color, 2)
            cv2.line(frame, (x + size//2, y + size//2), (x + size, y + size//2), color, 2)
        elif icon_type == "overall":
            # Draw a shield icon for overall threat level
            # Draw shield outline
            shield_points = [
                (x + size//2, y),  # top
                (x + size - 2, y + size//3),  # top right
                (x + size - 2, y + 2*size//3),  # bottom right
                (x + size//2, y + size),  # bottom
                (x + 2, y + 2*size//3),  # bottom left
                (x + 2, y + size//3)  # top left
            ]
            cv2.polylines(frame, [np.array(shield_points)], True, color, 2)
            # Add warning symbol inside (exclamation mark)
            cv2.line(frame, (x + size//2, y + size//4), (x + size//2, y + size//2), color, 2)
            cv2.circle(frame, (x + size//2, y + 3*size//4), 1, color, -1)
        elif icon_type == "sync":
            # Draw a simple sync icon (two arrows)
            cv2.arrowedLine(frame, (x, y + size//2), (x + size//2, y + size//2), color, 2, tipLength=0.3)
            cv2.arrowedLine(frame, (x + size//2, y + size//2), (x + size, y + size//2), color, 2, tipLength=0.3)

    def receive_video(self):
        """Receive and display video frames"""
        data = b""
        payload_size = struct.calcsize("L")
        
        # Set up display window only if not in headless mode
        if not self.headless_mode:
            cv2.namedWindow('Video Stream', cv2.WINDOW_NORMAL)
            
            # Set initial window size for better display on laptop
            initial_width = 960  # Standard laptop width
            initial_height = 720  # Height maintaining aspect ratio
            cv2.resizeWindow('Video Stream', initial_width, initial_height)
        
        #logger.info("Starting video stream reception...")
        print("Receiving video stream...")
        
        if not self.headless_mode:
            print("Press 'q' to quit")
        else:
            print("Running in headless mode - press Ctrl+C to quit")
            
        if self.nsfw_detection_available:
            print(f"NSFW detection: Active (every {self.nsfw_model_interval} sec)")
            #logger.info(f"NSFW detection is active and ready (every {self.nsfw_model_interval} sec)")
        else:
            print("NSFW detection: Disabled")
            #logger.warning("NSFW detection is disabled")
            
        if self.gun_detection_available:
            print(f"Local weapon detection: Active (every {self.gun_detection_interval} seconds)")
            logger.info(f"Local weapon detection is active and ready (processing every {self.gun_detection_interval} seconds)")
        else:
            print("Gun detection: Disabled")
            #logger.warning("Gun detection is disabled")
            
        if self.transcription_available:
            print(f"Audio transcription: Active (Whisper model)")
            print(f"Transcript file: {self.transcript_file}")
            print(f"Abusive language detection: Active ({len(self.profanity_words)} words)")
            print(f"Profanity decay factor: {self.abusive_decay_factor}")
            logger.info(f"Audio transcription is active and ready")
        else:
            print("Audio transcription: Disabled")
            logger.warning("Audio transcription is disabled")
            
        if self.sync_enabled:
            print(f"Audio-Video synchronization: Active")
            print(f"Max sync drift tolerance: {self.max_sync_drift:.1f}s")
            print(f"Max compensation limit: {self.max_compensation:.3f}s")
            print("Note: Press 's' during playback to toggle sync on/off")
            logger.info(f"Audio-Video synchronization is enabled")
        else:
            print("Audio-Video synchronization: Disabled")
            logger.warning("Audio-Video synchronization is disabled")
        
        frame_count = 0
        start_time = cv2.getTickCount()
        # For frame rate control (use configured target FPS)
        configured_fps = self.config.get('video', 'target_fps')
        try:
            target_fps = float(configured_fps) if configured_fps else 24.0
        except Exception:
            target_fps = 24.0
        frame_time = 1.0 / target_fps
        last_frame_time = time.time()
        
        try:
            while True:
                # Receive frame size
                while len(data) < payload_size:
                    data += self.socket.recv(4096)
                
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("L", packed_msg_size)[0]
                
                # Receive frame data
                while len(data) < msg_size:
                    data += self.socket.recv(4096)
                
                frame_data = data[:msg_size]
                data = data[msg_size:]
                
                # Decode frame
                try:
                    encoded_frame = pickle.loads(frame_data)
                    frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        frame_count += 1
                        current_time = time.time()
                        
                        # Debug: Print frame info every 30 frames
                        if frame_count % 30 == 0:
                            print(f"Frame {frame_count}: shape={frame.shape}, dtype={frame.dtype}")
                        
                        # Initialize or reinitialize base timestamp for synchronization at loop boundaries
                        with self.sync_lock:
                            if self.base_timestamp is None:
                                self.base_timestamp = current_time
                            # Detect likely video loop restart by large backwards jump in frame timing
                            elif self.video_sync_queue.qsize() > 0:
                                try:
                                    last_video_time = self.video_sync_queue.queue[-1]['timestamp']
                                    if current_time + 0.1 < last_video_time:  # timestamp went backwards notably
                                        self._reset_sync(reason="Detected video timestamp jump (loop restart)")
                                except Exception:
                                    pass

                        # Store video frame timing for synchronization
                        if self.sync_enabled:
                            try:
                                self.video_sync_queue.put_nowait({
                                    'timestamp': current_time,
                                    'frame_number': frame_count
                                })
                            except queue.Full:
                                # Remove oldest video timing if queue is full
                                try:
                                    self.video_sync_queue.get_nowait()
                                    self.video_sync_queue.put_nowait({
                                        'timestamp': current_time,
                                        'frame_number': frame_count
                                    })
                                except queue.Empty:
                                    pass
                        
                        # Calculate and apply synchronization compensation
                        if self.sync_enabled:
                            self.audio_delay_compensation = self._calculate_sync_compensation()
                        
                        # Check for NSFW content (rate limited by check_nsfw)
                        if self.nsfw_detection_available:
                            #logger.info(f"Checking frame #{frame_count} for NSFW content...")
                            self.current_nsfw_status, self.nsfw_confidence = self.check_nsfw(frame)
                            
                            if self.current_nsfw_status:
                                #print(f"NSFW detected! Confidence: {self.nsfw_confidence:.2f}")
                                pass
                            else:
                                #print(f"SFW - Confidence: {self.nsfw_confidence:.2f}")
                                pass

                        # Check for guns/weapons based on time interval (not frames)
                        current_time = time.time()
                        time_since_last_call = current_time - self.last_gun_detection_time
                        
                        if self.gun_detection_available and (time_since_last_call >= self.gun_detection_interval):
                            #logger.info(f"Checking frame #{frame_count} for weapons (time since last check: {time_since_last_call:.1f}s)...")
                            self.current_gun_status, self.gun_confidence, self.gun_objects = self.check_guns(frame)
                            
                            if self.current_gun_status:
                                #print(f"WEAPON detected! Confidence: {self.gun_confidence:.2f}")
                                pass
                            else:
                                #print(f"No weapons - Confidence: {self.gun_confidence:.2f}")
                                pass
                        else:
                            # If not checking this frame, still show status from previous check
                            if self.gun_detection_available:
                                time_remaining = max(0, self.gun_detection_interval - time_since_last_call)
                                #logger.info(f"Skipping weapon detection on frame #{frame_count} (next check in {time_remaining:.1f}s)")
                                if self.current_gun_status:
                                    #print(f"WEAPON detected! Confidence: {self.gun_confidence:.2f}")
                                    pass
                                else:
                                    #print(f"No weapons - Confidence: {self.gun_confidence:.2f}")
                                    pass
                        
                        # Update temperature values based on current detection status
                        self.update_temperatures()
                        
                        # In headless mode, log detection status periodically
                        if self.headless_mode and frame_count % 60 == 0:  # Every 2.5 seconds at 24 FPS
                            status_parts = []
                            if self.nsfw_detection_available:
                                status_parts.append(f"NSFW: {'DETECTED' if self.current_nsfw_status else 'SAFE'} ({self.nsfw_confidence:.2f})")
                            if self.gun_detection_available:
                                status_parts.append(f"Weapon: {'DETECTED' if self.current_gun_status else 'SAFE'} ({self.gun_confidence:.2f})")
                            
                            temp_parts = []
                            if self.nsfw_detection_available:
                                temp_parts.append(f"NSFW: {self.nsfw_temperature:.1f}°")
                            if self.gun_detection_available:
                                temp_parts.append(f"Weapon: {self.arm_temperature:.1f}°")
                            if self.transcription_available:
                                temp_parts.append(f"Abusive: {self.abusive_language_temperature:.1f}°")
                            
                            if status_parts:
                                print(f"[STATUS] {' | '.join(status_parts)}")
                            if temp_parts:
                                print(f"[TEMPS] {' | '.join(temp_parts)}")
                        
                        # Apply blur if NSFW content detected (even in headless mode for consistency)
                        if self.current_nsfw_status:
                            frame = self.blur_frame(frame)
                            
                        # Draw bounding boxes if guns are detected (even in headless mode for consistency)
                        if self.current_gun_status and self.gun_objects:
                            frame = self.draw_gun_boxes(frame, self.gun_objects)
                        
                        # Skip video display and related processing in headless mode
                        if not self.headless_mode:
                            try:
                                # Calculate and display FPS
                                current_tick = cv2.getTickCount()
                                elapsed_time = (current_tick - start_time) / cv2.getTickFrequency()
                                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                                
                                # === UPPER LEFT: TEMPERATURE DISPLAYS ===
                                temp_start_y = 40
                                temp_spacing = 60  # Increased spacing between temperature rows to prevent overlap
                                current_temp_y = temp_start_y
                                icon_size = 24
                                icon_margin = 15  # Space between icon and text
                                text_bar_spacing = 8  # Small space between text and bar
                                
                                # NSFW Temperature Display
                                if self.nsfw_detection_available:
                                    # Determine color first before using it
                                    temp_color = (0, 255, 0)  # Green for safe
                                    if self.nsfw_temperature > 50:
                                        temp_color = (0, 165, 255)  # Orange for warning
                                    if self.nsfw_temperature > 75:
                                        temp_color = (0, 0, 255)  # Red for danger
                                    
                                    # Draw NSFW icon
                                    icon_x = 15
                                    icon_y = current_temp_y - icon_size//2
                                    self.draw_icon(frame, icon_x, icon_y, "nsfw", temp_color, icon_size)
                                    
                                    # Draw NSFW text
                                    text_x = icon_x + icon_size + icon_margin
                                    cv2.putText(frame, f"NSFW: {self.nsfw_temperature:.1f}%", (text_x, current_temp_y), 
                                               cv2.FONT_HERSHEY_DUPLEX, 0.8, temp_color, 2)
                                    
                                    # Draw temperature bar BELOW the text
                                    bar_x = text_x
                                    bar_y = current_temp_y + text_bar_spacing  # Position bar below text
                                    bar_width = 200
                                    bar_height = 25
                                    # Draw bar background
                                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 2)
                                    # Draw temperature fill
                                    fill_width = int((self.nsfw_temperature / 100.0) * bar_width)
                                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), temp_color, -1)
                                    
                                    current_temp_y += temp_spacing
                                
                                # Weapon Temperature Display
                                if self.gun_detection_available:
                                    # Determine color first before using it
                                    temp_color = (0, 255, 0)  # Green for safe
                                    if self.arm_temperature > 50:
                                        temp_color = (0, 165, 255)  # Orange for warning
                                    if self.arm_temperature > 75:
                                        temp_color = (0, 0, 255)  # Red for danger
                                    
                                    # Draw Weapon icon
                                    icon_x = 15
                                    icon_y = current_temp_y - icon_size//2
                                    self.draw_icon(frame, icon_x, icon_y, "weapon", temp_color, icon_size)
                                    
                                    # Draw Weapon text
                                    text_x = icon_x + icon_size + icon_margin
                                    cv2.putText(frame, f"Weapon: {self.arm_temperature:.1f}%", (text_x, current_temp_y), 
                                               cv2.FONT_HERSHEY_DUPLEX, 0.8, temp_color, 2)
                                    
                                    # Draw temperature bar BELOW the text
                                    bar_x = text_x
                                    bar_y = current_temp_y + text_bar_spacing  # Position bar below text
                                    bar_width = 200
                                    bar_height = 25
                                    # Draw bar background
                                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 2)
                                    # Draw temperature fill
                                    fill_width = int((self.arm_temperature / 100.0) * bar_width)
                                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), temp_color, -1)
                                    
                                    current_temp_y += temp_spacing
                                
                                # Abusive Language Temperature Display
                                if self.transcription_available and self.config.is_enabled('profanity_filter'):
                                    # Determine color first before using it
                                    temp_color = (0, 255, 0)  # Green for safe
                                    if self.abusive_language_temperature > 50:
                                        temp_color = (0, 165, 255)  # Orange for warning
                                    if self.abusive_language_temperature > 75:
                                        temp_color = (0, 0, 255)  # Red for danger
                                    
                                    # Draw Abusive icon
                                    icon_x = 15
                                    icon_y = current_temp_y - icon_size//2
                                    self.draw_icon(frame, icon_x, icon_y, "abusive", temp_color, icon_size)
                                    
                                    # Draw Abusive text
                                    text_x = icon_x + icon_size + icon_margin
                                    cv2.putText(frame, f"Abusive: {self.abusive_language_temperature:.1f}%", (text_x, current_temp_y), 
                                               cv2.FONT_HERSHEY_DUPLEX, 0.8, temp_color, 2)
                                    
                                    # Draw temperature bar BELOW the text
                                    bar_x = text_x
                                    bar_y = current_temp_y + text_bar_spacing  # Position bar below text
                                    bar_width = 200
                                    bar_height = 25
                                    # Draw bar background
                                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 2)
                                    # Draw temperature fill
                                    fill_width = int((self.abusive_language_temperature / 100.0) * bar_width)
                                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), temp_color, -1)
                                    
                                    current_temp_y += temp_spacing
                                
                                # === OVERALL THREAT LEVEL ===
                                # Calculate overall threat level as maximum of all enabled temperatures
                                enabled_temperatures = []
                                if self.nsfw_detection_available:
                                    enabled_temperatures.append(self.nsfw_temperature)
                                if self.gun_detection_available:
                                    enabled_temperatures.append(self.arm_temperature)
                                if self.transcription_available and self.config.is_enabled('profanity_filter'):
                                    enabled_temperatures.append(self.abusive_language_temperature)
                                
                                # Only show overall temperature if at least one individual temperature is enabled
                                if enabled_temperatures:
                                    overall_temperature = max(enabled_temperatures)
                                    
                                    # Determine color based on overall temperature
                                    temp_color = (0, 255, 0)  # Green for safe
                                    if overall_temperature > 50:
                                        temp_color = (0, 165, 255)  # Orange for warning
                                    if overall_temperature > 75:
                                        temp_color = (0, 0, 255)  # Red for danger
                                    
                                    # Draw Overall threat icon
                                    icon_x = 15
                                    icon_y = current_temp_y - icon_size//2
                                    self.draw_icon(frame, icon_x, icon_y, "overall", temp_color, icon_size)
                                    
                                    # Draw Overall threat text
                                    text_x = icon_x + icon_size + icon_margin
                                    cv2.putText(frame, f"Overall: {overall_temperature:.1f}%", (text_x, current_temp_y), 
                                               cv2.FONT_HERSHEY_DUPLEX, 0.8, temp_color, 2)
                                    
                                    # Draw temperature bar BELOW the text
                                    bar_x = text_x
                                    bar_y = current_temp_y + text_bar_spacing  # Position bar below text
                                    bar_width = 200
                                    bar_height = 25
                                    # Draw bar background
                                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 2)
                                    # Draw temperature fill
                                    fill_width = int((overall_temperature / 100.0) * bar_width)
                                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), temp_color, -1)
                                    
                                    current_temp_y += temp_spacing
                                
                                # === UPPER RIGHT: TEXT INFO AND STATUS ===
                                info_start_x = frame.shape[1] - 320  # Slightly wider for better spacing
                                info_start_y = 40
                                info_spacing = 30  # Increased spacing
                                current_info_y = info_start_y
                                
                                # Performance Info with icon
                                icon_x = info_start_x - 30
                                icon_y = current_info_y - 12
                                self.draw_icon(frame, icon_x, icon_y, "fps", (0, 255, 0), 20)
                                cv2.putText(frame, f"FPS: {fps:.1f}", (info_start_x, current_info_y), 
                                           cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
                                current_info_y += info_spacing
                                
                                cv2.putText(frame, f"Frames: {frame_count}", (info_start_x, current_info_y), 
                                           cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
                                current_info_y += info_spacing
                                
                                # NSFW Detection Status
                                if self.nsfw_detection_available:
                                    if self.current_nsfw_status:
                                        cv2.putText(frame, f"NSFW: {self.nsfw_confidence:.2f}", (info_start_x, current_info_y), 
                                                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                                        current_info_y += info_spacing
                                        cv2.putText(frame, "BLURRED", (info_start_x, current_info_y), 
                                                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                                    else:
                                        cv2.putText(frame, f"SFW: {self.nsfw_confidence:.2f}", (info_start_x, current_info_y), 
                                                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
                                    current_info_y += info_spacing
                                else:
                                    cv2.putText(frame, "NSFW: Disabled", (info_start_x, current_info_y), 
                                               cv2.FONT_HERSHEY_DUPLEX, 0.7, (128, 128, 128), 2)
                                    current_info_y += info_spacing
                                
                                # Gun Detection Status
                                if self.gun_detection_available:
                                    if self.current_gun_status:
                                        cv2.putText(frame, f"WEAPON: {self.gun_confidence:.2f}", (info_start_x, current_info_y), 
                                                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                                        current_info_y += info_spacing
                                        cv2.putText(frame, f"Objects: {len(self.gun_objects)}", (info_start_x, current_info_y), 
                                                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                                    else:
                                        cv2.putText(frame, "No weapons detected", (info_start_x, current_info_y), 
                                                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
                                    current_info_y += info_spacing
                                else:
                                    cv2.putText(frame, "Weapon: Disabled", (info_start_x, current_info_y), 
                                               cv2.FONT_HERSHEY_DUPLEX, 0.7, (128, 128, 128), 2)
                                    current_info_y += info_spacing
                                
                                # Show recently detected abusive words (if available)
                                if self.transcription_available and self.abusive_words_detected:
                                    recent_words = self.abusive_words_detected[-3:]  # Last 3 words to fit better
                                    words_text = "Recent: " + ", ".join([f"*{word}*" for word in recent_words])
                                    cv2.putText(frame, words_text[:40], (info_start_x, current_info_y), 
                                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), 2)
                                    current_info_y += info_spacing
                                
                                # Display live subtitles at the bottom of the frame
                                if self.transcription_available:
                                    current_time = time.time()
                                    with self.transcription_lock:
                                        # Check if we should still display the current subtitle
                                        if current_time - self.last_subtitle_time <= self.subtitle_display_time and self.current_subtitle:
                                            # Get frame dimensions
                                            frame_height, frame_width = frame.shape[:2]
                                            
                                            # Subtitle settings
                                            subtitle_text = self.current_subtitle
                                            font = cv2.FONT_HERSHEY_DUPLEX  # Better font for subtitles
                                            font_scale = 0.9  # Slightly larger
                                            thickness = 2
                                            
                                            # Split long text into multiple lines
                                            max_chars_per_line = 60
                                            words = subtitle_text.split()
                                            lines = []
                                            current_line = ""
                                            
                                            for word in words:
                                                if len(current_line + " " + word) <= max_chars_per_line:
                                                    current_line += " " + word if current_line else word
                                                else:
                                                    if current_line:
                                                        lines.append(current_line)
                                                    current_line = word
                                            
                                            if current_line:
                                                lines.append(current_line)
                                            
                                            # Display subtitle lines
                                            y_offset = frame_height - 80  # Moved up slightly
                                            for i, line in enumerate(reversed(lines[-3:])):  # Show max 3 lines
                                                # Get text size for background rectangle
                                                (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
                                                
                                                # Calculate position (center horizontally)
                                                x_pos = (frame_width - text_width) // 2
                                                y_pos = y_offset - (i * (text_height + 15))  # Increased line spacing
                                                
                                                # Draw background rectangle with better styling
                                                cv2.rectangle(frame, 
                                                            (x_pos - 15, y_pos - text_height - 8),
                                                            (x_pos + text_width + 15, y_pos + baseline + 8),
                                                            (0, 0, 0), -1)  # Black background
                                                
                                                # Draw subtitle text with better contrast
                                                cv2.putText(frame, line, (x_pos, y_pos), 
                                                           font, font_scale, (255, 255, 255), thickness)
                                
                                # Add transcription status indicator (moved to upper right info section)
                                if self.transcription_available:
                                    # Draw transcription icon
                                    icon_x = info_start_x - 30
                                    icon_y = current_info_y - 12
                                    self.draw_icon(frame, icon_x, icon_y, "abusive", (0, 255, 255), 20)
                                    cv2.putText(frame, "Transcription: ON", (info_start_x, current_info_y), 
                                               cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
                                else:
                                    cv2.putText(frame, "Transcription: OFF", (info_start_x, current_info_y), 
                                               cv2.FONT_HERSHEY_DUPLEX, 0.7, (128, 128, 128), 2)
                                current_info_y += info_spacing
                                
                                # Add synchronization status indicator (moved to upper right info section)
                                if self.sync_enabled:
                                    # Draw sync icon
                                    icon_x = info_start_x - 30
                                    icon_y = current_info_y - 12
                                    sync_color = (0, 255, 0)  # Green for good sync
                                    if abs(self.audio_delay_compensation) > 0.05:
                                        sync_color = (0, 165, 255)  # Orange for moderate drift
                                    if abs(self.audio_delay_compensation) > 0.1:
                                        sync_color = (0, 0, 255)  # Red for significant drift
                                    
                                    self.draw_icon(frame, icon_x, icon_y, "sync", sync_color, 20)
                                    sync_text = f"Sync: {self.audio_delay_compensation:+.3f}s"
                                    cv2.putText(frame, sync_text, (info_start_x, current_info_y), 
                                               cv2.FONT_HERSHEY_DUPLEX, 0.7, sync_color, 2)
                                else:
                                    cv2.putText(frame, "Sync: OFF", (info_start_x, current_info_y), 
                                               cv2.FONT_HERSHEY_DUPLEX, 0.7, (128, 128, 128), 2)
                                
                                # Resize frame to fit better on laptop screens
                                # Use a scale factor to make the window smaller
                                scale_factor = 0.9  # Reduce to 90% of original size
                                display_width = int(frame.shape[1] * scale_factor)
                                display_height = int(frame.shape[0] * scale_factor)
                                display_frame = cv2.resize(frame, (display_width, display_height))
                                
                                # Display resized frame
                                cv2.imshow('Video Stream', display_frame)
                                
                            except Exception as e:
                                # If there's an error in the overlay drawing, still show the basic frame
                                print(f"Error drawing overlays: {e}")
                                # Fallback: show basic frame without overlays
                                scale_factor = 0.9
                                display_width = int(frame.shape[1] * scale_factor)
                                display_height = int(frame.shape[0] * scale_factor)
                                display_frame = cv2.resize(frame, (display_width, display_height))
                                cv2.imshow('Video Stream', display_frame)
                        
                        # Frame rate control and user input handling (constant frame pacing)
                        if not self.headless_mode:
                            # Synchronized frame rate control
                            current_time = time.time()
                            elapsed = current_time - last_frame_time
                            
                            # Calculate target delay with minimal sync adjustment
                            target_delay = frame_time
                            
                            # Apply only very small sync adjustments to avoid disruption
                            if self.sync_enabled and abs(self.audio_delay_compensation) > 0.02:
                                # Only apply very small video timing adjustments
                                if self.audio_delay_compensation < -0.05:
                                    # Audio is significantly behind, speed up video very slightly
                                    target_delay = max(0.001, target_delay * 0.95)
                                elif self.audio_delay_compensation > 0.05:
                                    # Audio is significantly ahead, slow down video very slightly
                                    target_delay = target_delay * 1.05
                            
                            sleep_time = max(0.0, target_delay - elapsed)
                            
                            # Check for quit key with precise timing
                            # Ensure at least 1ms waitKey to allow window events
                            key = cv2.waitKey(max(1, int(sleep_time * 1000))) & 0xFF
                            if key == ord('q'):
                                #logger.info("Quit requested by user")
                                print("Quit requested by user")
                                break
                            elif key == ord('s'):
                                # Toggle synchronization
                                self.sync_enabled = not self.sync_enabled
                                status = "ENABLED" if self.sync_enabled else "DISABLED"
                                print(f"Synchronization {status}")
                                #logger.info(f"Synchronization toggled: {status}")
                                if not self.sync_enabled:
                                    self.audio_delay_compensation = 0.0  # Reset compensation
                                    
                            last_frame_time = time.time()
                        else:
                            # In headless mode, just maintain frame timing without user input
                            current_time = time.time()
                            elapsed = current_time - last_frame_time
                            target_delay = frame_time
                            sleep_time = max(0.0, target_delay - elapsed)
                            if sleep_time > 0:
                                time.sleep(sleep_time)
                            last_frame_time = time.time()
                            
                        # Log synchronization status periodically (reduced frequency)
                        if frame_count % 240 == 0:  # Every 10 seconds at 24 FPS
                            actual_fps = frame_count / (time.time() - start_time)
                            if self.headless_mode:
                                print(f"[PERF] FPS: {actual_fps:.1f}, Frames: {frame_count}")
                            #logger.info(f"Performance - FPS: {actual_fps:.1f}, Sync compensation: {self.audio_delay_compensation:.3f}s")
                    else:
                        #logger.warning("Error: Received invalid frame")
                        print("Error: Received invalid frame")
                        
                except Exception as e:
                    #logger.error(f"Error decoding frame: {e}")
                    print(f"Error decoding frame: {e}")
                    continue
                    
        except (socket.error, ConnectionResetError):
            #logger.error("Connection lost to server")
            print("Connection lost to server")
        except KeyboardInterrupt:
            #logger.info("Stopping receiver...")
            print("\nStopping receiver...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Stop the NSFW detection thread if it's running
            if self.nsfw_detection_running:
                #logger.info("Stopping NSFW detection thread...")
                self.nsfw_detection_running = False
                
                # Wait for the thread to finish (with timeout)
                if self.nsfw_detection_thread and self.nsfw_detection_thread.is_alive():
                    self.nsfw_detection_thread.join(timeout=2.0)
                    logger.info("NSFW detection thread stopped" if not self.nsfw_detection_thread.is_alive() 
                              else "NSFW detection thread timeout - continuing cleanup")
            
            # Clear GPU resources if using GPU
            if GPU_UTILS_AVAILABLE and hasattr(self, 'nsfw_device') and self.nsfw_device != 'cpu':
                logger.info("Clearing GPU resources...")
                clear_gpu_cache()
                
            # Clear model references to free memory
            if hasattr(self, 'nsfw_pipeline'):
                del self.nsfw_pipeline
            if hasattr(self, 'whisper_model'):
                del self.whisper_model
            if hasattr(self, 'weapon_model'):
                del self.weapon_model
            
            # Stop the gun detection thread if it's running
            if self.gun_detection_running:
                #logger.info("Stopping gun detection thread...")
                self.gun_detection_running = False
                
                # Wait for the thread to finish (with timeout)
                if self.gun_detection_thread and self.gun_detection_thread.is_alive():
                    self.gun_detection_thread.join(timeout=2.0)
                    logger.info("Gun detection thread stopped" if not self.gun_detection_thread.is_alive() 
                              else "Gun detection thread timeout - continuing cleanup")
            
            # Stop the transcription thread if it's running
            if hasattr(self, 'transcription_running') and self.transcription_running:
                #logger.info("Stopping transcription thread...")
                self.transcription_running = False
                
                # Wait for the thread to finish (with timeout)
                if hasattr(self, 'transcription_thread') and self.transcription_thread and self.transcription_thread.is_alive():
                    self.transcription_thread.join(timeout=2.0)
                    logger.info("Transcription thread stopped" if not self.transcription_thread.is_alive() 
                              else "Transcription thread timeout - continuing cleanup")
                
                # Finalize transcript file
                if hasattr(self, 'transcript_file'):
                    try:
                        with open(self.transcript_file, 'a', encoding='utf-8') as f:
                            f.write(f"\n=== Transcript Ended at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                        logger.info(f"Transcript finalized and saved to: {self.transcript_file}")
                    except Exception as e:
                        logger.error(f"Error finalizing transcript: {e}")
            
            # Stop the audio playback thread if it's running
            if self.audio_playing:
                logger.info("Stopping audio playback thread...")
                self.audio_playing = False
                
                # Terminate audio process if running
                if self.audio_process:
                    try:
                        if self.audio_process.stdin:
                            self.audio_process.stdin.close()
                        self.audio_process.terminate()
                        self.audio_process.wait(timeout=2.0)
                        logger.info("Audio process terminated")
                    except:
                        self.audio_process.kill()
                        logger.info("Audio process killed")
                
                # Wait for the thread to finish (with timeout)
                if hasattr(self, 'audio_thread') and self.audio_thread.is_alive():
                    self.audio_thread.join(timeout=2.0)
                    logger.info("Audio playback thread stopped" if not self.audio_thread.is_alive() 
                              else "Audio playback thread timeout - continuing cleanup")
            
            cv2.destroyAllWindows() if not self.headless_mode else None
            self.socket.close()
            self.audio_socket.close()
            logger.info("Receiver closed successfully")
            print("Receiver closed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            pass

def main():
    # Parse command line arguments
    args = parse_command_line_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Apply command line overrides
    apply_cli_overrides(config, args)
    
    # Save configuration if requested
    if args.save_config:
        config.save_config()
        print("Configuration saved. Exiting.")
        return
    
    # Log GPU information at startup
    if GPU_UTILS_AVAILABLE:
        log_gpu_info()
    
    # Print current configuration
    logger.info("=== Starting Video Receiver with Content Detection ===")
    logger.info("Current configuration:")
    logger.info(f"  Headless Mode: {'Enabled' if config.get('video', 'headless_mode') else 'Disabled'}")
    logger.info(f"  NSFW Detection: {'Enabled' if config.is_enabled('nsfw_detection') else 'Disabled'}")
    if config.is_enabled('nsfw_detection'):
        nsfw_device = config.get('nsfw_detection', 'device')
        logger.info(f"    Device: {nsfw_device}")
    logger.info(f"  Gun Detection: {'Enabled' if config.is_enabled('gun_detection') else 'Disabled'}")
    logger.info(f"  Audio Transcription: {'Enabled' if config.is_enabled('transcription') else 'Disabled'}")
    if config.is_enabled('transcription'):
        whisper_device = config.get('transcription', 'device')
        compute_type = config.get('transcription', 'compute_type')
        logger.info(f"    Device: {whisper_device}")
        logger.info(f"    Compute Type: {compute_type}")
    logger.info(f"  Profanity Filter: {'Enabled' if config.is_enabled('profanity_filter') else 'Disabled'}")
    logger.info(f"  Network: {config.get('network', 'host')}:{config.get('network', 'video_port')}")
    
    # Create receiver with configuration
    receiver = VideoReceiver(config)
    
    if receiver.connect_to_server():
        try:
            receiver.receive_video()
        except KeyboardInterrupt:
            logger.info("Shutting down receiver...")
            print("\nShutting down receiver...")
            receiver.cleanup()

if __name__ == "__main__":
    main()