"""
Configuration module for video streaming with content detection
"""

import argparse
import json
import os
from typing import Dict, Any

# Import GPU utilities for automatic device detection
try:
    from gpu_utils import detect_best_device, optimize_whisper_settings, optimize_transformers_device
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False

# Get optimal device settings
if GPU_UTILS_AVAILABLE:
    _device_info = detect_best_device()
    _optimal_device, _optimal_compute_type = optimize_whisper_settings(_device_info)
    _transformers_device = optimize_transformers_device(_device_info)
else:
    _optimal_device = "cpu"
    _optimal_compute_type = "int8"
    _transformers_device = "cpu"

# Default configuration values with auto-detected optimal settings
DEFAULT_CONFIG = {
    "audio": {
        "enabled": True
    },
    "nsfw_detection": {
        "enabled": True,
        "time_interval": 0.5,  # Check every 0.5 seconds
        "blur_strength": 51,   # Blur strength for NSFW frames
        "device": _transformers_device  # Auto-detected optimal device for transformers
    },
    "gun_detection": {
        "enabled": True,
        "time_interval": 2.0,  # Check every 2 seconds
        "confidence_threshold": 0.4  # Minimum confidence for detection
    },
    "transcription": {
        "enabled": True,
        "whisper_model": "small",  # Options: tiny, base, small, medium, large
        "device": _optimal_device,  # Auto-detected optimal device
        "compute_type": _optimal_compute_type,  # Auto-detected optimal compute type
        "chunk_duration": 2.0,  # Process audio in chunks (seconds)
        "subtitle_duration": 3.0  # How long to show subtitles
    },
    "profanity_filter": {
        "enabled": True,
        "wordlist_file": "profanity_wordlist.txt",
        "replacement_char": "*"
    },
    "audio_sync": {
        "enabled": True,
        "max_drift": 0.05,  # Maximum allowed drift in seconds (tighter sync)
        "delay_compensation": 0.0,  # Audio delay compensation
        "buffer_size": 3  # Smaller buffer for tighter sync
    },
    "network": {
        "host": "localhost",
        "video_port": 9999,
        "audio_port": 9998
    },
    "video": {
        "target_fps": 25,
        "jpeg_quality": 60,
        "scale_factor": 0.5,
        "headless_mode": False  # Run without video display window
    }
}

class Config:
    """Configuration manager for the video streaming application"""
    
    def __init__(self, config_file: str = "stream_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # Merge with defaults to ensure all keys exist
                return self._merge_configs(DEFAULT_CONFIG, config)
            except Exception as e:
                print(f"Error loading config file: {e}")
                print("Using default configuration")
        
        return DEFAULT_CONFIG.copy()
    
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge user config with defaults"""
        result = default.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            print(f"Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, section: str, key: str = None):
        """Get configuration value"""
        if key is None:
            return self.config.get(section, {})
        return self.config.get(section, {}).get(key)
    
    def set(self, section: str, key: str, value: Any):
        """Set configuration value"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def is_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled"""
        return self.config.get(feature, {}).get('enabled', False)

def parse_command_line_args():
    """Parse command line arguments and return configuration overrides"""
    parser = argparse.ArgumentParser(
        description="Video Streaming with Content Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with all features enabled (default)
  python receiver.py
  
  # Disable NSFW detection
  python receiver.py --no-nsfw
  
  # Disable all content detection
  python receiver.py --no-nsfw --no-gun --no-transcription --no-profanity
  
  # Use custom ports
  python receiver.py --video-port 8888 --audio-port 8887
  
  # Use different Whisper model
  python receiver.py --whisper-model base
        """
    )
    
    # Feature toggles
    parser.add_argument('--no-audio', action='store_true',
                        help='Disable audio entirely (also disables sync and transcription)')
    parser.add_argument('--no-nsfw', action='store_true',
                        help='Disable NSFW detection')
    parser.add_argument('--no-gun', action='store_true',
                        help='Disable gun/weapon detection')
    parser.add_argument('--no-transcription', action='store_true',
                        help='Disable audio transcription')
    parser.add_argument('--no-profanity', action='store_true',
                        help='Disable profanity filtering')
    
    # Enable flags (for when default is disabled in config)
    parser.add_argument('--enable-nsfw', action='store_true',
                        help='Enable NSFW detection (overrides config)')
    parser.add_argument('--enable-gun', action='store_true',
                        help='Enable gun detection (overrides config)')
    parser.add_argument('--enable-transcription', action='store_true',
                        help='Enable transcription (overrides config)')
    parser.add_argument('--enable-profanity', action='store_true',
                        help='Enable profanity filtering (overrides config)')
    
    # Network settings
    parser.add_argument('--host', type=str, default=None,
                        help='Server host address')
    parser.add_argument('--video-port', type=int, default=None,
                        help='Video streaming port')
    parser.add_argument('--audio-port', type=int, default=None,
                        help='Audio streaming port')
    
    # Model settings
    parser.add_argument('--whisper-model', type=str, 
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size for transcription')
    parser.add_argument('--device', type=str,
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device for model inference (auto=best available, cpu, cuda for NVIDIA GPU, mps for Apple Silicon)')
    parser.add_argument('--compute-type', type=str,
                        choices=['auto', 'int8', 'float16', 'float32'],
                        help='Compute type for models (auto=optimal for device)')
    # Audio/Video sync settings
    parser.add_argument('--no-sync', action='store_true',
                        help='Disable audio-video synchronization (fixes timing issues)')
    parser.add_argument('--chunk-duration', type=float,
                        help='Audio processing chunk duration in seconds (smaller = faster response)')
    
    # Video/Display settings
    parser.add_argument('--headless', action='store_true',
                        help='Run without video display window (logs only)')
    parser.add_argument('--fps', type=int,
                        help='Target video framerate')
    parser.add_argument('--jpeg-quality', type=int,
                        help='JPEG compression quality (1-100)')
    
    # Configuration file
    parser.add_argument('--config', type=str, default='stream_config.json',
                        help='Configuration file path')
    parser.add_argument('--save-config', action='store_true',
                        help='Save current settings to config file')
    
    return parser.parse_args()

def apply_cli_overrides(config: Config, args):
    """Apply command line argument overrides to configuration"""
    
    # Feature toggles (disable flags)
    if args.no_audio:
        config.set('audio', 'enabled', False)
        config.set('audio_sync', 'enabled', False)
        config.set('transcription', 'enabled', False)
    if args.no_nsfw:
        config.set('nsfw_detection', 'enabled', False)
    if args.no_gun:
        config.set('gun_detection', 'enabled', False)
    if args.no_transcription:
        config.set('transcription', 'enabled', False)
    if args.no_profanity:
        config.set('profanity_filter', 'enabled', False)
    
    # Feature toggles (enable flags - these override disable flags)
    if args.enable_nsfw:
        config.set('nsfw_detection', 'enabled', True)
    if args.enable_gun:
        config.set('gun_detection', 'enabled', True)
    if args.enable_transcription:
        config.set('transcription', 'enabled', True)
    if args.enable_profanity:
        config.set('profanity_filter', 'enabled', True)
    
    # Network settings
    if args.host:
        config.set('network', 'host', args.host)
    if args.video_port:
        config.set('network', 'video_port', args.video_port)
    if args.audio_port:
        config.set('network', 'audio_port', args.audio_port)
    
    # Model settings
    if args.whisper_model:
        config.set('transcription', 'whisper_model', args.whisper_model)
    if args.chunk_duration:
        config.set('transcription', 'chunk_duration', args.chunk_duration)
    
    # Audio/Video sync settings
    if args.no_sync:
        config.set('audio_sync', 'enabled', False)
    
    # Video/Display settings
    if args.headless:
        config.set('video', 'headless_mode', True)
    if args.fps:
        config.set('video', 'target_fps', args.fps)
    if args.jpeg_quality:
        config.set('video', 'jpeg_quality', args.jpeg_quality)
    
    # Device and compute type settings
    if args.device:
        if args.device == 'auto' and GPU_UTILS_AVAILABLE:
            # Auto-detect optimal device
            device_info = detect_best_device()
            optimal_device, optimal_compute = optimize_whisper_settings(device_info)
            transformers_device = optimize_transformers_device(device_info)
            
            config.set('transcription', 'device', optimal_device)
            config.set('transcription', 'compute_type', optimal_compute)
            config.set('nsfw_detection', 'device', transformers_device)
            print(f"Auto-detected optimal device: {optimal_device} (compute: {optimal_compute})")
        else:
            config.set('transcription', 'device', args.device)
            config.set('nsfw_detection', 'device', args.device)
    
    if args.compute_type:
        if args.compute_type == 'auto' and GPU_UTILS_AVAILABLE:
            # Auto-detect optimal compute type
            device_info = detect_best_device()
            _, optimal_compute = optimize_whisper_settings(device_info)
            config.set('transcription', 'compute_type', optimal_compute)
            print(f"Auto-detected optimal compute type: {optimal_compute}")
        else:
            config.set('transcription', 'compute_type', args.compute_type)

def create_default_config_file():
    """Create a default configuration file"""
    config = Config()
    config.save_config()
    print("Default configuration file created!")

if __name__ == "__main__":
    # Create default config file when run directly
    create_default_config_file()
