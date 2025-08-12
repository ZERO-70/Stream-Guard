import cv2
import socket
import struct
import pickle
import time
import sys
import os
import numpy as np
import threading
import subprocess
from config import Config, parse_command_line_args, apply_cli_overrides

class VideoSender:
    def __init__(self, config: Config = None):
        # Use provided config or create default
        self.config = config if config else Config()
        
        # Network settings from config
        self.host = self.config.get('network', 'host')
        self.port = self.config.get('network', 'video_port')
        self.audio_port = self.config.get('network', 'audio_port')
        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Audio streaming socket
        self.audio_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.audio_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Audio streaming parameters
        self.chunk_size = 1024
        self.streaming = False
        
        # Synchronization for looping
        self.loop_sync = threading.Event()
        self.video_duration = 0
        self.audio_duration = 0
        self.loop_start_time = 0
        
    def start_server(self):
        """Start the video and audio servers and wait for client connections"""
        try:
            # Start video server
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            print(f"Video server started on {self.host}:{self.port}")
            
            # Start audio server
            self.audio_socket.bind((self.host, self.audio_port))
            self.audio_socket.listen(1)
            print(f"Audio server started on {self.host}:{self.audio_port}")
            
            print("Waiting for client connections...")
            
            # Accept video connection
            self.client_socket, self.client_address = self.socket.accept()
            print(f"Client connected to video stream from {self.client_address}")
            
            # Accept audio connection
            self.audio_client_socket, self.audio_client_address = self.audio_socket.accept()
            print(f"Client connected to audio stream from {self.audio_client_address}")
            
        except Exception as e:
            print(f"Error starting server: {e}")
            sys.exit(1)
    
    def extract_audio(self, video_path):
        """Extract audio from video file using FFmpeg"""
        print("Extracting audio from video file...")
        audio_path = 'temp_audio.raw'
        
        try:
            # Use FFmpeg to extract raw audio data
            cmd = [
                'ffmpeg', '-i', video_path,
                '-f', 's16le',  # 16-bit signed little-endian
                '-ac', '2',     # stereo
                '-ar', '44100', # 44.1kHz sample rate
                '-y',           # overwrite output files
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Audio extracted to {audio_path}")
                return audio_path
            else:
                print(f"FFmpeg error: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None
    
    def stream_audio(self, audio_path):
        """Stream raw audio data to the client with synchronized looping"""
        try:
            # Send audio parameters to client
            params = {
                'channels': 2,
                'rate': 44100,
                'chunk_size': self.chunk_size
            }
            params_data = pickle.dumps(params)
            self.audio_client_socket.sendall(struct.pack("L", len(params_data)))
            self.audio_client_socket.sendall(params_data)
            
            print(f"Starting audio stream: 2 channels, 44100Hz")
            
            # Calculate audio duration
            audio_file_size = os.path.getsize(audio_path)
            samples_per_second = 44100 * 2 * 2  # 44100Hz * 2 channels * 2 bytes per sample
            self.audio_duration = audio_file_size / samples_per_second
            print(f"Audio duration: {self.audio_duration:.2f} seconds")
            
            # Stream raw audio data
            with open(audio_path, 'rb') as f:
                audio_start_time = time.time()
                
                while self.streaming:
                    try:
                        # Read chunk of audio data
                        data = f.read(self.chunk_size * 4)  # 4 bytes per sample (2 channels * 2 bytes)
                        
                        if len(data) == 0:
                            # Wait for video to signal loop restart
                            print("Audio reached end, waiting for video sync...")
                            self.loop_sync.wait()  # Wait for video to signal restart
                            self.loop_sync.clear()  # Reset the event
                            
                            # Reset audio position and timing
                            f.seek(0)
                            audio_start_time = time.time()
                            data = f.read(self.chunk_size * 4)
                            print("Audio loop restarted in sync with video")
                        
                        if len(data) > 0:
                            # Send audio chunk size and data
                            self.audio_client_socket.sendall(struct.pack("L", len(data)))
                            self.audio_client_socket.sendall(data)
                            
                            # Control audio playback speed (roughly 44100 samples per second)
                            time.sleep(self.chunk_size / 44100.0)
                        
                    except (socket.error, ConnectionResetError):
                        print("Client disconnected from audio stream")
                        break
            
        except Exception as e:
            print(f"Error streaming audio: {e}")
    
    def stream_video(self, video_path):
        """Stream video frames and audio to the client"""
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' not found!")
            print("Please place a video.mp4 file in the same directory as this script.")
            sys.exit(1)
        
        # Extract audio from video
        audio_path = self.extract_audio(video_path)
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{video_path}'")
            sys.exit(1)
        
        # Get video properties
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Use configured target FPS if provided; fall back to source FPS
        configured_fps = self.config.get('video', 'target_fps')
        target_fps = float(configured_fps) if configured_fps else (float(source_fps) if source_fps and source_fps > 0 else 24.0)
        frame_delay = 1.0 / target_fps
        # Duration is based on source fps if available
        self.video_duration = (total_frames / float(source_fps)) if source_fps and source_fps > 0 else (total_frames * frame_delay)
        
        print(f"Video FPS (source): {source_fps}")
        print(f"Target playback FPS: {target_fps}")
        print(f"Video duration: {self.video_duration:.2f} seconds")
        print(f"Frame delay: {frame_delay:.3f} seconds")
        print("Starting video and audio streams...")
        
        # Flag to control audio streaming
        self.streaming = True
        
        # Start audio streaming in a separate thread
        audio_thread = threading.Thread(target=self.stream_audio, args=(audio_path,), daemon=True)
        audio_thread.start()
        
        frame_count = 0
        loop_count = 0
        
        try:
            while True:
                loop_start_time = time.time()
                frame_count = 0
                
                # Reset video position
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                print(f"Starting video loop #{loop_count + 1}")
                
                # Signal audio to restart its loop
                if loop_count > 0:  # Don't signal on first loop
                    self.loop_sync.set()
                
                # Stream one complete video loop
                while True:
                    # Maintain a constant frame rate by skipping/dropping frames when behind schedule
                    elapsed_since_loop = time.time() - loop_start_time
                    expected_index = int(elapsed_since_loop / frame_delay)

                    # If we're behind, jump ahead to the expected frame index to keep real-time pace
                    if expected_index > frame_count:
                        # Fast seek to the expected frame index (drop frames)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, min(expected_index, total_frames - 1))
                        frame_count = expected_index

                    ret, frame = cap.read()
                    if not ret:
                        # End of video reached
                        print(f"Video loop #{loop_count + 1} completed")
                        break
                    
                    # Optional: downscale to reduce bandwidth/CPU if configured
                    try:
                        scale_factor = self.config.get('video', 'scale_factor')
                        if scale_factor and float(scale_factor) > 0 and float(scale_factor) != 1.0:
                            frame = cv2.resize(
                                frame,
                                (int(frame.shape[1] * float(scale_factor)), int(frame.shape[0] * float(scale_factor)))
                            )
                    except Exception:
                        pass

                    # Encode frame as JPEG with quality from config
                    jpeg_quality = self.config.get('video', 'jpeg_quality')
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                    _, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
                    
                    # Prepare frame data
                    frame_data = pickle.dumps(encoded_frame)
                    frame_size = len(frame_data)
                    
                    # Send frame size first, then frame data
                    try:
                        self.client_socket.sendall(struct.pack("L", frame_size))
                        self.client_socket.sendall(frame_data)
                        
                        frame_count += 1
                        
                        # Maintain real-time playback schedule for next frame
                        elapsed_time = time.time() - loop_start_time
                        next_frame_time = frame_count * frame_delay
                        if next_frame_time > elapsed_time:
                            time.sleep(next_frame_time - elapsed_time)
                        
                    except (socket.error, ConnectionResetError):
                        print("Client disconnected")
                        self.streaming = False
                        break
                
                if not self.streaming:
                    break
                    
                loop_count += 1
                
                # Small pause between loops to ensure sync
                time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nStreaming stopped by user")
        finally:
            self.streaming = False
            cap.release()
            # Clean up temporary audio file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    print(f"Removed temporary audio file: {audio_path}")
                except:
                    pass
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.streaming = False
            if hasattr(self, 'client_socket'):
                self.client_socket.close()
            if hasattr(self, 'audio_client_socket'):
                self.audio_client_socket.close()
            self.socket.close()
            self.audio_socket.close()
            print("Video and audio servers closed")
        except Exception as e:
            print(f"Error during cleanup: {e}")
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
    
    # Print current configuration
    print("=== Starting Video Sender ===")
    print("Current configuration:")
    print(f"  Network: {config.get('network', 'host')}:{config.get('network', 'video_port')}")
    print(f"  Video Quality: {config.get('video', 'jpeg_quality')}%")
    print(f"  Target FPS: {config.get('video', 'target_fps')}")
    
    # Create sender with configuration
    sender = VideoSender(config)
    
    try:
        sender.start_server()
        sender.stream_video('video.mp4')
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sender.cleanup()

if __name__ == "__main__":
    main() 