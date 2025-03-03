import os
import cv2
import threading
import queue
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import torch
import librosa
import soundfile as sf
import moviepy
from transformers import pipeline
import pygame

# Import local modules
from copyright_detection import detect_copyright
from keyframe import extract_key_frames,blur_copyright_frames
from asr import transcribe_audio, create_folders, extract_audio, split_audio, save_transcript
from audio_copyright import predict_audio
from nsfw_detection import detect_nsfw_content, blur_nsfw_regions
from nsfw_detection import process_nsfw_temporal


os.environ["IMAGEIO_FFMPEG_EXE"] = "ffmpeg"


class VideoProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Content Moderator")
        self.root.geometry("1280x720")
        
        # Create UI elements
        self.setup_ui()
        
        # Check for GPU availability
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU detected: {gpu_name}")
            # Update status after UI is created
            self.root.after(100, lambda: self.status_var.set(f"GPU acceleration available: {gpu_name}"))
        else:
            print("No GPU detected, using CPU")
        
        # Initialize processing variables
        self.video_path = None
        self.blurred_video_path = None
        self.is_playing = False
        self.cap = None
        self.processing_results = {
            "copyright_images": [],  # Will store (timestamp, frame_path, result)
            "copyright_audio": "Unknown",
            "transcriptions": []
        }
        
        # Create output directories
        os.makedirs("temp", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
        # Initialize thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Setup ASR model (load once)
        if self.use_gpu:
            self.asr_model = pipeline(model="openai/whisper-base", device="cuda")
        else:
            self.asr_model = pipeline(model="openai/whisper-base")
            
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video selection frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Select video button
        select_btn = ttk.Button(control_frame, text="Select Video", command=self.select_video)
        select_btn.pack(side=tk.LEFT, padx=5)
        
        # Process button
        self.process_btn = ttk.Button(control_frame, text="Process Video", command=self.process_video, state=tk.DISABLED)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        
        # Play button
        self.play_btn = ttk.Button(control_frame, text="Play", command=self.toggle_play, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        # Add Preview Blur button after the Play button
        self.preview_blur_btn = ttk.Button(
            control_frame, 
            text="Preview & Edit Blur", 
            command=self.preview_blur, 
            state=tk.DISABLED
        )
        self.preview_blur_btn.pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_var = tk.StringVar(value="Select a video to begin")
        status_label = ttk.Label(control_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT, padx=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, length=200, mode="determinate")
        self.progress.pack(side=tk.RIGHT, padx=5)
        
        # Create split view
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video frame (left side)
        video_container = ttk.LabelFrame(content_frame, text="Video")
        video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.video_canvas = tk.Canvas(video_container, background="black")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Results frame (right side)
        results_frame = ttk.LabelFrame(content_frame, text="Analysis Results")
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create notebook for different result types
        result_tabs = ttk.Notebook(results_frame)
        result_tabs.pack(fill=tk.BOTH, expand=True)
        
        # Copyright tab
        copyright_frame = ttk.Frame(result_tabs)
        result_tabs.add(copyright_frame, text="Copyright")
        
        # Scrollable copyright text
        copyright_scroll = ttk.Scrollbar(copyright_frame)
        copyright_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.copyright_text = tk.Text(copyright_frame, height=20, width=50, wrap=tk.WORD)
        self.copyright_text.pack(fill=tk.BOTH, expand=True)
        self.copyright_text.config(yscrollcommand=copyright_scroll.set)
        copyright_scroll.config(command=self.copyright_text.yview)
        
        # Transcript tab
        transcript_frame = ttk.Frame(result_tabs)
        result_tabs.add(transcript_frame, text="Transcript")
        
        # Scrollable transcript text
        transcript_scroll = ttk.Scrollbar(transcript_frame)
        transcript_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.transcript_text = tk.Text(transcript_frame, height=20, width=50, wrap=tk.WORD)
        self.transcript_text.pack(fill=tk.BOTH, expand=True)
        self.transcript_text.config(yscrollcommand=transcript_scroll.set)
        transcript_scroll.config(command=self.transcript_text.yview)

    def select_video(self):
        """Open file dialog to select video file"""
        self.video_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov")]
        )
        
        if self.video_path:
        # Check if video has audio
            try:
                import subprocess
                
                # Use ffprobe to check for audio streams
                cmd = [
                    "ffprobe", 
                    "-v", "error", 
                    "-select_streams", "a:0", 
                    "-show_entries", "stream=codec_type", 
                    "-of", "default=noprint_wrappers=1:nokey=1", 
                    self.video_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                has_audio = "audio" in result.stdout
                
                if not has_audio:
                    self.status_var.set("Warning: This video doesn't appear to have audio!")
                    print("No audio stream detected in video file")
                else:
                    print("Audio stream detected in video file")
                    
            except Exception as e:
                print(f"Error checking for audio streams: {e}")
            
            self.status_var.set(f"Selected: {os.path.basename(self.video_path)}")
            self.process_btn.config(state=tk.NORMAL)
            self.copyright_text.delete(1.0, tk.END)
            self.copyright_text.insert(tk.END, "Ready for analysis")
            self.transcript_text.delete(1.0, tk.END)
            
            # Reset processing results
            self.processing_results = {
                "copyright_images": [],
                "copyright_audio": "Unknown",
                "transcriptions": []
            }
    
    def process_video(self):
        """Process the entire video before playback"""
        self.process_btn.config(state=tk.DISABLED)
        self.play_btn.config(state=tk.DISABLED)
        self.status_var.set("Processing video... Please wait")
        self.progress["value"] = 0
        
        # Run processing in a separate thread to keep UI responsive
        processing_thread = threading.Thread(target=self._process_video_thread)
        processing_thread.daemon = True
        processing_thread.start()
    
    def _process_video_thread(self):
        """Background thread to process the video with parallel tasks"""
        try:
            # Create session folder
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_folder = os.path.join("results", session_id)
            os.makedirs(temp_folder, exist_ok=True)
            
            self.root.after(0, lambda: self.progress.config(value=5))
            
            # Step 1: Start key frame extraction (will be processed in parallel)
            self.root.after(0, lambda: self.status_var.set("Extracting key frames & audio..."))
            key_frames_folder = os.path.join(temp_folder, "key_frames")
            
            key_frame_future = self.thread_pool.submit(
                extract_key_frames, 
                self.video_path, 
                key_frames_folder, 
                frame_interval=24,
                threshold=0.5,                 # Slightly more sensitive
                #adaptive_threshold=True,        # Enable adaptive thresholding
                min_scene_length=15,            # Require more frames between keyframes
                use_gpu=self.use_gpu
                )
            

            # Step 2: Extract audio in parallel
            audio_path = os.path.join(temp_folder, "audio.wav")
            def extract_audio_task():
                """Extract audio from video with more detailed error logging"""
                try:
                    # Add path check
                    print(f"Extracting audio from: {self.video_path} to {audio_path}")
                    print(f"Video path exists: {os.path.exists(self.video_path)}")
                    
                    # First attempt - try with proper VideoFileClip import
                    try:
                        from moviepy import VideoFileClip
                        print("Successfully imported MoviePy")
                        
                        # Try loading the video with more verbose logging
                        print("Loading video with MoviePy...")
                        video = VideoFileClip(self.video_path)
                        print(f"Video loaded. Has audio: {video.audio is not None}")
                        
                        if video.audio:
                            print(f"Audio tracks found, writing to {audio_path}")
                            # Try writing audio without parameters first (most compatible)
                            try:
                                print("Attempting to write audio with default parameters...")
                                video.audio.write_audiofile(audio_path)
                                print(f"Audio successfully written to {audio_path}")
                            except Exception as e1:
                                print(f"Basic audio extraction failed: {e1}")
                                try:
                                    # Try with verbose=False
                                    print("Attempting with verbose=False...")
                                    video.audio.write_audiofile(audio_path, verbose=False)
                                    print("Success with verbose=False")
                                except Exception as e2:
                                    print(f"Second attempt failed: {e2}")
                                    # Try with logger=None (newer moviepy)
                                    try:
                                        print("Attempting with logger=None...")
                                        video.audio.write_audiofile(audio_path, logger=None)
                                        print("Success with logger=None")
                                    except Exception as e3:
                                        print(f"Third attempt failed: {e3}")
                            video.close()
                            
                            # Check if audio file was created and is valid
                            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                                print(f"Audio file exists with size: {os.path.getsize(audio_path)} bytes")
                                return audio_path
                            else:
                                print("Audio file was not created properly")
                        else:
                            print("No audio track found in video")
                            
                    except Exception as e:
                        print(f"MoviePy audio extraction failed with exception type {type(e).__name__}: {e}")
                        import traceback
                        traceback.print_exc()

                    # Second attempt - try using ffmpeg directly
                    print("Trying direct ffmpeg extraction...")
                    try:
                        import subprocess
                        
                        # Try with more specific ffmpeg command
                        print("Running ffmpeg with audio codec specification...")
                        cmd = [
                            "ffmpeg", 
                            "-i", self.video_path, 
                            "-vn",  # No video
                            "-acodec", "pcm_s16le",  # Force audio codec
                            "-ar", "44100",  # Sample rate
                            "-ac", "2",  # Two channels
                            "-y",  # Overwrite
                            audio_path
                        ]
                        
                        process = subprocess.run(
                            cmd, 
                            check=False, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE
                        )
                        
                        print(f"FFmpeg return code: {process.returncode}")
                        if process.stderr:
                            print(f"FFmpeg stderr: {process.stderr.decode('utf-8', errors='ignore')[:500]}")
                        
                        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                            print(f"FFmpeg created audio file: {os.path.getsize(audio_path)} bytes")
                            return audio_path
                        elif os.path.exists(audio_path):
                            print("FFmpeg created empty audio file")
                        else:
                            print("FFmpeg did not create audio file")
                            
                    except Exception as ffmpeg_error:
                        print(f"FFmpeg extraction failed: {ffmpeg_error}")
                    
                    # If we get here, create a placeholder
                    print("All extraction methods failed - creating placeholder")
                    with open(audio_path, 'w') as f:
                        f.write('')
                    return audio_path
                    
                except Exception as e:
                    print(f"Unhandled exception in audio extraction: {e}")
                    import traceback
                    traceback.print_exc()
                    open(audio_path, 'w').close()
                    return audio_path
            
            audio_future = self.thread_pool.submit(extract_audio_task)
            
            # Wait for audio extraction to complete
            audio_path = audio_future.result()
            self.root.after(0, lambda: self.progress.config(value=30))
            
            # Step 3: Process audio for copyright and prepare for transcription in parallel
            audio_copyright_results = None
            audio_copyright_future = None
            
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                # Start enhanced copyright detection
                self.root.after(0, lambda: self.status_var.set("Analyzing audio for copyright content..."))
                
                # Use the process_audio_copyright function
                from audio_copyright import process_audio_copyright
                audio_copyright_future = self.thread_pool.submit(
                    process_audio_copyright, 
                    audio_path, 
                    temp_folder
                )
                
                # Prepare audio for transcription in parallel
                chunks_folder = os.path.join(temp_folder, "audio_chunks")
                os.makedirs(chunks_folder, exist_ok=True)
                self.root.after(0, lambda: self.status_var.set("Preparing audio for transcription..."))
                audio_chunks_future = self.thread_pool.submit(split_audio, audio_path, chunks_folder)
            else:
                print("No valid audio found - skipping audio copyright check")
                # Create a dummy future that returns empty results
                audio_chunks_future = self.thread_pool.submit(lambda: [])
            
            # Wait for key frames extraction to complete
            keyframes_list = key_frame_future.result()
            self.root.after(0, lambda: self.progress.config(value=40))
            
            # Get video properties for later processing
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            print(f"Video properties: {total_frames} frames, {fps} FPS, {duration:.2f}s duration")
            
            # Step 4: Process key frames for copyright in parallel batches
            self.root.after(0, lambda: self.status_var.set("Checking frames for copyright and NSFW content..."))
            copyright_keyframes = []  # Will store keyframes with copyright issues
            nsfw_keyframes = []       # Will store keyframes with NSFW content
            self.copyright_keyframes = copyright_keyframes
            self.nsfw_keyframes = nsfw_keyframes

            total_keyframes = len(keyframes_list)
            if total_keyframes == 0:
                print("No keyframes extracted. Check video file and keyframe extraction parameters.")
                self.root.after(0, lambda: self.status_var.set("No keyframes extracted. Process failed."))
                self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
                return

            # Make sure nsfw_images exists in the processing results dictionary
            if not isinstance(self.processing_results, dict):
                self.processing_results = {}
                
            if "nsfw_images" not in self.processing_results:
                self.processing_results["nsfw_images"] = []
                
            # Process frames in parallel batches
            batch_size = 4  # Process 4 frames at a time
            for i in range(0, total_keyframes, batch_size):
                batch = keyframes_list[i:i+batch_size]
                copyright_futures = []
                nsfw_futures = []
                
                for frame_number, timestamp_seconds, frame_path in batch:
                    # Submit each frame for copyright detection
                    copyright_futures.append((
                        frame_number, 
                        timestamp_seconds, 
                        frame_path, 
                        self.thread_pool.submit(detect_copyright, frame_path, use_gpu=self.use_gpu)
                    ))
                    
                    # Submit each frame for NSFW detection
                    nsfw_futures.append((
                        frame_number, 
                        timestamp_seconds, 
                        frame_path, 
                        self.thread_pool.submit(detect_nsfw_content, frame_path, confidence_threshold=0.5, use_gpu=self.use_gpu)
                    ))
                
                # Process copyright detection results
                for frame_number, timestamp_seconds, frame_path, future in copyright_futures:
                    try:
                        result = future.result()
                        
                        # Format timestamp for display
                        minutes = int(timestamp_seconds // 60)
                        seconds = int(timestamp_seconds % 60)
                        timestamp = f"[{minutes:02}:{seconds:02}]"
                        
                        if "Copyright Issue" in result:
                            self.processing_results["copyright_images"].append(
                                (timestamp, frame_path, result)
                            )
                            copyright_keyframes.append((frame_number, timestamp_seconds, frame_path))
                            
                            # Display the timestamp in real-time
                            self.root.after(0, lambda ts=timestamp: self.copyright_text.insert(tk.END, 
                                                                    f"Copyright detected at {ts}\n"))
                    except Exception as e:
                        print(f"Error processing copyright for frame {frame_path}: {e}")
                
                # Process NSFW detection results
                for frame_number, timestamp_seconds, frame_path, future in nsfw_futures:
                    try:
                        nsfw_result = future.result()
                        
                        # Format timestamp for display
                        minutes = int(timestamp_seconds // 60)
                        seconds = int(timestamp_seconds % 60)
                        timestamp = f"[{minutes:02}:{seconds:02}]"
                        
                        if nsfw_result["has_nsfw"]:
                            self.processing_results["nsfw_images"].append(
                                (timestamp, frame_path, f"NSFW Content (Confidence: {nsfw_result['confidence']:.2f})")
                            )
                            
                            # Add to frames that need blurring
                            nsfw_keyframes.append((frame_number, timestamp_seconds, frame_path, nsfw_result["detections"]))
                            
                            # Display the timestamp in real-time
                            self.root.after(0, lambda ts=timestamp: self.copyright_text.insert(tk.END, 
                                                                    f"NSFW content detected at {ts}\n"))
                    except Exception as e:
                        print(f"Error processing NSFW detection for frame {frame_path}: {e}")
                
                # Update progress
                progress_value = 40 + ((i + len(batch)) / total_keyframes * 20)
                self.root.after(0, lambda v=progress_value: self.progress.config(value=v))
                
            # Debug info after initial NSFW detection
            print(f"Initial NSFW detection: found {len(nsfw_keyframes)} frames with NSFW content")
            
            # Step 5: Process NSFW temporal extension AFTER all keyframes have been analyzed
            if nsfw_keyframes:
                self.root.after(0, lambda: self.status_var.set("Extending NSFW detection temporally..."))
                self.root.after(0, lambda: self.progress.config(value=60))
                
                # Create a folder for temporal analysis
                nsfw_temporal_folder = os.path.join(temp_folder, "nsfw_temporal")
                os.makedirs(nsfw_temporal_folder, exist_ok=True)
                
                # Adjust confidence threshold for short videos
                is_short_video = duration < 30.0
                current_confidence = 0.6 if is_short_video else 0.5
                
                print(f"Running temporal NSFW analysis (confidence threshold: {current_confidence})")
                
                # Run temporal analysis
                extended_nsfw_frames = process_nsfw_temporal(
                    self.video_path,
                    nsfw_keyframes,
                    nsfw_temporal_folder,
                    frame_buffer=int(fps * 1.5),  # Check 1.5 seconds before and after
                    confidence_threshold=current_confidence,
                    use_gpu=self.use_gpu
                )
                
                # If we found additional frames with NSFW content
                if extended_nsfw_frames:
                    # Get frame numbers from original NSFW keyframes
                    original_frame_numbers = set(frame[0] for frame in nsfw_keyframes)
                    
                    # Count newly detected frames
                    new_frame_count = 0
                    for frame_num in extended_nsfw_frames:
                        if frame_num not in original_frame_numbers:
                            new_frame_count += 1
                            
                    print(f"Temporal analysis found {new_frame_count} additional frames with NSFW content")
                    
                    if new_frame_count > 0:
                        # Create new list with both original and additional frames
                        new_nsfw_keyframes = []
                        
                        # Keep original keyframes
                        for frame_number, timestamp_seconds, frame_path, detections in nsfw_keyframes:
                            new_nsfw_keyframes.append((frame_number, timestamp_seconds, frame_path, detections))
                        
                        # Add new frames
                        for frame_num, detections in extended_nsfw_frames.items():
                            if frame_num not in original_frame_numbers:  # Not in original list
                                timestamp_seconds = frame_num / fps
                                minutes = int(timestamp_seconds // 60)
                                seconds = int(timestamp_seconds % 60)
                                timestamp = f"[{minutes:02}:{seconds:02}]"
                                
                                # Use a placeholder path
                                frame_path = "extended_frame"
                                
                                # Add to keyframes list
                                new_nsfw_keyframes.append((frame_num, timestamp_seconds, frame_path, detections))
                                
                                # Add to results for reporting
                                self.processing_results["nsfw_images"].append(
                                    (timestamp, frame_path, f"NSFW Content (Extended detection)")
                                )
                        
                        # Replace with extended list
                        nsfw_keyframes = new_nsfw_keyframes
                        
                        # Update UI
                        self.root.after(0, lambda cnt=new_frame_count: self.copyright_text.insert(tk.END, 
                                        f"\nExtended NSFW protection: {cnt} additional frames will be blurred\n"))
                
            # Step 6: Get audio copyright results
            self.root.after(0, lambda: self.progress.config(value=70))
            clean_audio_path = None
            copyright_audio_segments = []
            
            if audio_copyright_future:
                try:
                    audio_copyright_results = audio_copyright_future.result()
                    
                    # Store detailed results
                    self.processing_results["copyright_audio"] = audio_copyright_results["result"]
                    
                    # If we have a clean audio version, use that for the blurred video
                    clean_audio_path = audio_copyright_results.get("clean_audio")
                    
                    # Store copyright segments for reporting
                    copyright_audio_segments = audio_copyright_results.get("copyright_segments", [])
                    self.processing_results["copyright_audio_segments"] = copyright_audio_segments
                    
                    # Update UI with copyright segment information
                    if copyright_audio_segments:
                        self.root.after(0, lambda: self.copyright_text.insert(tk.END, 
                                            "\nCopyrighted Audio Segments:\n"))
                        for i, (start, end) in enumerate(copyright_audio_segments):
                            start_min = int(start // 60)
                            start_sec = int(start % 60)
                            end_min = int(end // 60)
                            end_sec = int(end % 60)
                            segment_info = f"{i+1}. [{start_min:02}:{start_sec:02} - {end_min:02}:{end_sec:02}]\n"
                            self.root.after(0, lambda info=segment_info: self.copyright_text.insert(tk.END, info))
                except Exception as e:
                    print(f"Error checking audio copyright: {e}")
                    self.processing_results["copyright_audio"] = "Error checking audio copyright"
            else:
                self.processing_results["copyright_audio"] = "No audio content to check"
            
            # Step 7: Create clean version if needed
            self.blurred_video_path = None
            
            # Print verification of detected issues
            print(f"Final check before creating clean version:")
            print(f"  - Copyright frames: {len(copyright_keyframes)}")
            print(f"  - NSFW frames: {len(nsfw_keyframes)}")
            print(f"  - Copyright audio segments: {len(copyright_audio_segments)}")
            
            # Either copyright or NSFW content triggers cleaning
            if copyright_keyframes or copyright_audio_segments or nsfw_keyframes:
                self.root.after(0, lambda: self.status_var.set("Creating clean version of content..."))
                blurred_output = os.path.join(temp_folder, "clean_video.mp4")
                
                try:
                    # Call the blur function with the clean audio path if available
                    # and include NSFW keyframes
                    blur_copyright_frames(
                        self.video_path, 
                        blurred_output, 
                        self.copyright_keyframes if hasattr(self, 'copyright_keyframes') else copyright_keyframes,
                        keyframes_list, 
                        use_gpu=self.use_gpu,
                        audio_path=clean_audio_path,  # Pass the clean audio
                        nsfw_keyframes=self.nsfw_keyframes if hasattr(self, 'nsfw_keyframes') else nsfw_keyframes  # Pass NSFW frames
                    )
                    
                    self.blurred_video_path = blurred_output
                    
                    # Update UI to indicate clean version is available
                    message = "\n\nClean video created with:"
                    if copyright_keyframes:
                        message += "\n- Copyright visual content blurred"
                    if copyright_audio_segments:
                        message += "\n- Copyright audio content muted"
                    if nsfw_keyframes:
                        message += "\n- NSFW content blurred"
                    message += "\n"
                    
                    self.root.after(0, lambda msg=message: self.copyright_text.insert(tk.END, msg))
                except Exception as e:
                    print(f"Error creating clean video: {e}")
                    import traceback
                    traceback.print_exc()
            
            self.root.after(0, lambda: self.progress.config(value=80))
            
            # Step 8: Process audio chunks for transcription
            self.root.after(0, lambda: self.status_var.set("Transcribing audio..."))
            try:
                # Get audio chunks
                audio_chunks = audio_chunks_future.result()
                
                # Process chunks in parallel
                transcription_futures = []
                for chunk_path, start_time in audio_chunks:
                    # The lambda function is incorrectly defined - fix it to properly pass the chunk path
                    transcription_futures.append((
                        start_time,
                        self.thread_pool.submit(lambda p=chunk_path: self.asr_model(p)["text"], )
                    ))
                
                # Collect results
                for start_time, future in transcription_futures:
                    try:
                        transcription = future.result()
                        
                        # Format timestamp
                        minutes = int(start_time // 60)
                        seconds = int(start_time % 60)
                        timestamp = f"[{minutes:02}:{seconds:02}]"
                        
                        self.processing_results["transcriptions"].append(
                            (start_time, f"{timestamp} {transcription}")
                        )
                    except Exception as e:
                        print(f"Error transcribing chunk: {e}")
                
                # Sort transcriptions by start time
                self.processing_results["transcriptions"].sort(key=lambda x: x[0])
                
            except Exception as e:
                print(f"Error in transcription: {e}")
            
            self.root.after(0, lambda: self.progress.config(value=95))
            
            # Update the UI with all results
            self._update_results_ui()
            
            # Save results to file
            self._save_report(temp_folder)
            
            # Enable play button
            # Near the end of the method where you enable the play button:
            self.root.after(0, lambda: self.play_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.preview_blur_btn.config(state=tk.NORMAL))  # Enable preview button
            self.root.after(0, lambda: self.status_var.set(f"Processing complete! Results saved to {temp_folder}"))
            self.root.after(0, lambda: self.progress.config(value=100))
            
        except Exception as e:
            print(f"Error processing video: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))

    


    def _update_results_ui(self):
        """Update UI with processing results"""
        # Update copyright info
        self.root.after(0, lambda: self.copyright_text.delete(1.0, tk.END))
        
        copyright_text = f"Audio: {self.processing_results['copyright_audio']}\n\n"
        
        # Add detailed audio copyright segments if available
        if self.processing_results.get("copyright_audio_segments"):
            copyright_text += "Copyrighted Audio Segments:\n"
            for i, (start, end) in enumerate(self.processing_results["copyright_audio_segments"]):
                start_min = int(start // 60)
                start_sec = int(start % 60)
                end_min = int(end // 60)
                end_sec = int(end % 60)
                duration = end - start
                copyright_text += f"{i+1}. [{start_min:02}:{start_sec:02} - {end_min:02}:{end_sec:02}] " 
                copyright_text += f"(Duration: {duration:.1f}s)\n"
            copyright_text += "\n"
        
        # Create clickable timestamps for copyright frames
        self.root.after(0, lambda: self.copyright_text.insert(tk.END, copyright_text))
        
        # Show review button if we have copyright frames
        if self.processing_results["copyright_images"] or self.processing_results.get("nsfw_images"):
            # Add a review button
            self.root.after(0, lambda: self.copyright_text.insert(tk.END, 
                        "Click on a timestamp below to manually review detection:\n\n"))
        
        # Add each copyright frame as a clickable timestamp
        self.copyright_text.tag_configure("clickable", foreground="blue", underline=True)
        self.copyright_text.tag_bind("clickable", "<Button-1>", self.on_timestamp_click)
        
        # Video Frames with Copyright Issues section
        self.root.after(0, lambda: self.copyright_text.insert(tk.END, "Video Frames with Copyright Issues:\n"))
        
        # Store frame references for click handling
        self.clickable_frames = []
        
        if self.processing_results["copyright_images"]:
            for i, (timestamp, frame_path, result) in enumerate(self.processing_results["copyright_images"]):
                # Mark the frame for potential review
                frame_index = i
                frame_entry = (i, timestamp, frame_path, result, "copyright")
                self.clickable_frames.append(frame_entry)
                
                # Insert clickable timestamp
                self.root.after(0, lambda i=i, ts=timestamp, res=result, path=os.path.basename(frame_path):
                    self.copyright_text.insert(tk.END, f"{i+1}. ", "", ts, "clickable", 
                                            f" - {res} ({path})\n", ""))
        else:
            self.root.after(0, lambda: self.copyright_text.insert(tk.END, "No copyright issues detected in video frames.\n"))
        
        # NSFW content information
        self.root.after(0, lambda: self.copyright_text.insert(tk.END, "\nVideo Frames with NSFW Content:\n"))
        
        if self.processing_results.get("nsfw_images"):
            for i, (timestamp, frame_path, result) in enumerate(self.processing_results.get("nsfw_images")):
                # Add to clickable frames
                frame_index = i + len(self.processing_results["copyright_images"])
                frame_entry = (frame_index, timestamp, frame_path, result, "nsfw")
                self.clickable_frames.append(frame_entry)
                
                # Insert clickable timestamp
                self.root.after(0, lambda i=i, ts=timestamp, res=result, path=os.path.basename(frame_path):
                    self.copyright_text.insert(tk.END, f"{i+1}. ", "", ts, "clickable", 
                                            f" - {result} ({path})\n", ""))
        else:
            self.root.after(0, lambda: self.copyright_text.insert(tk.END, "No NSFW content detected in video frames.\n"))
        
        # Add Review All button if we have frames to review
        if self.clickable_frames:
            self.root.after(0, lambda: self.copyright_text.insert(tk.END, "\n"))
            self.review_button_position = self.copyright_text.index(tk.INSERT)
            self.root.after(0, lambda: self.copyright_text.insert(tk.END, "Review All Detections", "review_button"))
            self.copyright_text.tag_configure("review_button", foreground="white", background="blue")
            self.copyright_text.tag_bind("review_button", "<Button-1>", lambda e: self.start_review_workflow())
        
        # Update transcript
        self.root.after(0, lambda: self.transcript_text.delete(1.0, tk.END))
        transcript_text = "\n".join([entry[1] for entry in self.processing_results["transcriptions"]])
        self.root.after(0, lambda: self.transcript_text.insert(tk.END, transcript_text))

    def create_clean_video_with_decisions(self):
        """Create a clean video based on user decisions about what to blur"""
        # Create session folder
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_folder = os.path.join("results", session_id)
        os.makedirs(temp_folder, exist_ok=True)
        
        # Show progress
        self.root.after(0, lambda: self.status_var.set("Creating clean version with your selections..."))
        self.root.after(0, lambda: self.progress.config(value=0))
        
        # Log what we're using
        print(f"Creating clean video with user selections:")
        print(f"Copyright frames to blur: {len(self.copyright_keyframes)}")
        print(f"NSFW frames to blur: {len(self.nsfw_keyframes)}")
        
        # Create clean video
        blurred_output = os.path.join(temp_folder, "clean_video.mp4")
        
        try:
            # Call blur function with the user-selected frames
            blur_copyright_frames(
                self.video_path, 
                blurred_output, 
                self.copyright_keyframes,  # User-approved copyright frames
                [],  # No need for all keyframes  
                use_gpu=self.use_gpu,
                audio_path=None,  # Use original audio
                nsfw_keyframes=self.nsfw_keyframes  # User-approved NSFW frames
            )
            
            self.blurred_video_path = blurred_output
            
            # Update UI to indicate clean version is available
            message = "\n\nCustom clean video created with your selections."
            if self.copyright_keyframes:
                message += f"\n- {len(self.copyright_keyframes)} copyright frames blurred"
            if self.nsfw_keyframes:
                message += f"\n- {len(self.nsfw_keyframes)} NSFW frames blurred"
            
            self.root.after(0, lambda msg=message: self.copyright_text.insert(tk.END, msg))
            self.root.after(0, lambda: self.play_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.status_var.set("Custom clean video created successfully!"))
            self.root.after(0, lambda: self.progress.config(value=100))
            
        except Exception as e:
            print(f"Error creating clean video: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.status_var.set(f"Error creating clean video: {str(e)}"))
            
    def on_timestamp_click(self, event):
        """Handle click on a timestamp to review detection"""
        # Get the clicked position
        index = self.copyright_text.index(f"@{event.x},{event.y}")
        line_start = self.copyright_text.index(f"{index} linestart")
        line_end = self.copyright_text.index(f"{index} lineend")
        line_content = self.copyright_text.get(line_start, line_end)
        
        # Extract frame index from line content
        try:
            # Extract the frame number that was clicked
            frame_idx = int(line_content.split(".")[0]) - 1
            
            frame_type = None
            if "NSFW Content" in line_content:
                # Adjust index to account for copyright frames
                copyright_count = len(self.processing_results["copyright_images"])
                if frame_idx >= copyright_count:
                    frame_idx -= copyright_count
                frame_type = "nsfw"
            else:
                frame_type = "copyright"
            
            # Show the review dialog for this specific frame
            if frame_type == "copyright":
                frames_to_review = [(i, ts, path, res) for i, (ts, path, res) in 
                                enumerate(self.processing_results["copyright_images"]) 
                                if i == frame_idx]
                self.review_specific_frame(frame_idx, frames_to_review, "copyright")
            else:
                frames_to_review = [(i, ts, path, res) for i, (ts, path, res) in 
                                enumerate(self.processing_results.get("nsfw_images", [])) 
                                if i == frame_idx]
                self.review_specific_frame(frame_idx, frames_to_review, "nsfw")
                
        except Exception as e:
            print(f"Error handling timestamp click: {e}")
            import traceback
            traceback.print_exc()


    def review_frames(self, frames_to_review):
        """Review all detected frames in a unified workflow"""
        if not frames_to_review:
            print("No frames to review")
            return
            
        # Create review window
        review_window = tk.Toplevel(self.root)
        review_window.title("Review Content Detections")
        review_window.geometry("900x700")
        review_window.grab_set()  # Make modal
        
        # Initialize frames to blur
        if not hasattr(self, "copyright_keyframes") or self.copyright_keyframes is None:
            self.copyright_keyframes = []
        if not hasattr(self, "nsfw_keyframes") or self.nsfw_keyframes is None:
            self.nsfw_keyframes = []
        
        # Create a working copy of frames to blur
        frames_to_blur = {
            "copyright": list(self.copyright_keyframes[:]),
            "nsfw": list(self.nsfw_keyframes[:])
        }
        
        # Create frames
        top_frame = ttk.Frame(review_window, padding=10)
        top_frame.pack(fill=tk.X)
        
        image_frame = ttk.Frame(review_window)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        button_frame = ttk.Frame(review_window, padding=10)
        button_frame.pack(fill=tk.X)
        
        # Add instructions
        ttk.Label(top_frame, text="Review each detection and decide whether to blur it or not.", 
                wraplength=880).pack(pady=5)
        
        # Information about current frame
        info_var = tk.StringVar()
        info_label = ttk.Label(top_frame, textvariable=info_var, font=("Arial", 10, "bold"))
        info_label.pack(pady=5)
        
        # Decision status
        decision_var = tk.StringVar(value="Please make a decision for each frame")
        decision_label = ttk.Label(top_frame, textvariable=decision_var)
        decision_label.pack(pady=5)
        
        # Progress indicator
        progress_var = tk.StringVar(value="Frame 0/0")
        progress_label = ttk.Label(top_frame, textvariable=progress_var)
        progress_label.pack(pady=5)
        
        # Create canvas for image display
        canvas = tk.Canvas(image_frame, background="black")
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # Current frame index and decision tracking
        current_idx = [0]  # Use list for modification in nested functions
        decisions = {}  # frame_index: "blur" or "ignore"
        
        def show_current_frame():
            """Display the current frame"""
            if 0 <= current_idx[0] < len(frames_to_review):
                frame_info = frames_to_review[current_idx[0]]
                frame_path = frame_info["frame_path"]
                
                # Update progress indicator
                progress_var.set(f"Frame {current_idx[0] + 1}/{len(frames_to_review)}")
                
                # Display frame info
                frame_type = frame_info["type"].upper()
                timestamp = frame_info["timestamp"]
                result = frame_info["result"]
                
                info_var.set(f"{timestamp} - {frame_type}: {result}")
                
                # Show decision status if already decided
                if current_idx[0] in decisions:
                    action = "WILL BE BLURRED" if decisions[current_idx[0]] == "blur" else "WILL NOT BE BLURRED"
                    decision_var.set(f"Decision: {action}")
                else:
                    decision_var.set("Please make a decision")
                
                # Load and display image
                try:
                    # Skip for extended frames that don't have a path
                    if frame_path == "extended_frame":
                        canvas.delete("all")
                        canvas.create_text(
                            canvas.winfo_width() // 2, 
                            canvas.winfo_height() // 2,
                            text="Extended detection frame - no preview available",
                            fill="white", 
                            font=("Arial", 14)
                        )
                        return
                    
                    img = Image.open(frame_path)
                    
                    # Calculate display size
                    width, height = img.size
                    canvas_width = canvas.winfo_width()
                    canvas_height = canvas.winfo_height()
                    
                    if canvas_width > 1 and canvas_height > 1:
                        ratio = min(canvas_width / width, canvas_height / height)
                        new_width = int(width * ratio)
                        new_height = int(height * ratio)
                        img = img.resize((new_width, new_height), Image.LANCZOS)
                    
                    # Display the image
                    photo = ImageTk.PhotoImage(img)
                    canvas.delete("all")  # Clear previous image
                    canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo, anchor=tk.CENTER)
                    canvas.image = photo  # Keep reference
                    
                except Exception as e:
                    print(f"Error displaying image: {e}")
                    canvas.delete("all")
                    canvas.create_text(400, 300, text=f"Error displaying image: {e}", fill="white")
        
        # Update the mark_blur function to handle extended frames correctly
        def mark_blur():
            """Mark current frame to be blurred"""
            if 0 <= current_idx[0] < len(frames_to_review):
                frame_info = frames_to_review[current_idx[0]]
                frame_type = frame_info["type"]
                frame_path = frame_info["frame_path"]
                
                decisions[current_idx[0]] = "blur"
                decision_var.set("Decision: WILL BE BLURRED")
                
                # Handle extended frames (no path) specially
                if frame_path == "extended_frame":
                    # For extended frames, we'll create a new entry instead of trying to find existing one
                    if frame_type == "nsfw":
                        # Extract frame number and timestamp from the frame info
                        frame_number = frame_info.get("index", 0)
                        timestamp = frame_info["timestamp"]
                        # Parse timestamp like "[MM:SS]" into seconds
                        try:
                            ts_parts = timestamp.strip("[]").split(":")
                            timestamp_seconds = int(ts_parts[0]) * 60 + int(ts_parts[1])
                        except:
                            timestamp_seconds = 0
                        
                        # Add a new entry to frames_to_blur
                        new_entry = (frame_number, timestamp_seconds, frame_path, [])
                        if new_entry not in frames_to_blur["nsfw"]:
                            frames_to_blur["nsfw"].append(new_entry)
                else:
                    # For normal frames, check by path
                    if frame_type == "copyright":
                        for keyframe in self.copyright_keyframes:
                            if len(keyframe) >= 3 and os.path.exists(keyframe[2]) and os.path.exists(frame_path) and os.path.samefile(keyframe[2], frame_path):
                                if keyframe not in frames_to_blur["copyright"]:
                                    frames_to_blur["copyright"].append(keyframe)
                                break
                    elif frame_type == "nsfw":
                        for keyframe in self.nsfw_keyframes:
                            if len(keyframe) >= 3 and os.path.exists(keyframe[2]) and os.path.exists(frame_path) and os.path.samefile(keyframe[2], frame_path):
                                if keyframe not in frames_to_blur["nsfw"]:
                                    frames_to_blur["nsfw"].append(keyframe)
                                break
                
                # Automatically go to next frame after a short delay
                review_window.after(500, next_frame)
        
        def mark_ignore():
            """Mark current frame to be ignored (not blurred)"""
            if 0 <= current_idx[0] < len(frames_to_review):
                frame_info = frames_to_review[current_idx[0]]
                frame_type = frame_info["type"]
                frame_path = frame_info["frame_path"]
                
                decisions[current_idx[0]] = "ignore"
                decision_var.set("Decision: WILL NOT BE BLURRED")
                
                # Remove from frames to blur if present
                if frame_path == "extended_frame":
                    # For extended frames, locate by timestamp
                    if frame_type == "nsfw":
                        timestamp = frame_info["timestamp"]
                        frames_to_remove = []
                        
                        for i, keyframe in enumerate(frames_to_blur["nsfw"]):
                            # Convert timestamp to [MM:SS] format for comparison
                            timestamp_seconds = keyframe[1]
                            minutes = int(timestamp_seconds // 60)
                            seconds = int(timestamp_seconds % 60)
                            keyframe_timestamp = f"[{minutes:02}:{seconds:02}]"
                            
                            if keyframe_timestamp == timestamp:
                                frames_to_remove.append(i)
                        
                        # Remove entries in reverse order to avoid index shifting
                        for idx in sorted(frames_to_remove, reverse=True):
                            frames_to_blur["nsfw"].pop(idx)
                else:
                    # For normal frames with paths
                    if frame_type == "copyright":
                        frames_to_remove = []
                        for i, keyframe in enumerate(frames_to_blur["copyright"]):
                            if len(keyframe) >= 3 and os.path.exists(keyframe[2]) and os.path.exists(frame_path) and os.path.samefile(keyframe[2], frame_path):
                                frames_to_remove.append(i)
                        
                        # Remove frames in reverse order
                        for idx in sorted(frames_to_remove, reverse=True):
                            frames_to_blur["copyright"].pop(idx)
                                
                    elif frame_type == "nsfw":
                        frames_to_remove = []
                        for i, keyframe in enumerate(frames_to_blur["nsfw"]):
                            if len(keyframe) >= 3 and os.path.exists(keyframe[2]) and os.path.exists(frame_path) and os.path.samefile(keyframe[2], frame_path):
                                frames_to_remove.append(i)
                        
                        # Remove frames in reverse order
                        for idx in sorted(frames_to_remove, reverse=True):
                            frames_to_blur["nsfw"].pop(idx)
                
                # Automatically go to next frame after a short delay
                review_window.after(500, next_frame)
        
        def prev_frame():
            """Go to previous frame"""
            if current_idx[0] > 0:
                current_idx[0] -= 1
                show_current_frame()
        
        def next_frame():
            """Go to next frame"""
            if current_idx[0] < len(frames_to_review) - 1:
                current_idx[0] += 1
                show_current_frame()
            else:
                # Check if all frames have been reviewed
                if len(decisions) == len(frames_to_review):
                    decision_var.set("All frames reviewed - click Finish Review")
                else:
                    decision_var.set("Please review all frames before finishing")
        
        def finish_review():
            """Complete the review process"""
            # Check if all frames have been reviewed
            if len(decisions) < len(frames_to_review):
                missing = len(frames_to_review) - len(decisions)
                decision_var.set(f"Please review all frames. {missing} frames still need decisions.")
                return
            
            # Store the final lists of frames to blur
            self.copyright_keyframes = frames_to_blur["copyright"]
            self.nsfw_keyframes = frames_to_blur["nsfw"]
            
            # Close the window
            review_window.destroy()
            
            # Create final clean video with user decisions
            self.create_clean_video_with_decisions()
        
        # Navigation buttons
        nav_frame = ttk.Frame(button_frame)
        nav_frame.pack(side=tk.LEFT)
        
        ttk.Button(nav_frame, text="Previous", command=prev_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next", command=next_frame).pack(side=tk.LEFT, padx=5)
        
        # Decision buttons
        decision_frame = ttk.Frame(button_frame)
        decision_frame.pack(side=tk.LEFT, padx=20)
        
        blur_btn = ttk.Button(decision_frame, text="Blur This Content", command=mark_blur)
        blur_btn.pack(side=tk.LEFT, padx=5)
        
        ignore_btn = ttk.Button(decision_frame, text="Keep Content (Don't Blur)", command=mark_ignore)
        ignore_btn.pack(side=tk.LEFT, padx=5)
        
        # Finish button
        finish_btn = ttk.Button(button_frame, text="Finish Review", command=finish_review)
        finish_btn.pack(side=tk.RIGHT, padx=10)
        
        # Wait for window to fully load then update
        review_window.update()
        review_window.after(100, show_current_frame)
        
        # Wait for review to complete
        self.root.wait_window(review_window)


    def update_video_frame(self, frame):
        """Update the video frame in the UI"""
        h, w = frame.shape[:2]
        
        # Calculate aspect ratio to fit in canvas
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:  # Avoid division by zero
            # Keep aspect ratio
            ratio = min(canvas_width / w, canvas_height / h)
            new_width = int(w * ratio)
            new_height = int(h * ratio)
            
            # Resize the frame to fit the canvas
            frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert to PhotoImage
            img = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=img)
            
            # Update canvas
            self.video_canvas.create_image(
                canvas_width // 2, canvas_height // 2, 
                image=photo, anchor=tk.CENTER
            )
            self.video_canvas.image = photo  # Keep a reference


    def start_review_workflow(self):
        """Start the full review workflow for all detected frames"""
        print("Starting full content review workflow")
        
        # Initialize keyframes if not already set
        if not hasattr(self, "copyright_keyframes") or self.copyright_keyframes is None:
            # Get original keyframes from the processing results
            self.copyright_keyframes = []
            
            # Rebuild from processing results if needed
            for timestamp, frame_path, result in self.processing_results["copyright_images"]:
                # Find the frame number from the filename if possible
                frame_number = 0
                try:
                    # Try to extract frame number from filename like "frame_XXX.jpg"
                    frame_name = os.path.basename(frame_path)
                    if "frame_" in frame_name:
                        frame_number = int(frame_name.split("frame_")[1].split(".")[0])
                except:
                    pass
                
                # Convert timestamp to seconds
                timestamp_seconds = 0
                try:
                    # Convert "[MM:SS]" format to seconds
                    ts = timestamp.strip("[]").split(":")
                    timestamp_seconds = int(ts[0]) * 60 + int(ts[1])
                except:
                    pass
                
                # Add to copyright keyframes
                self.copyright_keyframes.append((frame_number, timestamp_seconds, frame_path))
        
        if not hasattr(self, "nsfw_keyframes") or self.nsfw_keyframes is None:
            self.nsfw_keyframes = []
            
            # Rebuild from processing results if needed
            for timestamp, frame_path, result in self.processing_results.get("nsfw_images", []):
                # Find the frame number from the filename if possible
                frame_number = 0
                try:
                    frame_name = os.path.basename(frame_path)
                    if "frame_" in frame_name:
                        frame_number = int(frame_name.split("frame_")[1].split(".")[0])
                except:
                    pass
                
                # Convert timestamp to seconds
                timestamp_seconds = 0
                try:
                    ts = timestamp.strip("[]").split(":")
                    timestamp_seconds = int(ts[0]) * 60 + int(ts[1])
                except:
                    pass
                
                # Add to NSFW keyframes with empty detections if it's not an extended frame
                if frame_path != "extended_frame":
                    self.nsfw_keyframes.append((frame_number, timestamp_seconds, frame_path, []))
                else:
                    # For extended frames, just use the information we have
                    self.nsfw_keyframes.append((frame_number, timestamp_seconds, frame_path, []))
        
        # Create a list of frames to review with their metadata
        frames_to_review = []
        
        # Add copyright frames
        for i, (timestamp, frame_path, result) in enumerate(self.processing_results["copyright_images"]):
            frames_to_review.append({
                "type": "copyright",
                "index": i,
                "frame_path": frame_path,
                "timestamp": timestamp,
                "result": result
            })
        
        # Add NSFW frames
        for i, (timestamp, frame_path, result) in enumerate(self.processing_results.get("nsfw_images", [])):
            frames_to_review.append({
                "type": "nsfw",
                "index": i,
                "frame_path": frame_path,
                "timestamp": timestamp,
                "result": result
            })
        
        # Launch review dialog
        self.review_frames(frames_to_review)

    def review_specific_frame(self, frame_index, frame_data, frame_type):
        """Review a specific clicked frame"""
        if not frame_data:
            print(f"No frame data for index {frame_index}, type {frame_type}")
            return
        
        idx, timestamp, frame_path, result = frame_data[0]
        
        # Get the matching keyframe from our processing results
        target_frame = None
        original_keyframes = None
        
        if frame_type == "copyright":
            # Find the relevant keyframe data from the copyright_keyframes list
            for keyframe in self.copyright_keyframes:
                frame_number, ts, path = keyframe
                if path == frame_path:
                    target_frame = keyframe
                    break
        elif frame_type == "nsfw":
            # Find in NSFW keyframes
            for keyframe in self.nsfw_keyframes:
                frame_number, ts, path, detections = keyframe
                if path == frame_path:
                    target_frame = keyframe
                    break
        
        if not target_frame:
            print(f"Could not find matching keyframe data for {frame_path}")
            return
        
        # Create a simple review dialog
        review_window = tk.Toplevel(self.root)
        review_window.title(f"Review {frame_type.upper()} Detection")
        review_window.geometry("800x600")
        
        # Create frame for image display
        image_frame = ttk.Frame(review_window)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Display the frame image
        canvas = tk.Canvas(image_frame, background="black")
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # Load and display the image
        try:
            img = Image.open(frame_path)
            
            # Calculate display size
            width, height = img.size
            canvas_width, canvas_height = 780, 450  # Approximate canvas size
            
            # Scale to fit
            ratio = min(canvas_width / width, canvas_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Display image
            photo = ImageTk.PhotoImage(img)
            canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo, anchor=tk.CENTER)
            canvas.image = photo  # Keep reference
            
        except Exception as e:
            print(f"Error displaying image: {e}")
            canvas.create_text(400, 300, text=f"Error displaying image: {e}", fill="white")
        
        # Info panel
        info_frame = ttk.Frame(review_window, padding=10)
        info_frame.pack(fill=tk.X)
        
        ttk.Label(info_frame, text=f"Timestamp: {timestamp}").pack(side=tk.LEFT, padx=5)
        ttk.Label(info_frame, text=f"Detection: {result}").pack(side=tk.LEFT, padx=5)
        
        # Decision buttons
        button_frame = ttk.Frame(review_window, padding=10)
        button_frame.pack(fill=tk.X)
        
        # Store the decision
        decision_var = tk.StringVar(value="undecided")
        
        def mark_keep():
            decision_var.set("keep")
            # Add this frame to the list that will be blurred
            if frame_type == "copyright":
                if not hasattr(self, "frames_to_blur"):
                    self.frames_to_blur = {"copyright": set(), "nsfw": set()}
                self.frames_to_blur["copyright"].add(target_frame)
            elif frame_type == "nsfw":
                if not hasattr(self, "frames_to_blur"):
                    self.frames_to_blur = {"copyright": set(), "nsfw": set()}
                self.frames_to_blur["nsfw"].add(target_frame)
            review_window.destroy()
        
        def mark_ignore():
            decision_var.set("ignore")
            # Remove this frame from the list that will be blurred
            if frame_type == "copyright":
                if hasattr(self, "frames_to_blur") and "copyright" in self.frames_to_blur:
                    self.frames_to_blur["copyright"].discard(target_frame)
            elif frame_type == "nsfw":
                if hasattr(self, "frames_to_blur") and "nsfw" in self.frames_to_blur:
                    self.frames_to_blur["nsfw"].discard(target_frame)
            review_window.destroy()
        
        ttk.Button(button_frame, text=f"Blur this {frame_type} content", command=mark_keep).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Do not blur (safe content)", command=mark_ignore).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=review_window.destroy).pack(side=tk.RIGHT, padx=5)
        
        # Wait for user decision
        self.root.wait_window(review_window)


    def review_copyright_detections(self):
        """Opens a window for reviewing copyright/NSFW detections with low confidence"""
        if not hasattr(self, "review_complete"):
            self.review_complete = False
        
        # Create review window
        review_window = tk.Toplevel(self.root)
        review_window.title("Review Content Detections")
        review_window.geometry("800x600")
        review_window.grab_set()  # Make modal
        
        # Track user decisions
        self.user_decisions = {
            "copyright": {},  # frame_number: is_copyright (True/False)
            "nsfw": {}        # frame_number: is_nsfw (True/False)
        }
        
        # Create frames
        top_frame = ttk.Frame(review_window, padding=10)
        top_frame.pack(fill=tk.X)
        
        image_frame = ttk.Frame(review_window)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        button_frame = ttk.Frame(review_window, padding=10)
        button_frame.pack(fill=tk.X)
        
        # Add instructions
        ttk.Label(top_frame, text="Review each detection and confirm if it contains copyright or sensitive content.", 
                wraplength=780).pack(pady=5)
        
        # Create canvas for image display
        image_canvas = tk.Canvas(image_frame, background="black")
        image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Prepare list of frames to review (combine copyright and nsfw)
        frames_to_review = []
        
        # Add low confidence copyright frames
        if hasattr(self, "low_confidence_copyright"):
            for frame_num, timestamp_seconds, frame_path, confidence in self.low_confidence_copyright:
                frames_to_review.append({
                    "type": "copyright",
                    "frame_number": frame_num,
                    "timestamp": timestamp_seconds,
                    "path": frame_path,
                    "confidence": confidence,
                    "description": f"Potential copyright content (confidence: {confidence:.2f})"
                })
        
        # Add low confidence NSFW frames
        if hasattr(self, "low_confidence_nsfw"):
            for frame_num, timestamp_seconds, frame_path, confidence, detections in self.low_confidence_nsfw:
                frames_to_review.append({
                    "type": "nsfw",
                    "frame_number": frame_num,
                    "timestamp": timestamp_seconds,
                    "path": frame_path,
                    "confidence": confidence,
                    "detections": detections,
                    "description": f"Potential sensitive content (confidence: {confidence:.2f})"
                })
        
        if not frames_to_review:
            ttk.Label(image_frame, text="No content requires review.").pack(pady=20)
            ttk.Button(button_frame, text="Close", command=review_window.destroy).pack(side=tk.RIGHT)
            return
        
        # Sort by frame number
        frames_to_review.sort(key=lambda x: x["frame_number"])
        
        # Current frame index
        current_idx = [0]  # Use list to allow modification in nested functions
        
        # Information about current frame
        info_var = tk.StringVar()
        ttk.Label(top_frame, textvariable=info_var).pack(pady=5)
        
        # Decision status
        decision_var = tk.StringVar(value="Please make a decision")
        ttk.Label(top_frame, textvariable=decision_var).pack(pady=5)
        
        def show_current_frame():
            """Display the current frame"""
            if 0 <= current_idx[0] < len(frames_to_review):
                frame_info = frames_to_review[current_idx[0]]
                frame_path = frame_info["path"]
                
                # Load image
                try:
                    # Clear canvas
                    image_canvas.delete("all")
                    
                    # Load and display image
                    img = Image.open(frame_path)
                    width, height = img.size
                    
                    # Resize to fit canvas
                    canvas_width = image_canvas.winfo_width()
                    canvas_height = image_canvas.winfo_height()
                    
                    if canvas_width > 1 and canvas_height > 1:
                        ratio = min(canvas_width / width, canvas_height / height)
                        new_width = int(width * ratio)
                        new_height = int(height * ratio)
                        img = img.resize((new_width, new_height), Image.LANCZOS)
                    
                    photo = ImageTk.PhotoImage(img)
                    image_canvas.create_image(
                        canvas_width // 2, canvas_height // 2, 
                        image=photo, anchor=tk.CENTER
                    )
                    image_canvas.image = photo  # Keep reference
                    
                    # Draw bounding boxes for NSFW content if available
                    if frame_info["type"] == "nsfw" and "detections" in frame_info:
                        for det in frame_info["detections"]:
                            x1, y1, x2, y2 = det["bbox"]
                            x1 = int(x1 * ratio)
                            y1 = int(y1 * ratio)
                            x2 = int(x2 * ratio)
                            y2 = int(y2 * ratio)
                            
                            # Convert coordinates to canvas space
                            x1 += canvas_width // 2 - new_width // 2
                            y1 += canvas_height // 2 - new_height // 2
                            x2 += canvas_width // 2 - new_width // 2
                            y2 += canvas_height // 2 - new_height // 2
                            
                            # Draw rectangle
                            image_canvas.create_rectangle(
                                x1, y1, x2, y2, 
                                outline="red", width=2
                            )
                    
                    # Update info label
                    minutes = int(frame_info["timestamp"] // 60)
                    seconds = int(frame_info["timestamp"] % 60)
                    timestamp_str = f"{minutes:02}:{seconds:02}"
                    
                    info_var.set(f"Frame {current_idx[0] + 1}/{len(frames_to_review)} - {timestamp_str} - {frame_info['description']}")
                    
                    # Check if already decided
                    frame_number = frame_info["frame_number"]
                    if frame_info["type"] == "copyright" and frame_number in self.user_decisions["copyright"]:
                        decision = "COPYRIGHT" if self.user_decisions["copyright"][frame_number] else "NOT COPYRIGHT"
                        decision_var.set(f"Decision: {decision}")
                    elif frame_info["type"] == "nsfw" and frame_number in self.user_decisions["nsfw"]:
                        decision = "NSFW" if self.user_decisions["nsfw"][frame_number] else "NOT NSFW"
                        decision_var.set(f"Decision: {decision}")
                    else:
                        decision_var.set("Please make a decision")
                    
                except Exception as e:
                    print(f"Error displaying frame: {e}")
                    info_var.set(f"Error displaying frame: {e}")
        
        def next_frame():
            """Go to next frame"""
            if current_idx[0] < len(frames_to_review) - 1:
                current_idx[0] += 1
                show_current_frame()
            else:
                # Check if all frames have been reviewed
                all_reviewed = True
                for frame_info in frames_to_review:
                    frame_number = frame_info["frame_number"]
                    frame_type = frame_info["type"]
                    if frame_type == "copyright" and frame_number not in self.user_decisions["copyright"]:
                        all_reviewed = False
                        break
                    elif frame_type == "nsfw" and frame_number not in self.user_decisions["nsfw"]:
                        all_reviewed = False
                        break
                
                if all_reviewed:
                    info_var.set("All frames have been reviewed!")
                    decision_var.set("Review complete - click Finish")
                else:
                    info_var.set("Please review all frames before finishing")
        
        def prev_frame():
            """Go to previous frame"""
            if current_idx[0] > 0:
                current_idx[0] -= 1
                show_current_frame()
        
        def mark_as_copyright():
            """Mark current frame as copyright content"""
            if 0 <= current_idx[0] < len(frames_to_review):
                frame_info = frames_to_review[current_idx[0]]
                frame_type = frame_info["type"]
                frame_number = frame_info["frame_number"]
                
                if frame_type == "copyright":
                    self.user_decisions["copyright"][frame_number] = True
                    decision_var.set("Decision: COPYRIGHT")
                elif frame_type == "nsfw":
                    self.user_decisions["nsfw"][frame_number] = True
                    decision_var.set("Decision: NSFW")
                
                # Go to next frame after short delay
                review_window.after(500, next_frame)
        
        def mark_as_not_copyright():
            """Mark current frame as not copyright content"""
            if 0 <= current_idx[0] < len(frames_to_review):
                frame_info = frames_to_review[current_idx[0]]
                frame_type = frame_info["type"]
                frame_number = frame_info["frame_number"]
                
                if frame_type == "copyright":
                    self.user_decisions["copyright"][frame_number] = False
                    decision_var.set("Decision: NOT COPYRIGHT")
                elif frame_type == "nsfw":
                    self.user_decisions["nsfw"][frame_number] = False
                    decision_var.set("Decision: NOT NSFW")
                
                # Go to next frame after short delay
                review_window.after(500, next_frame)
        
        def finish_review():
            """Apply user decisions and close window"""
            # Check if all frames have been reviewed
            all_reviewed = True
            for frame_info in frames_to_review:
                frame_number = frame_info["frame_number"]
                frame_type = frame_info["type"]
                if frame_type == "copyright" and frame_number not in self.user_decisions["copyright"]:
                    all_reviewed = False
                    info_var.set("Please review all frames before finishing")
                    return
                elif frame_type == "nsfw" and frame_number not in self.user_decisions["nsfw"]:
                    all_reviewed = False
                    info_var.set("Please review all frames before finishing")
                    return
            
            # Apply decisions - update copyright_keyframes and nsfw_keyframes lists
            if hasattr(self, "copyright_keyframes"):
                # Keep high confidence frames
                confirmed_copyright_keyframes = [frame for frame in self.copyright_keyframes 
                                                if frame[0] not in self.user_decisions["copyright"]]
                
                # Add user confirmed copyright frames
                for frame_info in frames_to_review:
                    if (frame_info["type"] == "copyright" and 
                        frame_info["frame_number"] in self.user_decisions["copyright"] and
                        self.user_decisions["copyright"][frame_info["frame_number"]]):
                        # Add to copyright frames
                        confirmed_copyright_keyframes.append(
                            (frame_info["frame_number"], frame_info["timestamp"], frame_info["path"])
                        )
                
                # Replace the list
                self.copyright_keyframes = confirmed_copyright_keyframes
            
            if hasattr(self, "nsfw_keyframes"):
                # Keep high confidence frames
                confirmed_nsfw_keyframes = [frame for frame in self.nsfw_keyframes 
                                        if frame[0] not in self.user_decisions["nsfw"]]
                
                # Add user confirmed nsfw frames
                for frame_info in frames_to_review:
                    if (frame_info["type"] == "nsfw" and 
                        frame_info["frame_number"] in self.user_decisions["nsfw"] and
                        self.user_decisions["nsfw"][frame_info["frame_number"]]):
                        # Add to nsfw frames (need to maintain format)
                        confirmed_nsfw_keyframes.append(
                            (frame_info["frame_number"], frame_info["timestamp"], 
                            frame_info["path"], frame_info.get("detections", []))
                        )
                
                # Replace the list
                self.nsfw_keyframes = confirmed_nsfw_keyframes
            
            # Mark review as complete
            self.review_complete = True
            
            # Close window
            review_window.destroy()
        
        # Add navigation buttons
        nav_frame = ttk.Frame(button_frame)
        nav_frame.pack(side=tk.LEFT)
        
        ttk.Button(nav_frame, text="Previous", command=prev_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next", command=next_frame).pack(side=tk.LEFT, padx=5)
        
        # Add decision buttons
        decision_frame = ttk.Frame(button_frame)
        decision_frame.pack(side=tk.LEFT, padx=20)
        
        ttk.Button(decision_frame, text="Mark as Copyright/NSFW", command=mark_as_copyright).pack(side=tk.LEFT, padx=5)
        ttk.Button(decision_frame, text="Mark as Safe", command=mark_as_not_copyright).pack(side=tk.LEFT, padx=5)
        
        # Add finish button
        ttk.Button(button_frame, text="Finish Review", command=finish_review).pack(side=tk.RIGHT)
        
        # Display first frame once the window is loaded
        review_window.update()
        review_window.after(100, show_current_frame)
        
        # Wait for window to close
        self.root.wait_window(review_window)
        return self.review_complete
    
    def preview_blur(self):
        """Preview and edit what will be blurred before creating the final video"""
        if not hasattr(self, 'processing_results') or not self.processing_results:
            self.status_var.set("No processing results available. Process the video first.")
            return
        
        # Check if we have any detected content
        has_content = (len(self.processing_results.get("copyright_images", [])) > 0 or 
                    len(self.processing_results.get("nsfw_images", [])) > 0)
        
        if not has_content:
            self.status_var.set("No copyright or NSFW content detected to preview.")
            return
        
        # Initialize keyframes if not already set
        if not hasattr(self, "copyright_keyframes") or self.copyright_keyframes is None:
            self.copyright_keyframes = []
            
            # Rebuild from processing results
            for timestamp, frame_path, result in self.processing_results["copyright_images"]:
                # Find the frame number from the filename if possible
                frame_number = 0
                try:
                    frame_name = os.path.basename(frame_path)
                    if "frame_" in frame_name:
                        frame_number = int(frame_name.split("frame_")[1].split(".")[0])
                except:
                    pass
                    
                # Convert timestamp to seconds
                timestamp_seconds = 0
                try:
                    ts = timestamp.strip("[]").split(":")
                    timestamp_seconds = int(ts[0]) * 60 + int(ts[1])
                except:
                    pass
                    
                # Add to copyright keyframes
                self.copyright_keyframes.append((frame_number, timestamp_seconds, frame_path))
        
        if not hasattr(self, "nsfw_keyframes") or self.nsfw_keyframes is None:
            self.nsfw_keyframes = []
            
            # Rebuild from processing results
            for timestamp, frame_path, result in self.processing_results.get("nsfw_images", []):
                # Find the frame number from the filename if possible
                frame_number = 0
                try:
                    frame_name = os.path.basename(frame_path)
                    if "frame_" in frame_name:
                        frame_number = int(frame_name.split("frame_")[1].split(".")[0])
                except:
                    pass
                    
                # Convert timestamp to seconds
                timestamp_seconds = 0
                try:
                    ts = timestamp.strip("[]").split(":")
                    timestamp_seconds = int(ts[0]) * 60 + int(ts[1])
                except:
                    pass
                    
                # Add to NSFW keyframes with empty detections if needed
                if frame_path != "extended_frame":
                    self.nsfw_keyframes.append((frame_number, timestamp_seconds, frame_path, []))
                else:
                    # For extended frames, just use the information we have
                    self.nsfw_keyframes.append((frame_number, timestamp_seconds, frame_path, []))
        
        # Create message box asking what the user wants to do
        preview_window = tk.Toplevel(self.root)
        preview_window.title("Preview & Edit Blur Options")
        preview_window.geometry("500x400")
        preview_window.grab_set()
        
        # Add instructions
        ttk.Label(preview_window, text="What would you like to do with the detected content?", 
                font=("Arial", 12, "bold"), wraplength=450).pack(pady=20)
        
        # Frame for summary
        summary_frame = ttk.Frame(preview_window, padding=10)
        summary_frame.pack(fill=tk.X, pady=10)
        
        # Display summary of detected content
        ttk.Label(summary_frame, text=f"Copyright frames: {len(self.copyright_keyframes)}").pack(anchor=tk.W)
        ttk.Label(summary_frame, text=f"NSFW frames: {len(self.nsfw_keyframes)}").pack(anchor=tk.W)
        if hasattr(self, 'processing_results') and 'copyright_audio_segments' in self.processing_results:
            ttk.Label(summary_frame, text=f"Audio copyright segments: {len(self.processing_results['copyright_audio_segments'])}").pack(anchor=tk.W)
        
        # Frame for buttons
        button_frame = ttk.Frame(preview_window, padding=10)
        button_frame.pack(fill=tk.X, pady=20)
        
        # Action buttons
        ttk.Button(
            button_frame, 
            text="Review and Edit Detections", 
            command=lambda: [preview_window.destroy(), self.start_review_workflow()]
        ).pack(fill=tk.X, pady=5)
        
        ttk.Button(
            button_frame, 
            text="Create Clean Video with All Detections", 
            command=lambda: [preview_window.destroy(), self.create_clean_video_with_decisions()]
        ).pack(fill=tk.X, pady=5)
        
        ttk.Button(
            button_frame, 
            text="Cancel", 
            command=preview_window.destroy
        ).pack(fill=tk.X, pady=5)
        
        # Wait for user decision
        self.root.wait_window(preview_window)

    def _save_report(self, folder):
        """Save processing results to a report file"""
        report_path = os.path.join(folder, "analysis_report.txt")
        
        with open(report_path, "w", encoding="utf-8") as f:
            # Write header
            f.write("=" * 50 + "\n")
            f.write("CONTENT MODERATION REPORT\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Video: {os.path.basename(self.video_path)}\n")
            f.write("=" * 50 + "\n\n")
            
            # Write copyright info
            f.write("COPYRIGHT ANALYSIS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Audio: {self.processing_results['copyright_audio']}\n")
            
            # Add detailed audio copyright segments if available
            if self.processing_results.get("copyright_audio_segments"):
                f.write("\nAudio Copyright Segments:\n")
                for i, (start, end) in enumerate(self.processing_results["copyright_audio_segments"]):
                    start_min = int(start // 60)
                    start_sec = int(start % 60)
                    end_min = int(end // 60)
                    end_sec = int(end % 60)
                    f.write(f"{i+1}. [{start_min:02}:{start_sec:02} - {end_min:02}:{end_sec:02}]\n")
            
            f.write("\nVideo Frames with Copyright Issues:\n")
            
            if self.processing_results["copyright_images"]:
                for i, (timestamp, frame_path, result) in enumerate(self.processing_results["copyright_images"]):
                    f.write(f"{i+1}. {timestamp} - {result} ({os.path.basename(frame_path)})\n")
            else:
                f.write("No copyright issues detected in video frames.\n")
                
            # Add NSFW content information
            f.write("\nVideo Frames with NSFW Content:\n")
            if self.processing_results.get("nsfw_images"):
                for i, (timestamp, frame_path, result) in enumerate(self.processing_results["nsfw_images"]):
                    f.write(f"{i+1}. {timestamp} - {result}")
                    if "extended_frame" not in frame_path:
                        f.write(f" ({os.path.basename(frame_path)})")
                    f.write("\n")
            else:
                f.write("No NSFW content detected in video frames.\n")
            
            # Add information about the clean video if created
            if self.blurred_video_path:
                f.write(f"\nClean version created: {os.path.basename(self.blurred_video_path)}\n")
                if self.processing_results.get("copyright_audio_segments"):
                    f.write("- Copyright audio segments muted\n")
                if self.processing_results["copyright_images"]:
                    f.write("- Copyright visual content blurred\n")
                if self.processing_results.get("nsfw_images"):
                    f.write("- NSFW content blurred\n") 
            
            f.write("\n" + "=" * 50 + "\n\n")
            
            # Write transcript
            f.write("TRANSCRIPT:\n")
            f.write("-" * 50 + "\n")
            for _, text in self.processing_results["transcriptions"]:
                f.write(f"{text}\n")
    
    def toggle_play(self):
        """Start/stop video playback"""
        if not self.is_playing and self.video_path:
            self.is_playing = True
            self.play_btn.config(text="Stop")
            
            # Use blurred video if available, otherwise use original
            play_path = self.blurred_video_path if self.blurred_video_path else self.video_path
            if self.blurred_video_path:
                self.status_var.set("Playing blurred version with copyright content removed")
            
            # Start video playing thread
            self.play_thread = threading.Thread(target=self.play_video, args=(play_path,))
            self.play_thread.daemon = True
            self.play_thread.start()
        else:
            self.is_playing = False
            self.play_btn.config(text="Play")
            
            # Clean up
            self.cleanup()

    # Replace the play_video method with this improved version
    def play_video(self, video_path=None):
        """Thread function to play the video with synchronized audio"""
        import cv2
        import pygame
        import time
        import numpy as np
        
        # Initialize pygame for audio playback
        pygame.init()
        pygame.mixer.init()
        
        path_to_play = video_path if video_path else self.video_path
        self.cap = cv2.VideoCapture(path_to_play)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open video file {path_to_play}")
            self.root.after(0, lambda: self.status_var.set(f"Error: Could not open video file"))
            self.root.after(0, lambda: self.play_btn.config(text="Play"))
            self.is_playing = False
            return
        
        # Get video properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_time = 1 / fps if fps > 0 else 0.033  # Default to 30fps
        
        # Start audio playback using pygame
        try:
            # Try to extract audio to a temporary file first (more compatible)
            import tempfile
            import os
            import subprocess
            
            # Create temp file with appropriate extension
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio.close()
            
            # Use ffmpeg to extract audio to the temp file
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", path_to_play, "-vn", 
                    "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                    temp_audio.name
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Check if the extracted audio file exists and is not empty
                if os.path.exists(temp_audio.name) and os.path.getsize(temp_audio.name) > 1000:
                    pygame.mixer.music.load(temp_audio.name)
                    pygame.mixer.music.play()
                    print(f"Playing audio from extracted temporary file: {temp_audio.name}")
                else:
                    raise Exception("Extracted audio file is missing or empty")
                    
            except Exception as e:
                # Fallback: try playing directly from the video file
                print(f"Audio extraction failed: {e}, trying direct playback")
                pygame.mixer.music.load(path_to_play)
                pygame.mixer.music.play()
                
        except Exception as e:
            print(f"Audio playback error: {e}")
            # Continue with video only if audio fails
            self.root.after(0, lambda: self.status_var.set("Playing video only (audio error)"))
            
        # Store temp audio path for cleanup
        self.temp_audio_path = temp_audio.name if 'temp_audio' in locals() else None
        
        # Synchronization variables
        start_time = time.time()
        frame_count = 0
        
        while self.is_playing and self.cap.isOpened():
            # Calculate the target frame based on elapsed time for better sync
            elapsed = time.time() - start_time
            target_frame = int(elapsed * fps)
            
            # Skip frames if we're behind
            if target_frame > frame_count:
                # Set the frame position
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                frame_count = target_frame
            
            # Read the frame
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Convert to RGB for tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update frame in UI
            self.update_video_frame(rgb_frame)
            
            # Control playback speed for smoother video
            actual_elapsed = time.time() - start_time
            expected_elapsed = frame_count / fps
            delay = max(0, expected_elapsed - actual_elapsed)
            if delay > 0:
                time.sleep(delay)
            
            frame_count += 1
        
        # Cleanup when video ends or is stopped
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        # Video ended naturally (not stopped by user)
        if self.is_playing:
            self.root.after(0, lambda: self.play_btn.config(text="Play"))
            self.is_playing = False
        
        pygame.mixer.quit()
            


    def cleanup(self):
        """Clean up resources"""
        # Stop audio if playing
        try:
            if hasattr(pygame, 'mixer') and pygame.mixer.get_init():
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
        except Exception:
            pass
            
        # Close video capture
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()
            
        # Delete temp audio file
        if hasattr(self, 'temp_audio_path') and self.temp_audio_path:
            try:
                if os.path.exists(self.temp_audio_path):
                    os.remove(self.temp_audio_path)
                    self.temp_audio_path = None
            except Exception as e:
                print(f"Error removing temp audio file: {e}")


    

def main():
    root = tk.Tk()
    app = VideoProcessor(root)
    root.mainloop()

def __del__(self):
    """Destructor to ensure cleanup"""
    self.cleanup()

if __name__ == "__main__":
    main()