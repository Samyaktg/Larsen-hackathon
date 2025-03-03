import cv2
import numpy as np
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from nsfw_detection import blur_nsfw_regions
from PIL import Image
from scipy.spatial.distance import cosine

# Global model for feature extraction
_feature_extractor = None

def get_feature_extractor(use_gpu=False):
    """Returns a pre-trained model for feature extraction"""
    global _feature_extractor
    
    if _feature_extractor is None:
        print("Loading feature extraction model...")
        # Use ResNet18 as it's smaller and faster than larger models
        model = models.resnet18(pretrained=True)
        # Remove the classification layer
        _feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
        _feature_extractor.eval()
        
        # Move to GPU if available and requested
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        _feature_extractor = _feature_extractor.to(device)
        print(f"Feature extraction model loaded on {device}")
    
    return _feature_extractor

def extract_features(frame, model, use_gpu=False):
    """Extract deep features from a frame using pre-trained model"""
    # Define image transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Convert OpenCV BGR to RGB and then to PIL Image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # Preprocess image
    img_tensor = preprocess(pil_image).unsqueeze(0)
    
    # Move to GPU if available and requested
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    img_tensor = img_tensor.to(device)
    
    # Extract features
    with torch.no_grad():
        features = model(img_tensor)
    
    return features.cpu().numpy().flatten()

def calculate_histogram(frame):
    """Calculate color histogram for a frame"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Calculate histogram for H and S channels
    hist_h = cv2.calcHist([hsv], [0], None, [30], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    # Normalize and flatten
    cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
    return np.concatenate([hist_h.flatten(), hist_s.flatten()])

def extract_key_frames(video_path, output_folder, frame_interval=5, threshold=0.5, 
                       adaptive_threshold=True, min_scene_length=5, use_gpu=False):
    """
    Extracts key frames from a video using a combination of histogram and deep features.
    
    Parameters:
        video_path (str): Path to the video file.
        output_folder (str): Directory to save key frames.
        frame_interval (int): Process every nth frame for efficiency.
        threshold (float): Base threshold for scene change detection (0-1).
        adaptive_threshold (bool): Whether to adapt threshold based on video length.
        min_scene_length (int): Minimum frames between detected scenes.
        use_gpu (bool): Whether to use GPU for feature extraction.
        
    Returns:
        list: List of extracted keyframe information (frame_number, timestamp_seconds, filepath)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load feature extraction model
    feature_extractor = get_feature_extractor(use_gpu)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default to 30fps if unable to get actual FPS
    
    # Get total frame count and duration
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Adjust parameters for short videos - much more aggressive settings
    is_short_video = duration < 30.0
    if is_short_video:
        print(f"Short video detected ({duration:.2f}s). Using highly sensitive keyframe detection.")
        frame_interval = 1  # Process every frame for short videos
        threshold = threshold * 0.4  # Much lower threshold for short videos (60% reduction)
        min_scene_length = 2  # Allow keyframes to be very close together
    
    # Initialize variables
    frame_count = 0
    prev_hist = None
    prev_features = None
    last_keyframe = -min_scene_length  # Ensure first frame is considered
    keyframes = []  # Will store (frame_number, timestamp_seconds, filepath)
    
    # Track scene change scores for adaptive thresholding
    if adaptive_threshold:
        scores_history = []
    
    print(f"Starting key frame extraction from {video_path}")
    print(f"Parameters: frame_interval={frame_interval}, threshold={threshold}, min_scene_length={min_scene_length}")
    
    # Always capture the first frame as a keyframe
    ret, first_frame = cap.read()
    if ret:
        # Save first frame as keyframe
        timestamp_seconds = 0
        minutes = 0
        seconds = 0
        milliseconds = 0
        timestamp_str = f"{minutes:02}_{seconds:02}_{milliseconds:03}"
        
        frame_filename = os.path.join(output_folder, f"frame_{timestamp_str}_{frame_count}.jpg")
        cv2.imwrite(frame_filename, first_frame)
        
        print(f"First keyframe at {timestamp_seconds:.2f}s")
        keyframes.append((frame_count, timestamp_seconds, frame_filename))
        
        # Initialize with first frame data
        prev_hist = calculate_histogram(first_frame)
        prev_features = extract_features(first_frame, feature_extractor, use_gpu)
        last_keyframe = frame_count
        
        # For short videos, sample more points throughout the video
        if is_short_video and duration > 3:
            # Add several more points for very short videos
            sample_points = [
                int(fps * 1.0),  # ~1 second in
                int(fps * 2.0),  # ~2 seconds in
                int(total_frames // 2),  # middle of video
                int(total_frames - fps * 2)  # 2 seconds from end
            ]
            
            for point in sample_points:
                if 0 < point < total_frames and abs(point - frame_count) > fps:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, point)
                    ret, sample_frame = cap.read()
                    if ret:
                        timestamp_seconds = point / fps
                        minutes = int(timestamp_seconds // 60)
                        seconds = int(timestamp_seconds % 60)
                        milliseconds = int((timestamp_seconds % 1) * 1000)
                        timestamp_str = f"{minutes:02}_{seconds:02}_{milliseconds:03}"
                        
                        frame_filename = os.path.join(output_folder, f"frame_{timestamp_str}_{point}.jpg")
                        cv2.imwrite(frame_filename, sample_frame)
                        
                        print(f"Sample keyframe at {timestamp_seconds:.2f}s")
                        keyframes.append((point, timestamp_seconds, frame_filename))
        # For longer videos, just get an early frame  
        elif min_scene_length > 5 and total_frames > 10:
            early_frame_number = min(int(fps * 1.5), total_frames//10)  # ~1.5 seconds in or 10% of video
            cap.set(cv2.CAP_PROP_POS_FRAMES, early_frame_number)
            ret, early_frame = cap.read()
            if ret:
                timestamp_seconds = early_frame_number / fps
                minutes = int(timestamp_seconds // 60)
                seconds = int(timestamp_seconds % 60)
                milliseconds = int((timestamp_seconds % 1) * 1000)
                timestamp_str = f"{minutes:02}_{seconds:02}_{milliseconds:03}"
                
                frame_filename = os.path.join(output_folder, f"frame_{timestamp_str}_{early_frame_number}.jpg")
                cv2.imwrite(frame_filename, early_frame)
                
                print(f"Early keyframe at {timestamp_seconds:.2f}s")
                keyframes.append((early_frame_number, timestamp_seconds, frame_filename))
        
        # Reset to beginning for normal processing
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
    
    # Main frame processing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every nth frame (or every frame for short videos)
        if frame_count % frame_interval == 0:
            # Calculate timestamp
            timestamp_seconds = frame_count / fps
            
            # For short videos, consider processing more frames
            check_this_frame = frame_count - last_keyframe >= min_scene_length
            
            # Special handling for very short videos - check more frames
            if is_short_video and duration < 10 and frame_count % max(1, min_scene_length//2) == 0:
                check_this_frame = True
            
            # Only analyze frames after the minimum scene length
            if check_this_frame:
                # Calculate histogram
                hist = calculate_histogram(frame)
                
                # Extract deep features
                features = extract_features(frame, feature_extractor, use_gpu)
                
                # Compute changes if we have previous frames
                if prev_hist is not None and prev_features is not None:
                    # Histogram difference (correlation-based)
                    hist_diff = cv2.compareHist(hist.astype(np.float32), 
                                                prev_hist.astype(np.float32), 
                                                cv2.HISTCMP_CORREL)
                    hist_score = 1.0 - max(0, hist_diff)  # Convert to distance
                    
                    # Feature vector difference (cosine distance)
                    feature_score = cosine(features, prev_features)
                    
                    # Combined score (weighted average)
                    # For short videos, weight histogram changes more heavily
                    if is_short_video:
                        combined_score = 0.5 * hist_score + 0.5 * feature_score
                    else:
                        combined_score = 0.4 * hist_score + 0.6 * feature_score
                    
                    # If adaptive threshold is enabled, update score history
                    if adaptive_threshold:
                        scores_history.append(combined_score)
                        # Use percentile-based threshold after collecting enough samples
                        if len(scores_history) > 10:
                            # For short videos, use a more sensitive percentile
                            if is_short_video:
                                percentile = 60  # Lower percentile = more keyframes
                                scaling = 1.0    # Less scaling = more keyframes
                            else:
                                percentile = 75
                                scaling = 1.2
                            
                            dynamic_threshold = np.percentile(scores_history, percentile) * scaling
                            effective_threshold = min(threshold, dynamic_threshold) if is_short_video else max(threshold, dynamic_threshold)
                        else:
                            effective_threshold = threshold
                    else:
                        effective_threshold = threshold
                    
                    # Detect scene change
                    if combined_score > effective_threshold:
                        # Format timestamp for filename
                        minutes = int(timestamp_seconds // 60)
                        seconds = int(timestamp_seconds % 60)
                        milliseconds = int((timestamp_seconds % 1) * 1000)
                        timestamp_str = f"{minutes:02}_{seconds:02}_{milliseconds:03}"
                        
                        # Save keyframe
                        frame_filename = os.path.join(output_folder, f"frame_{timestamp_str}_{frame_count}.jpg")
                        cv2.imwrite(frame_filename, frame)
                        print(f"Keyframe at {timestamp_seconds:.2f}s (score: {combined_score:.3f}, threshold: {effective_threshold:.3f})")
                        
                        # Add to keyframes list
                        keyframes.append((frame_count, timestamp_seconds, frame_filename))
                        
                        # Update last keyframe position
                        last_keyframe = frame_count
            
                # Update previous frame data
                prev_hist = hist
                prev_features = features
        
        frame_count += 1
        
        # Progress indicator for long videos
        if frame_count % 500 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"Processed {frame_count} frames ({progress:.1f}%)")
    
    # Final check: If no keyframes detected in last portion of video, add one more
    if keyframes:
        last_keyframe_time = keyframes[-1][1]
        end_threshold = 0.8 if is_short_video else 0.7  # Check closer to the end for short videos
        
        if last_keyframe_time < (end_threshold * duration) and total_frames > 0:
            try:
                # Jump to near end of video (90% for short videos, 85% for longer)
                late_frame_position = int(total_frames * (0.9 if is_short_video else 0.85))
                cap.set(cv2.CAP_PROP_POS_FRAMES, late_frame_position)
                ret, late_frame = cap.read()
                if ret:
                    timestamp_seconds = late_frame_position / fps
                    minutes = int(timestamp_seconds // 60)
                    seconds = int(timestamp_seconds % 60)
                    milliseconds = int((timestamp_seconds % 1) * 1000)
                    timestamp_str = f"{minutes:02}_{seconds:02}_{milliseconds:03}"
                    
                    frame_filename = os.path.join(output_folder, f"frame_{timestamp_str}_{late_frame_position}.jpg")
                    cv2.imwrite(frame_filename, late_frame)
                    print(f"Added final keyframe at {timestamp_seconds:.2f}s for better coverage")
                    keyframes.append((late_frame_position, timestamp_seconds, frame_filename))
            except:
                print("Could not add final keyframe")
    
    cap.release()
    
    # For very short videos with few keyframes, add more evenly spaced keyframes
    if is_short_video and len(keyframes) < 4 and total_frames > 0:
        print(f"Adding more keyframes for short video with only {len(keyframes)} keyframes")
        try:
            # Reopen video
            cap = cv2.VideoCapture(video_path)
            
            # Get existing keyframe positions
            existing_positions = set(kf[0] for kf in keyframes)
            
            # Calculate 3 evenly spaced positions
            even_positions = [
                int(total_frames * 0.25),
                int(total_frames * 0.5),
                int(total_frames * 0.75)
            ]
            
            for pos in even_positions:
                if pos not in existing_positions and pos < total_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                    ret, frame = cap.read()
                    if ret:
                        timestamp_seconds = pos / fps
                        minutes = int(timestamp_seconds // 60)
                        seconds = int(timestamp_seconds % 60)
                        milliseconds = int((timestamp_seconds % 1) * 1000)
                        timestamp_str = f"{minutes:02}_{seconds:02}_{milliseconds:03}"
                        
                        frame_filename = os.path.join(output_folder, f"frame_{timestamp_str}_{pos}.jpg")
                        cv2.imwrite(frame_filename, frame)
                        print(f"Added extra keyframe at {timestamp_seconds:.2f}s for better coverage")
                        keyframes.append((pos, timestamp_seconds, frame_filename))
            
            cap.release()
        except Exception as e:
            print(f"Error adding extra keyframes: {e}")
    
    print(f"Key frame extraction complete. Found {len(keyframes)} keyframes.")
    return keyframes

def blur_copyright_frames(video_path, output_path, copyright_frames, all_keyframes, use_gpu=False, audio_path=None, nsfw_keyframes=None):
    """
    Creates a copy of the video with copyright frames blurred and optionally replaces audio.
    Blurs all frames from a copyrighted keyframe until the next significant keyframe.
    
    Parameters:
        video_path (str): Path to the input video.
        output_path (str): Path to save the output video.
        copyright_frames (list): List of (frame_number, timestamp, frame_path) tuples for frames with copyright.
        all_keyframes (list): List of all extracted keyframes.
        use_gpu (bool): Whether to use GPU acceleration.
        audio_path (str): Optional path to an audio file to use instead of the video's audio.
        nsfw_keyframes (list): List of (frame_number, timestamp, path, detections) tuples with NSFW content.
        
    Returns:
        str: Path to the output video.
    """
    import cv2
    import subprocess
    import os
    import tempfile
    import shutil
    
    try:
        print(f"Creating blurred video at: {output_path}")
        
        # Debug information for troubleshooting
        print(f"Copyright frames count: {len(copyright_frames) if copyright_frames else 0}")
        print(f"NSFW frames count: {len(nsfw_keyframes) if nsfw_keyframes else 0}")
        
        if nsfw_keyframes and len(nsfw_keyframes) > 0:
            print(f"First NSFW frame: {nsfw_keyframes[0][0]}, detections: {len(nsfw_keyframes[0][3])}")
            
        # Open the input video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames, {duration:.2f}s")
        
        # For shorter videos, adjust blur parameters
        is_short_video = duration < 30.0
        if is_short_video:
            print(f"Short video detected (duration: {duration:.2f}s). Using adjusted blur settings.")
        
        # Create a temporary directory for frames
        temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory: {temp_dir}")
        
        # Sort all keyframes by frame number
        sorted_keyframes = sorted(all_keyframes, key=lambda x: x[0])
        
        # Extract copyright frame numbers for quick lookup
        copyright_frame_numbers = set([frame[0] for frame in copyright_frames]) if copyright_frames else set()
        
        # Prepare NSFW frame information
        nsfw_frame_dict = {}
        if nsfw_keyframes:
            for frame_number, _, _, detections in nsfw_keyframes:
                nsfw_frame_dict[frame_number] = detections
            print(f"Found {len(nsfw_frame_dict)} frames with NSFW content to blur")
        
        # Calculate blur zones with transition frames
        blur_zones = []  # Will store (start_frame, end_frame, center_frame) tuples
        
        # Set transition duration (in frames) based on video length
        fade_frames = int(fps * (0.5 if not is_short_video else 0.2))
        
        # Create blur zones between copyrighted keyframe and next keyframe
        for i in range(len(sorted_keyframes) - 1):
            current_keyframe = sorted_keyframes[i]
            next_keyframe = sorted_keyframes[i+1]
            
            current_frame_num = current_keyframe[0]
            next_frame_num = next_keyframe[0]
            
            # If the current keyframe has copyright
            if current_frame_num in copyright_frame_numbers:
                # Add small margin to ensure we blur enough
                blur_start = current_frame_num
                blur_end = next_frame_num
                center = (blur_start + blur_end) // 2
                
                # Create transition zones
                fade_in_start = max(0, blur_start - fade_frames)
                fade_out_end = min(total_frames, blur_end + fade_frames)
                
                # Add blur zone with fade in/out frame ranges
                blur_zones.append((fade_in_start, fade_out_end, center, blur_start, blur_end))
                
                print(f"Created blur zone from frame {blur_start} to {blur_end} with transitions")
        
        # Special case for the last keyframe
        last_keyframe = sorted_keyframes[-1]
        if last_keyframe[0] in copyright_frame_numbers:
            blur_start = last_keyframe[0]
            blur_end = min(total_frames, blur_start + int(fps * 5))  # Blur 5 seconds after or to end
            center = (blur_start + blur_end) // 2
            
            fade_in_start = max(0, blur_start - fade_frames)
            fade_out_end = blur_end  # No fade out for end of video
            
            blur_zones.append((fade_in_start, fade_out_end, center, blur_start, blur_end))
            print(f"Created final blur zone from frame {blur_start} to {blur_end}")
            
        print(f"Created {len(blur_zones)} inter-keyframe blur zones with smooth transitions")
        
        # Process all frames
        frame_count = 0
        nsfw_blurred_frames = 0  # Count how many NSFW frames we blur
        copyright_blurred_frames = 0  # Count copyright frames blurred
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = frame.copy()
            apply_blur = False
            blur_strength = 0
            is_copyright_frame = False
            is_nsfw_frame = False
            
            # Check if this frame is in any blur zone
            for zone_start, zone_end, center, hard_start, hard_end in blur_zones:
                if zone_start <= frame_count <= zone_end:
                    # In a transition zone
                    if frame_count < hard_start:
                        # Fade in - gradually increase blur strength
                        progress = 1.0 - ((hard_start - frame_count) / fade_frames)
                        blur_strength = max(blur_strength, progress)
                    elif frame_count > hard_end:
                        # Fade out - gradually decrease blur strength
                        progress = 1.0 - ((frame_count - hard_end) / fade_frames)
                        blur_strength = max(blur_strength, progress)
                    else:
                        # Inside hard blur zone - full strength
                        blur_strength = 1.0
                    
                    apply_blur = True
                    is_copyright_frame = True
                    copyright_blurred_frames += 1
                    break
            
            # Apply gradual blur based on strength
            if apply_blur:
                # Calculate appropriate blur kernel size based on strength and video size
                base_kernel = 25 if is_short_video else 45
                max_kernel = 99 if not is_short_video else 75
                
                # Scale kernel by blur strength (odd values only)
                kernel_size = int(base_kernel + blur_strength * (max_kernel - base_kernel))
                kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
                
                # Scale sigma by blur strength
                sigma = 5 + 25 * blur_strength
                
                # Apply Gaussian blur
                blurred = cv2.GaussianBlur(processed_frame, (kernel_size, kernel_size), sigma)
                
                # Blend original and blurred based on strength for smoother transition
                if blur_strength < 1.0:
                    processed_frame = cv2.addWeighted(
                        processed_frame, 1.0 - blur_strength, 
                        blurred, blur_strength, 0
                    )
                else:
                    processed_frame = blurred
                
                # Add text overlay for copyright blur
                if is_copyright_frame and blur_strength > 0.5:
                    # Calculate position (bottom of screen)
                    height, width = processed_frame.shape[:2]
                    text_position = (int(width * 0.05), int(height * 0.95))
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = width / 1000  # Scale based on video width
                    font_color = (255, 255, 255)  # White
                    text_thickness = max(1, int(width / 500))
                    outline_thickness = text_thickness + 1
                    
                    # Add text with outline for visibility
                    text = "COPYRIGHTED CONTENT BLURRED"
                    cv2.putText(processed_frame, text, text_position, font, font_scale, 
                               (0, 0, 0), outline_thickness)  # Black outline
                    cv2.putText(processed_frame, text, text_position, font, font_scale, 
                               font_color, text_thickness)
            
            # Check if this frame has NSFW regions to blur
            if frame_count in nsfw_frame_dict:
                # Debug - print info about this frame
                if frame_count % 30 == 0:  # Only print every 30th frame to avoid too much output
                    print(f"Applying NSFW blur to frame {frame_count}, detections: {len(nsfw_frame_dict[frame_count])}")
                
                # Apply targeted blur only to the NSFW regions
                detections = nsfw_frame_dict[frame_count]
                
                # Adjust blur strength for short videos
                region_blur_strength = 25 if is_short_video else 35
                
                # Use our blur_nsfw_regions function
                from nsfw_detection import blur_nsfw_regions
                processed_frame = blur_nsfw_regions(processed_frame, detections, blur_strength=region_blur_strength)
                nsfw_blurred_frames += 1
                is_nsfw_frame = True
                
                # Add text overlay for NSFW blur
                if is_nsfw_frame:
                    # Calculate position (bottom of screen, above copyright text if both present)
                    height, width = processed_frame.shape[:2]
                    y_position = int(height * 0.90) if is_copyright_frame else int(height * 0.95)
                    text_position = (int(width * 0.05), y_position)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = width / 1000  # Scale based on video width
                    font_color = (255, 255, 255)  # White
                    text_thickness = max(1, int(width / 500))
                    outline_thickness = text_thickness + 1
                    
                    # Add text with outline for visibility
                    text = "SENSITIVE CONTENT BLURRED"
                    cv2.putText(processed_frame, text, text_position, font, font_scale, 
                               (0, 0, 0), outline_thickness)  # Black outline
                    cv2.putText(processed_frame, text, text_position, font, font_scale, 
                               font_color, text_thickness)
            
            # Save the processed frame
            cv2.imwrite(os.path.join(temp_dir, f"frame_{frame_count:06d}.png"), processed_frame)
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Processed {frame_count} frames ({progress:.1f}%)")
                
        cap.release()
        print(f"Total frames processed: {frame_count}")
        print(f"Applied copyright blur to {copyright_blurred_frames} frames")
        print(f"Applied NSFW blur to {nsfw_blurred_frames} frames")
        
        # Use FFmpeg to combine frames into video
        print("Creating video from frames...")
        frame_pattern = os.path.join(temp_dir, "frame_%06d.png")
        
        # Determine which audio to use
        audio_option = []
        if audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            print(f"Using provided clean audio: {audio_path}")
            audio_option = ["-i", audio_path, "-c:a", "aac", "-map", "0:v", "-map", "1:a"]
        else:
            print("Using original audio from video")
            audio_option = ["-c:a", "copy"]
            
        # Construct FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-framerate", str(fps),
            "-i", frame_pattern
        ]
        
        # Add audio input if using custom audio
        if audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            cmd.extend(["-i", audio_path])
            
        # Add output options
        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p"
        ])
        
        # Add mapping options for audio
        if audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            cmd.extend(["-map", "0:v", "-map", "1:a"])
        else:
            # Try to copy original audio, but continue if it fails
            try:
                cmd.extend(["-i", video_path, "-map", "0:v", "-map", "1:a", "-c:a", "aac"])
            except:
                cmd.extend(["-an"])  # No audio if we can't copy or add new audio
                
        # Add output file
        cmd.append(output_path)
        
        print(f"Running FFmpeg command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"Blurred video created at: {output_path}")
        
        return output_path
    
    except Exception as e:
        print(f"Error creating blurred video: {e}")
        import traceback
        traceback.print_exc()
        # Try to clean up
        try:
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except:
            pass
        return None