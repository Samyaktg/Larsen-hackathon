import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Global model reference for singleton pattern
_nsfw_model = None

def get_nsfw_model(model_path="best.pt"):
    """
    Load YOLO NSFW detection model as a singleton to avoid reloading.
    
    Parameters:
        model_path (str): Path to the trained YOLO model weights
        
    Returns:
        YOLO: Loaded model instance
    """
    global _nsfw_model
    
    if _nsfw_model is None:
        print(f"Loading NSFW detection model from {model_path}")
        try:
            _nsfw_model = YOLO(model_path)
            print("NSFW detection model loaded successfully")
        except Exception as e:
            print(f"Error loading NSFW model: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    return _nsfw_model

def detect_nsfw_content(image_path, confidence_threshold=0.5, use_gpu=False):
    """
    Detect NSFW content in an image.
    
    Parameters:
        image_path (str): Path to the image file
        confidence_threshold (float): Minimum confidence for detection
        use_gpu (bool): Whether to use GPU for inference
        
    Returns:
        dict: Detection results containing:
            - has_nsfw (bool): Whether NSFW content was detected
            - detections (list): List of detection details
            - confidence (float): Highest confidence score (0-1)
    """
    try:
        # Load the model
        model = get_nsfw_model()
        if model is None:
            return {"has_nsfw": False, "detections": [], "confidence": 0, "error": "Model not loaded"}
        
        # Set device
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        # Run inference
        print(f"Running NSFW detection on {image_path}")
        results = model(image_path, device=device)
        
        # Process results
        if results and len(results) > 0:
            result = results[0]  # Get result for first image
            detections = []
            highest_confidence = 0
            
            # Extract detection information
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    # Get coordinates, confidence and class
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    class_name = model.names[class_id]
                    
                    # Only include detections above threshold
                    if confidence > confidence_threshold:
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": confidence,
                            "class_id": class_id, 
                            "class_name": class_name
                        })
                        highest_confidence = max(highest_confidence, confidence)
            
            has_nsfw = len(detections) > 0
            return {
                "has_nsfw": has_nsfw,
                "detections": detections,
                "confidence": highest_confidence
            }
            
        return {"has_nsfw": False, "detections": [], "confidence": 0}
        
    except Exception as e:
        print(f"Error in NSFW detection: {e}")
        import traceback
        traceback.print_exc()
        return {"has_nsfw": False, "detections": [], "confidence": 0, "error": str(e)}

def blur_nsfw_regions(image, detections, blur_strength=25):
    """
    Apply blur to the entire image when NSFW regions are detected.
    
    Parameters:
        image: Image as numpy array (BGR format for OpenCV)
        detections: List of detection dictionaries with bbox coordinates
        blur_strength: Intensity of the blur (higher = stronger blur)
        
    Returns:
        numpy array: Image with full-frame blur
    """
    try:
        # If there are detections, apply full-frame blur
        if detections:
            # Calculate appropriate kernel size based on image dimensions
            height, width = image.shape[:2]
            # Create kernel size based on image dimensions but maintain odd numbers
            kernel_size = min(blur_strength * 2 + 1, min(height, width) // 3)
            kernel_size = max(25, kernel_size)
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd number for kernel size
            
            # Apply strong Gaussian blur to the entire image
            blurred_image = cv2.GaussianBlur(image.copy(), (kernel_size, kernel_size), 0)
            print(f"Applied full-frame blur with kernel size {kernel_size}")
            return blurred_image
        
        # If no detections, return the original image
        return image
        
    except Exception as e:
        print(f"Error applying full-frame blur: {e}")
        return image  # Return original image in case of failure

def process_nsfw_temporal(video_path, nsfw_keyframes, output_folder, frame_buffer=30, confidence_threshold=0.5, use_gpu=False):
    """
    Process NSFW detections with temporal extension to ensure complete coverage.
    For each NSFW frame, extract forward and backward frames and check if they also contain NSFW content.
    
    Parameters:
        video_path (str): Path to the video file
        nsfw_keyframes (list): List of (frame_number, timestamp, path, detections) tuples with NSFW content
        output_folder (str): Where to save extracted frames for analysis
        frame_buffer (int): How many frames to check before and after an NSFW detection
        confidence_threshold (float): Confidence threshold for NSFW detection
        use_gpu (bool): Whether to use GPU acceleration
    
    Returns:
        dict: Extended NSFW frame data, mapping frame numbers to detections
    """
    if not nsfw_keyframes:
        print("No NSFW keyframes provided for temporal analysis")
        return {}
        
    try:
        os.makedirs(output_folder, exist_ok=True)
        
        # Extract frame numbers where NSFW content was detected
        nsfw_frame_numbers = [frame[0] for frame in nsfw_keyframes]
        print(f"Starting temporal analysis with {len(nsfw_frame_numbers)} NSFW frames")
        
        # Get initial detection data
        nsfw_frame_dict = {}
        for frame_number, _, _, detections in nsfw_keyframes:
            nsfw_frame_dict[frame_number] = detections
            print(f"Initial NSFW frame {frame_number} has {len(detections)} detections")
            
        # Open the video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Adjust frame buffer for short videos (less than 30s)
        if total_frames / fps < 30.0:
            frame_buffer = max(5, int(frame_buffer * 0.5))
            print(f"Short video detected, using reduced temporal buffer: {frame_buffer} frames")
        
        # For each NSFW frame, determine frames to check before and after
        frames_to_check = set()
        for frame_num in nsfw_frame_numbers:
            # Add surrounding frames within buffer
            start = max(0, frame_num - frame_buffer)
            end = min(total_frames - 1, frame_num + frame_buffer)
            frames_to_check.update(range(start, end + 1))
            
        # Remove frames we already know have NSFW content
        frames_to_check = frames_to_check - set(nsfw_frame_numbers)
        
        if not frames_to_check:
            print("No additional frames to check in temporal analysis")
            return nsfw_frame_dict
            
        print(f"Expanding NSFW detection to {len(frames_to_check)} adjacent frames")
        
        # Load detection model
        model = get_nsfw_model()
        if model is None:
            print("Failed to load NSFW model for temporal analysis")
            return nsfw_frame_dict
            
        # Get device for inference
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"Using {device} for NSFW temporal analysis")
        
        # Process each frame to check for NSFW content
        found_frames = 0
        for frame_num in sorted(frames_to_check):
            # Set video position to the frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Could not read frame {frame_num}")
                continue
                
            # Save frame to disk for analysis
            temp_frame_path = os.path.join(output_folder, f"temp_frame_{frame_num:06d}.jpg")
            cv2.imwrite(temp_frame_path, frame)
            
            # Run detection on the frame
            try:
                results = model(temp_frame_path, device=device)
                
                if results and len(results) > 0:
                    result = results[0]
                    detections = []
                    highest_confidence = 0
                    
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        for box in result.boxes:
                            # Get coordinates, confidence and class
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            confidence = box.conf[0].item()
                            class_id = int(box.cls[0].item())
                            class_name = model.names[class_id]
                            
                            # Only include detections above threshold
                            if confidence > confidence_threshold:
                                detections.append({
                                    "bbox": [x1, y1, x2, y2],
                                    "confidence": confidence,
                                    "class_id": class_id,
                                    "class_name": class_name
                                })
                                highest_confidence = max(highest_confidence, confidence)
                    
                    # If NSFW content detected in this frame, add to our dict
                    if detections:
                        nsfw_frame_dict[frame_num] = detections
                        found_frames += 1
                        print(f"Found additional NSFW content in frame {frame_num} with confidence {highest_confidence:.2f}")
                
            except Exception as e:
                print(f"Error analyzing frame {frame_num}: {e}")
            
            # Clean up temp file
            try:
                os.remove(temp_frame_path)
            except Exception as e:
                print(f"Could not remove temp file: {e}")
                
        cap.release()
        print(f"Temporal analysis complete. Original NSFW frames: {len(nsfw_keyframes)}, Additional found: {found_frames}")
        print(f"Total frames to blur: {len(nsfw_frame_dict)}")
        return nsfw_frame_dict
        
    except Exception as e:
        print(f"Error in temporal NSFW processing: {e}")
        import traceback
        traceback.print_exc()
        return nsfw_frame_dict