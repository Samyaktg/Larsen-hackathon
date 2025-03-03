import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import numpy as np
import librosa
import os
from pydub import AudioSegment
import tempfile

model_name = "facebook/wav2vec2-base"
processor = Wav2Vec2Processor.from_pretrained(model_name)

def load_audio(file_path, target_length=160000):
    """
    Loads an audio file and ensures a fixed length by padding or truncating.
    
    Parameters:
        file_path (str): Path to the audio file.
        target_length (int): Fixed length for all audio samples (e.g., 10s at 16kHz = 160000 samples).

    Returns:
        torch.Tensor: Fixed-length waveform tensor.
    """
    waveform, sample_rate = torchaudio.load(file_path)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)  # Convert to 16kHz

    # Ensure waveform is mono (1 channel)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Pad or truncate to target length
    if waveform.shape[1] < target_length:
        pad_amount = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    else:
        waveform = waveform[:, :target_length]  # Truncate

    return waveform.squeeze(0)

def analyze_audio_segments(audio_path, model_path="./wav2vec2_copyright", segment_length=10.0, overlap=2.0, confidence_threshold=0.85):
    """
    Analyzes audio in segments and identifies copyrighted portions with confidence threshold.
    
    Parameters:
        audio_path (str): Path to the audio file.
        model_path (str): Path to the fine-tuned model.
        segment_length (float): Length of each segment in seconds.
        overlap (float): Overlap between segments in seconds.
        confidence_threshold (float): Minimum confidence to consider a segment as copyrighted.
    
    Returns:
        list: List of tuples containing (start_time, end_time) of copyrighted segments.
    """
    print(f"Analyzing audio copyright in segments: {audio_path}")
    
    try:
        # Load the model
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        model.eval()
        
        # Load audio file with librosa (better for segmenting)
        y, sr = librosa.load(audio_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"Audio duration: {duration:.2f} seconds")
        
        # Adjust segment length for very short videos
        if duration < 15:
            # Use segments of 1/3 the duration with minimal overlap
            segment_length = max(2.0, duration / 3)  # At least 2 seconds
            overlap = max(0.5, segment_length / 4)   # Smaller overlap
            print(f"Short audio detected. Using segment length: {segment_length:.1f}s, overlap: {overlap:.1f}s")
        
        # Calculate segments
        segment_samples = int(segment_length * 16000)
        hop_samples = int((segment_length - overlap) * 16000)
        
        # Store copyright segments
        copyright_segments = []
        all_segments = []
        
        # Process each segment
        for start_sample in range(0, len(y), hop_samples):
            end_sample = min(start_sample + segment_samples, len(y))
            segment = y[start_sample:end_sample]
            
            # Skip if segment is too short
            if len(segment) < segment_samples / 2:
                continue
                
            # Convert to tensor
            segment_tensor = torch.FloatTensor(segment)
            
            # Pad if necessary
            if len(segment) < segment_samples:
                pad_amount = segment_samples - len(segment)
                segment_tensor = torch.nn.functional.pad(segment_tensor, (0, pad_amount))
            
            # Process with the model
            inputs = processor(segment_tensor, return_tensors="pt", padding=True, sampling_rate=16000)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits).item()
                confidence = torch.softmax(logits, dim=1)[0][predicted_class].item()
            
            # Calculate time stamps
            start_time = start_sample / 16000
            end_time = end_sample / 16000
            
            # Store segment info
            segment_info = {
                "start_time": start_time,
                "end_time": end_time,
                "copyright": predicted_class == 1,
                "confidence": confidence
            }
            all_segments.append(segment_info)
            
            # Only consider high confidence copyright predictions
            if predicted_class == 1 and confidence >= confidence_threshold:
                print(f"⚠️ Copyright detected at [{start_time:.2f}s - {end_time:.2f}s], confidence: {confidence:.2f}")
                copyright_segments.append((start_time, end_time))
            elif predicted_class == 1:
                print(f"ℹ️ Potential copyright at [{start_time:.2f}s - {end_time:.2f}s], but confidence too low: {confidence:.2f}")
        
        # Merge overlapping segments
        if copyright_segments:
            merged_segments = []
            current_start, current_end = copyright_segments[0]
            
            for next_start, next_end in copyright_segments[1:]:
                if next_start <= current_end:  # Overlap
                    current_end = max(current_end, next_end)
                else:
                    merged_segments.append((current_start, current_end))
                    current_start, current_end = next_start, next_end
                    
            merged_segments.append((current_start, current_end))
            
            # Add buffer around segments (0.5 seconds)
            # For short videos, use a smaller buffer
            buffer_size = 0.25 if duration < 15 else 0.5
            
            buffered_segments = []
            for start, end in merged_segments:
                buffered_segments.append((max(0, start - buffer_size), min(duration, end + buffer_size)))
                
            copyright_segments = buffered_segments
        
        # Calculate percentage of copyrighted content
        total_copyright_duration = sum(end - start for start, end in copyright_segments)
        copyright_percentage = (total_copyright_duration / duration) * 100 if duration > 0 else 0
        
        print(f"Analysis complete: {len(copyright_segments)} high confidence copyright segments found.")
        print(f"Copyright content: {total_copyright_duration:.2f}s ({copyright_percentage:.1f}%)")
        
        return {
            "segments": copyright_segments,
            "percentage": copyright_percentage,
            "total_duration": duration,
            "copyright_duration": total_copyright_duration,
            "result": "Copyrighted" if copyright_segments else "Not Copyrighted"
        }
    
    except Exception as e:
        print(f"Error analyzing audio segments: {e}")
        import traceback
        traceback.print_exc()
        return {
            "segments": [],
            "percentage": 0,
            "total_duration": 0,
            "copyright_duration": 0,
            "result": f"Error: {str(e)}"
        }

def predict_audio(audio_path, model_path="./wav2vec2_copyright"):
    """
    Predicts whether an audio clip is copyrighted and returns detailed analysis.
    """
    try:
        # Simple check to see if file exists and is valid
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return "Error: File not found"
            
        # Check if file has content
        if os.path.getsize(audio_path) < 1000:  # Less than 1KB
            print(f"Audio file is too small or empty: {audio_path}")
            return "No audio content"
        
        # Run segment analysis
        results = analyze_audio_segments(audio_path, model_path)
        
        # Simplified return for backward compatibility
        return results["result"]
        
    except Exception as e:
        print(f"Error in audio copyright prediction: {e}")
        return f"Error: {str(e)}"

def mute_copyright_segments(input_audio_path, output_audio_path, copyright_segments):
    """
    Creates a new audio file with copyrighted segments muted.
    
    Parameters:
        input_audio_path (str): Path to the input audio file.
        output_audio_path (str): Path to save the output audio file.
        copyright_segments (list): List of (start_time, end_time) tuples to mute.
    
    Returns:
        str: Path to the output audio file.
    """
    try:
        print(f"Processing audio: {input_audio_path}")
        print(f"Muting {len(copyright_segments)} copyright segments")
        
        # Load audio with pydub (supports more formats)
        audio = AudioSegment.from_file(input_audio_path)
        
        # Create silent audio of same length
        silent_segment = AudioSegment.silent(duration=len(audio))
        
        # If no copyright segments, just return the original
        if not copyright_segments:
            audio.export(output_audio_path, format="wav")
            return output_audio_path
            
        # Process each segment for muting
        output_audio = audio
        
        for start_time, end_time in copyright_segments:
            # Convert seconds to milliseconds
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            
            # Ensure within bounds
            start_ms = max(0, start_ms)
            end_ms = min(len(audio), end_ms)
            
            if start_ms >= end_ms:
                continue
                
            # Replace segment with silence
            output_audio = output_audio[:start_ms] + silent_segment[start_ms:end_ms] + output_audio[end_ms:]
        
        # Export the modified audio
        output_audio.export(output_audio_path, format="wav")
        print(f"Created modified audio: {output_audio_path}")
        return output_audio_path
        
    except Exception as e:
        print(f"Error muting copyright segments: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_audio_copyright(audio_path, output_folder=None, confidence_threshold=0.85):
    """
    Complete processing function that detects copyright segments and creates a clean version.
    Only high-confidence segments are considered for muting.
    
    Parameters:
        audio_path (str): Path to the input audio file.
        output_folder (str): Optional folder to save output files.
        confidence_threshold (float): Minimum confidence to consider a segment as copyrighted.
    
    Returns:
        dict: Results including paths and copyright analysis.
    """
    try:
        print("Initializing audio copyright detection model...")
        # Create output folder if needed
        if output_folder is None:
            output_folder = os.path.dirname(audio_path)
        os.makedirs(output_folder, exist_ok=True)
        
        # Analyze audio for copyright content
        print(f"Analyzing audio for copyright segments (confidence threshold: {confidence_threshold:.2f})...")
        analysis = analyze_audio_segments(audio_path, confidence_threshold=confidence_threshold)
        
        # Create output path
        base_name = os.path.basename(audio_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_folder, f"{name}_clean.wav")
        
        # If copyright segments found, create clean version
        if analysis["segments"]:
            muted_audio_path = mute_copyright_segments(audio_path, output_path, analysis["segments"])
            
            return {
                "original_audio": audio_path,
                "clean_audio": muted_audio_path,
                "copyright_segments": analysis["segments"],
                "copyright_percentage": analysis["percentage"],
                "result": f"Copyrighted content muted (confidence threshold: {confidence_threshold:.2f})"
            }
        else:
            # No copyright, just copy the file
            audio = AudioSegment.from_file(audio_path)
            audio.export(output_path, format="wav")
            
            return {
                "original_audio": audio_path,
                "clean_audio": output_path,
                "copyright_segments": [],
                "copyright_percentage": 0,
                "result": "No high-confidence copyright content found"
            }
    
    except Exception as e:
        print(f"Error processing audio copyright: {e}")
        return {
            "original_audio": audio_path,
            "clean_audio": None,
            "error": str(e),
            "result": f"Error: {str(e)}"
        }