import os
import librosa
import soundfile as sf
import torch
import moviepy as mp
from transformers import pipeline
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_folders(video_name):
    """
    Creates a structured directory for saving audio chunks and transcript.

    Parameters:
        video_name (str): Name of the video file (without extension).

    Returns:
        str: Path to the created directory.
    """
    base_dir = os.path.join("processed_videos", video_name)
    chunks_dir = os.path.join(base_dir, "audio_chunks")

    os.makedirs(chunks_dir, exist_ok=True)
    logger.info(f"Created directory structure at {base_dir}")

    return base_dir, chunks_dir

def extract_audio(video_path, output_folder):
    """
    Extracts audio from a video file and saves it as an MP3 file.

    Parameters:
        video_path (str): Path to the input video file.
        output_folder (str): Directory where the audio will be saved.

    Returns:
        str: Path to the saved audio file.
    """
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        audio_output = os.path.join(output_folder, "extracted_audio.mp3")
        logger.info(f"Extracting audio from {video_path}")
        
        video = mp.VideoFileClip(video_path)
        audio = video.audio
        if audio is None:
            raise ValueError("No audio found in the video!")

        audio.write_audiofile(audio_output, verbose=False, logger=None)
        video.close()  # Properly close the video to release resources
        logger.info(f"Audio extracted successfully to {audio_output}")
        return audio_output
        
    except Exception as e:
        logger.error(f"Error extracting audio: {e}")
        raise

def split_audio(input_audio, chunks_folder, chunk_length=30):
    """
    Splits an audio file into smaller chunks for efficient transcription.

    Parameters:
        input_audio (str): Path to the input audio file.
        chunks_folder (str): Directory where audio chunks will be saved.
        chunk_length (int): Length of each chunk in seconds.

    Returns:
        list: List of tuples (chunk file path, start time).
    """
    chunk_files = []
    
    try:
        # Validate input file
        if not os.path.exists(input_audio):
            logger.error(f"Input audio file not found: {input_audio}")
            return []
            
        # Check file size to ensure it's not empty
        if os.path.getsize(input_audio) < 1000:  # Less than 1KB
            logger.warning(f"Input audio file too small: {input_audio}")
            return []
        
        logger.info(f"Loading audio file: {input_audio}")
        # Load the entire audio file
        y, sr = librosa.load(input_audio, sr=16000)
        duration = len(y) / sr
        logger.info(f"Audio duration: {duration:.2f} seconds")
        
        # Calculate chunks
        chunks = int(duration / chunk_length) + 1
        logger.info(f"Creating {chunks} audio chunks of {chunk_length}s each")
        
        # Split into chunks
        for i in range(chunks):
            start_sample = int(i * chunk_length * sr)
            end_sample = int(min((i + 1) * chunk_length * sr, len(y)))
            
            if end_sample <= start_sample:
                continue
                
            chunk = y[start_sample:end_sample]
            chunk_duration = len(chunk)/sr
            
            # Only process chunks with meaningful length
            if chunk_duration < 0.5:  # Skip chunks shorter than 0.5 seconds
                continue
                
            chunk_file = os.path.join(chunks_folder, f"chunk_{i:03d}.wav")
            sf.write(chunk_file, chunk, sr)
            
            start_time = i * chunk_length
            chunk_files.append((chunk_file, start_time))
            logger.info(f"Created chunk {i}: {chunk_duration:.1f}s starting at {start_time}s")
            
        return chunk_files
        
    except Exception as e:
        logger.error(f"Error splitting audio: {str(e)}", exc_info=True)
        return chunk_files  # Return any chunks successfully created

def transcribe_audio(audio_chunks):
    """
    Transcribes multiple audio chunks using Hugging Face's Whisper ASR model.

    Parameters:
        audio_chunks (list): List of tuples (audio file path, start time).

    Returns:
        list: List of transcriptions with timestamps.
    """
    transcriptions = []
    
    if not audio_chunks:
        logger.error("No valid audio chunks to transcribe")
        return transcriptions
    
    try:
        logger.info(f"Setting up ASR model on {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model with explicit task specification
        asr_model = pipeline("automatic-speech-recognition", 
                            model="openai/whisper-base", 
                            device=device)
        
        logger.info(f"Transcribing {len(audio_chunks)} audio chunks")
        for chunk_path, start_time in audio_chunks:
            try:
                # Validate chunk file
                if not os.path.exists(chunk_path):
                    logger.warning(f"Audio chunk file not found: {chunk_path}")
                    continue
                    
                if os.path.getsize(chunk_path) < 1000:  # Less than 1KB
                    logger.warning(f"Audio chunk file too small: {chunk_path}")
                    continue
                
                logger.info(f"Transcribing chunk starting at {start_time}s")
                # Use the simplified API call
                result = asr_model(chunk_path)
                transcription = result["text"].strip()
                
                if transcription:  # Only add non-empty transcriptions
                    transcriptions.append((start_time, transcription))
                    logger.info(f"Transcribed chunk at {start_time}s: '{transcription[:30]}...'")
                else:
                    logger.warning(f"Empty transcription for chunk at {start_time}s")
                    
            except Exception as e:
                logger.error(f"Error transcribing chunk {chunk_path}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error setting up transcription model: {str(e)}", exc_info=True)
    
    return transcriptions

def save_transcript(transcriptions, output_folder):
    """
    Saves the transcribed text along with timestamps to a file.

    Parameters:
        transcriptions (list): List of (timestamp, text) tuples.
        output_folder (str): Directory where the transcript will be saved.
    """
    if not transcriptions:
        logger.warning("No transcriptions to save")
        return None
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    transcript_file = os.path.join(output_folder, f"transcription_{timestamp}.txt")

    try:
        with open(transcript_file, "w", encoding="utf-8") as f:
            for start_time, text in sorted(transcriptions):  # Sort by timestamp
                minutes = int(start_time // 60)
                seconds = int(start_time % 60)
                timestamp = f"[{minutes:02d}:{seconds:02d}]"
                f.write(f"{timestamp} {text}\n")

        logger.info(f"Transcript saved at: {transcript_file}")
        return transcript_file
    except Exception as e:
        logger.error(f"Error saving transcript: {str(e)}")
        return None

def main(video_path):
    """
    Main function to process video and transcribe audio with timestamps.

    Parameters:
        video_path (str): Path to the video file.
    """
    try:
        # Get video name without extension
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        logger.info(f"Processing video: {video_name}")

        # Create necessary folders
        base_folder, chunks_folder = create_folders(video_name)

        logger.info("Extracting audio...")
        audio_path = extract_audio(video_path, base_folder)

        logger.info("Splitting audio into chunks...")
        audio_chunks = split_audio(audio_path, chunks_folder)

        if not audio_chunks:
            logger.error("No audio chunks were created. Cannot proceed with transcription.")
            return

        logger.info("Transcribing audio...")
        transcriptions = transcribe_audio(audio_chunks)

        if not transcriptions:
            logger.error("No transcriptions were generated.")
            return

        # Save the final transcript
        transcript_file = save_transcript(transcriptions, base_folder)

        if transcript_file:
            logger.info(f"✅ Processing complete! All files are saved in: {base_folder}")
        else:
            logger.error("Failed to save transcript.")

    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}", exc_info=True)

if __name__ == "__main__":
    video_file = input("Enter path to video file: ")
    main(video_file)