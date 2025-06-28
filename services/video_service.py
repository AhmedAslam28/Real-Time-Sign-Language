
"""
Video processing and combination service.
"""
import os
import streamlit as st
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import concatenate_videoclips
from typing import List, Optional

def combine_videos(video_paths: List[str], output_path: str) -> Optional[str]:
    """Combine multiple videos into a single video file using MoviePy."""
    if not video_paths:
        return None
    
    # Filter out None/invalid paths
    valid_paths = [path for path in video_paths if path and os.path.exists(path)]
    
    if not valid_paths:
        return None
    
    clips = []
    try:
        # Load all video clips
        for path in valid_paths:
            clips.append(VideoFileClip(path))
        
        # Concatenate the clips
        final_clip = concatenate_videoclips(clips, method="compose")
        
        # Write the result to a file
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        
        # Close all clips to free up resources
        for clip in clips:
            clip.close()
        final_clip.close()
        
        return output_path
    
    except Exception as e:
        st.error(f"Error combining videos: {str(e)}")
        # Close any open clips in case of error
        for clip in clips:
            try:
                clip.close()
            except:
                pass
        if 'final_clip' in locals():
            try:
                final_clip.close()
            except:
                pass
        return None

def validate_video_file(video_path: str) -> bool:
    """Validate if a video file exists and is accessible."""
    return os.path.exists(video_path) and os.path.isfile(video_path)
