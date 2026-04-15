# Install dependencies
# The error indicates a numpy version incompatibility.
# The installed numpy (1.26.4) conflicts with pre-installed packages requiring numpy>=2.0.
# We will try to explicitly install numpy>=2.0 along with other dependencies.
# If gradio==4.11.0 is not compatible with numpy>=2.0, this might lead to further issues or require
# upgrading gradio to a newer version.
!pip install transformers==4.41.2 youtube-transcript-api sentencepiece torch==2.10.0 gradio==4.11.0 numpy>=2.0 --upgrade

# Save your app.py code into the notebook
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import torch
import gradio as gr
from transformers import pipeline

# Load summarization model
text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summary(input_text):
    # Handle long transcripts by chunking
    max_chunk = 800
    chunks = [input_text[i:i+max_chunk] for i in range(0, len(input_text), max_chunk)]
    summaries = [text_summary(chunk)[0]['summary_text'] for chunk in chunks]
    return " ".join(summaries)

def extract_video_id(url):
    regex = r"(?:youtube\.com/(?:[^/\n\s]+/\S+|(?:v|e(?:mbed)?)|\S+?[?&]v=)|youtu\.be/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None

def get_youtube_transcript(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        return "Video ID could not be extracted."

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        formatter = TextFormatter()
        text_transcript = formatter.format_transcript(transcript)
        summary_text = summary(text_transcript)
        return summary_text
    except Exception as e:
        return f"An error occurred: {e}"

# Launch Gradio app
demo = gr.Interface(
    fn=get_youtube_transcript,
    inputs=[gr.Textbox(label="Input YouTube URL to summarize", lines=1)],
    outputs=[gr.Textbox(label="Summarized text", lines=6)],
    title="YouTube Script Summarizer",
    description="Summarizes YouTube video transcripts using DistilBART."
)

demo.launch(share=True)
