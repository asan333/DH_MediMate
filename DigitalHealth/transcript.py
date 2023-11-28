import json
import os
import csv
import subprocess
import pandas as pd
from typing import Optional, List, Dict, Any
import time
import psutil
import GPUtil
from pytube import YouTube
import matplotlib.pyplot as plt
import whisperx
from whisperx import load_align_model, align
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from collections import Counter
from openai import OpenAI
import csv

from diarization import run_diarization, save_results_to_csv, csv_to_dialogue
from gpt import generate_outputs

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

def build_transcripts(audio_file):

    results = run_diarization(audio_file)
    csv_file_path = 'whisperx_script.csv'
    save_results_to_csv(results,csv_file_path)
    dialogue = csv_to_dialogue(csv_file_path)
    print(dialogue)
    doctor_notes, clean_lines, summ_dialogue, doctor_dialogue, patient_dialogue, first_speaker = generate_outputs(dialogue)

    return doctor_notes, summ_dialogue, doctor_dialogue, patient_dialogue, first_speaker

def show_notes(doctor_notes):
    # Open a file where you want to store the output
    with open('doctor_notes.txt', 'w') as file:
        # Use the file argument in print to write to the file
        print(doctor_notes, file=file)

def notes_report():
    # Define the path to your image
    image_path = 'logo_image.jpg'  # Replace with your image path

    # Create a PDF canvas
    c = canvas.Canvas("doctor_notes.pdf", pagesize=letter)
    width, height = letter  # Get the dimensions of the page

    # Add the image - adjust x, y, width, and height as needed
    # Here, we place the image at the top center of the page
    image_width = 2 * inch  # Example width of the image
    image_height = 2 * inch  # Example height of the image
    c.drawImage(image_path, (width - image_width) / 2, height - image_height - 30, width=image_width, height=image_height)

    # Now, add the text from the file below the image
    text_y_position = height - image_height - 50  # Position of the text start
    with open('doctor_notes.txt', 'r') as file:
        for line in file:
            c.drawString(72, text_y_position, line.strip())  # 72 is a margin here
            text_y_position -= 20  # Move down for the next line

    # Save the PDF
    c.save()