import json
import os
import subprocess
from typing import Optional, List, Dict, Any
import time
import psutil
import GPUtil
# from pytube import YouTube
import matplotlib.pyplot as plt
import whisperx
from whisperx import load_align_model, align
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
import csv
from openai import OpenAI

# define calling openAI
def generate_text(prompt,model_name):
  client = OpenAI(api_key="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")

  response = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    temperature=0,
    max_tokens=1024
  )

  return response.choices[0].message.content

def generate_outputs(dialogue):
    
    model_name = "gpt-3.5-turbo"

    doctor_notes_prompt = "Generate a clear and concise doctor's notes from the following conversation in structured format (ps: ignore or correct any inaccuracies in speaker assignment) : " + dialogue
    doctor_notes = generate_text(doctor_notes_prompt,model_name)
    print("Doctor's notes:\n",doctor_notes)

    clean_lines_prompt = "Rewrite the following dialogue while cleaning it up so that correct speaker is assigned to the correct line. Identify the speaker from the context as well as the classification already mentioned. The speakers are either the 'Doctor' or the 'Patient' (some lines might be misclassified to the wrong speaker, be careful and smart with the corrections). Start of dialogue : " + dialogue
    clean_lines = generate_text(clean_lines_prompt,model_name)
    print("Cleaned Dialogue:\n",clean_lines)

    summ_dialogue_prompt = "Summarize the following dialogue into another cogent dialogue. Only show the summary. Make sure there is no loss of important information. Be careful and smart about it. Start of dialogue: " + clean_lines
    summ_dialogue = generate_text(summ_dialogue_prompt,model_name)
    print("Summarized:\n",summ_dialogue)

    # doctor_lines_prompt = " In the following dialogue, put whatever the Doctor says in a single string and output it. Don't put in anything said by the Patient. Make sure all the lines are in a single string. Start of dialogue : " + summ_dialogue
    # doctor_lines = generate_text(doctor_lines_prompt)
    # print("Doctor Lines:\n",doctor_lines)

    # patient_lines_prompt = " In the following dialogue, put whatever the Patient says in a single string and output it. Don't put in anything said by the Doctor. Make sure all the lines are in a single string. Start of dialogue : " + summ_dialogue
    # patient_lines = generate_text(patient_lines_prompt)
    # print("Patient Lines:\n",patient_lines)

    # dialogue_to_csv(summ_dialogue, 'final_dialogue.csv')

    doctor_dialogue, patient_dialogue, first_speaker = separate_dialogue(summ_dialogue)

    return doctor_notes, clean_lines, summ_dialogue, doctor_dialogue, patient_dialogue, first_speaker

def dialogue_to_csv(dialogue, filename):
    """
    Function to convert a dialogue string into a CSV file.
    Each line of dialogue is assumed to be in the format 'Speaker: Speech'.

    :param dialogue: A string containing the dialogue.
    :param filename: Name of the file to save the CSV.
    """
    # Splitting the dialogue into lines
    lines = dialogue.split('\n')

    # Parsing each line into (Speaker, Speech)
    data = []
    for line in lines:
        if line:  # checking if the line is not empty
            speaker, speech = line.split(': ', 1)
            data.append([speaker, speech])

    # Writing to CSV
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Speaker', 'Speech'])  # Writing header
        writer.writerows(data)

def rem_slash(string):
    """Remove the first three characters from a string."""
    # If the string is at least 2 characters long, remove the first three
    if len(string) > 2:
        return string[2:]
    else:
      print("String is too short. Error.")

def separate_dialogue(input_dialogue):
    # Splitting the input dialogue into individual lines
    lines = input_dialogue.split('\n')

    # Initializing strings for doctor's and patient's dialogue
    doctor_dialogue = ''
    patient_dialogue = ''
    last_speaker = None
    first_speaker = None
    # Iterating through each line
    for line in lines:
        # Splitting each line into speaker and speech
        if ': ' in line:
            speaker, speech = line.split(': ', 1)

            # Adding speech to the respective dialogue string
            if speaker == 'Doctor':
                if last_speaker and last_speaker != 'Doctor':
                    doctor_dialogue += ' / '
                else:
                  doctor_dialogue += ' '
                doctor_dialogue += speech
            elif speaker == 'Patient':
                if last_speaker and last_speaker != 'Patient':
                    patient_dialogue += ' / '
                else:
                  patient_dialogue += ' '
                patient_dialogue += speech

            # Updating the last speaker
            last_speaker = speaker

    if doctor_dialogue[2] !="/":
      first_speaker = "Doctor"
      patient_dialogue = rem_slash(patient_dialogue)
    else:
      first_speaker = "Patient"
      doctor_dialogue = rem_slash(doctor_dialogue)


    return doctor_dialogue, patient_dialogue, first_speaker