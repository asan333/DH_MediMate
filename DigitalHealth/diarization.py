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
from collections import Counter
import csv
import pandas as pd

# Converts audio file to .wav format using ffmpeg

def convert_to_wav(input_file: str, output_file: Optional[str] = None) -> None:
    """
    Converts an audio file to WAV format using FFmpeg.

    Args:
        input_file: The path of the input audio file to convert.
        output_file: The path of the output WAV file. If None, the output file will be created by replacing the input file
        extension with ".wav".

    Returns:
        None
    """
    if not output_file:
        output_file = os.path.splitext(input_file)[0] + ".wav"

    command = f'ffmpeg -i "{input_file}" -vn -acodec pcm_s16le -ar 44100 -ac 1 "{output_file}"'

    try:
        subprocess.run(command, shell=True, check=True)
        print(f'Successfully converted "{input_file}" to "{output_file}"')
    except subprocess.CalledProcessError as e:
        print(f'Error: {e}, could not convert "{input_file}" to "{output_file}"')



# Transcribes audio using Whisper

def transcribe(audio_file: str, model_name: str, device: str = "cpu") -> Dict[str, Any]:
    """
    Transcribe an audio file using a speech-to-text model.

    Args:
        audio_file: Path to the audio file to transcribe.
        model_name: Name of the model to use for transcription.
        device: The device to use for inference (e.g., "cpu" or "cuda").

    Returns:
        A dictionary representing the transcript, including the segments, the language code, and the duration of the audio file.
    """

    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
    batch_size = 32 # reduce if low on GPU mem
    model = whisperx.load_model(model_name, device, compute_type=compute_type)
    result = model.transcribe(audio_file,batch_size=batch_size)

    language_code = result["language"]
    return {
        "segments": result["segments"],
        "language_code": language_code,
    }

# Aligns segments using Whisper X

def align_segments(
    segments: List[Dict[str, Any]],
    language_code: str,
    audio_file: str,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Align the transcript segments using a pretrained alignment model.

    Args:
        segments: List of transcript segments to align.
        language_code: Language code of the audio file.
        audio_file: Path to the audio file containing the audio data.
        device: The device to use for inference (e.g., "cpu" or "cuda").

    Returns:
        A dictionary representing the aligned transcript segments.
    """
    model_a, metadata = load_align_model(language_code=language_code, device=device)
    result_aligned = whisperx.align(segments, model_a, metadata, audio_file, device, return_char_alignments=False)
    return result_aligned


# Diarization using Pyannote HuggingFace API

def diarize(audio_file: str, hf_token: str) -> Dict[str, Any]:
    """
    Perform speaker diarization on an audio file.

    Args:
        audio_file: Path to the audio file to diarize.
        hf_token: Authentication token for accessing the Hugging Face API.

    Returns:
        A dictionary representing the diarized audio file, including the speaker embeddings and the number of speakers.
    """
    diarization_pipeline = DiarizationPipeline(use_auth_token=hf_token,device="cuda")
    diarization_result = diarization_pipeline(audio_file,min_speakers=2)
    return diarization_result

# Assign speaker to each transcript segment

def assign_speakers(
    diarization_result: Dict[str, Any], aligned_segments: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Assign speakers to each transcript segment based on the speaker diarization result.

    Args:
        diarization_result: Dictionary representing the diarized audio file, including the speaker embeddings and the number of speakers.
        aligned_segments: Dictionary representing the aligned transcript segments.

    Returns:
        A list of dictionaries representing each segment of the transcript, including the start and end times, the
        spoken text, and the speaker ID.
    """
    results_segments_w_speakers = assign_word_speakers(diarization_result, aligned_segments)

    return results_segments_w_speakers

# Function that uses the previously defined functions to transcribe and diarize audio file

def transcribe_and_diarize(
    audio_file: str,
    hf_token: str,
    model_name: str,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """
    Transcribe an audio file and perform speaker diarization to determine which words were spoken by each speaker.

    Args:
        audio_file: Path to the audio file to transcribe and diarize.
        hf_token: Authentication token for accessing the Hugging Face API.
        model_name: Name of the model to use for transcription.
        device: The device to use for inference (e.g., "cpu" or "cuda").

    Returns:
        A list of dictionaries representing each segment of the transcript, including the start and end times, the
        spoken text, and the speaker ID.
    """
    transcript = transcribe(audio_file, model_name, device)
    aligned_segments = align_segments(
        transcript["segments"], transcript["language_code"], audio_file, device
    )
    diarization_result = diarize(audio_file, hf_token)
    results_segments_w_speakers = assign_speakers(diarization_result, aligned_segments)

    print(results_segments_w_speakers)
    print("")

    return results_segments_w_speakers

def plot_results(data: Dict[str, Dict[str, float]]) -> None:
    """
    Plot the execution time and memory usage for each combination of model and device.

    Args:
        data: A dictionary containing the execution time and memory usage for each combination of model and device.
    """
    model_names = list(data.keys())
    devices = list(data[model_names[0]].keys())

    # Separate data for execution time and memory usage
    execution_times = [[data[model][device]["execution_time"] for device in devices] for model in model_names]
    memory_usages = [[data[model][device]["memory_usage"] for device in devices] for model in model_names]

    # Create bar plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    x = list(range(len(devices)))
    bar_width = 0.25
    colors = ["b", "g", "r"]

    for i, model in enumerate(model_names):
        ax[0].bar(
            [elem + i * bar_width for elem in x],
            execution_times[i],
            color=colors[i],
            width=bar_width,
            edgecolor="white",
            label=model,
        )

        ax[1].bar(
            [elem + i * bar_width for elem in x],
            memory_usages[i],
            color=colors[i],
            width=bar_width,
            edgecolor="white",
            label=model,
        )

    # Set plot parameters
    ax[0].set_title("Execution Time")
    ax[0].set_xlabel("Device")
    ax[0].set_ylabel("Execution Time (s)")
    ax[0].set_xticks([elem + bar_width for elem in x])
    ax[0].set_xticklabels(devices)
    ax[0].legend()

    ax[1].set_title("Memory Usage")
    ax[1].set_xlabel("Device")
    ax[1].set_ylabel("Memory Usage (GB)")
    ax[1].set_xticks([elem + bar_width for elem in x])
    ax[1].set_xticklabels(devices)
    ax[1].legend()

    plt.tight_layout()
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    fig.savefig(os.path.join(plots_dir, "whisper_models_comparison.png"), dpi=300)
    plt.close(fig)


def get_gpu_memory_usage():
    gpus = GPUtil.getGPUs()
    if len(gpus) > 0:
        return gpus[0].memoryUsed / 1024
    else:
        return 0

def run_diarization(audio_file):
    os.environ["HUGGINGFACE_TOKEN"] = "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"

    # model_names = ["tiny","base", "medium", "large"]
    # devices = ["cpu", "cuda"]

    model_names = ["medium"]
    devices = ["cuda"]

    hf_token = os.environ["HUGGINGFACE_TOKEN"]
    language_code = "en"

    # convert_to_wav("./sample_audio/DH_Records/Short_Version_3_people.m4a")
    convert_to_wav(audio_file)

    audio_file = ("./sample_audio.wav")
    # audio_file = ("sample.wav")
    results = {}

    for model_name in model_names:
        results[model_name] = {}
        for device in devices:
            print(f"Testing {model_name} model on {device}")

            start_time = time.time()
            results_segments_w_speakers = transcribe_and_diarize(
                audio_file, hf_token, model_name, device
            )
            end_time = time.time()

            if device == "cpu":
                memory_usage = psutil.Process().memory_info().rss / (1024 ** 3)
            else:
                memory_usage = get_gpu_memory_usage()

            results[model_name][device] = {
                "execution_time": end_time - start_time,
                "memory_usage": memory_usage,
            }

            print(f"Execution time for {model_name} on {device}: {results[model_name][device]['execution_time']:.2f} seconds")
            print(f"Memory usage for {model_name} on {device}: {results[model_name][device]['memory_usage']:.2f}GB")
            print("\n")

    plot_results(results)

    results = {}

    return results_segments_w_speakers

def save_results_to_csv(results_dict, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        headers = [
            'Start', 'End', 'Text', 'Speaker'
        ]
        writer.writerow(headers)

        for segment_list in results_dict.values():
            for segment in segment_list:
                speaker = ''
                if 'words' in segment:
                    speaker_counter = Counter(word.get('speaker', '') for word in segment['words'])
                    speaker = speaker_counter.most_common(1)[0][0]

                if segment.get('text'):
                  writer.writerow([
                    segment.get('start', ''),
                    segment.get('end', ''),
                    segment.get('text', ''),
                    speaker
                  ])

def csv_to_dialogue(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Initialize an empty string for the dialogue
    dialogue = ''

    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        # Append the speaker's name and text to the dialogue string
        dialogue += str(row['Speaker']) + ': ' + str(row['Text']) + '\n'

    return dialogue
