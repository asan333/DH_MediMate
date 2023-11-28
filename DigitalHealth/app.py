from flask import Flask, jsonify, request, send_file
import os
import base64

from transcript import build_transcripts, show_notes, notes_report

app = Flask(__name__)

def encode_file_to_base64(file_path):
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode('utf-8')


app = Flask(__name__)

@app.route('/process_audio', methods=['POST'])
def process_audio():
    # Check if the request contains an audio file
    if 'audio' not in request.files:
        return "No audio file in request", 400

    audio_file = request.files['audio']
    
    # Define the path for saving the audio file
    audio_file_path = os.path.join(os.getcwd(), 'sample_audio.wav')

    # Save the audio file
    # audio_file.save(audio_file_path)
    # audio_file_path = os.path.join(os.getcwd(), 'sample_audio.wav')
    if os.path.exists(audio_file_path):
        os.remove(audio_file_path)
    audio_file.save(audio_file_path)


    print(audio_file_path)

    doctor_notes, summ_dialogue, doctor_dialogue, patient_dialogue, first_speaker = build_transcripts(audio_file_path)
    show_notes(doctor_notes)
    notes_report()

    transcript_file_path = os.path.join(os.getcwd(), 'transcript.txt')
    with open(transcript_file_path, 'w') as file:
        file.write(summ_dialogue)


    doctor_notes_base64 = encode_file_to_base64("doctor_notes.pdf")
    transcript_base64 = encode_file_to_base64("transcript.txt")  # Assuming transcript.txt is already generated

    response_data = {
        'doctor_notes': doctor_notes_base64,  # Assuming the file is in a static directory
        'summ_dialogue': transcript_base64,
        'doctor_dialogue': doctor_dialogue,
        'patient_dialogue': patient_dialogue,
        'first_speaker': first_speaker
    }

    return jsonify(response_data)

    # You can process the audio_file here as needed
    # For now, we'll just return a pre-defined transcript

    # transcript_file_path = os.path.join(os.getcwd(), 'transcript.txt')
    
    # # Check if the transcript file exists
    # if not os.path.exists(transcript_file_path):
    #     return "Transcript file not found", 404

    # return send_file(transcript_file_path, as_attachment=True, download_name='transcript.txt')


@app.route('/process_transcript', methods=['POST'])
def process_transcript():
    if 'transcript' not in request.files:
        return "No transcript file in request", 400

    transcript_file = request.files['transcript']

    # Process the transcript file to generate a video
    # For demonstration, we'll just return a pre-existing video file

    video_file_path = os.path.join(os.getcwd(), 'Video.mp4')

    # Check if the video file exists
    if not os.path.exists(video_file_path):
        return "Video file not found", 404

    return send_file(video_file_path, as_attachment=True, download_name='Video.mp4')



if __name__ == '__main__':
    app.run(debug=True)