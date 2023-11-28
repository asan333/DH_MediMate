import streamlit as st
# from streamlit_lottie import st_lottie
import requests
import base64
import os
from io import BytesIO

def download_button_with_filename(label, data, file_name, mime):
    """Helper function to create a download button with a specific filename."""
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:{mime};base64,{b64}" download="{file_name}">{label}</a>'
    return st.markdown(href, unsafe_allow_html=True)

# UI layout and styling
st.title("Digital Health Conversation Analysis")
st.markdown("Upload your audio file to get a transcript.")

# Initialize session state
if 'response_data' not in st.session_state:
    st.session_state['response_data'] = None
if 'audio_processed' not in st.session_state:
    st.session_state['audio_processed'] = False

# Step 1: Upload Audio and Get Transcript
uploaded_audio = st.file_uploader("Choose an audio file...", type=["mp3", "wav"], key="audio_uploader")

if uploaded_audio is not None:
    # Reset session state if a new file is uploaded
    st.session_state['response_data'] = None
    st.session_state['audio_processed'] = False

if uploaded_audio is not None and not st.session_state['audio_processed']:
    with st.spinner('Processing audio to get transcript...'):
        transcript_response = requests.post('http://127.0.0.1:5000/process_audio', files={'audio': uploaded_audio.getvalue()})

        if transcript_response.status_code == 200:
            st.session_state['response_data'] = transcript_response.json()
            st.session_state['audio_processed'] = True


# Process and display the response data
if st.session_state['response_data']:
    response_data = st.session_state['response_data']

    # Display and download transcript
    transcript_file = base64.b64decode(response_data['summ_dialogue'])
    st.download_button(label="Download Transcript", data=transcript_file, file_name="transcript.txt", mime='text/plain')

    # Display and download doctor's notes
    doctor_notes = base64.b64decode(response_data['doctor_notes'])
    st.download_button(label="Download Doctor's Notes", data=doctor_notes, file_name="doctor_notes.pdf", mime='application/pdf')

    # Prepare data for the next API call
    patient_dialogue = response_data['patient_dialogue']
    doctor_dialogue = response_data['doctor_dialogue']
    first_speaker = response_data['first_speaker']

# Step 2: Upload Transcript and Get Video
uploaded_transcript = st.file_uploader("Upload the transcript file...", type=["txt"], key="transcript_uploader")

if uploaded_transcript is not None:
    with st.spinner('Processing transcript to generate video...'):
        json_payload = {
            "patient_dialogue": patient_dialogue,
            "doctor_dialogue": doctor_dialogue,
            "first_speaker": first_speaker
        }

        video_response = requests.post('http://8418-34-132-186-100.ngrok-free.app', json=json_payload, verify=False, timeout=300)

        if video_response.status_code == 200:
            # Display video
            video_base64 = base64.b64encode(video_response.content).decode("utf-8")
            
            # Set up the video container with a specific width using HTML and CSS
            video_html = f"""
            <div style="display: flex; justify-content: center; margin: auto;">
                <video width="480" controls>
                    <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
            """

            # Display video using custom HTML
            st.markdown(video_html, unsafe_allow_html=True)
        else:
            st.error(f"Failed to process transcript. Status code: {video_response.status_code}")

# Footer or additional information
st.markdown("Thank you for using our service!")





# # Initialize session state
# if 'response_data' not in st.session_state:
#     st.session_state['response_data'] = None

# # Step 1: Upload Audio and Get Transcript
# uploaded_audio = st.file_uploader("Choose an audio file...", type=["mp3", "wav"], key="audio_uploader")

# if uploaded_audio is not None:
#     # Reset session state if a new file is uploaded
#     st.session_state['response_data'] = None

#     with st.spinner('Processing audio to get transcript...'):
#         transcript_response = requests.post('http://127.0.0.1:5000/process_audio', files={'audio': uploaded_audio.getvalue()})

#         if transcript_response.status_code == 200:
#             st.session_state['response_data'] = transcript_response.json()

# # Process and display the response data
# if st.session_state['response_data']:
#     response_data = st.session_state['response_data']

#     # Display and download transcript
#     transcript_file = base64.b64decode(response_data['summ_dialogue'])
#     st.download_button(label="Download Transcript", data=transcript_file, file_name="transcript.txt", mime='text/plain')

#     # Display and download doctor's notes
#     doctor_notes = base64.b64decode(response_data['doctor_notes'])
#     st.download_button(label="Download Doctor's Notes", data=doctor_notes, file_name="doctor_notes.pdf", mime='application/pdf')

#     # Prepare data for the next API call
#     patient_dialogue = response_data['patient_dialogue']
#     doctor_dialogue = response_data['doctor_dialogue']
#     first_speaker = response_data['first_speaker']

# # Step 2: Upload Transcript and Get Video
# uploaded_transcript = st.file_uploader("Upload the transcript file...", type=["txt"], key="transcript_uploader")

# if uploaded_transcript is not None:
#     with st.spinner('Processing transcript to generate video...'):
#         json_payload = {
#             "patient_dialogue": patient_dialogue,
#             "doctor_dialogue": doctor_dialogue,
#             "first_speaker": first_speaker
#         }

#         video_response = requests.post('http://8418-34-132-186-100.ngrok-free.app', json=json_payload, verify=False, timeout=300)

#         if video_response.status_code == 200:
#             # Display video
#             video_base64 = base64.b64encode(video_response.content).decode("utf-8")
            
#             # Set up the video container with a specific width using HTML and CSS
#             video_html = f"""
#             <div style="display: flex; justify-content: center; margin: auto;">
#                 <video width="480" controls>
#                     <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
#                     Your browser does not support the video tag.
#                 </video>
#             </div>
#             """

#             # Display video using custom HTML
#             st.markdown(video_html, unsafe_allow_html=True)
#         else:
#             st.error(f"Failed to process transcript. Status code: {video_response.status_code}")

# # Footer or additional information
# st.markdown("Thank you for using our service!")
