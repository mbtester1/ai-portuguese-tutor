import whisper
from transformers import pipeline
import pyaudio
import wave

# Load Whisper model
whisper_model = whisper.load_model("base")

# Load the text generation model
generator = pipeline("text-generation", model="microsoft/DialoGPT-medium")

# Function to record audio from microphone
def record_audio(filename, duration=5, rate=44100, channels=1):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=1024)

    print("Recording...")
    frames = []

    for _ in range(0, int(rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# Function to transcribe audio to text using Whisper
def transcribe_audio(filename):
    result = whisper_model.transcribe(filename)
    return result["text"]

# Function to generate a response using the language model
def generate_response(text):
    response = generator(text, max_length=50, num_return_sequences=1)
    return response[0]['generated_text']

if __name__ == "__main__":
    audio_filename = "recorded_audio.wav"
    record_audio(audio_filename)
    text = transcribe_audio(audio_filename)
    print(f"Transcribed Text: {text}")
    response = generate_response(text)
    print(f"AI Response: {response}")

