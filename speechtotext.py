from faster_whisper import WhisperModel

# Load model (tiny, base, small, medium, large-v2)
model = WhisperModel("base", device="cpu")

segments, info = model.transcribe("16 Mar task recording.m4a")

print("Detected language:", info.language)

text = ""
for segment in segments:
    text += segment.text + " "

print("Transcription:")
print(text.strip())