# import sounddevice as sd
# import numpy as np
# from faster_whisper import WhisperModel
# import queue

# # Load model
# model = WhisperModel("small", compute_type="int8")  # good for CPU

# # Audio settings
# samplerate = 16000
# block_duration = 2  # seconds
# q = queue.Queue()

# def audio_callback(indata, frames, time, status):
#     q.put(indata.copy())

# print("🎙️ Listening... Speak now")

# with sd.InputStream(samplerate=samplerate,
#                     channels=1,
#                     callback=audio_callback,
#                     blocksize=int(samplerate * block_duration)):

#     while True:
#         audio_data = q.get()
#         audio_data = np.squeeze(audio_data)

#         segments, _ = model.transcribe(audio_data)

#         for segment in segments:
#             print(segment.text)

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue

# Load model
model = WhisperModel("small", compute_type="int8")

samplerate = 16000
block_duration = 3  # better context
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

print("🎙️ Listening... Speak now")

with sd.InputStream(
        samplerate=samplerate,
        channels=1,
        dtype='float32',
        callback=audio_callback,
        blocksize=int(samplerate * block_duration)):

    while True:
        audio_data = q.get()

        # flatten audio
        audio_data = audio_data.flatten()

        # skip very quiet audio (silence filter)
        if np.abs(audio_data).mean() < 0.01:
            continue

        segments, _ = model.transcribe(
            audio_data,
            beam_size=5,
            temperature=0
        )

        for segment in segments:
            print("You said:", segment.text)