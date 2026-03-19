import wave
import json
from vosk import Model, KaldiRecognizer

wf = wave.open("test1.mp3", "rb")

model = Model("vosk-model-small-en-us-0.15")
rec = KaldiRecognizer(model, wf.getframerate())

while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        print(json.loads(rec.Result()))

print(json.loads(rec.FinalResult()))