import cv2, os, time, subprocess
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
from google import genai
from google.genai import types, errors
from groq import Groq

gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
groq_client = Groq()
  
def capture_frame():
    cam = cv2.VideoCapture(0)
    for _ in range(5):                 # warm-up: let the camera auto-adjust exposure
        cam.read()
    ok, frame = cam.read()
    cam.release()
    if not ok or frame is None:
        return None
    cv2.imwrite("shot.jpg", frame)
    with open("shot.jpg", "rb") as f:
        return f.read()

def listen(seconds=5):
    fs = 16000
    print("🎙️   Listening... speak now")
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    write_wav("audio.wav", fs, audio)
    with open("audio.wav", "rb") as f:
        tx = groq_client.audio.transcriptions.create(
            file=("audio.wav", f.read()), model="whisper-large-v3")
    return tx.text.strip()

def ask_gemini(question, img_bytes, tries=4):
    contents = [question, types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")]
    for i in range(tries):
        try:
            return gemini.models.generate_content(
                model="gemini-3.5-flash", contents=contents).text
        except errors.ServerError:
            if i < tries - 1:
                time.sleep(2 ** i)
            else:
                raise

print("👋 Assistant ready. Press Enter to ask, or type 'q' to quit.")
try:
    while True:
        cmd = input("\nPress Enter to ask (or 'q' to quit)... ")
        if cmd.strip().lower() in ("q", "quit", "exit"):
            print("👋 Bye!")
            break
        img = capture_frame()
        if img is None:
            print("❌ Camera not available.")
            continue
        question = listen()
        print(f"🗣️   You: {question}")
        answer = ask_gemini(question, img)
        print(f"🤖 {answer}")
        subprocess.run(["say", answer])
except KeyboardInterrupt:
    print("\n👋 Bye!")

