
import cv2, os, time, subprocess
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
from google import genai
from google.genai import types, errors
from groq import Groq

gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
groq_client = Groq()

  # built-in OpenCV face detector (ships with opencv-python — no install)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

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

def handle(frame):
    cv2.imwrite("shot.jpg", frame)
    with open("shot.jpg", "rb") as f:
        img = f.read()
    question = listen()
    print(f"🗣️   You: {question}")
    answer = ask_gemini(question, img)
    print(f"🤖 {answer}")
    subprocess.run(["say", answer])

  # ── watch the camera; trigger when a face is held ~1 second ──
cam = cv2.VideoCapture(0)
print("👀 Watching for a face... (Ctrl+C to quit)")
face_since = None
last_trigger = 0
try:
    while True:
        ok, frame = cam.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        now = time.time()

        if len(faces) > 0:
            if face_since is None:
                face_since = now                        # face just appeared           
            elif now - face_since >= 1.0 and now - last_trigger > 8:
                print("✅ Face detected — ask away!")
                handle(frame)
                last_trigger = time.time()              # start cooldown
                face_since = None
        else:
            face_since = None                           # face gone → reset            
except KeyboardInterrupt:
    cam.release()
    print("\n👋 Bye!")

