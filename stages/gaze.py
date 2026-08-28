import cv2, os, time, subprocess
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
from google import genai
from google.genai import types, errors
from groq import Groq

# ── clients (same as vision.py) ──
gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
groq_client = Groq()

# ── the two OpenCV "shape-spotters" (no install) ──
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")


# ── EYE → LEFT / CENTER / RIGHT ──
def pupil_direction(eye_gray):
    h, w = eye_gray.shape
    m, n = int(w * 0.15), int(h * 0.15)            # trim edges (eyebrow/lashes)
    roi = eye_gray[n:h - n, m:w - m]
    roi = cv2.GaussianBlur(roi, (7, 7), 0)         # blur so a stray dark pixel can't win

    _, _, min_loc, _ = cv2.minMaxLoc(roi)          # darkest spot = pupil
    ratio = min_loc[0] / roi.shape[1]

    pupil_point = (m + min_loc[0], n + min_loc[1])
    if ratio < 0.40:
        return "LEFT", pupil_point
    elif ratio > 0.60:
        return "RIGHT", pupil_point
    else:
        return "CENTER", pupil_point


# ── VOICE + VISION pipeline (copied from vision.py) ──
def listen(seconds=5):
    fs = 16000
    print("🎙️   Listening... speak now")
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()

    # How loud was the recording? (RMS = average loudness)
    volume = float(np.sqrt(np.mean(audio ** 2)))
    print(f"🔊 volume: {volume:.4f}")
    if volume < 0.01:                 # basically silence → you didn't really speak
        return ""                     # tell handle() there's nothing to send

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
    question = listen()
    if not question:                              # silence → don't bother Gemini
        print("🤷 didn't catch anything — try again")
        subprocess.run(["say", "I didn't catch that"])
        return

    cv2.imwrite("shot.jpg", frame)
    with open("shot.jpg", "rb") as f:
        img = f.read()
    print(f"🗣️   You: {question}")
    answer = ask_gemini(question, img)
    print(f"🤖 {answer}")
    subprocess.run(["say", answer])


# ── watch the eyes; wake when LEFT is held ~1s ──
cam = cv2.VideoCapture(0)
print("👀 Look LEFT and hold ~1s to WAKE, then speak.  (press q in the window to quit)")

left_since = None        # stopwatch: when did I start looking LEFT?
last_trigger = 0         # for the 8-second cooldown

while True:
    ok, frame = cam.read()
    if not ok:
        continue

    frame = cv2.flip(frame, 1)                     # mirror, so it feels natural
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    current_dir = None
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)

        upper_gray = gray[fy:fy + int(fh * 0.6), fx:fx + fw]
        eyes = eye_cascade.detectMultiScale(upper_gray, 1.1, 6, minSize=(30, 30))
        for (ex, ey, ew, eh) in eyes[:2]:
            ax, ay = fx + ex, fy + ey
            cv2.rectangle(frame, (ax, ay), (ax + ew, ay + eh), (0, 255, 0), 2)

            eye_gray = gray[ay:ay + eh, ax:ax + ew]
            direction, (px, py) = pupil_direction(eye_gray)
            if current_dir is None:
                current_dir = direction            # first eye = our reading

            cv2.circle(frame, (ax + px, ay + py), 3, (0, 0, 255), -1)
            cv2.putText(frame, direction, (ax, ay - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # --- WAKE: look LEFT and hold ~1s → run the assistant ---
    now = time.time()
    if current_dir == "LEFT":
        if left_since is None:
            left_since = now
        elif now - left_since >= 1.0 and now - last_trigger > 8:
            print("🔔 WAKE!")
            subprocess.run(["say", "yes"])
            handle(frame)                          # listen → Gemini → speak
            last_trigger = time.time()             # start cooldown
            left_since = None
    else:
        left_since = None

    cv2.putText(frame, "WAKE: look LEFT and hold", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("gaze", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
