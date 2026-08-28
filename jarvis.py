import cv2, os, time, subprocess, json, atexit
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
import ollama
from groq import Groq

# ── check the key early, with a friendly message instead of a raw crash ──
if not os.environ.get("GROQ_API_KEY"):
    raise SystemExit("⚠️  Set GROQ_API_KEY first:  export GROQ_API_KEY=your_key")

groq_client = Groq()          # cloud: voice→text + the thinking agent (Groq is private)
VISION_MODEL = "moondream"    # LOCAL eyes via Ollama — the image NEVER leaves your Mac

cam = cv2.VideoCapture(0)     # kept only for the look() tool (webcam vision)


def _cleanup():
    """Always free the camera AND delete the saved recording/snapshot on exit."""
    cam.release()
    cv2.destroyAllWindows()
    for f in ("audio.wav", "shot.jpg"):
        if os.path.exists(f):
            os.remove(f)


atexit.register(_cleanup)   # runs even on crash / Ctrl-C → camera light never stuck on


# ══════════════════════════════════════════════════════════════════
#  SENSES  (all local, on your Mac)
# ══════════════════════════════════════════════════════════════════
def listen(seconds=5):
    fs = 16000
    print("🎙️   Listening... speak now")
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    peak = float(np.abs(audio).max())                 # loudest moment (works for short words)
    print(f"🔊 peak: {peak:.3f}")
    if peak < 0.02:                                   # basically silence → don't send garbage
        return ""
    write_wav("audio.wav", fs, audio)
    with open("audio.wav", "rb") as f:
        tx = groq_client.audio.transcriptions.create(
            file=("audio.wav", f.read()), model="whisper-large-v3")
    return tx.text.strip()


def speak(text):
    print(f"🤖 {text}")
    subprocess.run(["say", "--", text])       # "--" so a reply starting with '-' isn't read as a flag


def see(question, image_path):
    """Ask the LOCAL vision model (Ollama/moondream) about an image. Nothing leaves the Mac."""
    resp = ollama.chat(model=VISION_MODEL, messages=[
        {"role": "user", "content": question, "images": [image_path]}])
    try:
        return resp.message.content.strip()
    except AttributeError:
        return resp["message"]["content"].strip()


# ══════════════════════════════════════════════════════════════════
#  HANDS  (the tools the agent can use)
# ══════════════════════════════════════════════════════════════════
# obvious foot-guns we refuse to run at all (a basic guard, NOT bulletproof)
DANGEROUS = ("rm -rf", "mkfs", ":(){", "| bash", "|bash", "| sh", "|sh",
             "curl", "wget", "base64", "dd if=", "sudo", "jarvis.py")


def run_bash(command):
    """DO something on the Mac. Every command must be approved by you."""
    # show hidden characters so a command can't DISGUISE what it really is
    safe_view = command.encode("unicode_escape").decode()
    print(f"\n🛠️  the agent wants to run:\n     {safe_view}")

    if any(bad in command.lower() for bad in DANGEROUS):
        print("     ⛔ blocked — matches a dangerous pattern, not running.")
        return "BLOCKED: command looked dangerous; not run."

    ok = input("     approve? [y/N] ").strip().lower()
    if ok != "y":
        return "DENIED by the user."
    r = subprocess.run(command, shell=True, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    return out[:3000] if out else "(done, no output)"


def look(question="What do you see?"):
    """LOOK through the camera (the room) and answer a question about the scene."""
    try:
        ok, frame = cam.read()
        if not ok:
            return "camera not available"
        frame = cv2.flip(frame, 1)
        cv2.imwrite("shot.jpg", frame)
        return see(question, "shot.jpg") or "(no description)"
    except Exception as e:
        return f"look failed: {e}"


def frontmost_app():
    """Ask macOS which app is in front — exact, no AI."""
    script = ('tell application "System Events" to get name of '
              'first application process whose frontmost is true')
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    return r.stdout.strip() or "(unknown)"


def see_screen(question="What is on the screen?"):
    """SCREEN-EYES: which app is in front + a local moondream look at the desktop."""
    try:
        subprocess.run(["screencapture", "-x", "/tmp/screen.png"])
        app = frontmost_app()
        desc = see(question, "/tmp/screen.png")
        return f"Front app: {app}. Screen shows: {desc}"
    except Exception as e:
        return f"see_screen failed: {e}"


TOOLS = [
    {"type": "function", "function": {
        "name": "run_bash",
        "description": "Run a shell command on the user's Mac to DO things "
                       "(create/open files, launch apps, check status). Returns the output.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "look",
        "description": "Look through the Mac CAMERA (the room / the user) and answer a question.",
        "parameters": {"type": "object",
                       "properties": {"question": {"type": "string"}},
                       "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "see_screen",
        "description": "See the user's DESKTOP: which app is in front and what is on the screen. "
                       "Use this for anything about the computer/screen/apps (not the room).",
        "parameters": {"type": "object",
                       "properties": {"question": {"type": "string"}},
                       "required": ["question"]}}},
]


def run_tool(name, args):
    if name == "run_bash":
        return run_bash(args.get("command", ""))
    if name == "look":
        return look(args.get("question", "What do you see?"))
    if name == "see_screen":
        return see_screen(args.get("question", "What is on the screen?"))
    return f"unknown tool: {name}"


# ══════════════════════════════════════════════════════════════════
#  BRAIN  (the agent loop — think, use tools, repeat)
# ══════════════════════════════════════════════════════════════════
SYSTEM = ("You are a hands-free assistant living on the user's Mac. "
          "You can run bash commands to do things, look through the camera (the room), "
          "and see the user's screen/desktop (which app is in front + what is on screen). "
          "For questions about apps or what's on the computer, use see_screen. "
          "Prefer simple, safe commands. Keep spoken replies short.")


def call_model(messages, tries=3):
    for i in range(tries):
        try:
            return groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages, tools=TOOLS, tool_choice="auto")
        except Exception:
            if i < tries - 1:
                time.sleep(2 ** i)
            else:
                raise


def agent(user_text):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_text}]
    for _ in range(6):                                    # at most 6 think/act steps
        msg = call_model(messages).choices[0].message
        if not msg.tool_calls:
            return msg.content or "(no reply)"
        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = run_tool(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": str(result)})
    return "Stopped after too many steps."


# ══════════════════════════════════════════════════════════════════
#  PUSH-TO-TALK  (press ENTER → listen → act → speak)
# ══════════════════════════════════════════════════════════════════
def handle():
    command = listen()
    if not command:
        speak("I didn't catch that")
        return
    print(f"🗣️   You: {command}")
    speak(agent(command))


print("🤖 jarvis ready.  Press ENTER to talk, then speak your command.  (Ctrl+C to quit)")

while True:
    try:
        input()                          # ENTER = push-to-talk
    except (KeyboardInterrupt, EOFError):
        print("\n👋 bye!")
        break
    subprocess.run(["say", "yes"])       # quick cue so you know it's listening
    try:
        handle()
    except Exception as e:
        print(f"⚠️ error: {e}")
        speak("Sorry, something went wrong")
