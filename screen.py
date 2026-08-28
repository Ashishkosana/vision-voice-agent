import subprocess
import time
import ollama


def frontmost_app():
    """Ask macOS which app is in front. Rock-solid — no AI, no guessing."""
    script = ('tell application "System Events" to get name of '
              'first application process whose frontmost is true')
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    return r.stdout.strip() or "(unknown)"


def screenshot(path="/tmp/screen.png"):
    """Grab the screen silently (-x = no camera-shutter sound)."""
    subprocess.run(["screencapture", "-x", path])
    return path


def describe_screen(path):
    """Let the LOCAL moondream say what's on the screen. Nothing leaves the Mac."""
    resp = ollama.chat(model="moondream", messages=[
        {"role": "user",
         "content": "Describe what is shown on this computer screen in one sentence.",
         "images": [path]}])
    try:
        return resp.message.content.strip()
    except AttributeError:
        return resp["message"]["content"].strip()


# --- test ---
print("👉 Switch to any app (Finder, Chrome, your editor)... capturing in 3 seconds")
time.sleep(3)

app = frontmost_app()
print(f"🎯 front app: {app}")

path = screenshot()
print(f"📸 screenshot saved: {path}")

print("👁️  looking (moondream)...")
print(f"🖼️  {describe_screen(path)}")
