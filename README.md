# 👁️🎙️ Vision-Voice Agent (jarvis)

A **hands-free desktop assistant** for macOS. Your eyes wake it, your voice commands it,
it can **see** and **act**, and it talks back — with the private/heavy work kept **on your Mac**.

Built brick by brick on an 8 GB M1 MacBook Air, no paid APIs.

---

## What it does

```
  look LEFT (hold ~1s) ──► "yes" ──► 🎙️ you speak a command
                                            │
                                     🧠 the agent plans
                                            │
                          ┌─────────────────┼─────────────────┐
                       🛠️ run_bash        👁️ look          (or just answer)
                     (asks your y/N)   (camera → vision)
                                            │
                                    🗣️ it speaks the result
```

## The parts (and where each runs)

| Part | Job | Tech | Runs |
|------|-----|------|------|
| 👁️ eyes | gaze LEFT/CENTER/RIGHT → wake gesture | OpenCV (Haar + pupil) | **local** |
| 👂 ears | speech → text | Groq Whisper | cloud *(private: no training, not stored)* |
| 🧠 brain | plans + calls tools | Groq llama-3.3-70b | cloud *(private)* |
| 🛠️ hands | run shell commands (approval-gated) | `subprocess` | **local** |
| 🖼️ vision | describe what it sees | moondream via Ollama | **local — never leaves the Mac** |
| 🗣️ mouth | speak the reply | macOS `say` | **local** |

**Privacy:** images stay on-device (local moondream). Only your voice + text go to Groq,
which by policy doesn't train on them or store them by default.

## Setup

```bash
# 1. Python deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Local vision model (the "eyes")
brew install ollama && brew services start ollama
ollama pull moondream

# 3. Your Groq key (free tier)
export GROQ_API_KEY=your_key
```

macOS will ask for **Camera** and **Microphone** permissions the first time — allow them.

## Run

```bash
python3 jarvis.py
```

Look **LEFT** and hold ~1s → it says *"yes"* → speak a command
(e.g. *"what do you see?"*, *"make a file called notes.txt on my desktop"*).
Press **q** in the window to quit.

## Files

```
jarvis.py       ⭐ the assistant (eyes + ears + brain + hands + vision + voice)
screen.py       🖥️ screen-eyes (WIP): which app is in front + a screenshot moondream describes
requirements.txt
stages/         how it was built, brick by brick
  vision.py       stage 1 — webcam face-trigger → voice → vision
  gaze.py         stage 2 — pupil tracking → "look LEFT" wake gesture
```

## Roadmap

- [ ] **Screen-eyes** — merge `screen.py` so it sees the *desktop* (which app / what's on screen), not just the room
- [ ] Command **allowlist** instead of free-form bash (safer against prompt-injection)
- [ ] Fully local voice (local Whisper) so *nothing* leaves the Mac

## Known limitations (honest)

- Webcam gaze is **region-level**, not icon-precise (that's physics, not a bug).
- The bash tool is approval-gated + blocks obvious foot-guns, but still **trusts what it hears/sees** — don't run it around others or point the camera at untrusted text.
