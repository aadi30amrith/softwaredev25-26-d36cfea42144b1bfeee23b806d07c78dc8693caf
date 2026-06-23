import asyncio
import threading
import tkinter as tk
from tkinter import font
from PIL import Image
from playwright.async_api import async_playwright
from gtts import gTTS
import numpy as np
import sounddevice as sd
import subprocess
import io
import speech_recognition as sr
import json
import os

# ============================================
# GLOBAL AUDIO TOGGLE
# ============================================
ECHONAV_ENABLED = True  # Global flag for whether audio is on or off

# ============================================
# SPEECH SETTINGS (adjustable from the control panel)
# ============================================
SPEECH_RATE = 1.0  # afplay playback rate: 0.75 = slower, 2.0 = faster

# gTTS doesn't offer separate "voices", but the tld parameter changes
# which Google TTS server is used, which changes the accent.
VOICE_TLD = "com"  # default = American English

VOICE_OPTIONS = [
    ("com",    "American"),
    ("co.uk",  "British"),
    ("com.au", "Australian"),
    ("co.in",  "Indian"),
    ("ie",     "Irish"),
    ("co.za",  "South African"),
    ("ca",     "Canadian"),
]

# ============================================
# BOOKMARKS
# ============================================
# Path to the JSON file that persists bookmarks between sessions.
# Stored in the same directory as EchoNav.py itself.
BOOKMARKS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "echonav_bookmarks.json")

# Default bookmarks loaded the very first time the app runs (before the
# user has saved their own file).
DEFAULT_BOOKMARKS = [
    "https://tsaweb.org/",
    "https://www.linkedin.com/",
    "https://www.nbc.com/",
    "https://www.yahoo.com/",
    "https://www.cnn.com/",
    "https://www.amazon.com/",
    "https://www.youtube.com/",
    "https://www.google.com/",
    "https://www.instagram.com/",
    "https://www.wikipedia.org/",
]


def load_bookmarks() -> list[str]:
    """
    Load the saved bookmark list from disk.

    Reads BOOKMARKS_FILE (a simple JSON array of URL strings). If the file
    doesn't exist yet (first run) returns DEFAULT_BOOKMARKS instead and
    immediately saves them so the file is created for future sessions.
    """
    if os.path.exists(BOOKMARKS_FILE):
        try:
            with open(BOOKMARKS_FILE, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception as e:
            print("EchoNav: couldn't read bookmarks file:", e)
    # First run or corrupted file — use defaults and save them
    save_bookmarks(DEFAULT_BOOKMARKS)
    return list(DEFAULT_BOOKMARKS)


def save_bookmarks(bookmarks: list[str]) -> None:
    """
    Persist the current bookmark list to disk as a JSON file.

    Writes BOOKMARKS_FILE so bookmarks survive between app restarts.
    Called whenever a bookmark is added or removed.
    """
    try:
        with open(BOOKMARKS_FILE, "w") as f:
            json.dump(bookmarks, f, indent=2)
    except Exception as e:
        print("EchoNav: couldn't save bookmarks:", e)


def add_bookmark(url: str) -> tuple[bool, str]:
    """
    Add a URL to the bookmark list if it isn't already there.

    Returns a (success, message) tuple:
      (True,  "Saved!")             — bookmark was added
      (False, "Already bookmarked") — URL was already in the list
    Saves to disk immediately after adding.
    """
    bookmarks = load_bookmarks()
    if url in bookmarks:
        return False, "Already bookmarked"
    bookmarks.append(url)
    save_bookmarks(bookmarks)
    return True, "Saved!"


# In-memory bookmark list loaded once at startup; the bookmark window
# refreshes from disk each time it opens so additions are always reflected.
BOOKMARKS: list[str] = load_bookmarks()

# ============================================
# BRIDGE TO THE RUNNING PLAYWRIGHT BROWSER
# ============================================
# These let the Tkinter control panel (main thread) talk to the
# Playwright browser, which runs its own asyncio event loop in a
# background thread.
PLAYWRIGHT_LOOP = None
PLAYWRIGHT_PAGE = None


def toggle_audio():
    """
    Flip the global ECHONAV_ENABLED flag on or off.

    Called from JavaScript via window.echoNavToggleAudio(), which is bound
    to the Ctrl+M / Cmd+M keyboard shortcut in the browser. Also updates
    the flag when the user mutes from the control panel.

    Speaks a confirmation either way so the user knows the toggle worked.
    For the 'muted' confirmation it bypasses the enabled check by calling
    _speak_now() directly — otherwise the user would hear nothing at all.
    """
    global ECHONAV_ENABLED
    ECHONAV_ENABLED = not ECHONAV_ENABLED
    if ECHONAV_ENABLED:
        print("Audio toggled. Now: ON")
        speak("EchoNav unmuted")
    else:
        print("Audio toggled. Now: OFF")
        # Kill any currently-playing audio immediately so long voice
        # command responses stop the moment the user presses Ctrl+M.
        _kill_current_speech()
        # Bypass the mute check just for this one confirmation,
        # otherwise the user would hear nothing at all.
        _speak_now("EchoNav muted")


def _do_bring_panel_to_front():
    """
    Raise the Tkinter control panel window to the top and give it focus.

    Must only be called from the Tkinter main thread (which is why
    bring_panel_to_front() schedules this via root.after() rather than
    calling it directly from a background thread).

    Sets -topmost True briefly so the window pops above the browser, then
    removes that flag after 300ms so it doesn't stay pinned forever.
    """
    try:
        root.deiconify()
        root.lift()
        root.attributes("-topmost", True)
        root.focus_force()
        # Drop the topmost flag shortly after so it doesn't stay pinned above everything
        root.after(300, lambda: root.attributes("-topmost", False))
    except Exception as e:
        print("EchoNav: couldn't bring panel to front:", e)


def bring_panel_to_front():
    """
    Thread-safe wrapper to raise the control panel window.

    Called from JavaScript via window.echoNavShowPanel() (bound to
    Ctrl+Shift+E / Cmd+Shift+E). Since this is triggered from the
    browser's background thread, we can't call Tkinter directly —
    root.after(0, fn) safely hands the work off to Tkinter's own thread.
    """
    try:
        root.after(0, _do_bring_panel_to_front)
    except Exception as e:
        print("EchoNav: couldn't schedule bring-to-front:", e)


# ============================================
# URL FILTER
# ============================================
def is_url(text: str) -> bool:
    """
    Return True if the given text looks like a URL.

    Used to prevent EchoNav from reading raw URLs aloud (e.g. when an
    image's alt text is accidentally set to its src URL, or when a link's
    visible label is a full HTTP address). Checks for http://, https://,
    or www. prefixes.
    """
    if not text:
        return False
    t = text.strip().lower()
    return t.startswith("http://") or t.startswith("https://") or t.startswith("www.")


# ============================================
# SPEECH
# ============================================
# Counter used by _speak_now and speak_and_wait to generate unique temp
# filenames. A pool of 10 rotating names is enough since speech calls
# don't overlap more than a couple at a time.
_speech_counter = {"n": 0}

# Holds a reference to the currently-playing afplay subprocess.
# Both _speak_now (non-blocking) and speak_and_wait (blocking) store their
# process here so toggle_audio() can kill it immediately when Ctrl+M fires.
_current_afplay = {"proc": None}


def _kill_current_speech():
    """
    Kill the afplay process that is currently playing audio, if any.

    Called by toggle_audio() when the user mutes so that long spoken
    responses (nav options, headings, summaries) stop immediately rather
    than finishing the whole sentence first.
    """
    proc = _current_afplay.get("proc")
    if proc and proc.poll() is None:   # poll() returns None if still running
        try:
            proc.kill()
        except Exception:
            pass
    _current_afplay["proc"] = None


def _speak_now(text):
    """
    Synthesize text to speech and play it, bypassing the mute check.

    Uses gTTS to send the text to Google's TTS servers and save the result
    as a uniquely-named MP3 file (using a counter so concurrent calls never
    overwrite each other), then plays it using afplay at the current
    SPEECH_RATE speed.

    Stores the afplay subprocess in _current_afplay so toggle_audio() can
    kill it immediately if Ctrl+M is pressed while it's still playing.

    Only called directly for mute/unmute confirmations. All other speech
    goes through speak() which checks ECHONAV_ENABLED first.
    """
    if not text:
        return
    if is_url(text):
        print("Blocked URL:", text)
        return

    text = text.strip()
    if text:
        _speech_counter["n"] += 1
        filename = f"echonav_speech_{_speech_counter['n'] % 10}.mp3"
        try:
            tts = gTTS(text=text, lang='en', tld=VOICE_TLD)
            tts.save(filename)
            rate = str(SPEECH_RATE)
            def _play(f=filename):
                proc = subprocess.Popen(["afplay", "-r", rate, f])
                _current_afplay["proc"] = proc
                proc.wait()
                _current_afplay["proc"] = None
            threading.Thread(target=_play, daemon=True).start()
        except Exception as e:
            print("Speech error:", e)


def speak(text):
    """
    Speak text aloud if EchoNav is not muted.

    Checks ECHONAV_ENABLED and filters out blank text and URLs before
    passing to _speak_now(). This is the main entry point for all speech
    in the application — page event handlers, voice command responses,
    and control panel feedback all call this function.
    """
    if not ECHONAV_ENABLED:
        return
    if not text:
        return
    if is_url(text):
        print("Blocked URL:", text)
        return
    _speak_now(text)


def speak_and_wait(text):
    """
    Speak text aloud and BLOCK until afplay finishes (or is killed).

    Used for voice command results and the repeat-offer flow so the
    "Would you like me to repeat that?" question only plays after the
    main content has fully finished. Uses subprocess.Popen so the process
    is stored in _current_afplay — pressing Ctrl+M while this is running
    kills the process and unblocks this function immediately.

    Respects ECHONAV_ENABLED — does nothing if muted.
    """
    if not ECHONAV_ENABLED:
        return
    if not text or is_url(text):
        return
    text = text.strip()
    if not text:
        return
    _speech_counter["n"] += 1
    filename = f"echonav_speech_{_speech_counter['n'] % 10}.mp3"
    try:
        tts = gTTS(text=text, lang='en', tld=VOICE_TLD)
        tts.save(filename)
        proc = subprocess.Popen(["afplay", "-r", str(SPEECH_RATE), filename])
        _current_afplay["proc"] = proc
        proc.wait()          # blocks until finished OR killed by toggle_audio()
        _current_afplay["proc"] = None
    except Exception as e:
        print("Speech error (blocking):", e)
        _current_afplay["proc"] = None


# ============================================
# PING SOUND
# ============================================
def play_ping(direction='center'):
    """
    Play a short directional audio ping through the speakers.

    Generates a pure sine wave tone mathematically using numpy, then plays
    it as a stereo signal via sounddevice. The direction controls which
    channel gets the tone:
      'left'   — tone in left ear only (440 Hz) — used for form errors
      'right'  — tone in right ear only (220 Hz) — used for links
      'center' — tone in both ears (440 Hz) — used for buttons and forms

    This gives users a spatial audio cue that complements the spoken label.
    Does nothing if ECHONAV_ENABLED is False.
    """
    if not ECHONAV_ENABLED:
        return

    duration = 0.2
    freq = 440 if direction == 'center' else 220
    sample_rate = 44100

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)

    if direction == 'left':
        stereo = np.column_stack((tone, np.zeros_like(tone)))
    elif direction == 'right':
        stereo = np.column_stack((np.zeros_like(tone), tone))
    else:
        stereo = np.column_stack((tone, tone))

    sd.play(stereo, samplerate=sample_rate)


# ============================================
# VOICE COMMAND PARSER
# ============================================

# Words that signal navigation intent — stripped before matching
COMMAND_TRIGGERS = [
    "navigate to", "navigate", "go to", "go", "take me to", "take me",
    "open", "click", "press", "find", "show me", "show",
    "bring me to", "bring me", "jump to", "jump", "visit", "head to",
    "select", "tap", "hit", "load", "launch", "scroll to",
]

# Phrases that trigger a page summary instead of navigation
SUMMARY_TRIGGERS = [
    "summarize the page", "summarize this page", "summarize page",
    "what's on this page", "what is on this page", "what's on the page",
    "describe the page", "describe this page", "page summary",
    "what do i have", "what's here", "what is here",
    "list the page", "read the page", "overview",
]

# Phrases that ask about the top navigation bar / menu (and sub-menus)
NAV_TRIGGERS = [
    "what are the options on the top bar", "what's on the top bar",
    "what is on the top bar", "what are the top bar options",
    "top bar options", "what's in the top bar", "what is in the top bar",
    "what's in the navigation", "what is in the navigation",
    "what are the navigation options", "navigation options",
    "what are the menu options", "menu options",
    "what's in the menu", "what is in the menu",
    "list the menu", "list the navigation", "list the nav",
    "what are the nav options", "nav options",
    "what's in the nav bar", "what is in the nav bar", "nav bar options",
    "what options are on the top bar", "what are my navigation options",
]

# Phrases that ask about the page's headings
HEADING_TRIGGERS = [
    "what are the headings", "what is the heading", "what's the heading",
    "what are the headings on this page", "what are the headings on the page",
    "list the headings", "what headings are there", "what headings are on this page",
    "what big text is on this page", "what are the section titles",
    "what is the main heading", "what are the main headings",
    "read the headings", "what titles are on this page", "list the section titles",
]

# Phrases that trigger adding the current page to bookmarks
BOOKMARK_ADD_TRIGGERS = [
    "add this website to my folder",
    "add this to my folder",
    "save this website",
    "save this page",
    "bookmark this",
    "bookmark this page",
    "bookmark this website",
    "add to bookmarks",
    "add to my bookmarks",
    "save to bookmarks",
    "add this site to my folder",
    "save this site",
]

# Maps a detected intent to the JS function (exposed on window) that
# gathers the info and speaks it.
# Note: bookmark_add is handled directly in run_voice_command (Python side)
# because it needs to write to disk, not just read DOM info.
INTENT_JS_FUNCTIONS = {
    "summarize": "window.echoNavSummarize",
    "nav_options": "window.echoNavGetNavOptions",
    "headings": "window.echoNavGetHeadings",
}

# Status messages shown on the control panel while each intent runs
INTENT_STATUS_MESSAGES = {
    "summarize": "📋 Summarizing page…",
    "nav_options": "🧭 Checking the top bar…",
    "headings": "📑 Checking the page headings…",
}

# Confirmation status shown once the JS function has run
INTENT_DONE_MESSAGES = {
    "summarize": "📋 Page summarized.",
    "nav_options": "🧭 Read out the top bar options.",
    "headings": "📑 Read out the page headings.",
}


def detect_intent(text: str) -> str | None:
    """
    Check a spoken transcript against all known informational command patterns.

    Compares the lowercased transcript against SUMMARY_TRIGGERS,
    NAV_TRIGGERS, and HEADING_TRIGGERS. Returns the intent name string
    ('summarize', 'nav_options', or 'headings') if a match is found,
    or None if the transcript doesn't match any informational pattern
    (meaning it should be treated as a navigation command instead).
    """
    t = text.lower().strip()

    intent_triggers = {
        "summarize": SUMMARY_TRIGGERS,
        "nav_options": NAV_TRIGGERS,
        "headings": HEADING_TRIGGERS,
        "bookmark_add": BOOKMARK_ADD_TRIGGERS,
    }

    for intent, triggers in intent_triggers.items():
        if any(t == trig or t.startswith(trig) for trig in triggers):
            return intent

    return None


# Words that are too generic to match against page elements
STOP_WORDS = {
    "the", "a", "an", "page", "section", "link", "button",
    "tab", "menu", "item", "please", "can", "you", "i", "want",
    "to", "on", "in", "at", "of", "and", "or", "this", "that",
    "website", "site", "here",
}


def extract_target(command: str) -> str:
    """
    Strip navigation trigger words from a spoken command to get the target.

    Removes phrases like "navigate to", "go to", "open", etc. from the
    start of the command (trying longest phrases first to avoid partial
    matches), then removes common stop words like "the", "a", "page".

    Example: "navigate to the home page" → "home"
    Example: "click the sign in button" → "sign in"

    The returned string is then matched against page element labels by
    find_best_match().
    """
    text = command.lower().strip()

    # Remove trigger phrases (longest first to avoid partial matches)
    for trigger in sorted(COMMAND_TRIGGERS, key=len, reverse=True):
        if text.startswith(trigger):
            text = text[len(trigger):].strip()
            break

    # Remove stop words
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)


def word_overlap_score(target: str, candidate: str) -> int:
    """
    Count how many words in 'target' also appear in 'candidate'.

    Used as the primary scoring method in find_best_match(). Both strings
    are lowercased and split into word sets before comparing.

    Example: word_overlap_score("home", "Home Page") → 1
    Example: word_overlap_score("sign in", "Sign In Button") → 2
    """
    target_words = set(target.lower().split())
    candidate_words = set(candidate.lower().split())
    return len(target_words & candidate_words)


def find_best_match(target: str, elements: list) -> dict | None:
    """
    Find the page element whose label best matches the spoken target.

    Scores each element using word_overlap_score() as the base, then adds:
      +2 if the target string appears as a substring of the label
      +1 if the label starts with the first word of the target

    Only returns a match if the best score is greater than zero — this
    prevents false positives when nothing on the page is remotely relevant.

    Each element in the list is a dict with 'text', 'tag', and 'index'
    keys (as returned by the JS getPageElements() function).
    """
    if not target or not elements:
        return None

    best = None
    best_score = 0

    for el in elements:
        label = (el.get("text") or "").strip()
        if not label or is_url(label):
            continue

        score = word_overlap_score(target, label)

        # Bonus: target appears as substring
        if target in label.lower():
            score += 2

        # Bonus: first word of target matches start of label
        target_words = target.lower().split()
        if target_words and label.lower().startswith(target_words[0]):
            score += 1

        if score > best_score:
            best_score = score
            best = el

    # Only return a match if there's at least some overlap
    return best if best_score > 0 else None


def listen_and_parse() -> dict:
    """
    Listen to the microphone and convert speech to a structured command.

    Adjusts for ambient noise, listens for up to 6 seconds of silence
    before timing out, and limits each phrase to 8 seconds. Sends the
    recorded audio to Google's speech-to-text API via the SpeechRecognition
    library.

    Once a transcript is received, calls detect_intent() to check for
    informational commands (summarize, nav_options, headings). If none
    match, falls through to extract_target() for navigation commands.

    Returns a dict with:
      'transcript' — the raw spoken text (or empty string on error)
      'target'     — the extracted navigation target (navigation intent only)
      'intent'     — 'summarize', 'nav_options', 'headings', or 'navigate'
      'error'      — 'timeout', 'unclear', 'stt_error: ...', or '' for success
    """
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 0.8   # stop after 0.8s silence
    recognizer.energy_threshold = 300  # microphone sensitivity

    try:
        with sr.Microphone() as source:
            print("EchoNav: Listening for voice command...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=6, phrase_time_limit=8)

        transcript = recognizer.recognize_google(audio)
        print(f"EchoNav heard: {transcript}")

        intent = detect_intent(transcript)
        if intent:
            print(f"EchoNav intent: {intent}")
            return {"transcript": transcript, "target": "", "intent": intent, "error": ""}

        target = extract_target(transcript)
        print(f"EchoNav target: {target}")
        return {"transcript": transcript, "target": target, "intent": "navigate", "error": ""}

    except sr.WaitTimeoutError:
        return {"transcript": "", "target": "", "intent": "", "error": "timeout"}
    except sr.UnknownValueError:
        return {"transcript": "", "target": "", "intent": "", "error": "unclear"}
    except sr.RequestError as e:
        return {"transcript": "", "target": "", "intent": "", "error": f"stt_error: {e}"}
    except Exception as e:
        return {"transcript": "", "target": "", "intent": "", "error": str(e)}


# ============================================
# RUN JS ON THE LIVE PAGE — FOR DISPLAY CONTROLS
# ============================================
def run_on_page(js_code, arg=None, timeout=5):
    """
    Run a JavaScript expression in the live Playwright browser and return
    its result, called safely from a non-async thread (e.g. Tkinter).

    The browser runs in its own asyncio event loop on a background thread.
    You can't call async Playwright functions directly from Tkinter's thread,
    so this function uses asyncio.run_coroutine_threadsafe() to submit the
    JS evaluation to the browser's loop and then blocks until it completes.

    js_code  — a JavaScript expression string, e.g. "document.title"
    arg      — an optional Python value passed as the argument to a JS
               function expression, e.g. "(p) => { ... }" with arg=50
    timeout  — seconds to wait before giving up (default 5)

    Returns the JS result (converted to a Python type by Playwright), or
    None if the browser isn't connected or an error occurs.
    """
    loop = PLAYWRIGHT_LOOP
    page = PLAYWRIGHT_PAGE

    if loop is None or page is None:
        return None

    try:
        if arg is not None:
            future = asyncio.run_coroutine_threadsafe(page.evaluate(js_code, arg), loop)
        else:
            future = asyncio.run_coroutine_threadsafe(page.evaluate(js_code), loop)
        return future.result(timeout=timeout)
    except Exception as e:
        print("EchoNav run_on_page error:", e)
        return None


# ============================================
# REPEAT CONFIRMATION LISTENER
# ============================================

# Phrases that mean "yes, repeat it"
REPEAT_YES = [
    "yes", "yeah", "yep", "yup", "sure", "please", "yes please",
    "repeat", "repeat that", "say that again", "again", "once more",
    "repeat it", "say again",
]

# Phrases that mean "repeat only a specific numbered item"
# e.g. "just the second option", "option 3", "number two"
import re as _re

# Phrases that mean "no, don't repeat"
REPEAT_NO = [
    "no", "nope", "nah", "no thanks", "i'm good", "im good",
    "that's fine", "thats fine", "stop", "nevermind", "never mind",
    "don't repeat", "dont repeat", "skip",
]

ORDINALS = {
    "first": 1, "one": 1, "1": 1,
    "second": 2, "two": 2, "2": 2,
    "third": 3, "three": 3, "3": 3,
    "fourth": 4, "four": 4, "4": 4,
    "fifth": 5, "five": 5, "5": 5,
    "sixth": 6, "six": 6, "6": 6,
    "seventh": 7, "seven": 7, "7": 7,
    "eighth": 8, "eight": 8, "8": 8,
    "ninth": 9, "nine": 9, "9": 9,
    "tenth": 10, "ten": 10, "10": 10,
}


def listen_for_repeat(items: list[str]) -> str:
    """
    After reading a list of items aloud, ask the user if they want to
    hear it again and listen for their answer.

    Returns one of:
      "yes"       — repeat the whole thing
      "no"        — don't repeat
      "item:N"    — repeat only item N (1-based index)
      "unclear"   — couldn't understand the reply
      "timeout"   — no reply received
      "error"     — microphone problem

    The caller (run_voice_command) uses the return value to decide
    what to do next.
    """
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 0.8
    recognizer.energy_threshold = 300

    try:
        with sr.Microphone() as source:
            print("EchoNav: Listening for repeat confirmation...")
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio = recognizer.listen(source, timeout=6, phrase_time_limit=6)

        transcript = recognizer.recognize_google(audio).lower().strip()
        print(f"EchoNav repeat reply: {transcript}")

        # Check for "no" first so "no thanks" doesn't accidentally match a number
        if any(transcript == w or transcript.startswith(w) for w in REPEAT_NO):
            return "no"

        # Check for "yes" / "repeat all"
        if any(transcript == w or transcript.startswith(w) for w in REPEAT_YES):
            return "yes"

        # Check for a specific item number, e.g.
        # "just the second option", "number 3", "option two", "the third one"
        for word, number in ORDINALS.items():
            if word in transcript and number <= len(items):
                return f"item:{number}"

        # Also catch bare digits like "3"
        digits = _re.findall(r"\b(\d+)\b", transcript)
        for d in digits:
            n = int(d)
            if 1 <= n <= len(items):
                return f"item:{n}"

        return "unclear"

    except sr.WaitTimeoutError:
        return "timeout"
    except sr.UnknownValueError:
        return "unclear"
    except Exception:
        return "error"


# Minimum number of items in a spoken list before EchoNav offers to repeat.
# Lists shorter than this are easy enough to remember without a repeat offer.
REPEAT_THRESHOLD = 3


def speak_list_with_repeat_offer(items: list[str], intro: str, set_status) -> None:
    """
    Speak `intro` followed by each item in `items`, then ask the user
    whether they want any of it repeated.

    `intro`      — e.g. "The top bar has the following options:"
    `items`      — list of individual option strings
    `set_status` — callback to update the control panel status label

    If the list is shorter than REPEAT_THRESHOLD items, no repeat offer
    is made since it's short enough to remember easily.
    """
    # Build the full spoken sentence and say it
    full_text = intro + " " + ". ".join(items) + "."
    speak(full_text)

    # Don't bother offering a repeat for very short lists
    if len(items) < REPEAT_THRESHOLD:
        return

    # Offer the repeat
    import time
    time.sleep(0.4)   # small gap so the offer doesn't overlap the main audio
    speak("Would you like me to repeat that?")
    set_status("🔁 Say 'yes', 'no', or 'just the second option', etc.")

    reply = listen_for_repeat(items)

    if reply == "yes":
        speak(full_text)
        set_status("🔁 Repeated.")

    elif reply.startswith("item:"):
        n = int(reply.split(":")[1])
        speak(f"Option {n}: {items[n - 1]}")
        set_status(f"🔁 Repeated option {n}.")

    elif reply == "no":
        set_status("✅ Done.")

    elif reply == "timeout":
        # User didn't reply — that's fine, just move on silently
        set_status("✅ Done.")

    else:
        # unclear / error — move on
        set_status("✅ Done.")


# ============================================
# VOICE COMMAND — RUN FROM THE CONTROL PANEL
# ============================================
def run_voice_command(on_status=None):
    """
    Runs in a background thread, triggered by the control panel's mic button.
    Listens for a spoken command, detects its intent, and either calls the
    appropriate JS function on the live page (for informational queries) or
    matches and clicks a page element (for navigation commands).

    For informational results with 3+ items, asks the user if they want
    a repeat, then listens for "yes", "no", or "just the second option" etc.
    """
    def set_status(msg):
        """Update the control panel status label (thread-safe via on_status)."""
        if on_status:
            on_status(msg)

    loop = PLAYWRIGHT_LOOP
    page = PLAYWRIGHT_PAGE

    if loop is None or page is None:
        set_status("EchoNav isn't connected to a browser yet.")
        return

    set_status("🎤 Listening… speak your command")
    speak("Listening. Say your command.")

    result = listen_and_parse()

    if result["error"] == "timeout":
        set_status("⏱ No speech detected. Try again.")
        speak("No speech detected. Try again.")
        return

    if result["error"] == "unclear":
        set_status("❓ Couldn't understand. Please repeat.")
        speak("Couldn't understand. Please repeat.")
        return

    if result["error"]:
        set_status("⚠️ Microphone error. Check your mic.")
        speak("Microphone error.")
        return

    intent = result["intent"]

    # ------------------------------------------------------------------
    # Informational intents: summarize / nav_options / headings
    # Call the matching JS function, get back the summary string, speak
    # it (JS does the speaking), then offer a repeat if the list is long.
    # ------------------------------------------------------------------
    if intent in INTENT_JS_FUNCTIONS:
        fn = INTENT_JS_FUNCTIONS[intent]
        set_status(INTENT_STATUS_MESSAGES.get(intent, "Working…"))
        try:
            # The JS function speaks the result itself AND returns the
            # summary string so we can parse the items list for repeat.
            summary = asyncio.run_coroutine_threadsafe(
                page.evaluate(f"{fn} ? {fn}() : ''"),
                loop
            ).result(timeout=10)

            set_status(INTENT_DONE_MESSAGES.get(intent, "Done."))

            # Speak the summary ourselves (blocking) so we know exactly
            # when it finishes before asking the repeat question.
            if summary:
                speak_and_wait(summary)

                after_colon = summary.split(": ", 1)[-1].rstrip(".")
                items = [i.strip() for i in after_colon.split(". ") if i.strip()]
                if len(items) >= REPEAT_THRESHOLD:
                    # Small pause so the mic doesn't open while the room is
                    # still echoing from the last spoken word.
                    import time; time.sleep(0.6)
                    speak_and_wait("Would you like me to repeat that?")
                    import time; time.sleep(0.5)
                    set_status("🔁 Mic is on — say yes, no, or 'just the second option'")
                    reply = listen_for_repeat(items)
                    print(f"EchoNav repeat reply code: {reply}")

                    if reply == "yes":
                        speak_and_wait(summary)
                        set_status("🔁 Repeated.")
                    elif reply.startswith("item:"):
                        n = int(reply.split(":")[1])
                        speak_and_wait(f"Option {n}: {items[n - 1]}")
                        set_status(f"🔁 Repeated option {n}.")
                    else:
                        set_status("✅ Done.")

        except Exception as e:
            set_status("⚠️ Couldn't check this page.")
            print(f"EchoNav {intent} error:", e)
        return

    # ------------------------------------------------------------------
    # Bookmark add intent: get current page URL from the browser and
    # save it to the bookmark list on disk, then refresh the bookmark
    # window if it's currently open.
    # ------------------------------------------------------------------
    if intent == "bookmark_add":
        set_status("🔖 Adding this page to your folder…")
        try:
            url = asyncio.run_coroutine_threadsafe(
                PLAYWRIGHT_PAGE.evaluate("window.location.href"),
                PLAYWRIGHT_LOOP
            ).result(timeout=5)
            success, msg = add_bookmark(url)
            if success:
                speak(f"Added {url} to your folder.")
                set_status(f"🔖 Saved: {url}")
            else:
                speak("This website is already in your folder.")
                set_status("🔖 Already in your folder.")
        except Exception as e:
            print("EchoNav bookmark_add error:", e)
            set_status("⚠️ Couldn't get the current page URL.")
            speak("Sorry, I couldn't get the current page address.")
        return

    # ------------------------------------------------------------------
    # Navigation intent: find the best-matching link/button and click it
    # ------------------------------------------------------------------
    target = result["target"]
    if not target:
        set_status("Couldn't figure out a command from that.")
        speak("Sorry, I didn't catch a command.")
        return

    try:
        elements = asyncio.run_coroutine_threadsafe(
            page.evaluate("window.echoNavGetElements ? window.echoNavGetElements() : []"),
            loop
        ).result(timeout=10)
    except Exception as e:
        elements = []
        print("EchoNav get-elements error:", e)

    match = find_best_match(target, elements)

    if not match:
        heard = result["transcript"]
        if heard:
            set_status(f'🔍 Heard "{heard}" — nothing matched on this page.')
        else:
            set_status("🔍 Nothing matched on this page.")
        speak("Couldn't find that on this page.")
        return

    set_status(f"✅ Going to: {match['text']}")
    speak("Navigating to " + match["text"])

    try:
        asyncio.run_coroutine_threadsafe(
            page.evaluate(
                "([idx, label]) => window.echoNavClickElement(idx, label)",
                [match["index"], match["text"]]
            ),
            loop
        ).result(timeout=10)
    except Exception as e:
        print("EchoNav click error:", e)
        set_status("⚠️ Found it, but couldn't click it.")


# ============================================
# PLAYWRIGHT LOGIC
# ============================================
async def run_playwright(url):
    """
    Core browser coroutine — runs for the entire lifetime of the session.

    Launches a visible Chromium browser via Playwright, creates a fresh
    browser context (isolated cookies/storage), injects the EchoNav JS
    overlay into every page, navigates to the starting URL, then loops
    forever (checking once per second) until the browser window is closed.

    Also saves a reference to this asyncio event loop and the current page
    into the PLAYWRIGHT_LOOP and PLAYWRIGHT_PAGE globals so the Tkinter
    control panel can call run_on_page() from a different thread.

    Listens for new tabs via context.on("page", handle_new_page) and
    re-injects EchoNav into each new tab automatically.
    """
    global PLAYWRIGHT_LOOP, PLAYWRIGHT_PAGE

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=[
                # Disable Content Security Policy enforcement so EchoNav's
                # injected JS works on sites like ESPN that have strict CSPs.
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ]
        )
        context = await browser.new_context()

        echonav_script = """
        if (!window.__echoNavInjected) {
            window.__echoNavInjected = true;

            function setupEchoNav() {
                let lastSpokenSelection = "";

                // Collect all clickable elements on the page with their text + index
                function getPageElements() {
                    const results = [];
                    let index = 0;

                    document.querySelectorAll("a, button").forEach(el => {
                        const text = (
                            el.innerText ||
                            el.getAttribute("aria-label") ||
                            el.title ||
                            ""
                        ).trim();

                        if (text && text.length < 120) {
                            results.push({ text, tag: el.tagName.toLowerCase(), index });
                        }
                        index++;
                    });

                    return results;
                }

                // Summarize the page and speak it aloud
                function summarizePage() {
                    // Buttons
                    const buttons = [];
                    document.querySelectorAll("button").forEach(btn => {
                        const label = (btn.innerText || btn.getAttribute("aria-label") || "").trim();
                        if (label && !label.startsWith("http")) buttons.push(label);
                    });

                    // Links
                    const links = [];
                    document.querySelectorAll("a").forEach(a => {
                        const label = (a.innerText || a.getAttribute("aria-label") || a.title || "").trim();
                        if (label && !label.startsWith("http") && label.length < 80) links.push(label);
                    });

                    // Images
                    const images = document.querySelectorAll("img");

                    // Form inputs
                    const inputs = [];
                    document.querySelectorAll("input, textarea, select").forEach(field => {
                        const type = field.type || field.tagName.toLowerCase();
                        if (type !== "hidden" && type !== "submit" && type !== "button") {
                            const label = field.placeholder || field.getAttribute("aria-label") || field.name || type;
                            inputs.push(label);
                        }
                    });

                    // Build spoken summary
                    let parts = [];

                    // Buttons
                    if (buttons.length === 0) {
                        parts.push("no buttons");
                    } else {
                        const names = buttons.join(", ");
                        parts.push(`${buttons.length} button${buttons.length > 1 ? "s" : ""}: ${names}`);
                    }

                    // Links
                    if (links.length > 0) {
                        parts.push(`${links.length} link${links.length > 1 ? "s" : ""}`);
                    }

                    // Images
                    if (images.length > 0) {
                        parts.push(`${images.length} image${images.length > 1 ? "s" : ""}`);
                    }

                    // Inputs
                    if (inputs.length > 0) {
                        const inputNames = inputs.join(", ");
                        parts.push(`${inputs.length} form field${inputs.length > 1 ? "s" : ""}: ${inputNames}`);
                    }

                    const summary = "This page has " + parts.join(". It also has ") + ".";
                    return summary;
                }

                // ============================
                // ============================
                // TOP BAR / NAVIGATION OPTIONS (with sub-options)
                // ============================
                function getNavOptions() {

                    // --- Helpers ---

                    // Extract a clean single-line text label from an element.
                    // Uses innerText first (respects CSS visibility), falls back
                    // to aria-label, then title. Takes only the first line.
                    function cleanLabel(el) {
                        const raw = (
                            el.innerText ||
                            el.getAttribute("aria-label") ||
                            el.title || ""
                        ).trim();
                        return raw.split("\\n")[0].trim();
                    }

                    // Words that signal a logo, icon, utility, or promo link.
                    // Any nav link whose label contains one of these is skipped.
                    const JUNK = [
                        "logo", "icon", "skip", "search", "close", "modal",
                        "hamburger", "toggle", "account", "sign in", "sign out",
                        "login", "log in", "register", "cart", "provider",
                        "hulu", "disney", "try", "feedback", "edition",
                    ];
                    function isJunk(label) {
                        if (!label || label.length <= 1) return true;
                        const l = label.toLowerCase();
                        return JUNK.some(w => l.includes(w));
                    }

                    // Get the top-level label of a container element
                    // without including its nested submenu text.
                    function labelOf(el) {
                        if (el.tagName === "A" || el.tagName === "BUTTON") return cleanLabel(el);
                        const child = el.querySelector(":scope > a, :scope > button");
                        return cleanLabel(child || el);
                    }

                    // Get DIRECT sub-items of an <li> (one level deep only).
                    // :scope > ul > li prevents grandchild links (mega-menus) leaking in.
                    function getSubItems(liEl) {
                        const subUl = liEl.querySelector(":scope > ul");
                        if (!subUl) return [];
                        const seen = new Set();
                        const out = [];
                        subUl.querySelectorAll(":scope > li > a, :scope > li > button").forEach(s => {
                            const t = cleanLabel(s);
                            if (t && t.length > 1 && t.length < 50 && !seen.has(t) && !isJunk(t)) {
                                seen.add(t);
                                out.push(t);
                            }
                        });
                        return out;
                    }

                    // Filter a list of elements down to real nav links only.
                    function filterNavLinks(els) {
                        return els.filter(el => {
                            const label = cleanLabel(el);
                            // Must have visible text, be a reasonable length, not be junk,
                            // and not be a plain anchor pointing to "#" (utility links)
                            const href = el.getAttribute("href") || "";
                            return (
                                label.length >= 2 &&
                                label.length < 45 &&
                                !isJunk(label) &&
                                el.innerText.trim().length > 0 &&
                                href !== "#"
                            );
                        });
                    }

                    let items = [];

                    // ---- Strategy 0: visible <nav> containing a <ul> ----
                    // Pick the <nav> whose bounding box is closest to the top of the
                    // viewport — this handles CNN which has two <nav> elements
                    // (one hidden mobile nav and one visible desktop nav).
                    const allNavs = Array.from(document.querySelectorAll("nav, [role='navigation']"));
                    const visibleNavs = allNavs.filter(n => {
                        const r = n.getBoundingClientRect();
                        // Must be visible and near the top of the screen
                        return r.width > 0 && r.height > 0 && r.top < 200;
                    });
                    // Sort by vertical position — pick the topmost one
                    visibleNavs.sort((a, b) => a.getBoundingClientRect().top - b.getBoundingClientRect().top);

                    for (const navEl of visibleNavs) {
                        // Prefer a <ul> whose direct <li> children have nav links
                        const uls = Array.from(navEl.querySelectorAll("ul"));
                        for (const ul of uls) {
                            const lis = Array.from(ul.children).filter(c => c.tagName === "LI");
                            if (lis.length >= 2) {
                                // Make sure at least 2 of those <li> have readable links
                                const withLabels = lis.filter(li => {
                                    const label = labelOf(li);
                                    return label.length >= 2 && !isJunk(label);
                                });
                                if (withLabels.length >= 2) {
                                    items = lis;
                                    break;
                                }
                            }
                        }
                        if (items.length > 0) break;

                        // No <ul> — try direct link children of the nav
                        if (items.length === 0) {
                            const directLinks = filterNavLinks(
                                Array.from(navEl.querySelectorAll(":scope > a, :scope > button, :scope > li > a, :scope > div > a"))
                            );
                            if (directLinks.length >= 2) { items = directLinks; break; }
                        }
                    }

                    // ---- Strategy 1: Amazon-style top nav by id ----
                    // Amazon uses id="nav-main" or id="navbar" with a custom structure.
                    if (items.length === 0) {
                        const amznNav = document.querySelector("#nav-main, #navbar, #nav-bar");
                        if (amznNav) {
                            const links = filterNavLinks(Array.from(amznNav.querySelectorAll("a")));
                            if (links.length >= 2) items = links;
                        }
                    }

                    // ---- Strategy 2: <header> fallback (ABC-style) ----
                    // No <nav> tag — links sit directly in <header>.
                    // Strict junk filtering since headers mix logos/icons/promos.
                    if (items.length === 0) {
                        const header = document.querySelector("header");
                        if (header) {
                            const links = filterNavLinks(Array.from(header.querySelectorAll("a, button")));
                            if (links.length >= 2) items = links;
                        }
                    }

                    // ---- Strategy 3: fixed/sticky element near top of viewport ----
                    if (items.length === 0) {
                        for (const el of document.querySelectorAll("div, section, header")) {
                            const style = window.getComputedStyle(el);
                            const rect = el.getBoundingClientRect();
                            if (
                                (style.position === "fixed" || style.position === "sticky") &&
                                rect.top < 80 && rect.width > window.innerWidth * 0.4
                            ) {
                                const links = filterNavLinks(Array.from(el.querySelectorAll("a, button")));
                                if (links.length >= 2) { items = links; break; }
                            }
                        }
                    }

                    if (items.length === 0) {
                        window.echoNavSpeak("I couldn't identify the navigation bar on this page.");
                        return "No navigation bar found.";
                    }

                    // Build spoken result
                    const parts = [];
                    const seen = new Set();
                    items.forEach(item => {
                        const label = labelOf(item);
                        if (!label || seen.has(label) || isJunk(label) || label.length > 60) return;
                        seen.add(label);
                        const subs = item.tagName === "LI" ? getSubItems(item) : [];
                        if (subs.length > 0) {
                            parts.push(`${label}, with sub-options: ${subs.join(", ")}`);
                        } else {
                            parts.push(label);
                        }
                    });

                    if (parts.length === 0) {
                        window.echoNavSpeak("I found a navigation area but couldn't read the labels.");
                        return "No readable menu options found.";
                    }

                    return "The top bar has the following options: " + parts.join(". ") + ".";
                }

                // ============================
                // PAGE HEADINGS
                // ============================
                function getHeadings() {
                    const headings = Array.from(document.querySelectorAll("h1, h2, h3, h4, h5, h6"))
                        .map(h => ({
                            level: parseInt(h.tagName.substring(1), 10),
                            text: (h.innerText || "").trim()
                        }))
                        .filter(h => h.text.length > 0 && h.text.length < 150);

                    if (headings.length === 0) {
                        window.echoNavSpeak("This page doesn't have any headings.");
                        return "No headings found.";
                    }

                    // Group headings by level
                    const byLevel = {};
                    headings.forEach(h => {
                        if (!byLevel[h.level]) byLevel[h.level] = [];
                        byLevel[h.level].push(h.text);
                    });

                    const parts = [];
                    Object.keys(byLevel).sort((a, b) => a - b).forEach(level => {
                        const texts = byLevel[level];
                        let levelName;
                        if (level === "1") {
                            levelName = texts.length > 1 ? "main headings" : "main heading";
                        } else {
                            levelName = `level ${level} heading${texts.length > 1 ? "s" : ""}`;
                        }
                        parts.push(`${texts.length} ${levelName}: ${texts.join(", ")}`);
                    });

                    const summary = "This page has " + parts.join(". It also has ") + ".";
                    return summary;
                }

                // Click the element on the page that matches the chosen result
                function clickMatchedElement(matchIndex, matchText) {
                    let index = 0;
                    let clicked = false;

                    document.querySelectorAll("a, button").forEach(el => {
                        if (clicked) return;
                        const text = (
                            el.innerText ||
                            el.getAttribute("aria-label") ||
                            el.title ||
                            ""
                        ).trim();

                        if (index === matchIndex && text === matchText) {
                            // Visual highlight before clicking
                            el.style.outline = "3px solid #1a73e8";
                            el.style.outlineOffset = "2px";
                            setTimeout(() => {
                                el.style.outline = "";
                                el.style.outlineOffset = "";
                                el.click();
                            }, 600);
                            clicked = true;
                        }
                        index++;
                    });

                    return clicked;
                }

                // ============================
                // ZOOM CONTROL
                // ============================
                function setZoom(percent) {
                    document.documentElement.style.zoom = percent + "%";
                    return percent;
                }

                // ============================
                // HIGH CONTRAST TOGGLE
                // ============================
                function toggleHighContrast() {
                    let styleEl = document.getElementById("echonav-contrast-style");
                    if (styleEl) {
                        styleEl.remove();
                        return false;
                    } else {
                        styleEl = document.createElement("style");
                        styleEl.id = "echonav-contrast-style";
                        styleEl.textContent = `
                            html {
                                filter: invert(1) hue-rotate(180deg) !important;
                                background: #fff !important;
                            }
                            img, video, picture, svg, canvas, iframe {
                                filter: invert(1) hue-rotate(180deg) !important;
                            }
                        `;
                        document.head.appendChild(styleEl);
                        return true;
                    }
                }

                // Expose these so the Python-side control panel can call them
                // via page.evaluate(...)
                window.echoNavGetElements = getPageElements;
                window.echoNavSummarize = summarizePage;
                window.echoNavClickElement = clickMatchedElement;
                window.echoNavSetZoom = setZoom;
                window.echoNavToggleHighContrast = toggleHighContrast;
                window.echoNavGetNavOptions = getNavOptions;
                window.echoNavGetHeadings = getHeadings;

                // ============================
                // KEYBOARD SHORTCUTS
                // Ctrl+M / ⌘+M           -> mute/unmute
                // Ctrl+Shift+E / ⌘+Shift+E -> bring control panel to front
                // ============================
                document.addEventListener("keydown", (e) => {
                    const isMac = navigator.platform.toUpperCase().includes("MAC");
                    const key = e.key.toLowerCase();

                    const muteCombo = isMac
                        ? (e.metaKey && !e.shiftKey && key === "m")
                        : (e.ctrlKey && !e.shiftKey && key === "m");

                    const panelCombo = isMac
                        ? (e.metaKey && e.shiftKey && key === "e")
                        : (e.ctrlKey && e.shiftKey && key === "e");

                    if (muteCombo) {
                        window.echoNavToggleAudio();
                        e.preventDefault();
                    } else if (panelCombo) {
                        window.echoNavShowPanel();
                        e.preventDefault();
                    }
                });

                // ============================
                // BUTTONS
                // ============================
                document.querySelectorAll("button").forEach(btnEl => {
                    const label =
                        btnEl.innerText ||
                        btnEl.getAttribute("aria-label") ||
                        "Button";

                    btnEl.addEventListener("mouseenter", () => {
                        window.echoNavSpeak(label);
                        window.echoNavPing("center");
                    });

                    btnEl.addEventListener("focus", () => {
                        window.echoNavSpeak(label);
                        window.echoNavPing("center");
                    });
                });

                // ============================
                // LINKS
                // ============================
                document.querySelectorAll("a").forEach(link => {
                    function labelFor(linkEl) {
                        return (
                            linkEl.innerText ||
                            linkEl.getAttribute("aria-label") ||
                            linkEl.title ||
                            "Link"
                        );
                    }

                    link.addEventListener("mouseenter", () => {
                        window.echoNavSpeak(labelFor(link));
                        window.echoNavPing("right");
                    });

                    link.addEventListener("focus", () => {
                        window.echoNavSpeak(labelFor(link));
                        window.echoNavPing("right");
                    });
                });

                // ============================
                // SELECTED TEXT (GLOBAL)
                // ============================
                document.addEventListener("mouseup", () => {
                    const text = window.getSelection().toString().trim();
                    if (!text) {
                        lastSpokenSelection = "";
                        return;
                    }
                    if (text === lastSpokenSelection) return;
                    lastSpokenSelection = text;
                    window.echoNavSpeak(text);
                });

                // ============================
                // FORM FIELDS
                // ============================
                document.querySelectorAll("input, textarea").forEach(field => {
                    field.addEventListener("focus", () => {
                        let label =
                            field.placeholder ||
                            field.getAttribute("aria-label") ||
                            field.name ||
                            "Form field";
                        window.echoNavSpeak("Focused on " + label);
                        window.echoNavPing("center");
                    });

                    field.addEventListener("blur", () => {
                        if (field.value) {
                            window.echoNavSpeak("You entered: " + field.value);
                            window.echoNavPing("center");
                        }
                    });
                });

                // ============================
                // FORMS
                // ============================
                document.querySelectorAll("form").forEach(form => {
                    form.addEventListener("submit", (e) => {
                        let invalid = false;
                        form.querySelectorAll("input[required], textarea[required]").forEach(field => {
                            if (!field.value.trim()) {
                                invalid = true;
                                window.echoNavSpeak("Error: Required field is empty");
                                window.echoNavPing("left");
                            }
                        });
                        if (invalid) {
                            e.preventDefault();
                        } else {
                            window.echoNavSpeak("Form submitted successfully");
                            window.echoNavPing("center");
                        }
                    });
                });

                // ============================
                // IMAGES + IMAGE FOCUS BOX
                // ============================
                document.querySelectorAll("img").forEach(img => {
                    img.addEventListener("click", (event) => {
                        event.preventDefault();
                        event.stopPropagation();

                        img.style.outline = "3px solid #FFD500";
                        img.style.outlineOffset = "2px";
                        setTimeout(() => {
                            img.style.outline = "";
                            img.style.outlineOffset = "";
                        }, 1500);

                        const rect = img.getBoundingClientRect();
                        const meta = {
                            src: img.src || "",
                            alt: img.alt || "",
                            aria: img.getAttribute("aria-label") || "",
                            title: img.title || "",
                            echonav: img.getAttribute("data-echonav-desc") || "",
                            width: img.naturalWidth || rect.width || 0,
                            height: img.naturalHeight || rect.height || 0
                        };
                        window.echoNavDescribeImage(meta);
                    });
                });
            }

            if (document.readyState === "loading") {
                document.addEventListener("DOMContentLoaded", setupEchoNav);
            } else {
                setupEchoNav();
            }
        }
        """

        async def attach_echonav(page):
            """
            Wire up the EchoNav JS ↔ Python bridge for a given browser page.

            Exposes Python functions (speak, play_ping, etc.) as callable
            globals on the page's window object via page.expose_function().
            Then injects the EchoNav JS script via page.add_init_script() so
            it runs automatically on every navigation within that page.

            Called once for the initial page, and again for every new tab
            via handle_new_page().
            """

            async def describe_image(meta: dict):
                try:
                    from PIL import Image as PILImage

                    detail = (meta.get("echonav") or "").strip()
                    alt = (meta.get("alt") or "").strip()
                    aria = (meta.get("aria") or "").strip()
                    title = (meta.get("title") or "").strip()
                    src = (meta.get("src") or "").strip()
                    width = int(meta.get("width") or 0)
                    height = int(meta.get("height") or 0)

                    if is_url(alt):   alt = ""
                    if is_url(aria):  aria = ""
                    if is_url(title): title = ""

                    if detail:
                        speak(detail)
                        return
                    if alt:
                        speak(f"Image: {alt}")
                        return
                    if aria:
                        speak(f"Image: {aria}")
                        return
                    if title:
                        speak(f"Image: {title}")
                        return

                    img_format = None
                    real_w, real_h = width, height
                    aspect = None

                    if src:
                        try:
                            resp = await context.request.get(src)
                            if resp.ok:
                                body = await resp.body()
                                img = PILImage.open(io.BytesIO(body))
                                real_w, real_h = img.size
                                img_format = img.format
                        except:
                            pass

                    if real_w and real_h:
                        aspect = real_w / real_h

                    size_phrase = "medium-sized "
                    if real_w >= 800 or real_h >= 800:
                        size_phrase = "large "
                    elif real_w <= 128 and real_h <= 128:
                        size_phrase = "small "

                    shape_phrase = ""
                    if aspect:
                        if aspect > 1.3:
                            shape_phrase = "wide "
                        elif aspect < 0.8:
                            shape_phrase = "tall "
                        else:
                            shape_phrase = "square "

                    format_phrase = f"{img_format.upper()} " if img_format else ""
                    desc = f"Image: {size_phrase}{shape_phrase}{format_phrase}image."
                    speak(desc)

                except:
                    speak("Image: descriptive information is not available.")

            await page.expose_function("echoNavSpeak", speak)
            await page.expose_function("echoNavPing", play_ping)
            await page.expose_function("echoNavDescribeImage", describe_image)
            await page.expose_function("echoNavToggleAudio", lambda: toggle_audio())
            await page.expose_function("echoNavShowPanel", lambda: bring_panel_to_front())
            await page.add_init_script(echonav_script)

        page = await context.new_page()
        await attach_echonav(page)

        # Make this page reachable from the Tkinter control panel
        PLAYWRIGHT_LOOP = asyncio.get_event_loop()
        PLAYWRIGHT_PAGE = page

        def handle_new_page(new_page):
            asyncio.create_task(attach_echonav(new_page))
            global PLAYWRIGHT_PAGE
            PLAYWRIGHT_PAGE = new_page

        context.on("page", handle_new_page)

        try:
            await page.goto(url, timeout=60000, wait_until="domcontentloaded")
        except Exception as e:
            print(f"EchoNav: Page load warning — {e}")
            speak("Warning: the page took a long time to load and may not be fully ready.")

        while True:
            if browser.is_connected():
                await asyncio.sleep(1)
            else:
                break


# ============================================
# THREAD WRAPPER
# ============================================
def start_playwright_thread(url):
    """
    Launch the Playwright browser in a background daemon thread.

    Tkinter's mainloop() blocks the main thread, so the browser must run
    on its own thread. asyncio.run() starts a fresh event loop inside that
    thread and runs the entire run_playwright() coroutine inside it.

    The thread is a daemon so it automatically dies when the main program
    exits — no cleanup needed if the user closes the launcher window.
    """
    def runner():
        """Inner wrapper that starts the asyncio event loop for the browser."""
        asyncio.run(run_playwright(url))

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()


# ============================================
# COLOURS & THEME
# ============================================
BG        = "#0d1117"   # near-black background
CARD      = "#161b22"   # slightly lighter card surface
ACCENT    = "#1d8cf8"   # electric blue
ACCENT2   = "#0e6ac7"   # darker blue for hover
TEXT      = "#e6edf3"   # off-white primary text
SUBTEXT   = "#8b949e"   # muted grey secondary text
BORDER    = "#30363d"   # subtle border colour
SUCCESS   = "#3fb950"   # green for the launch button
SUCCESS2  = "#2ea043"   # darker green hover


# ============================================
# HELP TEXT (shown in the control panel's help window)
# ============================================
HELP_TEXT = """What EchoNav does

• Speaks labels for buttons and links when you hover or focus them.
• Reads selected text aloud.
• Describes images when you click them.
• Gives feedback on forms and required fields.

Voice Commands

Click "Voice Command" on the control panel, then say a command like:
  "Navigate to Home"
  "Go to Shop"
  "Open Contact"
  "Click Sign In"

To get a page overview, say:
  "Summarize the page"
  "What's on this page?"

To hear the top navigation bar (including any dropdown sub-options), say:
  "What are the options on the top bar?"
  "What's in the navigation?"

To hear the page's headings, say:
  "What are the headings?"
  "What is the main heading?"

To add the current page to your folder, say:
  "Add this website to my folder"
  "Bookmark this"
  "Save this page"

EchoNav will save the URL and confirm aloud. You can view, open, or
remove bookmarks any time by clicking "My Folder" on the launcher or
control panel.

Display Controls

Use the A− / A+ buttons to zoom the page out or in, or "Reset Zoom"
to return to 100%. Use "High Contrast" to invert the page's colors
for easier reading.

Speech Settings

Choose a playback speed (0.75x - 2x) and a voice/accent. Tap the
voice button to cycle through American, British, Australian,
Indian, Irish, South African, and Canadian accents — EchoNav will
say a sample phrase so you can hear the change.

Keyboard Shortcuts

  Ctrl+M  (Cmd+M on Mac)        — mute / unmute EchoNav
  Ctrl+Shift+E  (Cmd+Shift+E)   — bring this control panel to the front

Images with detailed descriptions

Websites can add a data-echonav-desc attribute to images.
EchoNav will read that description aloud when you click the image.
"""


# ============================================
# SPLASH SCREEN
# ============================================
def show_splash():
    """
    Display a full-screen splash window for 2 seconds while the app loads.

    Creates a separate Tk root (overrideredirect removes the title bar),
    centres it on screen, shows the EchoNav logo and a 'Loading...' label,
    then auto-destroys after 2000ms. Uses splash.mainloop() which blocks
    until the window is destroyed, so start_app() continues only after
    the splash closes.
    """
    splash = tk.Tk()
    splash.overrideredirect(True)

    # Centre on screen
    sw = splash.winfo_screenwidth()
    sh = splash.winfo_screenheight()
    w, h = 520, 260
    x = (sw - w) // 2
    y = (sh - h) // 2
    splash.geometry(f"{w}x{h}+{x}+{y}")
    splash.configure(bg=BG)

    # Outer frame with border effect
    frame = tk.Frame(splash, bg=CARD, bd=0, highlightthickness=1,
                     highlightbackground=ACCENT)
    frame.place(relx=0.5, rely=0.5, anchor="center", width=w - 20, height=h - 20)

    tk.Label(
        frame,
        text="EchoNav",
        font=("Helvetica", 52, "bold"),
        bg=CARD,
        fg=TEXT,
    ).pack(pady=(30, 4))

    tk.Label(
        frame,
        text="Accessible Web Navigator",
        font=("Helvetica", 15),
        bg=CARD,
        fg=ACCENT,
    ).pack()

    tk.Label(
        frame,
        text="Loading...",
        font=("Helvetica", 14),
        bg=CARD,
        fg=SUBTEXT
    ).pack(pady=(18, 0))

    splash.after(2000, splash.destroy)
    splash.mainloop()


# ============================================
# HOVER HELPERS
# ============================================
def on_enter_launch(e, btn):
    """Darken the Launch button when the mouse hovers over it."""
    btn.config(bg=SUCCESS2)

def on_leave_launch(e, btn):
    """Restore the Launch button's normal colour when the mouse leaves."""
    btn.config(bg=SUCCESS)

def on_enter_entry(e, ent):
    """Highlight the URL entry box border in accent blue when it's focused."""
    ent.config(highlightbackground=ACCENT, highlightcolor=ACCENT)

def on_leave_entry(e, ent):
    """Restore the URL entry box border to its normal subtle colour on blur."""
    ent.config(highlightbackground=BORDER, highlightcolor=BORDER)

def on_enter_panel_btn(e, btn, base_color, hover_color):
    """Change a control panel button's background to hover_color on mouse-enter."""
    btn.config(bg=hover_color)

def on_leave_panel_btn(e, btn, base_color, hover_color):
    """Restore a control panel button's background to base_color on mouse-leave."""
    btn.config(bg=base_color)


def make_section_label(parent, text):
    """
    Add a bold section heading label to the control panel.

    Used to visually group the Voice, Display, Speech Settings, and Help
    sections. Larger and brighter than body text so users with low vision
    can quickly identify each section's purpose.
    """
    tk.Label(
        parent,
        text=text,
        font=("Helvetica", 15, "bold"),
        bg=BG,
        fg=TEXT,
        anchor="w",
    ).pack(fill="x", padx=40, pady=(20, 6))


# ============================================
# FEATURE ROW BUILDER
# ============================================
def make_feature_row(parent, icon, text):
    """
    Build one icon + description row in the launcher's Features card.

    Each row has a coloured emoji icon on the left and a plain-text
    description on the right. Called once per feature in the features
    list inside start_app().
    """
    row = tk.Frame(parent, bg=CARD, pady=4)
    row.pack(fill="x", padx=0, pady=2)

    tk.Label(
        row,
        text=icon,
        font=("Helvetica", 17),
        bg=CARD,
        fg=ACCENT,
        width=3,
        anchor="center"
    ).pack(side="left", padx=(8, 4))

    tk.Label(
        row,
        text=text,
        font=("Helvetica", 15),
        bg=CARD,
        fg=TEXT,
        anchor="w",
        justify="left"
    ).pack(side="left", fill="x", expand=True)


# ============================================
# CONTROL PANEL (shown after launch)
# ============================================
def show_bookmarks_window(root):
    """
    Open the bookmark folder window from the launcher screen or control panel.

    Displays all saved bookmarks as clickable rows. Each row shows the URL
    with two buttons: 'Open' (launches EchoNav with that URL, closing the
    launcher) and 'Remove' (deletes it from the list and refreshes the view).

    The window refreshes its list from disk each time it opens, so any
    bookmarks added via voice command ("Add this website to my folder")
    during a session are immediately visible.

    A text entry + 'Add' button at the bottom lets the user manually type
    and save a URL without needing to navigate to it first.
    """
    win = tk.Toplevel(root)
    win.title("EchoNav Folder")
    win.configure(bg=BG)
    win.geometry("560x540")
    win.resizable(False, False)

    # ── Header ───────────────────────────────────────────
    tk.Label(
        win, text="📁  My Folder",
        font=("Helvetica", 22, "bold"),
        bg=BG, fg=TEXT,
    ).pack(pady=(18, 2))

    tk.Label(
        win, text="Click a site to open it in EchoNav.",
        font=("Helvetica", 12),
        bg=BG, fg=SUBTEXT,
    ).pack()

    tk.Frame(win, bg=ACCENT, height=2).pack(fill="x", padx=30, pady=(10, 0))

    # ── Scrollable bookmark list ──────────────────────────
    list_frame = tk.Frame(win, bg=BG)
    list_frame.pack(fill="both", expand=True, padx=20, pady=(10, 0))

    canvas = tk.Canvas(list_frame, bg=BG, highlightthickness=0)
    scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
    inner = tk.Frame(canvas, bg=BG)

    inner.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def refresh_list():
        """Clear and rebuild the bookmark rows from the current saved list."""
        for widget in inner.winfo_children():
            widget.destroy()

        bookmarks = load_bookmarks()

        if not bookmarks:
            tk.Label(
                inner, text="No bookmarks yet.",
                font=("Helvetica", 13), bg=BG, fg=SUBTEXT,
            ).pack(pady=20)
            return

        for i, url in enumerate(bookmarks):
            row = tk.Frame(inner, bg=CARD, pady=6)
            row.pack(fill="x", pady=3)

            # URL label — truncate if very long
            display = url if len(url) <= 48 else url[:45] + "…"
            tk.Label(
                row, text=display,
                font=("Helvetica", 12),
                bg=CARD, fg=TEXT,
                anchor="w",
            ).pack(side="left", padx=(10, 4), fill="x", expand=True)

            # Remove button
            def make_remove(u):
                """Closure so each Remove button captures its own URL."""
                def remove():
                    bm = load_bookmarks()
                    if u in bm:
                        bm.remove(u)
                        save_bookmarks(bm)
                    refresh_list()
                return remove

            tk.Button(
                row, text="✕",
                font=("Helvetica", 11, "bold"),
                bg=CARD, fg="#e53935",
                activebackground=BORDER, activeforeground="#e53935",
                relief="flat", bd=0, cursor="hand2", padx=8,
                command=make_remove(url),
            ).pack(side="right", padx=(0, 4))

            # Open button
            def make_open(u):
                """Closure so each Open button captures its own URL."""
                def open_url():
                    win.destroy()
                    # If the browser is already running, navigate to the URL.
                    # If not (we're still on the launcher), launch it fresh.
                    if PLAYWRIGHT_PAGE is not None and PLAYWRIGHT_LOOP is not None:
                        try:
                            asyncio.run_coroutine_threadsafe(
                                PLAYWRIGHT_PAGE.goto(u, timeout=60000,
                                                     wait_until="domcontentloaded"),
                                PLAYWRIGHT_LOOP
                            )
                        except Exception as e:
                            print("EchoNav bookmark open error:", e)
                    else:
                        # Still on the launcher — start the browser
                        start_playwright_thread(u)
                        show_control_panel(root)
                return open_url

            tk.Button(
                row, text="Open",
                font=("Helvetica", 11, "bold"),
                bg=ACCENT, fg="black",
                activebackground=ACCENT2, activeforeground="black",
                relief="flat", bd=0, cursor="hand2", padx=10,
                command=make_open(url),
            ).pack(side="right", padx=4)

    refresh_list()

    # ── Manual add row ────────────────────────────────────
    tk.Frame(win, bg=BORDER, height=1).pack(fill="x", padx=20, pady=(10, 6))

    add_row = tk.Frame(win, bg=BG)
    add_row.pack(fill="x", padx=20, pady=(0, 16))

    add_entry = tk.Entry(
        add_row,
        font=("Helvetica", 12),
        bg=CARD, fg=TEXT,
        insertbackground=ACCENT,
        relief="flat", bd=0,
        highlightthickness=2,
        highlightbackground=BORDER,
        highlightcolor=ACCENT,
    )
    add_entry.insert(0, "https://")
    add_entry.pack(side="left", fill="x", expand=True, ipady=7, padx=(0, 8))

    def do_add():
        """Read the entry field, validate, save, and refresh the list."""
        url = add_entry.get().strip()
        if not url or url == "https://":
            return
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        success, msg = add_bookmark(url)
        if success:
            add_entry.delete(0, "end")
            add_entry.insert(0, "https://")
        refresh_list()

    add_entry.bind("<Return>", lambda e: do_add())

    tk.Button(
        add_row, text="Add",
        font=("Helvetica", 12, "bold"),
        bg=SUCCESS, fg="black",
        activebackground=SUCCESS2, activeforeground="black",
        relief="flat", bd=0, cursor="hand2", padx=14, pady=7,
        command=do_add,
    ).pack(side="right")



    """
    Open a scrollable help popup window from the control panel.

    Creates a tk.Toplevel (a child window) containing the HELP_TEXT string
    in a read-only Text widget with a scrollbar. A Close button at the bottom
    destroys the window. The window is independent of the main panel so the
    user can keep the panel open while reading the help text.
    """
    win = tk.Toplevel(root)
    win.title("EchoNav Help")
    win.configure(bg=CARD)
    win.geometry("460x560")
    win.resizable(False, False)

    text_frame = tk.Frame(win, bg=CARD)
    text_frame.pack(fill="both", expand=True, padx=16, pady=16)

    scrollbar = tk.Scrollbar(text_frame)
    scrollbar.pack(side="right", fill="y")

    text_widget = tk.Text(
        text_frame,
        wrap="word",
        bg=CARD,
        fg=TEXT,
        font=("Helvetica", 13),
        relief="flat",
        highlightthickness=0,
        yscrollcommand=scrollbar.set,
        padx=4,
        pady=4,
    )
    text_widget.insert("1.0", HELP_TEXT)
    text_widget.config(state="disabled")
    text_widget.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=text_widget.yview)

    close_btn = tk.Button(
        win,
        text="Close",
        font=("Helvetica", 13, "bold"),
        bg=ACCENT,
        fg="black",
        activebackground=ACCENT2,
        activeforeground="black",
        relief="flat",
        bd=0,
        cursor="hand2",
        pady=10,
        command=win.destroy,
    )
    close_btn.pack(fill="x", padx=16, pady=(0, 16))


def show_control_panel(root):
    """
    Replace the launcher window's contents with the EchoNav control panel.

    Called immediately after the user clicks 'Launch EchoNav'. Destroys all
    existing launcher widgets and rebuilds the window as a compact control
    panel that stays visible alongside the browser for the entire session.

    The panel contains:
      - A status label that shows what EchoNav is currently doing
      - A Voice Command button (triggers the mic in a background thread)
      - Display controls: zoom in/out/reset and high-contrast toggle
      - Speech settings: speed selector and voice/accent cycler
      - A Help button that opens show_help_window()
      - Keyboard shortcut reminders at the bottom

    All buttons that run code (mic, zoom, contrast) are connected to
    functions defined inside this function via closures, which lets them
    share the zoom_state and contrast_state dicts without global variables.
    """
    for widget in root.winfo_children():
        widget.destroy()

    root.title("EchoNav Control Panel")
    root.resizable(False, False)

    win_w, win_h = 520, 840
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{win_w}x{win_h}+{(sw - win_w)//2}+{(sh - win_h)//2}")

    # ── HEADER ──────────────────────────────────────────
    header = tk.Frame(root, bg=BG)
    header.pack(fill="x", pady=(28, 0))

    tk.Label(
        header,
        text="EchoNav",
        font=("Helvetica", 36, "bold"),
        bg=BG,
        fg=TEXT,
    ).pack()

    tk.Label(
        header,
        text="Control Panel",
        font=("Helvetica", 15),
        bg=BG,
        fg=ACCENT,
    ).pack(pady=(2, 0))

    tk.Frame(root, bg=ACCENT, height=2).pack(fill="x", padx=40, pady=(14, 0))

    # ── STATUS LABEL ─────────────────────────────────────
    status_frame = tk.Frame(root, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
    status_frame.pack(fill="x", padx=40, pady=(16, 0))

    status_label = tk.Label(
        status_frame,
        text="Ready. Use the controls below, Ctrl+M to mute,\nor Ctrl+Shift+E to bring this panel to the front.",
        font=("Helvetica", 13),
        bg=CARD,
        fg=TEXT,
        wraplength=420,
        justify="center",
        pady=14,
    )
    status_label.pack(fill="x")

    def set_status(msg):
        root.after(0, lambda: status_label.config(text=msg))

    # ============================================
    # VOICE COMMAND SECTION
    # ============================================
    make_section_label(root, "Voice")

    mic_btn = tk.Button(
        root,
        text="🎤  Voice Command",
        font=("Helvetica", 18, "bold"),
        bg=ACCENT,
        fg="black",
        activebackground=ACCENT2,
        activeforeground="black",
        relief="flat",
        bd=0,
        cursor="hand2",
        pady=18,
    )
    mic_btn.pack(fill="x", padx=40)
    mic_btn.bind("<Enter>", lambda e: on_enter_panel_btn(e, mic_btn, ACCENT, ACCENT2))
    mic_btn.bind("<Leave>", lambda e: on_leave_panel_btn(e, mic_btn, ACCENT, ACCENT2))

    def on_mic_click():
        mic_btn.config(state="disabled", bg="#e53935", text="🔴  Listening…")

        def worker():
            try:
                run_voice_command(on_status=set_status)
            finally:
                root.after(0, lambda: mic_btn.config(state="normal", bg=ACCENT, text="🎤  Voice Command"))

        threading.Thread(target=worker, daemon=True).start()

    mic_btn.config(command=on_mic_click)

    tk.Label(
        root,
        text=(
            'Try: "Go to Shop", "Summarize the page",\n'
            '"What\'s on the top bar?", or "What are the headings?"'
        ),
        font=("Helvetica", 11),
        bg=BG,
        fg=SUBTEXT,
        wraplength=420,
        justify="center",
    ).pack(fill="x", padx=40, pady=(8, 0))

    # ============================================
    # DISPLAY SECTION — ZOOM + HIGH CONTRAST
    # ============================================
    make_section_label(root, "Display")

    zoom_state = {"value": 100}

    zoom_row = tk.Frame(root, bg=BG)
    zoom_row.pack(fill="x", padx=40)

    def update_zoom_label():
        zoom_value_label.config(text=f"{zoom_state['value']}%")

    def apply_zoom():
        if PLAYWRIGHT_PAGE is None:
            set_status("The browser isn't open yet.")
            return
        run_on_page(
            "(p) => { document.documentElement.style.zoom = p + '%'; }",
            zoom_state["value"],
        )
        update_zoom_label()
        set_status(f"Zoom set to {zoom_state['value']}%")
        speak(f"Zoom {zoom_state['value']} percent")

    def zoom_out():
        zoom_state["value"] = max(50, zoom_state["value"] - 10)
        apply_zoom()

    def zoom_in():
        zoom_state["value"] = min(200, zoom_state["value"] + 10)
        apply_zoom()

    def zoom_reset():
        zoom_state["value"] = 100
        apply_zoom()
        set_status("Zoom reset to 100%")
        speak("Zoom reset")

    zoom_out_btn = tk.Button(
        zoom_row, text="A−", font=("Helvetica", 18, "bold"),
        bg=CARD, fg="black", activebackground=BORDER, activeforeground=TEXT,
        relief="flat", bd=0, cursor="hand2", pady=14, width=4,
        command=zoom_out,
    )
    zoom_out_btn.pack(side="left", padx=(0, 6))
    zoom_out_btn.bind("<Enter>", lambda e: on_enter_panel_btn(e, zoom_out_btn, CARD, BORDER))
    zoom_out_btn.bind("<Leave>", lambda e: on_leave_panel_btn(e, zoom_out_btn, CARD, BORDER))

    zoom_value_label = tk.Label(
        zoom_row, text="100%", font=("Helvetica", 16, "bold"),
        bg=BG, fg=TEXT, width=8,
    )
    zoom_value_label.pack(side="left", expand=True, fill="x")

    zoom_in_btn = tk.Button(
        zoom_row, text="A+", font=("Helvetica", 18, "bold"),
        bg=CARD, fg="black", activebackground=BORDER, activeforeground=TEXT,
        relief="flat", bd=0, cursor="hand2", pady=14, width=4,
        command=zoom_in,
    )
    zoom_in_btn.pack(side="left", padx=(6, 0))
    zoom_in_btn.bind("<Enter>", lambda e: on_enter_panel_btn(e, zoom_in_btn, CARD, BORDER))
    zoom_in_btn.bind("<Leave>", lambda e: on_leave_panel_btn(e, zoom_in_btn, CARD, BORDER))

    zoom_reset_btn = tk.Button(
        root, text="Reset Zoom", font=("Helvetica", 13, "bold"),
        bg=CARD, fg="black", activebackground=BORDER, activeforeground=TEXT,
        relief="flat", bd=0, cursor="hand2", pady=10,
        highlightthickness=1, highlightbackground=BORDER,
        command=zoom_reset,
    )
    zoom_reset_btn.pack(fill="x", padx=40, pady=(8, 0))
    zoom_reset_btn.bind("<Enter>", lambda e: on_enter_panel_btn(e, zoom_reset_btn, CARD, BORDER))
    zoom_reset_btn.bind("<Leave>", lambda e: on_leave_panel_btn(e, zoom_reset_btn, CARD, BORDER))

    # High contrast toggle
    contrast_state = {"on": False}

    def toggle_contrast():
        if PLAYWRIGHT_PAGE is None:
            set_status("The browser isn't open yet.")
            return
        result = run_on_page(
            "window.echoNavToggleHighContrast ? window.echoNavToggleHighContrast() : false"
        )
        contrast_state["on"] = bool(result)
        if contrast_state["on"]:
            contrast_btn.config(text="🌓  High Contrast: ON", bg=SUCCESS, fg="black")
            set_status("High contrast mode enabled")
            speak("High contrast mode on")
        else:
            contrast_btn.config(text="🌓  High Contrast: OFF", bg=CARD, fg=TEXT)
            set_status("High contrast mode disabled")
            speak("High contrast mode off")

    contrast_btn = tk.Button(
        root, text="🌓  High Contrast: OFF", font=("Helvetica", 14, "bold"),
        bg=CARD, fg="black", activebackground=BORDER, activeforeground=TEXT,
        relief="flat", bd=0, cursor="hand2", pady=14,
        highlightthickness=1, highlightbackground=BORDER,
        command=toggle_contrast,
    )
    contrast_btn.pack(fill="x", padx=40, pady=(10, 0))

    # ============================================
    # SPEECH SETTINGS SECTION — SPEED + VOICE
    # ============================================
    make_section_label(root, "Speech Settings")

    tk.Label(
        root, text="Speed", font=("Helvetica", 12, "bold"),
        bg=BG, fg=SUBTEXT, anchor="w",
    ).pack(fill="x", padx=40)

    speed_row = tk.Frame(root, bg=BG)
    speed_row.pack(fill="x", padx=40, pady=(4, 0))

    SPEED_OPTIONS = [0.75, 1.0, 1.25, 1.5, 2.0]
    speed_buttons = {}

    def set_speed(value):
        global SPEECH_RATE
        SPEECH_RATE = value
        for v, b in speed_buttons.items():
            if v == value:
                b.config(bg=ACCENT, fg="black")
            else:
                b.config(bg=CARD, fg=TEXT)
        set_status(f"Speech speed set to {value:g}x")
        speak("Speed set")

    for v in SPEED_OPTIONS:
        is_default = (v == SPEECH_RATE)
        b = tk.Button(
            speed_row, text=f"{v:g}x", font=("Helvetica", 13, "bold"),
            bg=ACCENT if is_default else CARD,
            fg="black" if is_default else TEXT,
            activebackground=ACCENT2, activeforeground="black",
            relief="flat", bd=0, cursor="hand2", pady=12,
            command=lambda v=v: set_speed(v),
        )
        b.pack(side="left", expand=True, fill="x", padx=3)
        speed_buttons[v] = b

    tk.Label(
        root, text="Voice", font=("Helvetica", 12, "bold"),
        bg=BG, fg=SUBTEXT, anchor="w",
    ).pack(fill="x", padx=40, pady=(14, 0))

    voice_state = {"index": 0}

    def cycle_voice():
        global VOICE_TLD
        voice_state["index"] = (voice_state["index"] + 1) % len(VOICE_OPTIONS)
        tld, name = VOICE_OPTIONS[voice_state["index"]]
        VOICE_TLD = tld
        voice_btn.config(text=f"🗣️  Voice: {name}  (tap to change)")
        set_status(f"Voice changed to {name}")
        speak(f"This is the {name} voice")

    current_tld, current_name = VOICE_OPTIONS[voice_state["index"]]
    voice_btn = tk.Button(
        root, text=f"🗣️  Voice: {current_name}  (tap to change)",
        font=("Helvetica", 14, "bold"),
        bg=CARD, fg="black", activebackground=BORDER, activeforeground=TEXT,
        relief="flat", bd=0, cursor="hand2", pady=14,
        highlightthickness=1, highlightbackground=BORDER,
        command=cycle_voice,
    )
    voice_btn.pack(fill="x", padx=40, pady=(4, 0))

    # ============================================
    # HELP + SHORTCUTS
    # ============================================
    make_section_label(root, "Help & Folder")

    # Folder / bookmarks button — lets user browse and open saved sites
    folder_btn_panel = tk.Button(
        root,
        text="📁  My Folder",
        font=("Helvetica", 15, "bold"),
        bg=CARD,
        fg="black",
        activebackground=BORDER,
        activeforeground="black",
        relief="flat",
        bd=0,
        cursor="hand2",
        pady=14,
        highlightthickness=1,
        highlightbackground=BORDER,
        command=lambda: show_bookmarks_window(root),
    )
    folder_btn_panel.pack(fill="x", padx=40, pady=(0, 8))
    folder_btn_panel.bind("<Enter>", lambda e: on_enter_panel_btn(e, folder_btn_panel, CARD, BORDER))
    folder_btn_panel.bind("<Leave>", lambda e: on_leave_panel_btn(e, folder_btn_panel, CARD, BORDER))

    help_btn = tk.Button(
        root,
        text="?  Help",
        font=("Helvetica", 15, "bold"),
        bg=CARD,
        fg="black",
        activebackground=BORDER,
        activeforeground=TEXT,
        relief="flat",
        bd=0,
        cursor="hand2",
        pady=14,
        highlightthickness=1,
        highlightbackground=BORDER,
        # command=lambda: show_help_window(root),
    )
    help_btn.pack(fill="x", padx=40)
    help_btn.bind("<Enter>", lambda e: on_enter_panel_btn(e, help_btn, CARD, BORDER))
    help_btn.bind("<Leave>", lambda e: on_leave_panel_btn(e, help_btn, CARD, BORDER))

    tk.Label(
        root,
        text=(
            "Ctrl+M (⌘+M on Mac) — mute / unmute EchoNav\n"
            "Ctrl+Shift+E (⌘+Shift+E on Mac) — bring this panel to the front\n"
            'Voice: "Add this website to my folder" — bookmark current page'
        ),
        font=("Helvetica", 12),
        bg=BG,
        fg=SUBTEXT,
        justify="center",
    ).pack(pady=(14, 0))

    # ── FOOTER ───────────────────────────────────────────
    tk.Label(
        root,
        text="EchoNav  •  Built for accessibility",
        font=("Helvetica", 11),
        bg=BG,
        fg=SUBTEXT,
    ).pack(pady=(16, 14))


# ============================================
# MAIN LAUNCHER WINDOW
# ============================================
def start_app():
    """
    Entry point for the EchoNav application.

    Shows the splash screen, then builds and runs the main launcher window.
    The launcher lets the user type a URL and click 'Launch EchoNav'.

    When launched:
      1. start_playwright_thread(url) opens the browser in a background thread
      2. show_control_panel(root) replaces the launcher UI with the control panel
      3. root.mainloop() keeps Tkinter running until the window is closed

    Uses a global 'root' and 'entry' so other parts of the app (like
    bring_panel_to_front) can reference the main window from anywhere.
    """
    def launch():
        """
        Read the URL from the entry field, open the browser, and switch
        the window to the control panel. Prepends https:// if the user
        didn't type a protocol prefix.
        """
        url = entry.get().strip()
        if not url:
            return
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        start_playwright_thread(url)
        show_control_panel(root)

    def on_return(event):
        launch()

    show_splash()

    global root, entry
    root = tk.Tk()
    root.title("EchoNav")
    root.configure(bg=BG)
    root.resizable(False, False)

    # Centre window
    win_w, win_h = 780, 720
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{win_w}x{win_h}+{(sw - win_w)//2}+{(sh - win_h)//2}")

    # ── FOLDER BUTTON — top-right corner ────────────────
    # Placed using place() so it sits in the corner without affecting
    # the rest of the layout. Must be created before pack() calls so
    # it layers correctly.
    folder_corner_btn = tk.Button(
        root,
        text="📁",
        font=("Helvetica", 20),
        bg=BG,
        fg=TEXT,
        activebackground=CARD,
        activeforeground=TEXT,
        relief="flat",
        bd=0,
        cursor="hand2",
        command=lambda: show_bookmarks_window(root),
    )
    folder_corner_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-14, y=14)

    # ── HEADER ──────────────────────────────────────────
    header = tk.Frame(root, bg=BG)
    header.pack(fill="x", pady=(36, 0))

    tk.Label(
        header,
        text="EchoNav",
        font=("Helvetica", 46, "bold"),
        bg=BG,
        fg=TEXT,
    ).pack()

    tk.Label(
        header,
        text="Accessible Web Navigator",
        font=("Helvetica", 16),
        bg=BG,
        fg=ACCENT,
    ).pack(pady=(2, 0))

    # thin accent divider
    tk.Frame(root, bg=ACCENT, height=2).pack(fill="x", padx=60, pady=(18, 20))

    # ── FEATURES CARD ───────────────────────────────────
    card_outer = tk.Frame(root, bg=BORDER, bd=0)
    card_outer.pack(padx=50, fill="x")

    card = tk.Frame(card_outer, bg=CARD, bd=0)
    card.pack(padx=1, pady=1, fill="x")

    tk.Label(
        card,
        text="Features",
        font=("Helvetica", 13, "bold"),
        bg=CARD,
        fg=SUBTEXT,
        anchor="w"
    ).pack(fill="x", padx=14, pady=(10, 4))

    features = [
        ("🔊", "Spoken labels for buttons and links on hover"),
        ("📖", "Reads selected text aloud"),
        ("🖼️", "Click images for detailed descriptions"),
        ("📝", "Form feedback and required-field error detection"),
        ("🎤", "Voice commands - navigate, summarize, list the top bar, or list headings"),
        ("📁", "Bookmarks folder — open saved sites or say 'Add this website to my folder'"),
        ("🔍", "Zoom and high-contrast display controls"),
        ("🗣️", "Adjustable speech speed and voice/accent"),
        ("⌨️", "Ctrl+M to mute, Ctrl+Shift+E to open the control panel"),
    ]

    for icon, text in features:
        make_feature_row(card, icon, text)

    tk.Frame(card, bg=BG, height=10).pack()  # bottom padding

    # ── URL INPUT SECTION ────────────────────────────────
    input_section = tk.Frame(root, bg=BG)
    input_section.pack(pady=(28, 0), padx=50, fill="x")

    tk.Label(
        input_section,
        text="Enter a website URL",
        font=("Helvetica", 14, "bold"),
        bg=BG,
        fg=SUBTEXT,
        anchor="w"
    ).pack(fill="x", pady=(0, 8))

    # Entry with dark styling + blue focus ring
    entry = tk.Entry(
        input_section,
        font=("Helvetica", 20),
        bg=CARD,
        fg=TEXT,
        insertbackground=ACCENT,       # cursor colour
        relief="flat",
        bd=0,
        highlightthickness=2,
        highlightbackground=BORDER,
        highlightcolor=ACCENT,
    )
    entry.pack(fill="x", ipady=10, padx=0)
    entry.bind("<Return>", on_return)
    entry.bind("<FocusIn>",  lambda e: on_enter_entry(e, entry))
    entry.bind("<FocusOut>", lambda e: on_leave_entry(e, entry))
    entry.insert(0, "https://")

    # Clear placeholder on first click
    def clear_placeholder(e):
        if entry.get() == "https://":
            entry.delete(0, "end")
    entry.bind("<Button-1>", clear_placeholder)

    # ── LAUNCH BUTTON ────────────────────────────────────
    launch_btn = tk.Button(
        root,
        text="Launch EchoNav  →",
        font=("Helvetica", 18, "bold"),
        bg=SUCCESS,
        fg="black",
        activebackground=SUCCESS2,
        activeforeground="white",
        relief="flat",
        bd=0,
        cursor="hand2",
        pady=14,
        command=launch,
    )
    launch_btn.pack(fill="x", padx=50, pady=(22, 0))
    launch_btn.bind("<Enter>", lambda e: on_enter_launch(e, launch_btn))
    launch_btn.bind("<Leave>", lambda e: on_leave_launch(e, launch_btn))

    # ── FOOTER ───────────────────────────────────────────
    tk.Label(
        root,
        text="EchoNav  •  Built for accessibility",
        font=("Helvetica", 12),
        bg=BG,
        fg=SUBTEXT,
    ).pack(pady=(16, 0))

    entry.focus_set()
    root.mainloop()


# ============================================
# RUN
# ============================================
start_app()