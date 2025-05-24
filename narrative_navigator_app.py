import sys
import json
import numpy as np
import sounddevice as sd
import whisper
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QPushButton, QTextEdit, QTableWidget,
                           QTableWidgetItem, QInputDialog, QSplitter, QLineEdit, QLabel,
                           QTabWidget, QScrollArea, QFrame, QGridLayout)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import ollama
import queue
import threading
import time
import uuid
from duckduckgo_search import DDGS

# --- Helper Thread Classes (Copied from original app.py, with minor adjustments if necessary) ---

class AudioCaptureThread(QThread):
    """
    Captures audio from the specified input device and emits it as numpy arrays.
    Handles device selection and error reporting.
    """
    audio_data = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)

    def __init__(self, device_id=3, samplerate=16000):
        super().__init__()
        self.device_id = device_id
        self.samplerate = samplerate
        self._running = False
        self._stop_event = threading.Event()
        self.audio_queue = queue.Queue()

    def run(self):
        self._running = True
        self._stop_event.clear()

        def callback(indata, frames, time_info, status):
            """Callback function for sounddevice.InputStream."""
            if status:
                pass  # You might want to log status messages
            if self._running:
                self.audio_queue.put(indata.copy())

        try:
            devices = sd.query_devices()
            if self.device_id >= len(devices):
                raise ValueError(f"Device ID {self.device_id} is out of range. Available devices: {len(devices)}")
            device_info = sd.query_devices(self.device_id)
            if device_info['max_input_channels'] < 1:
                raise ValueError(f"Device {self.device_id} does not support audio input")

            with sd.InputStream(samplerate=self.samplerate,
                              device=self.device_id,
                              channels=1,
                              callback=callback):
                while self._running:
                    try:
                        # Get audio chunk from queue with a timeout to allow stopping
                        audio_chunk = self.audio_queue.get(timeout=0.1)
                        self.audio_data.emit(audio_chunk)
                    except queue.Empty:
                        # If queue is empty, check stop event
                        if self._stop_event.is_set():
                            break
        except Exception as e:
            error_msg = f"Audio Capture Critical Error: {str(e)}"
            if "device" in str(e).lower():
                error_msg += f"\nAvailable audio devices:\n"
                for i, dev in enumerate(sd.query_devices()):
                    if dev['max_input_channels'] > 0:
                        error_msg += f"ID {i}: {dev['name']}\n"
            self.error_signal.emit(error_msg)
            print(error_msg, file=sys.stderr)
        finally:
            self._running = False
            # Clear any remaining audio in the queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

    def stop(self):
        """Stops the audio capture thread."""
        self._running = False
        self._stop_event.set()

class TranscriptionThread(QThread):
    """
    Transcribes audio data using the Whisper model.
    Processes audio in chunks and emits transcriptions.
    """
    transcription = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        # Load the Whisper model (can be "tiny", "base", "small", "medium", "large")
        self.model = whisper.load_model("small.en")
        self.audio_buffer = np.array([], dtype=np.float32)
        self.running = False
        self.buffer_lock = threading.Lock()

    def run(self):
        self.running = True
        while self.running:
            audio_to_process = None
            # Safely access and clear a portion of the audio buffer
            with self.buffer_lock:
                # Process audio in 10-second chunks (16000 samples/sec * 10 sec)
                if len(self.audio_buffer) >= 16000 * 10:
                    audio_to_process = self.audio_buffer[:16000 * 10].copy()
                    self.audio_buffer = self.audio_buffer[16000 * 10:]

            if audio_to_process is not None:
                try:
                    # Perform transcription
                    result = self.model.transcribe(audio_to_process, language="en")
                    if result["text"].strip():
                        self.transcription.emit(result["text"])
                except Exception as e:
                    self.error_signal.emit(f"Transcription Error: {e}")
                    print(f"Transcription Error: {e}", file=sys.stderr)
            time.sleep(0.1) # Small delay to prevent busy-waiting

    def add_audio(self, audio_data):
        """Adds new audio data to the buffer for transcription."""
        with self.buffer_lock:
            self.audio_buffer = np.append(self.audio_buffer, audio_data.flatten())

    def stop(self):
        """Stops the transcription thread."""
        self.running = False

class WebSearchThread(QThread):
    """
    Performs a web search for external context using DuckDuckGo Search.
    Emits the search results or an error message.
    """
    context_ready = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, title):
        super().__init__()
        self.title = title

    def run(self):
        try:
            # Formulate a search query to get plot summary or overview
            search_query = f"{self.title} plot summary OR overview"
            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, max_results=5))

            context_lines = []
            for r in results:
                if r.get('body'):
                    context_lines.append(f"- {r['title']}: {r['body']}")
                elif r.get('link'):
                    context_lines.append(f"- {r['title']} ({r['link']})")

            full_context = "\n".join(context_lines)

            if not full_context:
                self.error_signal.emit(f"Warning: No significant web context found for '{self.title}'. LLM may operate with limited external knowledge.")
                full_context = f"No specific plot context found for the content titled '{self.title}'."

            self.context_ready.emit(full_context)
        except Exception as e:
            self.error_signal.emit(f"Error during web search for '{self.title}': {str(e)}. LLM will operate without external context.")
            self.context_ready.emit(f"Web search failed. No external context provided for '{self.title}'.")

class LLMThread(QThread):
    """
    Communicates with a local LLM (Ollama) to extract narrative entities
    from transcriptions and maintain a "cheat sheet".
    """
    entities_updated = pyqtSignal(list)
    llm_log = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.transcriptions = []
        self.entities = []
        self.running = False
        self.external_context = ""
        self.content_title = "Unknown Content"

        # Base system prompt for entity extraction
        self.base_system_prompt = """
You are an AI designed to extract *important* narrative entities from transcribed audio for a "cheat sheet" to help users understand a story.

The content you are analyzing is titled: "{content_title}".
Here is some external context about the content to help you identify relevant entities and their significance:
---
{external_context}
---

Analyze the provided text and identify named entities in the categories: Characters, Locations, Organizations, Key Objects, Concepts/Events.
Focus *only* on entities that are *significant* to the overall plot, character development, world-building, or major events. Avoid trivial mentions unless pivotal.

Return results in valid JSON format:
{{
  "entities": [
    {{"name": str, "type": str, "description": str}}
  ]
}}
- Ensure "name" is non-empty for valid entities.
- Only include entities explicitly mentioned in the transcript or context.
- Use exact category names: Characters, Locations, Organizations, Key Objects, Concepts/Events.
- If no *significant* entities are found, return an empty entities list.
- Maintain context from conversation history.
- Provide brief descriptions (max 10 words), focusing on their narrative role.
- If an entity already exists in the current cheat sheet (based on name and type), update its description instead of adding a new entry.
- Use the most complete, formal, or specific name (e.g., "Gordon Ramsay's Restaurant (Downing Street)" instead of "Restaurant").
"""
        # Initialize system prompt with placeholders
        self.system_prompt = self.base_system_prompt.format(
            content_title=self.content_title,
            external_context="No external context loaded yet. Please wait for the application to gather information."
        )

    def set_content_title(self, title):
        """Sets the content title for the LLM's system prompt."""
        self.content_title = title
        self._update_system_prompt()

    def set_external_context(self, context):
        """Sets the external context for the LLM's system prompt."""
        self.external_context = context
        self._update_system_prompt()

    def _update_system_prompt(self):
        """Updates the system prompt with current content title and external context."""
        self.system_prompt = self.base_system_prompt.format(
            content_title=self.content_title,
            external_context=self.external_context
        )

    def _normalize_entity_name(self, name):
        """Normalizes entity names for comparison to avoid duplicates."""
        normalized_name = name.lower()
        normalized_name = re.sub(r'\s*\([^)]*\)', '', normalized_name).strip() # Remove text in parentheses
        for article in ['the ', 'a ', 'an ']: # Remove common articles
            if normalized_name.startswith(article):
                normalized_name = normalized_name[len(article):].strip()
        normalized_name = re.sub(r"'s\b", '', normalized_name).strip() # Remove possessive 's
        normalized_name = re.sub(r"s'\b", 's', normalized_name).strip() # Handle plural possessives
        normalized_name = re.sub(r'[^a-z0-9\s]', '', normalized_name).strip() # Remove non-alphanumeric except space
        normalized_name = re.sub(r'\s+', ' ', normalized_name).strip() # Collapse multiple spaces
        return normalized_name

    def _is_similar_entity(self, entity1, entity2):
        """Checks if two entities are similar based on type and normalized name."""
        if entity1["type"] != entity2["type"]:
            return False
        name1_norm = self._normalize_entity_name(entity1["name"])
        name2_norm = self._normalize_entity_name(entity2["name"])
        return name1_norm == name2_norm

    def run(self):
        self.running = True
        # Wait for external context to be loaded before starting LLM processing
        if not self.external_context:
            self.llm_log.emit("LLM Thread waiting for external context...")
            while self.running and not self.external_context:
                time.sleep(1)
            if not self.running:
                return

        self.llm_log.emit("LLM Thread started with external context.")
        while self.running:
            if self.transcriptions:
                # Process the most recent transcription and some past context
                transcript_to_process = self.transcriptions[-1]
                recent_transcripts = self.transcriptions[-5:] if len(self.transcriptions) > 5 else self.transcriptions
                current_entities_json = json.dumps(self.entities, indent=2) # Current state of cheat sheet

                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content":
                     f"Current transcript snippet: {transcript_to_process}\n"
                     f"Recent context (last {len(recent_transcripts)} snippets): {recent_transcripts}\n"
                     f"Current narrative cheat sheet: {current_entities_json}\n"
                     f"Extract significant entities from the current transcript, updating the cheat sheet if entities already exist."}
                ]
                self.llm_log.emit(f"Prompt sent:\n{json.dumps(messages, indent=2)}")
                try:
                    # Call Ollama for chat completion
                    response = ollama.chat(model="phi4-mini:latest", messages=messages, stream=False)
                    content = response['message']['content']
                    self.llm_log.emit(f"LLM raw response:\n{content}")

                    # Extract JSON from the response (Ollama might wrap it in markdown)
                    json_start = content.find("```json")
                    json_end = content.rfind("```")
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        json_string = content[json_start + len("```json"):json_end].strip()
                    else:
                        json_string = content.strip()

                    try:
                        parsed_response = json.loads(json_string)
                        entities = parsed_response.get('entities', [])
                        if not isinstance(entities, list):
                            raise ValueError("Expected 'entities' to be a list.")

                        # Filter for valid entities and update/add to the main entities list
                        valid_entities = [e for e in entities if e.get("name") and e.get("type") in ["Characters", "Locations", "Organizations", "Key Objects", "Concepts/Events"]]
                        for new_entity in valid_entities:
                            found = False
                            for i, existing in enumerate(self.entities):
                                if self._is_similar_entity(new_entity, existing):
                                    # Update description if entity is similar
                                    self.entities[i]["description"] = new_entity.get("description", existing["description"])
                                    # Logic to update name if new name is more complete/formal
                                    new_name_is_longer = len(new_entity["name"]) > len(existing["name"])
                                    existing_name_normalized_in_new = self._normalize_entity_name(existing["name"]) in self._normalize_entity_name(new_entity["name"])
                                    new_name_starts_capital = new_entity["name"] and new_entity["name"][0].isupper()
                                    existing_name_starts_lower = existing["name"] and not existing["name"][0].isupper()
                                    if (new_name_is_longer and existing_name_normalized_in_new) or \
                                       (self._normalize_entity_name(new_entity["name"]) == self._normalize_entity_name(existing["name"]) and
                                        new_name_starts_capital and existing_name_starts_lower):
                                        self.entities[i]["name"] = new_entity["name"]
                                    found = True
                                    break
                            if not found:
                                self.entities.append(new_entity)
                        self.entities_updated.emit(self.entities) # Emit updated list to UI
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        self.llm_log.emit(f"Error: Invalid JSON response or parsing error - {str(e)}\nAttempted to parse:\n{json_string}")
                except Exception as e:
                    self.llm_log.emit(f"Error: LLM request failed - {str(e)}")
            time.sleep(2) # Wait before processing next transcription

    def add_transcription(self, text):
        """Adds a new transcription snippet to the LLM's buffer."""
        self.transcriptions.append(text)

    def get_transcriptions(self):
        """Returns the current list of transcriptions."""
        return self.transcriptions

    def get_entities(self):
        """Returns the current list of extracted entities."""
        return self.entities

    def stop(self):
        """Stops the LLM processing thread."""
        self.running = False

class ChatThread(QThread):
    """
    Handles chat interactions with the LLM, answering user questions
    based on transcription history and extracted entities.
    """
    chat_response = pyqtSignal(str)
    chat_log = pyqtSignal(str)

    def __init__(self, transcript_getter, entities_getter, content_title, external_context):
        super().__init__()
        self.chat_queue = queue.Queue()
        self.running = False
        self.transcript_getter = transcript_getter
        self.entities_getter = entities_getter
        self.content_title = content_title
        self.external_context = external_context

        # System prompt for the chat assistant
        self.system_prompt = """
You are an AI assistant helping a user understand a story by answering questions based on the transcript history and a narrative cheat sheet.

The content is titled: "{content_title}".
External context about the content:
---
{external_context}
---

Answer user questions using the provided transcript history and cheat sheet. Provide concise, relevant answers in plain text.
""".format(content_title=self.content_title, external_context=self.external_context)

    def add_chat_query(self, query):
        """Adds a user query to the chat queue."""
        self.chat_queue.put(query)

    def run(self):
        self.running = True
        while self.running:
            try:
                # Get query from queue without blocking
                query = self.chat_queue.get_nowait()
                transcripts = self.transcript_getter()
                recent_transcripts = transcripts[-5:] if len(transcripts) > 5 else transcripts
                current_entities = self.entities_getter()
                current_entities_json = json.dumps(current_entities, indent=2)

                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content":
                     f"Transcript history: {recent_transcripts}\n"
                     f"Current narrative cheat sheet: {current_entities_json}\n"
                     f"User question: {query}"}
                ]
                self.chat_log.emit(f"Chat prompt sent:\n{json.dumps(messages, indent=2)}")
                try:
                    # Call Ollama for chat completion
                    response = ollama.chat(model="phi4-mini:latest", messages=messages, stream=False)
                    content = response['message']['content']
                    self.chat_response.emit(content) # Emit AI's response
                    self.chat_log.emit(f"Chat response:\n{content}")
                except Exception as e:
                    self.chat_response.emit(f"Error: Unable to process query - {str(e)}")
                    self.chat_log.emit(f"Chat Error: {str(e)}")
            except queue.Empty:
                pass # No query in queue, continue loop
            time.sleep(0.5) # Small delay

    def stop(self):
        """Stops the chat thread."""
        self.running = False

# --- Main Application Window (Redesigned) ---

class NarrativeNavigator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Narrative Navigator AI")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(800, 600)

        # Initialize worker threads
        self.audio_thread = AudioCaptureThread()
        self.transcription_thread = TranscriptionThread()
        self.llm_thread = LLMThread()
        self.chat_thread = None # Initialized later once context is ready
        self.web_search_thread = None
        self.content_title = ""
        self.processing_active = False # Tracks if audio/LLM processing is active

        self.init_ui()

        # Connect signals from threads to UI update slots
        self.audio_thread.audio_data.connect(self.transcription_thread.add_audio)
        self.audio_thread.error_signal.connect(self.update_log)
        self.transcription_thread.transcription.connect(self.handle_transcription)
        self.transcription_thread.error_signal.connect(self.update_log)
        self.llm_thread.entities_updated.connect(self.update_story_elements_tab)
        self.llm_thread.llm_log.connect(self.update_log)

        # Prompt user for content title at startup
        self.get_content_title_and_context()

    def init_ui(self):
        """Initializes the main user interface components and layout."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # --- Header Section ---
        header_layout = QHBoxLayout()
        app_title = QLabel("Narrative Navigator AI")
        app_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #333;")
        header_layout.addWidget(app_title)

        header_layout.addStretch() # Pushes elements to the right

        # Status indicator (e.g., "Idle", "Recording...")
        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("font-size: 14px; color: gray; margin-right: 10px;")
        header_layout.addWidget(self.status_label)

        # Start/Stop Recording Button
        self.toggle_button = QPushButton("Start Recording")
        self.toggle_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold;")
        self.toggle_button.clicked.connect(self.toggle_processing)
        self.toggle_button.setEnabled(False) # Disabled until external context is loaded
        header_layout.addWidget(self.toggle_button)

        main_layout.addLayout(header_layout)

        # --- Tab Widget for Main Content ---
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane { /* The tab widget frame */
                border: 1px solid #ccc;
                border-radius: 8px;
                padding: 10px;
                background-color: #f9f9f9;
            }
            QTabBar::tab {
                background: #e0e0e0;
                border: 1px solid #c0c0c0;
                border-bottom-color: #c0c0c0; /* same as pane color */
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                min-width: 8ex;
                padding: 8px 15px;
                margin-right: 2px;
                color: #555;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #f9f9f9;
                border-color: #ccc;
                border-bottom-color: #f9f9f9; /* make selected tab appear connected to the pane */
                color: #333;
            }
            QTabBar::tab:hover {
                background: #d0d0d0;
            }
        """)
        main_layout.addWidget(self.tab_widget)

        # Create individual tabs
        self.create_overview_tab()
        self.create_story_elements_tab()
        self.create_live_transcript_tab()
        self.create_ai_chat_tab()

        # --- Global Log Area (for debugging/verbose output) ---
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setPlaceholderText("Application logs and debug messages will appear here...")
        self.log_display.setMaximumHeight(100) # Keep it compact
        self.log_display.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ddd; border-radius: 5px; padding: 5px;")
        main_layout.addWidget(self.log_display)


    def create_overview_tab(self):
        """Creates the 'Overview' tab content, mimicking Screenshot_14.png."""
        self.overview_tab = QWidget()
        overview_layout = QGridLayout(self.overview_tab)

        # Current Content Title Display
        self.current_content_title_label = QLabel("Analyzing: N/A")
        self.current_content_title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #444;")
        overview_layout.addWidget(self.current_content_title_label, 0, 0, 1, 2) # Span two columns

        # Card-like widgets for counts (similar to screenshot 1)
        # Using a helper function to create these styled cards
        self.char_count_label = self._create_info_card("Characters", "0", "#E0F7FA", "👤") # Light Cyan
        overview_layout.addWidget(self.char_count_label, 1, 0)

        self.loc_count_label = self._create_info_card("Locations", "0", "#E8F5E9", "📍") # Light Green
        overview_layout.addWidget(self.loc_count_label, 1, 1)

        self.transcript_line_count_label = self._create_info_card("Transcript Lines", "0", "#FFFDE7", "📝") # Light Yellow
        overview_layout.addWidget(self.transcript_line_count_label, 2, 0)

        self.total_elements_count_label = self._create_info_card("Total Elements", "0", "#FCE4EC", "✨") # Light Pink
        overview_layout.addWidget(self.total_elements_count_label, 2, 1)

        # Recent Activity section
        recent_activity_frame = QFrame()
        recent_activity_frame.setFrameShape(QFrame.StyledPanel)
        recent_activity_frame.setStyleSheet("background-color: white; border: 1px solid #eee; border-radius: 8px; padding: 10px;")
        recent_activity_layout = QVBoxLayout(recent_activity_frame)
        recent_activity_layout.addWidget(QLabel("<b>Recent Activity</b>"))
        self.recent_activity_text = QTextEdit()
        self.recent_activity_text.setReadOnly(True)
        self.recent_activity_text.setPlaceholderText("Latest story elements and transcript updates...")
        self.recent_activity_text.setStyleSheet("border: none; background-color: transparent;")
        recent_activity_layout.addWidget(self.recent_activity_text)
        overview_layout.addWidget(recent_activity_frame, 3, 0, 1, 2) # Span two columns

        # Key Characters section
        key_characters_frame = QFrame()
        key_characters_frame.setFrameShape(QFrame.StyledPanel)
        key_characters_frame.setStyleSheet("background-color: white; border: 1px solid #eee; border-radius: 8px; padding: 10px;")
        key_characters_layout = QVBoxLayout(key_characters_frame)
        key_characters_layout.addWidget(QLabel("<b>Key Characters</b>"))
        self.key_characters_list = QVBoxLayout() # Use a VBox to add character widgets dynamically
        key_characters_layout.addLayout(self.key_characters_list)
        key_characters_layout.addStretch() # Push content to top
        overview_layout.addWidget(key_characters_frame, 0, 2, 4, 1) # Occupy right side, span all rows

        # Set column stretch factors for responsive layout
        overview_layout.setColumnStretch(0, 1)
        overview_layout.setColumnStretch(1, 1)
        overview_layout.setColumnStretch(2, 2) # Make key characters section wider

        self.tab_widget.addTab(self.overview_tab, "Overview")

    def _create_info_card(self, title, count, color, icon):
        """Helper function to create a styled info card for the overview tab."""
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(f"background-color: {color}; border-radius: 8px; padding: 10px; margin: 5px; min-height: 100px;")
        layout = QVBoxLayout(frame)
        layout.setAlignment(Qt.AlignCenter) # Center content vertically and horizontally

        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 36px; margin-bottom: 5px;")
        layout.addWidget(icon_label)

        count_label = QLabel(count)
        count_label.setObjectName("count_label") # Add object name to easily find it later
        count_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #333;")
        layout.addWidget(count_label)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 14px; color: #555;")
        layout.addWidget(title_label)
        return frame


    def create_story_elements_tab(self):
        """Creates the 'Story Elements' tab content, mimicking Screenshot_15.png."""
        self.story_elements_tab = QWidget()
        elements_layout = QVBoxLayout(self.story_elements_tab)

        elements_layout.addWidget(QLabel("<b>Story Elements</b>"))
        elements_layout.addWidget(QLabel("AI extracted narrative elements organized by category"))

        self.cheat_sheet = QTableWidget()
        self.cheat_sheet.setColumnCount(3)
        self.cheat_sheet.setHorizontalHeaderLabels(["Name", "Type", "Description"])
        self.cheat_sheet.setEditTriggers(QTableWidget.NoEditTriggers)
        self.cheat_sheet.horizontalHeader().setStretchLastSection(True)
        self.cheat_sheet.verticalHeader().setVisible(False) # Hide row numbers
        self.cheat_sheet.setAlternatingRowColors(True) # For better readability
        self.cheat_sheet.setStyleSheet("""
            QTableWidget {
                border: 1px solid #ddd;
                border-radius: 8px;
                gridline-color: #f0f0f0;
                background-color: white;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 5px;
                border: 1px solid #ddd;
                font-weight: bold;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #CCE5FF; /* Light blue selection */
            }
        """)
        elements_layout.addWidget(self.cheat_sheet)

        self.tab_widget.addTab(self.story_elements_tab, "Story Elements")

    def create_live_transcript_tab(self):
        """Creates the 'Live Transcript' tab content, mimicking Screenshot_16.png."""
        self.live_transcript_tab = QWidget()
        transcript_layout = QVBoxLayout(self.live_transcript_tab)

        transcript_header_layout = QHBoxLayout()
        transcript_header_layout.addWidget(QLabel("<b>Live Transcript</b>"))
        # Add a "Processing" indicator if needed, similar to the screenshot
        self.processing_indicator = QLabel("Processing")
        self.processing_indicator.setStyleSheet("background-color: #d4edda; color: #155724; padding: 3px 8px; border-radius: 10px; font-size: 12px; font-weight: bold;")
        self.processing_indicator.setVisible(False) # Initially hidden
        transcript_header_layout.addWidget(self.processing_indicator)
        transcript_header_layout.addStretch()
        transcript_layout.addLayout(transcript_header_layout)

        transcript_layout.addWidget(QLabel("Real-time speech-to-text transcription"))

        self.transcript_display = QTextEdit()
        self.transcript_display.setReadOnly(True)
        self.transcript_display.setPlaceholderText("Live transcription will appear here...")
        self.transcript_display.setStyleSheet("background-color: white; border: 1px solid #ddd; border-radius: 8px; padding: 10px;")
        transcript_layout.addWidget(self.transcript_display)

        self.tab_widget.addTab(self.live_transcript_tab, "Live Transcript")

    def create_ai_chat_tab(self):
        """Creates the 'AI Chat' tab content, mimicking Screenshot_17.png."""
        self.ai_chat_tab = QWidget()
        chat_layout = QVBoxLayout(self.ai_chat_tab)

        chat_layout.addWidget(QLabel("<b>AI Story Assistant</b>"))
        chat_layout.addWidget(QLabel("Ask questions about the story, characters, and plot"))

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setPlaceholderText("Chat with the AI about the story...")
        self.chat_display.setStyleSheet("background-color: white; border: 1px solid #ddd; border-radius: 8px; padding: 10px;")
        chat_layout.addWidget(self.chat_display)

        chat_input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask about characters, plot, or story elements...")
        self.chat_input.returnPressed.connect(self.send_chat_query)
        self.chat_input.setStyleSheet("padding: 8px; border: 1px solid #ccc; border-radius: 5px;")
        chat_input_layout.addWidget(self.chat_input)

        self.send_chat_button = QPushButton("Send")
        # Using a standard icon for a send button
        self.send_chat_button.setIcon(self.style().standardIcon(self.style().SP_ArrowRight))
        self.send_chat_button.clicked.connect(self.send_chat_query)
        self.send_chat_button.setStyleSheet("background-color: #007BFF; color: white; padding: 8px 15px; border-radius: 5px; font-weight: bold;")
        chat_input_layout.addWidget(self.send_chat_button)

        chat_layout.addLayout(chat_input_layout)

        self.tab_widget.addTab(self.ai_chat_tab, "AI Chat")

    def get_content_title_and_context(self):
        """
        Prompts the user for the content title and initiates web search
        for external context.
        """
        title, ok = QInputDialog.getText(self, "Content Title", "Please enter the title of the content you are watching (e.g., Movie Title, Show Name, Game Title):")
        if ok and title:
            self.content_title = title.strip()
            self.current_content_title_label.setText(f"Analyzing: {self.content_title}") # Update overview tab
            if not self.content_title:
                self.update_log("Empty content title provided. LLM will operate without specific external context.")
                self.llm_thread.set_external_context("No external context provided by user.")
                self.init_chat_thread()
                self.toggle_button.setEnabled(True)
                return

            self.llm_thread.set_content_title(self.content_title)
            self.update_log(f"Searching for external context for: '{self.content_title}'...")
            self.web_search_thread = WebSearchThread(self.content_title)
            self.web_search_thread.context_ready.connect(self.set_llm_external_context)
            self.web_search_thread.error_signal.connect(self.update_log)
            self.web_search_thread.start()
        else:
            self.update_log("No content title provided. LLM will operate without specific external context.")
            self.llm_thread.set_external_context("No external context provided by user.")
            self.init_chat_thread()
            self.toggle_button.setEnabled(True)

    def set_llm_external_context(self, context):
        """
        Sets the external context for the LLM thread and enables the start button.
        """
        self.llm_thread.set_external_context(context)
        self.update_log(f"External context loaded for LLM (showing first 500 chars):\n{context[:500]}...")
        if len(context) > 500:
            self.update_log(f"...\n(Full context is {len(context)} characters long)")
        self.init_chat_thread() # Initialize chat thread after context is ready
        self.toggle_button.setEnabled(True)
        self.update_log("You can now click 'Start Recording' to begin audio processing and entity extraction.")

    def init_chat_thread(self):
        """Initializes or re-initializes the chat thread."""
        # Ensure chat thread is only initialized once or re-initialized correctly
        if self.chat_thread and self.chat_thread.isRunning():
            self.chat_thread.stop()
            self.chat_thread.wait()

        self.chat_thread = ChatThread(
            transcript_getter=self.llm_thread.get_transcriptions,
            entities_getter=self.llm_thread.get_entities,
            content_title=self.content_title,
            external_context=self.llm_thread.external_context
        )
        self.chat_thread.chat_response.connect(self.display_chat_response)
        self.chat_thread.chat_log.connect(self.update_log)
        self.chat_thread.start()

    def toggle_processing(self):
        """Starts or stops the audio capture, transcription, and LLM threads."""
        if not self.processing_active:
            # Start threads
            self.audio_thread.start()
            self.transcription_thread.start()
            self.llm_thread.start()
            self.processing_active = True
            # Update UI for "recording" state
            self.toggle_button.setText("Stop Recording")
            self.toggle_button.setStyleSheet("background-color: #FF5252; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold;") # Red for stop
            self.status_label.setText("Recording...")
            self.status_label.setStyleSheet("font-size: 14px; color: green; margin-right: 10px;")
            self.processing_indicator.setVisible(True) # Show processing indicator on transcript tab
            self.update_log("Processing started.")
        else:
            # Stop threads
            self.stop_processing()
            self.processing_active = False
            # Update UI for "idle" state
            self.toggle_button.setText("Start Recording")
            self.toggle_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold;") # Green for start
            self.status_label.setText("Idle")
            self.status_label.setStyleSheet("font-size: 14px; color: gray; margin-right: 10px;")
            self.processing_indicator.setVisible(False) # Hide processing indicator
            self.update_log("Processing stopped.")

    def stop_processing(self):
        """Gracefully stops all active processing threads."""
        self.llm_thread.stop()
        self.transcription_thread.stop()
        self.audio_thread.stop()
        # Wait for threads to finish their execution loop
        self.llm_thread.wait()
        self.transcription_thread.wait()
        self.audio_thread.wait()
        # Ensure audio buffer is cleared for the next run
        # This loop might be tricky if audio_buffer is constantly being written to.
        # A more robust solution might involve a clear() method in TranscriptionThread.
        # For now, we assume it will eventually empty or new audio will overwrite.
        # while not self.transcription_thread.audio_buffer.size == 0:
        #     time.sleep(0.01)
        self.update_log("All processing threads stopped.")

    def handle_transcription(self, text):
        """
        Receives transcription text, appends it to the display,
        and sends it to the LLM thread. Updates transcript line count.
        """
        self.transcript_display.append(text)
        self.llm_thread.add_transcription(text)
        # Update overview counts (approximate lines, could be more precise)
        # Assuming each transcription emit corresponds roughly to a "line" or segment
        current_lines = int(self.transcript_line_count_label.findChild(QLabel, "count_label").text())
        self.transcript_line_count_label.findChild(QLabel, "count_label").setText(str(current_lines + 1))


    def update_story_elements_tab(self, entities):
        """
        Updates the 'Story Elements' cheat sheet table and the 'Overview' tab's
        counts and key characters based on newly extracted entities.
        """
        # Sort entities by type, then name for consistent display
        entities.sort(key=lambda e: (e["type"], e["name"].lower()))
        self.cheat_sheet.setRowCount(len(entities))
        char_count = 0
        loc_count = 0
        total_elements = len(entities)

        # Clear existing key characters on overview to rebuild them
        # This ensures the list is always fresh and sorted
        for i in reversed(range(self.key_characters_list.count())):
            widget_to_remove = self.key_characters_list.itemAt(i).widget()
            if widget_to_remove:
                widget_to_remove.setParent(None) # Remove from layout and delete widget

        for i, entity in enumerate(entities):
            # Update the cheat sheet table
            self.cheat_sheet.setItem(i, 0, QTableWidgetItem(entity["name"]))
            self.cheat_sheet.setItem(i, 1, QTableWidgetItem(entity["type"]))
            description = entity.get("description", "No description available")
            self.cheat_sheet.setItem(i, 2, QTableWidgetItem(description))

            # Update counts for overview tab
            if entity["type"] == "Characters":
                char_count += 1
                # Add to Key Characters in Overview, similar to screenshot design
                self._add_key_character_to_overview(entity["name"], description)
            elif entity["type"] == "Locations":
                loc_count += 1
            # Add other types if you want to track them separately on overview

        self.cheat_sheet.resizeColumnsToContents()
        self.cheat_sheet.setColumnWidth(2, max(self.cheat_sheet.columnWidth(2), 250)) # Ensure description column is wide enough

        # Update overview cards with latest counts
        self.char_count_label.findChild(QLabel, "count_label").setText(str(char_count))
        self.loc_count_label.findChild(QLabel, "count_label").setText(str(loc_count))
        self.total_elements_count_label.findChild(QLabel, "count_label").setText(str(total_elements))


    def _add_key_character_to_overview(self, name, description):
        """Helper to add a character card to the 'Key Characters' section."""
        char_frame = QFrame()
        char_frame.setFrameShape(QFrame.StyledPanel)
        char_frame.setStyleSheet("background-color: #F8F8F8; border-radius: 5px; padding: 8px; margin-bottom: 5px; border: 1px solid #eee;")
        char_layout = QVBoxLayout(char_frame)
        char_layout.setContentsMargins(5,5,5,5) # Tight layout

        char_name_label = QLabel(f"<b>👤 {name}</b>") # Add a small icon
        char_name_label.setStyleSheet("font-size: 16px; color: #333;")
        char_layout.addWidget(char_name_label)

        char_desc_label = QLabel(description)
        char_desc_label.setWordWrap(True) # Allow text to wrap
        char_desc_label.setStyleSheet("font-size: 12px; color: #666;")
        char_layout.addWidget(char_desc_label)

        self.key_characters_list.addWidget(char_frame)
        # Also append to recent activity log
        self.recent_activity_text.append(f"New/Updated Entity: <b>{name}</b> ({description[:50]}...)")


    def update_log(self, log):
        """Appends a message to the global application log display."""
        self.log_display.append(log)
        # Scroll to the bottom automatically
        self.log_display.verticalScrollBar().setValue(self.log_display.verticalScrollBar().maximum())

    def send_chat_query(self):
        """Sends user's chat query to the ChatThread."""
        query = self.chat_input.text().strip()
        if not query:
            return
        self.chat_display.append(f"<b>User:</b> {query}") # Bold user query
        self.chat_thread.add_chat_query(query)
        self.chat_input.clear()
        self.send_chat_button.setEnabled(False) # Disable button while waiting for response

    def display_chat_response(self, response):
        """Displays AI's chat response."""
        self.chat_display.append(f"<b>AI:</b> {response}") # Bold AI response
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())
        self.send_chat_button.setEnabled(True) # Re-enable button after response

    def closeEvent(self, event):
        """Handles application shutdown, ensuring all threads are stopped."""
        try:
            self.stop_processing() # Stop main processing threads
            if self.chat_thread and self.chat_thread.isRunning():
                self.chat_thread.stop()
                self.chat_thread.wait()
            if self.web_search_thread and self.web_search_thread.isRunning():
                self.web_search_thread.quit() # QThread.quit() for non-event loop threads
                self.web_search_thread.wait()
            event.accept() # Accept the close event
        except Exception as e:
            print(f"Error during cleanup: {e}", file=sys.stderr)
            event.accept() # Still accept to prevent hanging

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NarrativeNavigator()
    window.show()
    sys.exit(app.exec_())