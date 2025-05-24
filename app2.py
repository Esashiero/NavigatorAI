import sys
import json
import numpy as np
import sounddevice as sd
import whisper
import re
import queue
import threading
import time
import ollama
from duckduckgo_search import DDGS
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QTextEdit, QTableWidget,
    QTableWidgetItem, QInputDialog, QSplitter, QLineEdit,
    QLabel, QFrame, QTabWidget, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QStyle # For standard icons

# --- Thread Classes (mostly unchanged, keeping for completeness) ---

class AudioCaptureThread(QThread):
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
            if status:
                pass
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
                        audio_chunk = self.audio_queue.get(timeout=0.1)
                        self.audio_data.emit(audio_chunk)
                    except queue.Empty:
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
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

    def stop(self):
        self._running = False
        self._stop_event.set()

class TranscriptionThread(QThread):
    transcription = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        try:
            self.model = whisper.load_model("small.en")
        except Exception as e:
            self.error_signal.emit(f"Failed to load Whisper model: {e}")
            self.model = None

        self.audio_buffer = np.array([], dtype=np.float32)
        self.running = False
        self.buffer_lock = threading.Lock()

    def run(self):
        self.running = True
        if self.model is None:
            self.error_signal.emit("Transcription model not loaded, cannot transcribe.")
            self.running = False
            return

        while self.running:
            audio_to_process = None
            with self.buffer_lock:
                if len(self.audio_buffer) >= 16000 * 10:
                    audio_to_process = self.audio_buffer[:16000 * 10].copy()
                    self.audio_buffer = self.audio_buffer[16000 * 10:]
            
            if audio_to_process is not None:
                try:
                    if audio_to_process.dtype != np.float32:
                        audio_to_process = audio_to_process.astype(np.float32) / 32768.0

                    result = self.model.transcribe(audio_to_process, language="en")
                    if result["text"].strip():
                        self.transcription.emit(result["text"])
                except Exception as e:
                    self.error_signal.emit(f"Transcription Error: {e}")
                    print(f"Transcription Error: {e}", file=sys.stderr)
            time.sleep(0.1)

    def add_audio(self, audio_data):
        with self.buffer_lock:
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / 32768.0
            self.audio_buffer = np.append(self.audio_buffer, audio_data.flatten())

    def stop(self):
        self.running = False

class WebSearchThread(QThread):
    context_ready = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, title):
        super().__init__()
        self.title = title

    def run(self):
        try:
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
    entities_updated = pyqtSignal(list)
    llm_log = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.transcriptions = []
        self.entities = []
        self.running = False
        self.external_context = ""
        self.content_title = "Unknown Content"

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
        self.system_prompt = self.base_system_prompt.format(
            content_title=self.content_title,
            external_context="No external context loaded yet. Please wait for the application to gather information."
        )
        self.last_transcript_processed_idx = -1

    def set_content_title(self, title):
        self.content_title = title
        self._update_system_prompt()

    def set_external_context(self, context):
        self.external_context = context
        self._update_system_prompt()

    def _update_system_prompt(self):
        self.system_prompt = self.base_system_prompt.format(
            content_title=self.content_title,
            external_context=self.external_context
        )

    def _normalize_entity_name(self, name):
        normalized_name = name.lower()
        normalized_name = re.sub(r'\s*\([^)]*\)', '', normalized_name).strip()
        for article in ['the ', 'a ', 'an ']:
            if normalized_name.startswith(article):
                normalized_name = normalized_name[len(article):].strip()
        normalized_name = re.sub(r"'s\b", '', normalized_name).strip()
        normalized_name = re.sub(r"s'\b", 's', normalized_name).strip()
        normalized_name = re.sub(r'[^a-z0-9\s]', '', normalized_name).strip()
        normalized_name = re.sub(r'\s+', ' ', normalized_name).strip()
        return normalized_name

    def _is_similar_entity(self, entity1, entity2):
        if entity1["type"] != entity2["type"]:
            return False
        name1_norm = self._normalize_entity_name(entity1["name"])
        name2_norm = self._normalize_entity_name(entity2["name"])
        return name1_norm == name2_norm

    def run(self):
        self.running = True
        if not self.external_context:
            self.llm_log.emit("LLM Thread waiting for external context...")
            while self.running and not self.external_context:
                time.sleep(1)
            if not self.running:
                return

        self.llm_log.emit("LLM Thread started with external context.")
        while self.running:
            if len(self.transcriptions) > self.last_transcript_processed_idx + 1:
                transcript_to_process = self.transcriptions[-1]
                self.last_transcript_processed_idx = len(self.transcriptions) - 1

                recent_transcripts = self.transcriptions[max(0, self.last_transcript_processed_idx - 4):self.last_transcript_processed_idx + 1]
                current_entities_json = json.dumps(self.entities, indent=2)

                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content":
                     f"Current transcript snippet: {transcript_to_process}\n"
                     f"Recent context (last {len(recent_transcripts)} snippets): {recent_transcripts}\n"
                     f"Current narrative cheat sheet: {current_entities_json}\n"
                     f"Extract significant entities from the current transcript, updating the cheat sheet if entities already exist."}
                ]
                self.llm_log.emit(f"Prompt sent (for transcript index {self.last_transcript_processed_idx}):\n{json.dumps(messages, indent=2)[:1000]}...")
                try:
                    response = ollama.chat(model="phi4-mini:latest", messages=messages, stream=False)
                    content = response['message']['content']
                    self.llm_log.emit(f"LLM raw response (for transcript index {self.last_transcript_processed_idx}):\n{content}")

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

                        valid_entities = [e for e in entities if e.get("name") and e.get("type") in ["Characters", "Locations", "Organizations", "Key Objects", "Concepts/Events"]]
                        for new_entity in valid_entities:
                            found = False
                            for i, existing in enumerate(self.entities):
                                if self._is_similar_entity(new_entity, existing):
                                    self.entities[i]["description"] = new_entity.get("description", existing["description"])
                                    
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
                        self.entities_updated.emit(self.entities)
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        self.llm_log.emit(f"Error: Invalid JSON response or parsing error - {str(e)}\nAttempted to parse:\n{json_string}")
                except Exception as e:
                    self.llm_log.emit(f"Error: LLM request failed - {str(e)}")
            time.sleep(2)

    def add_transcription(self, text):
        self.transcriptions.append(text)

    def get_transcriptions(self):
        return self.transcriptions

    def get_entities(self):
        return self.entities

    def stop(self):
        self.running = False

class ChatThread(QThread):
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
        self.chat_queue.put(query)

    def run(self):
        self.running = True
        while self.running:
            try:
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
                self.chat_log.emit(f"Chat prompt sent:\n{json.dumps(messages, indent=2)[:1000]}...")
                try:
                    response = ollama.chat(model="phi4-mini:latest", messages=messages, stream=False)
                    content = response['message']['content']
                    self.chat_response.emit(content)
                    self.chat_log.emit(f"Chat response:\n{content}")
                except Exception as e:
                    self.chat_response.emit(f"Error: Unable to process query - {str(e)}")
                    self.chat_log.emit(f"Chat Error: {str(e)}")
            except queue.Empty:
                pass
            time.sleep(0.5)

    def stop(self):
        self.running = False

# --- Main Application Window ---

class NarrativeNavigator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Narrative Navigator AI")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(800, 600)

        # Apply QSS stylesheet
        try:
            with open('style.qss', 'r') as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            print("Warning: 'style.qss' not found. UI will not be styled.", file=sys.stderr)
        except Exception as e:
            print(f"Error loading stylesheet: {e}", file=sys.stderr)

        self.audio_thread = AudioCaptureThread()
        self.transcription_thread = TranscriptionThread()
        self.llm_thread = LLMThread()
        self.chat_thread = None
        self.web_search_thread = None
        self.content_title = ""

        self.init_ui()

        # Connect signals and slots
        self.audio_thread.audio_data.connect(self.transcription_thread.add_audio)
        self.audio_thread.error_signal.connect(self.update_log)
        self.transcription_thread.transcription.connect(self.handle_transcription)
        self.transcription_thread.error_signal.connect(self.update_log)
        self.llm_thread.entities_updated.connect(self.update_entity_displays) 
        self.llm_thread.llm_log.connect(self.update_log)

        # Start content title prompt immediately
        self.get_content_title_and_context()

    def init_ui(self):
        main_container = QWidget()
        self.setCentralWidget(main_container)
        main_layout = QVBoxLayout(main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- 1. Header Bar ---
        header_widget = QWidget()
        header_widget.setObjectName("headerWidget")
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(20, 10, 20, 10)
        header_layout.setSpacing(15)

        app_logo = QLabel()
        app_logo.setPixmap(self.style().standardIcon(QStyle.SP_ComputerIcon).pixmap(QSize(32, 32)))
        
        app_name_label = QLabel("Narrative Navigator")
        app_name_label.setObjectName("appNameLabel")

        self.analysis_status_label = QLabel("Analyzing: [FULL] Beyond Boiling Point - Gordon Ramsay documentary (2000)")
        self.analysis_status_label.setObjectName("analysisStatusLabel")
        self.analysis_status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.analysis_status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        header_layout.addWidget(app_logo)
        header_layout.addWidget(app_name_label)
        header_layout.addWidget(self.analysis_status_label)

        self.volume_button = QPushButton()
        self.volume_button.setIcon(self.style().standardIcon(QStyle.SP_MediaVolume))
        self.volume_button.setObjectName("iconButton")
        self.volume_button.setIconSize(QSize(20, 20))
        self.volume_button.setFixedSize(36, 36)

        self.toggle_button = QPushButton("Stop Recording")
        self.toggle_button.setObjectName("stopRecordingButton")
        self.toggle_button.clicked.connect(self.toggle_processing)
        self.toggle_button.setEnabled(False)

        self.close_button = QPushButton() # Changed to icon button as per design
        self.close_button.setIcon(self.style().standardIcon(QStyle.SP_DialogCloseButton))
        self.close_button.setObjectName("iconButton")
        self.close_button.setIconSize(QSize(20,20))
        self.close_button.setFixedSize(36,36)
        self.close_button.clicked.connect(self.close)

        header_layout.addWidget(self.volume_button)
        header_layout.addWidget(self.toggle_button)
        header_layout.addWidget(self.close_button)

        main_layout.addWidget(header_widget)

        # --- 2. Main Content Splitter ---
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setContentsMargins(10, 0, 10, 10) # Margins around the splitter itself

        # Left Panel: Tab Widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("mainTabWidget")
        content_splitter.addWidget(self.tab_widget)

        # Create Tab Pages
        self.overview_tab_page = self._create_overview_tab()
        self.story_elements_tab_page = self._create_story_elements_tab()
        self.live_transcript_tab_page = self._create_live_transcript_tab()
        self.ai_chat_tab_page = self._create_ai_chat_tab()
        self.llm_log_tab_page = self._create_llm_log_tab()

        self.tab_widget.addTab(self.overview_tab_page, "Overview")
        self.tab_widget.addTab(self.story_elements_tab_page, "Story Elements")
        self.tab_widget.addTab(self.live_transcript_tab_page, "Live Transcript")
        self.tab_widget.addTab(self.ai_chat_tab_page, "AI Chat")
        self.tab_widget.addTab(self.llm_log_tab_page, "LLM Log")

        # Right Panel: Narrative Cheat Sheet (QTableWidget)
        right_panel = QWidget()
        right_panel.setObjectName("rightPanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 15, 15, 15)
        right_layout.setSpacing(10)

        cheat_sheet_label = QLabel("Narrative Cheat Sheet")
        cheat_sheet_label.setObjectName("sectionTitle") # Apply QSS for titles
        right_layout.addWidget(cheat_sheet_label)

        self.cheat_sheet_table = QTableWidget() # Renamed to avoid confusion with internal entities list
        self.cheat_sheet_table.setColumnCount(3)
        self.cheat_sheet_table.setHorizontalHeaderLabels(["Name", "Type", "Description"])
        self.cheat_sheet_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.cheat_sheet_table.horizontalHeader().setStretchLastSection(True)
        self.cheat_sheet_table.verticalHeader().setVisible(False)
        self.cheat_sheet_table.setAlternatingRowColors(True)
        self.cheat_sheet_table.setObjectName("cheatSheetTable") # QSS target
        right_layout.addWidget(self.cheat_sheet_table)

        content_splitter.addWidget(right_panel)
        content_splitter.setSizes([800, 400]) # Initial sizes
        main_layout.addWidget(content_splitter)

    # --- Helper methods to create tab content ---

    def _create_overview_tab(self):
        tab_page = QWidget()
        tab_layout = QVBoxLayout(tab_page)
        tab_layout.setContentsMargins(20, 20, 20, 20)
        tab_layout.setSpacing(15)

        title_label = QLabel("Overview")
        title_label.setObjectName("sectionTitle")
        tab_layout.addWidget(title_label)

        # Summary Metrics (Cards)
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(15)
        
        self.characters_count_label = QLabel("1")
        self.locations_count_label = QLabel("1")
        self.transcript_lines_count_label = QLabel("0")
        self.total_elements_count_label = QLabel("3")

        metrics_layout.addWidget(self._create_summary_card(self.characters_count_label, "Characters", QStyle.SP_MessageBoxInformation)) # Generic info icon
        metrics_layout.addWidget(self._create_summary_card(self.locations_count_label, "Locations", QStyle.SP_DirIcon)) # Folder icon for location
        metrics_layout.addWidget(self._create_summary_card(self.transcript_lines_count_label, "Transcript Lines", QStyle.SP_FileIcon)) # File icon
        metrics_layout.addWidget(self._create_summary_card(self.total_elements_count_label, "Total Elements", QStyle.SP_MessageBoxQuestion)) # Question icon
        tab_layout.addLayout(metrics_layout)

        # Recent Activity section
        recent_activity_widget = QFrame()
        recent_activity_widget.setProperty("class", "contentCard")
        recent_activity_layout = QVBoxLayout(recent_activity_widget)
        recent_activity_layout.setContentsMargins(15,15,15,15)
        recent_activity_layout.setSpacing(10)

        recent_activity_layout.addWidget(QLabel("Recent Activity"))
        self.recent_activity_display = QTextEdit()
        self.recent_activity_display.setReadOnly(True)
        self.recent_activity_display.setPlaceholderText("Latest story elements and transcript updates...")
        self.recent_activity_display.setFixedHeight(150)
        recent_activity_layout.addWidget(self.recent_activity_display)
        tab_layout.addWidget(recent_activity_widget)

        # Key Characters section
        key_characters_widget = QFrame()
        key_characters_widget.setProperty("class", "contentCard")
        key_characters_layout = QVBoxLayout(key_characters_widget)
        key_characters_layout.setContentsMargins(15,15,15,15)
        key_characters_layout.setSpacing(10)
        
        key_characters_layout.addWidget(QLabel("Key Characters"))
        self.key_characters_container_layout = QVBoxLayout()
        self.key_characters_container_layout.setContentsMargins(0,0,0,0)
        self.key_characters_container_layout.setSpacing(8)
        
        # Example initial character - will be cleared and repopulated by update_entity_displays
        example_char_card = self._create_entity_card(
            {"name": "Walter White", "type": "Characters", "description": "High school chemistry teacher turned meth cook"},
            "00:02:15"
        )
        self.key_characters_container_layout.addWidget(example_char_card)
        key_characters_layout.addLayout(self.key_characters_container_layout)
        tab_layout.addWidget(key_characters_widget)

        tab_layout.addStretch()
        return tab_page

    def _create_summary_card(self, count_label_ref, label_text, icon_style_hint):
        card = QFrame()
        card.setProperty("class", "contentCard")
        card.setFixedSize(160, 120)
        card_layout = QVBoxLayout(card)
        card_layout.setAlignment(Qt.AlignCenter)
        card_layout.setContentsMargins(10, 10, 10, 10)
        card_layout.setSpacing(5)

        icon_label = QLabel()
        icon_label.setPixmap(self.style().standardIcon(icon_style_hint).pixmap(QSize(30, 30)))
        card_layout.addWidget(icon_label, alignment=Qt.AlignCenter)

        count_label_ref.setStyleSheet("font-size: 28px; font-weight: bold; color: #6a0dad;")
        card_layout.addWidget(count_label_ref, alignment=Qt.AlignCenter)

        text_label = QLabel(label_text)
        text_label.setStyleSheet("font-size: 13px; color: #555555; font-weight: 500;")
        card_layout.addWidget(text_label, alignment=Qt.AlignCenter)
        return card

    def _create_story_elements_tab(self):
        tab_page = QWidget()
        tab_layout = QVBoxLayout(tab_page)
        tab_layout.setContentsMargins(20, 20, 20, 20)
        tab_layout.setSpacing(15)

        title_label = QLabel("Story Elements")
        title_label.setObjectName("sectionTitle")
        tab_layout.addWidget(title_label)

        # ScrollArea to contain dynamically added entity cards
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QFrame.NoFrame)

        self.story_elements_content_widget = QWidget()
        self.story_elements_container_layout = QVBoxLayout(self.story_elements_content_widget)
        self.story_elements_container_layout.setContentsMargins(0, 0, 0, 0)
        self.story_elements_container_layout.setSpacing(10)
        self.story_elements_container_layout.setAlignment(Qt.AlignTop)

        self.story_elements_container_layout.addStretch() # Keep this one inside the scroll area's content widget

        scroll_area.setWidget(self.story_elements_content_widget)
        tab_layout.addWidget(scroll_area)
        
        return tab_page
    
    # In NarrativeNavigator class
    def _create_entity_card(self, entity_data, first_mentioned_time="00:00:00"):
        card = QFrame()
        card.setProperty("class", "storyElementCard") # <-- NEW QSS CLASS

        # Change to QHBoxLayout for single-line content
        card_layout = QHBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 0) # Rely on QSS padding for the QFrame
        card_layout.setSpacing(5) # Tight spacing between elements

        icon_label = QLabel()
        # Set icon size to be smaller for compact display
        if entity_data["type"] == "Characters":
            icon_label.setPixmap(self.style().standardIcon(QStyle.SP_MessageBoxInformation).pixmap(QSize(16, 16)))
        elif entity_data["type"] == "Locations":
            icon_label.setPixmap(self.style().standardIcon(QStyle.SP_DirIcon).pixmap(QSize(16, 16)))
        elif entity_data["type"] == "Organizations":
            icon_label.setPixmap(self.style().standardIcon(QStyle.SP_DesktopIcon).pixmap(QSize(16, 16)))
        else: # Default for Key Objects, Concepts/Events
            icon_label.setPixmap(self.style().standardIcon(QStyle.SP_FileIcon).pixmap(QSize(16, 16)))
        card_layout.addWidget(icon_label)

        element_name = QLabel(entity_data["name"])
        element_name.setProperty("property", "nameLabel") # Custom property for QSS targeting
        card_layout.addWidget(element_name)
        
        category_tag = QLabel(entity_data["type"])
        category_tag.setProperty("class", "categoryTag") # Keep existing tag style
        category_tag.setObjectName(f"categoryTag-{entity_data['type'].replace(' ', '')}")
        card_layout.addWidget(category_tag)

        # Add a separator
        separator = QLabel("—") # A dash or em-dash
        separator.setStyleSheet("color: #aaaaaa; margin: 0 5px;")
        card_layout.addWidget(separator)

        description_text = entity_data.get("description", "No description available")
        # Truncate description to fit on one line gracefully
        max_desc_length = 80 # Adjust this value as needed
        if len(description_text) > max_desc_length:
            description_text = description_text[:max_desc_length].strip() + "..."
        
        description = QLabel(description_text)
        description.setProperty("property", "descriptionLabel") # Custom property for QSS targeting
        description.setWordWrap(False) # Crucial: Prevent word wrapping
        description.setTextFormat(Qt.PlainText) # Ensure plain text for truncation
        description.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred) # Allow description to expand
        card_layout.addWidget(description)

        card_layout.addStretch() # Push the timestamp to the far right

        first_mentioned = QLabel(f"First mentioned: {first_mentioned_time}")
        first_mentioned.setProperty("property", "timeLabel") # Custom property for QSS targeting
        card_layout.addWidget(first_mentioned)
        
        return card

    def _create_live_transcript_tab(self):
        tab_page = QWidget()
        tab_layout = QVBoxLayout(tab_page)
        tab_layout.setContentsMargins(20, 20, 20, 20)
        tab_layout.setSpacing(15)

        header_layout = QHBoxLayout()
        mic_icon = QLabel()
        mic_icon.setPixmap(self.style().standardIcon(QStyle.SP_MediaVolume).pixmap(QSize(24, 24)))
        header_layout.addWidget(mic_icon)
        
        title_label = QLabel("Live Transcript")
        title_label.setObjectName("sectionTitle")
        header_layout.addWidget(title_label)

        self.processing_tag = QLabel("Processing")
        self.processing_tag.setProperty("class", "processingTag")
        self.processing_tag.setVisible(False)
        header_layout.addWidget(self.processing_tag)
        header_layout.addStretch()
        tab_layout.addLayout(header_layout)

        self.transcript_display = QTextEdit()
        self.transcript_display.setReadOnly(True)
        self.transcript_display.setPlaceholderText("Live transcription will appear here...")
        tab_layout.addWidget(self.transcript_display)
        
        return tab_page

    def _create_ai_chat_tab(self):
        tab_page = QWidget()
        tab_layout = QVBoxLayout(tab_page)
        tab_layout.setContentsMargins(20, 20, 20, 20)
        tab_layout.setSpacing(15)

        header_layout = QHBoxLayout()
        ai_icon = QLabel()
        ai_icon.setPixmap(self.style().standardIcon(QStyle.SP_MessageBoxQuestion).pixmap(QSize(24, 24)))
        header_layout.addWidget(ai_icon)
        
        title_label = QLabel("AI Story Assistant")
        title_label.setObjectName("sectionTitle")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        tab_layout.addLayout(header_layout)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setPlaceholderText("Chat with the AI about the story...")
        tab_layout.addWidget(self.chat_display)

        chat_input_container = QFrame()
        chat_input_container.setObjectName("chatInputContainer")
        chat_input_layout = QHBoxLayout(chat_input_container)
        chat_input_layout.setContentsMargins(0, 0, 0, 0)
        chat_input_layout.setSpacing(0)

        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask about characters, plot, or story elements...")
        self.chat_input.returnPressed.connect(self.send_chat_query)
        chat_input_layout.addWidget(self.chat_input)

        send_button = QPushButton()
        send_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
        send_button.setIconSize(QSize(20, 20))
        send_button.setFixedSize(40, 40)
        send_button.setObjectName("sendButton")
        send_button.clicked.connect(self.send_chat_query)
        chat_input_layout.addWidget(send_button)

        tab_layout.addWidget(chat_input_container)
        return tab_page

    def _create_llm_log_tab(self):
        tab_page = QWidget()
        tab_layout = QVBoxLayout(tab_page)
        tab_layout.setContentsMargins(20, 20, 20, 20)
        tab_layout.setSpacing(15)

        title_label = QLabel("LLM Processing Log")
        title_label.setObjectName("sectionTitle")
        tab_layout.addWidget(title_label)

        self.llm_log_display = QTextEdit()
        self.llm_log_display.setReadOnly(True)
        self.llm_log_display.setPlaceholderText("LLM conversation and processing logs will appear here...")
        tab_layout.addWidget(self.llm_log_display)
        return tab_page

    # --- Core Logic Methods (adapted for new UI) ---

    def get_content_title_and_context(self):
        title, ok = QInputDialog.getText(self, "Content Title", "Please enter the title of the content you are watching:", QLineEdit.Normal, "The Outlast Trials Lore")
        if ok and title:
            self.content_title = title.strip()
            self.analysis_status_label.setText(f"Analyzing: [FULL] {self.content_title}")
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
        self.llm_thread.set_external_context(context)
        self.update_log(f"External context loaded for LLM (showing first 500 chars):\n{context[:500]}...")
        if len(context) > 500:
            self.update_log(f"...\n(Full context is {len(context)} characters long)")
        self.init_chat_thread()
        self.toggle_button.setEnabled(True)
        self.toggle_button.setText("Start Recording")
        self.update_log("You can now click 'Start Recording' to begin audio processing and entity extraction.")

    def init_chat_thread(self):
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
        self.update_log("Chat thread initialized.")

    def toggle_processing(self):
        if self.toggle_button.text() == "Start Recording":
            self.audio_thread.start()
            self.transcription_thread.start()
            self.llm_thread.start()
            self.toggle_button.setText("Stop Recording")
            self.processing_tag.setVisible(True)
            self.update_log("Processing started.")
        else:
            self.stop_processing()
            self.toggle_button.setText("Start Recording")
            self.processing_tag.setVisible(False)
            self.update_log("Processing stopped.")

    def stop_processing(self):
        self.llm_thread.stop()
        self.transcription_thread.stop()
        self.audio_thread.stop()
        self.llm_thread.wait()
        self.transcription_thread.wait()
        self.audio_thread.wait()
        self.update_log("All processing threads stopped.")

    def handle_transcription(self, text):
        if self.transcript_display:
            self.transcript_display.append(text)
            self.transcript_display.verticalScrollBar().setValue(self.transcript_display.verticalScrollBar().maximum())
            current_lines = int(self.transcript_lines_count_label.text())
            self.transcript_lines_count_label.setText(str(current_lines + 1))

        self.llm_thread.add_transcription(text)
        self.recent_activity_display.append(f"Transcript: {text[:80].strip()}...")
        self.recent_activity_display.verticalScrollBar().setValue(self.recent_activity_display.verticalScrollBar().maximum())

    def update_entity_displays(self, entities):
        # 1. Update Story Elements Tab (card view)
        for i in reversed(range(self.story_elements_container_layout.count())):
            widget_to_remove = self.story_elements_container_layout.itemAt(i).widget()
            if widget_to_remove:
                widget_to_remove.setParent(None)
                widget_to_remove.deleteLater()
        
        entities.sort(key=lambda e: (e["type"], e["name"].lower()))

        for entity in entities:
            # Note: Timestamp "00:00:00" is hardcoded as LLM does not provide it
            entity_card = self._create_entity_card(entity, "00:00:00") 
            self.story_elements_container_layout.addWidget(entity_card)
        
        # 2. Update Key Characters in Overview Tab
        for i in reversed(range(self.key_characters_container_layout.count())):
            widget_to_remove = self.key_characters_container_layout.itemAt(i).widget()
            if widget_to_remove:
                widget_to_remove.setParent(None)
                widget_to_remove.deleteLater()
        
        key_characters = [e for e in entities if e["type"] == "Characters"]
        for char_entity in key_characters:
            char_card = self._create_entity_card(char_entity, "00:00:00")
            self.key_characters_container_layout.addWidget(char_card)


        # 3. Update Narrative Cheat Sheet Table (right panel)
        self.cheat_sheet_table.setRowCount(len(entities))
        for i, entity in enumerate(entities):
            self.cheat_sheet_table.setItem(i, 0, QTableWidgetItem(entity["name"]))
            self.cheat_sheet_table.setItem(i, 1, QTableWidgetItem(entity["type"]))
            description = entity.get("description", "No description available")
            self.cheat_sheet_table.setItem(i, 2, QTableWidgetItem(description))
        self.cheat_sheet_table.resizeColumnsToContents()
        self.cheat_sheet_table.setColumnWidth(2, max(self.cheat_sheet_table.columnWidth(2), 250))


        # 4. Update Overview Tab counts
        char_count = sum(1 for e in entities if e["type"] == "Characters")
        loc_count = sum(1 for e in entities if e["type"] == "Locations")
        total_elements = len(entities)

        self.characters_count_label.setText(str(char_count))
        self.locations_count_label.setText(str(loc_count))
        self.total_elements_count_label.setText(str(total_elements))
        
        self.recent_activity_display.append(f"Entities updated: {total_elements} total elements found.")
        self.recent_activity_display.verticalScrollBar().setValue(self.recent_activity_display.verticalScrollBar().maximum())

    def update_log(self, log_message):
        if self.llm_log_display:
            self.llm_log_display.append(log_message)
            self.llm_log_display.verticalScrollBar().setValue(self.llm_log_display.verticalScrollBar().maximum())

    def send_chat_query(self):
        query = self.chat_input.text().strip()
        if not query:
            return
        self.chat_display.append(f"<div style='color: #333333; margin-bottom: 5px; font-weight: bold;'>User:</div><div style='background-color: #e0e7ff; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>{query}</div>")
        self.chat_thread.add_chat_query(query)
        self.chat_input.clear()
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())


    def display_chat_response(self, response):
        self.chat_display.append(f"<div style='color: #6a0dad; margin-bottom: 5px; font-weight: bold;'>AI:</div><div style='background-color: #f0f2f5; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>{response}</div>")
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())

    def closeEvent(self, event):
        try:
            self.stop_processing()
            if self.chat_thread:
                self.chat_thread.stop()
                self.chat_thread.wait()
            if self.web_search_thread and self.web_search_thread.isRunning():
                self.web_search_thread.quit()
                self.web_search_thread.wait()
            event.accept()
        except Exception as e:
            print(f"Error during cleanup: {e}", file=sys.stderr)
            event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NarrativeNavigator()
    window.show()
    sys.exit(app.exec_())