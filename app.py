import sys
import json
import numpy as np
import sounddevice as sd
import whisper
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QPushButton, QTextEdit, QTableWidget,
                           QTableWidgetItem, QInputDialog, QSplitter, QLineEdit, QLabel)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import ollama
import queue
import threading
import time
import uuid
from duckduckgo_search import DDGS

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
        self.model = whisper.load_model("small.en")
        self.audio_buffer = np.array([], dtype=np.float32)
        self.running = False
        self.buffer_lock = threading.Lock()

    def run(self):
        self.running = True
        while self.running:
            audio_to_process = None
            with self.buffer_lock:
                if len(self.audio_buffer) >= 16000 * 10:
                    audio_to_process = self.audio_buffer[:16000 * 10].copy()
                    self.audio_buffer = self.audio_buffer[16000 * 10:]
            
            if audio_to_process is not None:
                try:
                    result = self.model.transcribe(audio_to_process, language="en")
                    if result["text"].strip():
                        self.transcription.emit(result["text"])
                except Exception as e:
                    self.error_signal.emit(f"Transcription Error: {e}")
                    print(f"Transcription Error: {e}", file=sys.stderr)
            time.sleep(0.1)

    def add_audio(self, audio_data):
        with self.buffer_lock:
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
            if self.transcriptions:
                transcript_to_process = self.transcriptions[-1]
                recent_transcripts = self.transcriptions[-5:] if len(self.transcriptions) > 5 else self.transcriptions
                current_entities_json = json.dumps(self.entities, indent=2)

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
                    response = ollama.chat(model="phi4-mini:latest", messages=messages, stream=False)
                    content = response['message']['content']
                    self.llm_log.emit(f"LLM raw response:\n{content}")

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
                self.chat_log.emit(f"Chat prompt sent:\n{json.dumps(messages, indent=2)}")
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

class NarrativeNavigator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Narrative Navigator AI 2")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(800, 600)

        self.audio_thread = AudioCaptureThread()
        self.transcription_thread = TranscriptionThread()
        self.llm_thread = LLMThread()
        self.chat_thread = None
        self.web_search_thread = None
        self.content_title = ""

        self.init_ui()

        self.audio_thread.audio_data.connect(self.transcription_thread.add_audio)
        self.audio_thread.error_signal.connect(self.update_llm_log)
        self.transcription_thread.transcription.connect(self.handle_transcription)
        self.transcription_thread.error_signal.connect(self.update_llm_log)
        self.llm_thread.entities_updated.connect(self.update_cheat_sheet)
        self.llm_thread.llm_log.connect(self.update_llm_log)

        self.get_content_title_and_context()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setMinimumWidth(300)

        self.transcript_display = QTextEdit()
        self.transcript_display.setReadOnly(True)
        self.transcript_display.setPlaceholderText("Live transcription will appear here...")
        left_layout.addWidget(self.transcript_display)

        self.llm_log_display = QTextEdit()
        self.llm_log_display.setReadOnly(True)
        self.llm_log_display.setPlaceholderText("LLM conversation and processing logs will appear here...")
        left_layout.addWidget(self.llm_log_display)

        chat_widget = QWidget()
        chat_layout = QVBoxLayout()
        chat_widget.setLayout(chat_layout)

        chat_label = QLabel("Chat with AI:")
        chat_layout.addWidget(chat_label)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setPlaceholderText("Chat with the AI about the story...")
        chat_layout.addWidget(self.chat_display)

        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask a question about the story...")
        self.chat_input.returnPressed.connect(self.send_chat_query)
        chat_layout.addWidget(self.chat_input)

        left_layout.addWidget(chat_widget)

        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        right_panel.setMinimumWidth(300)

        self.cheat_sheet = QTableWidget()
        self.cheat_sheet.setColumnCount(3)
        self.cheat_sheet.setHorizontalHeaderLabels(["Name", "Type", "Description"])
        self.cheat_sheet.setEditTriggers(QTableWidget.NoEditTriggers)
        self.cheat_sheet.horizontalHeader().setStretchLastSection(True)
        self.cheat_sheet.verticalHeader().setVisible(False)
        self.cheat_sheet.setAlternatingRowColors(True)
        right_layout.addWidget(self.cheat_sheet)

        self.toggle_button = QPushButton("Start")
        self.toggle_button.clicked.connect(self.toggle_processing)
        self.toggle_button.setEnabled(False)
        right_layout.addWidget(self.toggle_button)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([720, 480])

    def get_content_title_and_context(self):
        title, ok = QInputDialog.getText(self, "Content Title", "Please enter the title of the content you are watching (e.g., Movie Title, Show Name, Game Title):")
        if ok and title:
            self.content_title = title.strip()
            if not self.content_title:
                self.update_llm_log("Empty content title provided. LLM will operate without specific external context.")
                self.llm_thread.set_external_context("No external context provided by user.")
                self.init_chat_thread()
                self.toggle_button.setEnabled(True)
                return

            self.llm_thread.set_content_title(self.content_title)
            self.update_llm_log(f"Searching for external context for: '{self.content_title}'...")
            self.web_search_thread = WebSearchThread(self.content_title)
            self.web_search_thread.context_ready.connect(self.set_llm_external_context)
            self.web_search_thread.error_signal.connect(self.update_llm_log)
            self.web_search_thread.start()
        else:
            self.update_llm_log("No content title provided. LLM will operate without specific external context.")
            self.llm_thread.set_external_context("No external context provided by user.")
            self.init_chat_thread()
            self.toggle_button.setEnabled(True)

    def set_llm_external_context(self, context):
        self.llm_thread.set_external_context(context)
        self.update_llm_log(f"External context loaded for LLM (showing first 500 chars):\n{context[:500]}...")
        if len(context) > 500:
            self.update_llm_log(f"...\n(Full context is {len(context)} characters long)")
        self.init_chat_thread()
        self.toggle_button.setEnabled(True)
        self.update_llm_log("You can now click 'Start' to begin audio processing and entity extraction.")

    def init_chat_thread(self):
        self.chat_thread = ChatThread(
            transcript_getter=self.llm_thread.get_transcriptions,
            entities_getter=self.llm_thread.get_entities,
            content_title=self.content_title,
            external_context=self.llm_thread.external_context
        )
        self.chat_thread.chat_response.connect(self.display_chat_response)
        self.chat_thread.chat_log.connect(self.update_llm_log)
        self.chat_thread.start()

    def toggle_processing(self):
        if self.toggle_button.text() == "Start":
            self.audio_thread.start()
            self.transcription_thread.start()
            self.llm_thread.start()
            self.toggle_button.setText("Stop")
            self.update_llm_log("Processing started.")
        else:
            self.stop_processing()
            self.toggle_button.setText("Start")
            self.update_llm_log("Processing stopped.")

    def stop_processing(self):
        self.llm_thread.stop()
        self.transcription_thread.stop()
        self.audio_thread.stop()
        self.llm_thread.wait()
        self.transcription_thread.wait()
        self.audio_thread.wait()

    def handle_transcription(self, text):
        self.transcript_display.append(text)
        self.llm_thread.add_transcription(text)

    def update_cheat_sheet(self, entities):
        entities.sort(key=lambda e: (e["type"], e["name"].lower()))
        self.cheat_sheet.setRowCount(len(entities))
        for i, entity in enumerate(entities):
            self.cheat_sheet.setItem(i, 0, QTableWidgetItem(entity["name"]))
            self.cheat_sheet.setItem(i, 1, QTableWidgetItem(entity["type"]))
            description = entity.get("description", "No description available")
            self.cheat_sheet.setItem(i, 2, QTableWidgetItem(description))
        self.cheat_sheet.resizeColumnsToContents()
        self.cheat_sheet.setColumnWidth(2, max(self.cheat_sheet.columnWidth(2), 250))

    def update_llm_log(self, log):
        self.llm_log_display.append(log)
        self.llm_log_display.verticalScrollBar().setValue(self.llm_log_display.verticalScrollBar().maximum())

    def send_chat_query(self):
        query = self.chat_input.text().strip()
        if not query:
            return
        self.chat_display.append(f"User: {query}")
        self.chat_thread.add_chat_query(query)
        self.chat_input.clear()

    def display_chat_response(self, response):
        self.chat_display.append(f"AI: {response}")
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