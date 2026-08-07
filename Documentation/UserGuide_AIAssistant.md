# AI Assistant (Ollama Q&A) — User Guide

This guide explains how to use the **AI Assistant** panel that has been added to
ITK-SNAP. The panel provides a chat-style dialog with a locally running
[Ollama](https://ollama.com) server, allowing you to ask questions about the
image currently displayed in your ITK-SNAP viewport using Vision-Language
Models (VLMs).

---

## 1. Prerequisites

### 1.1 Install and Start Ollama
Ollama must be installed and running on your local machine, listening on its
default port `11434`.

```bash
# Install (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Start the server (usually already running as a service)
ollama serve
```

By convention, ITK-SNAP assumes the server is available at:

```
http://localhost:11434
```

### 1.2 Pull a Vision-Language Model
For image Q&A you need a multimodal model. Recommended options:

```bash
ollama pull gemma3          # Google Gemma 3 (vision-capable)
ollama pull llava           # Popular open VLM
ollama pull llama3.2-vision # Meta Llama 3.2 vision variant
```

The default model shown in the panel is **`gemma4`**. Change this in the model
combo box to any model you have pulled locally.

---

## 2. Opening the AI Assistant Panel

1. Launch ITK-SNAP as usual and load an image.
2. Go to the menu bar: **Views → AI Assistant (Ollama)**.
3. A dockable panel titled **"AI Assistant"** will appear on the right-hand
   side of the main window.

The panel can be:
- Dragged to the left or right dock area.
- Floated as an independent window.
- Closed via the **×** button (reopen it any time from the Views menu).

---

## 3. Panel Layout

| Control | Purpose |
|---------|---------|
| **Server** | Ollama server URL. Defaults to `http://localhost:11434`. |
| **Model** | Editable combo box. Type any locally-pulled model name (default `gemma4`). |
| **Chat Display** | Scrollable conversation history with streaming responses. |
| **Attach Current View** | When enabled, the currently displayed view is sent as an image with your prompt. |
| **View Source** | Choose which panel to capture: Axial, Sagittal, Coronal, or 3D View. |
| **Prompt Editor** | Multi-line text area to type your question. |
| **Clear Chat** | Wipes the conversation memory (starts a fresh session). |
| **Stop** | Aborts a response that is currently streaming. |
| **Send** | Submits your prompt (and image, if attached) to the model. |

---

## 4. Asking Questions

### 4.1 Text-Only Question
1. Uncheck **Attach Current View**.
2. Type a question in the prompt editor (e.g. *"What is the ITK image
   coordinate convention?"*).
3. Click **Send**.

### 4.2 Question About the Displayed Image
1. Ensure a slice or 3D rendering is visible in the ITK-SNAP viewport.
2. Check **Attach Current View**.
3. Pick the view to capture (Axial / Sagittal / Coronal / 3D View).
4. Type a question — e.g. *"Describe the anatomy visible in this slice."*
5. Click **Send**.

The image is captured from the ITK-SNAP viewport, downscaled to at most
1024 px, JPEG-encoded, and sent as a Base64 string with your prompt.

### 4.3 Follow-Up Questions
The panel remembers the full conversation, including any images previously
attached. Subsequent questions can leave **Attach Current View** unchecked to
save bandwidth — the model still has the earlier image in context.

Example dialog:

```
You:        [with image attached] What do you see in this scan?
Assistant:  I see an axial slice of a brain MRI, T1-weighted...
You:        Are there any visible ventricular abnormalities?
Assistant:  The lateral ventricles appear slightly enlarged...
```

### 4.4 Starting a New Conversation
Click **Clear Chat** to reset the history. This is recommended when moving to a
different scan or a completely different topic.

---

## 5. Streaming Responses
Responses are streamed token-by-token in real time, so text appears as the
model generates it. If a response is too long or off-topic, click **Stop** to
abort.

---

## 6. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `[Error contacting Ollama: Connection refused]` | Ollama is not running. Start it with `ollama serve`. |
| Response says the model does not exist | Pull the model: `ollama pull <name>`. |
| Model responds to text but not to images | The model is not multimodal. Switch to `llava`, `gemma3`, or `llama3.2-vision`. |
| Panel does not appear | Enable it via **Views → AI Assistant (Ollama)**. |
| Payload is very slow | Large images take time to encode/transmit. The panel already downscales to 1024 px; consider a smaller model. |

---

## 7. Privacy & Security
All communication happens over **localhost** with your local Ollama server. No
image, prompt, or response is sent to any external service. If you point the
**Server** field at a remote URL, ensure it is trusted — the field allows any
HTTP(S) endpoint for advanced users.

---

## 8. Technical Details (for developers)
- Endpoint used: `POST {server}/api/chat`
- Streaming: newline-delimited JSON (`stream: true`)
- Payload schema:
  ```json
  {
    "model": "gemma4",
    "messages": [
      { "role": "user", "content": "…", "images": ["<base64>"] },
      { "role": "assistant", "content": "…" }
    ],
    "stream": true
  }
  ```
- Source files:
  - `GUI/Qt/Components/OllamaQAWidget.{h,cxx,ui}`
  - Integrated in `GUI/Qt/Windows/MainImageWindow.{h,cxx}`
  - Requires `Qt6::Network` (added in `CMake/standalone.cmake`).
