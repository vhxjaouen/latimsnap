/*=========================================================================

  Program:   ITK-SNAP
  Module:    OllamaQAWidget.cxx
  Language:  C++

  This file is part of ITK-SNAP

  ITK-SNAP is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

=========================================================================*/
#include "OllamaQAWidget.h"
#include "ui_OllamaQAWidget.h"

#include "MainImageWindow.h"
#include "SliceViewPanel.h"
#include "ViewPanel3D.h"

#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QUrl>
#include <QBuffer>
#include <QPixmap>
#include <QTextCursor>
#include <QDateTime>
#include <QScrollBar>
#include <QMessageBox>
#include <QLineEdit>
#include <QPushButton>
#include <QComboBox>
#include <QStringList>
#include <QFileDialog>
#include <QFile>

OllamaQAWidget::OllamaQAWidget(QWidget *parent)
  : QWidget(parent)
  , ui(new Ui::OllamaQAWidget)
{
  ui->setupUi(this);

  // Placeholder items shown until we successfully talk to the server.
  ui->comboModel->addItem("(fetching models...)");
  ui->comboModel->setCurrentIndex(0);

  m_Network = new QNetworkAccessManager(this);

  connect(ui->comboBackend, QOverload<int>::of(&QComboBox::currentIndexChanged),
          this, &OllamaQAWidget::onBackendChanged);
  connect(ui->btnSend,  &QPushButton::clicked, this, &OllamaQAWidget::onSendClicked);
  connect(ui->btnStop,  &QPushButton::clicked, this, &OllamaQAWidget::onStopClicked);
  connect(ui->btnClear, &QPushButton::clicked, this, &OllamaQAWidget::onClearClicked);
  connect(ui->btnExportJson, &QPushButton::clicked, this, &OllamaQAWidget::onExportJsonClicked);
  connect(ui->btnRefreshModels, &QPushButton::clicked,
          this, &OllamaQAWidget::onRefreshModelsClicked);
  connect(ui->btnFreeGpu, &QPushButton::clicked,
          this, &OllamaQAWidget::onFreeGpuClicked);
  connect(ui->btnLoadVram, &QPushButton::clicked,
          this, &OllamaQAWidget::onLoadVramClicked);
  connect(ui->checkVisionOnly, &QCheckBox::toggled,
          this, &OllamaQAWidget::onVisionFilterToggled);
  connect(ui->editServer, &QLineEdit::editingFinished,
          this, &OllamaQAWidget::onServerUrlChanged);

  ui->chatDisplay->setHtml("<i style='color:gray'>AI Assistant ready. "
                           "Type a question and click Send. "
                           "Enable <b>Attach Current View</b> to send the "
                           "displayed view to a Vision-Language Model.</i>");

  // Attempt an initial model list fetch from the default server.
  fetchModels();
}

OllamaQAWidget::~OllamaQAWidget()
{
  if (m_Reply)
  {
    m_Reply->abort();
    m_Reply->deleteLater();
  }
  if (m_ModelsReply)
  {
    m_ModelsReply->abort();
    m_ModelsReply->deleteLater();
  }
  if (m_PsReply)
  {
    m_PsReply->abort();
    m_PsReply->deleteLater();
  }
  if (m_PrewarmReply)
  {
    m_PrewarmReply->abort();
    m_PrewarmReply->deleteLater();
  }
  delete ui;
}

void
OllamaQAWidget::SetMainImageWindow(MainImageWindow *win)
{
  m_MainWindow = win;
}

OllamaQAWidget::BackendType
OllamaQAWidget::getBackendType() const
{
  return (ui->comboBackend->currentIndex() == 0) ? BACKEND_VLLM_OPENAI : BACKEND_OLLAMA;
}

void
OllamaQAWidget::onBackendChanged(int index)
{
  QString currServer = ui->editServer->text().trimmed();
  if (index == 0) // vLLM / OpenAI API
  {
    if (currServer == "http://localhost:11434" || currServer.isEmpty())
      ui->editServer->setText("http://localhost:8000");
  }
  else // Ollama
  {
    if (currServer == "http://localhost:8000" || currServer.isEmpty())
      ui->editServer->setText("http://localhost:11434");
  }
  fetchModels();
}

void
OllamaQAWidget::setBusy(bool busy)
{
  ui->btnSend->setEnabled(!busy);
  ui->btnStop->setEnabled(busy);
  ui->editPrompt->setReadOnly(busy);
  ui->labelStatus->setText(busy ? "Generating..." : "Ready.");
}

QImage
OllamaQAWidget::grabCurrentView() const
{
  if (!m_MainWindow)
    return QImage();

  int viewIdx = ui->comboViewSource->currentIndex(); // 0=Axial,1=Sagittal,2=Coronal,3=3D
  QWidget *target = nullptr;

  if (viewIdx >= 0 && viewIdx <= 2)
  {
    // Map: Axial=2, Coronal=1, Sagittal=0 in ITK-SNAP convention.
    // We take the panel index directly matching the combo order for simplicity.
    // Panels 0,1,2 correspond to the three orthogonal slice views.
    unsigned int panelIndex = static_cast<unsigned int>(viewIdx);
    SliceViewPanel *panel = m_MainWindow->GetSlicePanel(panelIndex);
    target = panel;
  }
  else
  {
    // 3D view - fall back to grabbing the whole main window central area
    target = m_MainWindow->centralWidget();
  }

  if (!target)
    return QImage();

  QPixmap pix = target->grab();
  return pix.toImage();
}

QString
OllamaQAWidget::encodeImageBase64(const QImage &img) const
{
  if (img.isNull())
    return QString();

  // Scale down for reasonable payload sizes if very large
  QImage scaled = img;
  const int maxDim = 1024;
  if (scaled.width() > maxDim || scaled.height() > maxDim)
    scaled = scaled.scaled(maxDim, maxDim, Qt::KeepAspectRatio, Qt::SmoothTransformation);

  QByteArray bytes;
  QBuffer buffer(&bytes);
  buffer.open(QIODevice::WriteOnly);
  scaled.save(&buffer, "JPEG", 85);
  return QString::fromLatin1(bytes.toBase64());
}

void
OllamaQAWidget::appendMessageToDisplay(const QString &role, const QString &text)
{
  QString color = (role == "user") ? "#0060c0" : "#008040";
  QString label = (role == "user") ? "You"      : "Assistant";
  QString safe  = text.toHtmlEscaped().replace("\n", "<br>");
  QString html  = QString("<div style='margin:6px 0;'>"
                          "<b style='color:%1'>%2:</b><br>%3"
                          "</div>")
                    .arg(color, label, safe);
  ui->chatDisplay->append(html);
  QScrollBar *sb = ui->chatDisplay->verticalScrollBar();
  if (sb) sb->setValue(sb->maximum());
}

void
OllamaQAWidget::updateStreamingAssistantText(const QString &delta)
{
  m_CurrentAssistantText += delta;

  QTextCursor cursor = ui->chatDisplay->textCursor();
  cursor.movePosition(QTextCursor::End);
  cursor.insertText(delta);
  QScrollBar *sb = ui->chatDisplay->verticalScrollBar();
  if (sb) sb->setValue(sb->maximum());
}

void
OllamaQAWidget::updateStreamingThinkingText(const QString &delta)
{
  m_CurrentThinkingText += delta;

  QTextCursor cursor = ui->thinkingDisplay->textCursor();
  cursor.movePosition(QTextCursor::End);
  cursor.insertText(delta);
  QScrollBar *sb = ui->thinkingDisplay->verticalScrollBar();
  if (sb) sb->setValue(sb->maximum());

  ui->tabWidgetMain->setTabText(1, QString("🧠 Thinking (%1 chars)").arg(m_CurrentThinkingText.length()));
}

void
OllamaQAWidget::appendVerbosityLog(const QString &htmlOrText)
{
  ui->verbosityDisplay->append(htmlOrText);
  QScrollBar *sb = ui->verbosityDisplay->verticalScrollBar();
  if (sb) sb->setValue(sb->maximum());
}

void
OllamaQAWidget::onSendClicked()
{
  if (m_Reply)
    return; // already in progress

  QString prompt = ui->editPrompt->toPlainText().trimmed();
  if (prompt.isEmpty())
    return;

  QString model  = ui->comboModel->currentText().trimmed();
  QString server = ui->editServer->text().trimmed();
  if (model.isEmpty() || server.isEmpty())
  {
    QMessageBox::warning(this, "AI Assistant",
                         "Please provide both a server URL and a model name.");
    return;
  }

  // Build the user message
  QJsonObject userMsg;
  userMsg["role"]    = "user";
  userMsg["content"] = prompt;

  int imgBytes = 0;
  int imgMaxDim = 0;

  if (ui->checkAttachView->isChecked())
  {
    QImage img = grabCurrentView();
    if (!img.isNull())
    {
      imgMaxDim = qMax(img.width(), img.height());
      QString b64 = encodeImageBase64(img);
      if (!b64.isEmpty())
      {
        imgBytes = b64.length() * 3 / 4; // approximate raw JPEG bytes
        QJsonArray imgArr;
        imgArr.append(b64);
        userMsg["images"] = imgArr;
      }
    }
  }

  m_Messages.append(userMsg);

  // Display the user's prompt (without embedding the base64)
  appendMessageToDisplay("user", prompt);
  ui->editPrompt->clear();

  // Prepare a placeholder for the assistant response
  m_CurrentAssistantText.clear();
  m_CurrentThinkingText.clear();
  ui->thinkingDisplay->clear();
  ui->tabWidgetMain->setTabText(1, "🧠 Thinking");

  QString color = "#008040";
  QString html  = QString("<div style='margin:6px 0;'>"
                          "<b style='color:%1'>Assistant:</b><br></div>")
                    .arg(color);
  ui->chatDisplay->append(html);

  m_RequestStartMs = QDateTime::currentMSecsSinceEpoch();
  m_FirstTokenMs = 0;

  QJsonObject body;
  QUrl requestUrl;
  QJsonArray prunedMessages;

  if (getBackendType() == BACKEND_VLLM_OPENAI)
  {
    // OpenAI / vLLM chat completions schema
    for (int i = 0; i < m_Messages.size(); ++i)
    {
      QJsonObject msg = m_Messages[i].toObject();
      QString role = msg["role"].toString();
      QString contentStr = msg["content"].toString();

      QJsonObject openAiMsg;
      openAiMsg["role"] = role;

      // Only attach vision image array on the current user turn to avoid prefill latency explosion
      if (role == "user" && i == m_Messages.size() - 1 && msg.contains("images"))
      {
        QJsonArray contentArr;
        QJsonObject textObj;
        textObj["type"] = "text";
        textObj["text"] = contentStr;
        contentArr.append(textObj);

        QJsonArray imgArr = msg["images"].toArray();
        for (const QJsonValue &iv : imgArr)
        {
          QJsonObject imgObj;
          imgObj["type"] = "image_url";
          QJsonObject urlObj;
          urlObj["url"] = QString("data:image/jpeg;base64,%1").arg(iv.toString());
          imgObj["image_url"] = urlObj;
          contentArr.append(imgObj);
        }
        openAiMsg["content"] = contentArr;
      }
      else
      {
        openAiMsg["content"] = contentStr;
      }
      prunedMessages.append(openAiMsg);
    }

    body["model"]    = model;
    body["messages"] = prunedMessages;
    body["stream"]   = true;

    QString chatUrlStr = server;
    if (!chatUrlStr.endsWith("/v1") && !chatUrlStr.contains("/v1/"))
    {
      if (chatUrlStr.endsWith("/"))
        chatUrlStr += "v1/chat/completions";
      else
        chatUrlStr += "/v1/chat/completions";
    }
    else if (!chatUrlStr.endsWith("/chat/completions"))
    {
      if (chatUrlStr.endsWith("/"))
        chatUrlStr += "chat/completions";
      else
        chatUrlStr += "/chat/completions";
    }
    requestUrl = QUrl(chatUrlStr);
  }
  else
  {
    // Ollama native chat schema
    for (int i = 0; i < m_Messages.size(); ++i)
    {
      QJsonObject msg = m_Messages[i].toObject();
      // Only keep images on the VERY LAST message (current turn)
      if (i < m_Messages.size() - 1 && msg.contains("images"))
      {
        msg.remove("images");
      }
      prunedMessages.append(msg);
    }

    body["model"]    = model;
    body["messages"] = prunedMessages;
    body["stream"]   = true;

    bool autoRelease = ui->checkAutoRelease->isChecked();
    if (autoRelease)
      body["keep_alive"] = 0;

    bool disableThinking = ui->checkDisableThinking->isChecked();
    if (disableThinking)
    {
      body["think"] = false;
      QJsonObject opts;
      opts["think"] = false;
      body["options"] = opts;
    }

    requestUrl = QUrl(server + "/api/chat");
  }

  QJsonDocument doc(body);
  QByteArray payload = doc.toJson(QJsonDocument::Compact);

  // Pre-flight verbosity logging
  QString nowStr = QDateTime::currentDateTime().toString("hh:mm:ss");
  QString reqLog = QString(
    "<hr><b style='color:#0060c0;'>[%1 REQUEST - %2]</b> <b>Model:</b> %3 | <b>Server:</b> %4<br>"
    "<b>Payload size:</b> %5 KB (%6 messages in context)%7")
    .arg(nowStr)
    .arg(getBackendType() == BACKEND_VLLM_OPENAI ? "vLLM / OpenAI API" : "Ollama Native API")
    .arg(model, requestUrl.toString())
    .arg(payload.size() / 1024.0, 0, 'f', 1)
    .arg(prunedMessages.size())
    .arg(imgBytes > 0 ? QString(" | Attached view: %1 KB JPEG, max dim %2px").arg(imgBytes / 1024).arg(imgMaxDim) : "");

  appendVerbosityLog(reqLog);

  QNetworkRequest req(requestUrl);
  req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

  m_LineBuffer.clear();
  m_InFlightModel = model;
  m_Reply = m_Network->post(req, payload);

  connect(m_Reply, &QNetworkReply::readyRead, this, &OllamaQAWidget::onReadyRead);
  connect(m_Reply, &QNetworkReply::finished,  this, &OllamaQAWidget::onFinished);
  connect(m_Reply, &QNetworkReply::errorOccurred,
          this, &OllamaQAWidget::onErrorOccurred);

  setBusy(true);
}

void
OllamaQAWidget::onReadyRead()
{
  if (!m_Reply)
    return;

  m_LineBuffer += m_Reply->readAll();

  // Ollama streams newline-delimited JSON objects
  while (true)
  {
    int nl = m_LineBuffer.indexOf('\n');
    if (nl < 0)
      break;
    QByteArray line = m_LineBuffer.left(nl).trimmed();
    m_LineBuffer.remove(0, nl + 1);

    if (line.isEmpty())
      continue;

    QJsonParseError err;
    QJsonDocument doc = QJsonDocument::fromJson(line, &err);
    if (err.error != QJsonParseError::NoError || !doc.isObject())
      continue;

    QJsonObject obj = doc.object();

    // Ollama /api/chat returns {"message":{"role":"assistant","content":"..."}, "done":false}
    if (obj.contains("message"))
    {
      QJsonObject msg = obj["message"].toObject();

      // Check if thinking content is present in message.think (e.g. DeepSeek-R1 / Qwen 3.6 / Ollama reasoning)
      if (msg.contains("think"))
      {
        QString thinkingDelta = msg["think"].toString();
        if (!thinkingDelta.isEmpty())
          updateStreamingThinkingText(thinkingDelta);
      }

      QString delta = msg["content"].toString();
      if (!delta.isEmpty())
      {
        // Some models embed thinking tokens directly inside <think>...</think> tags in content
        if (delta.contains("<think>"))
        {
          int startIdx = delta.indexOf("<think>");
          if (startIdx > 0)
            updateStreamingAssistantText(delta.left(startIdx));
          QString thinkPart = delta.mid(startIdx + 7);
          updateStreamingThinkingText(thinkPart);
        }
        else if (delta.contains("</think>"))
        {
          int endIdx = delta.indexOf("</think>");
          updateStreamingThinkingText(delta.left(endIdx));
          QString contentPart = delta.mid(endIdx + 8);
          if (!contentPart.isEmpty())
            updateStreamingAssistantText(contentPart);
        }
        else
        {
          updateStreamingAssistantText(delta);
        }
      }
    }

    if (obj["done"].toBool(false))
    {
      // Persist assistant message into history
      QJsonObject assistantMsg;
      assistantMsg["role"]    = "assistant";
      assistantMsg["content"] = m_CurrentAssistantText;
      if (!m_CurrentThinkingText.isEmpty())
        assistantMsg["thinking"] = m_CurrentThinkingText;
      m_Messages.append(assistantMsg);

      // Extract telemetry
      double totalSec  = obj.value("total_duration").toDouble(0) / 1e9;
      double loadSec   = obj.value("load_duration").toDouble(0) / 1e9;
      qint64 promptTok = static_cast<qint64>(obj.value("prompt_eval_count").toDouble(0));
      double promptSec = obj.value("prompt_eval_duration").toDouble(0) / 1e9;
      qint64 genTok    = static_cast<qint64>(obj.value("eval_count").toDouble(0));
      double genSec    = obj.value("eval_duration").toDouble(0) / 1e9;
      QString doneReason = obj.value("done_reason").toString("stop");

      double promptTokPerSec = (promptSec > 0.001) ? (promptTok / promptSec) : 0.0;
      double genTokPerSec    = (genSec > 0.001)    ? (genTok / genSec)       : 0.0;

      // Store in turn metrics history for JSON export
      QJsonObject metricsObj;
      metricsObj["total_duration_s"]      = totalSec;
      metricsObj["load_duration_s"]       = loadSec;
      metricsObj["prompt_eval_count"]     = promptTok;
      metricsObj["prompt_eval_duration_s"] = promptSec;
      metricsObj["prompt_eval_tok_s"]     = promptTokPerSec;
      metricsObj["eval_count"]            = genTok;
      metricsObj["eval_duration_s"]        = genSec;
      metricsObj["eval_tok_s"]            = genTokPerSec;
      metricsObj["done_reason"]           = doneReason;
      m_TurnMetricsHistory.append(metricsObj);

      QString loadNotice;
      if (loadSec > 2.0)
        loadNotice = QString(" <span style='color:#c00000;'>(Cold VRAM load: %1s)</span>").arg(loadSec, 0, 'f', 1);

      QString perfLog = QString(
        "<b style='color:#008040;'>[TELEMETRY]</b> <b>Total:</b> %1 s | "
        "<b>Load VRAM:</b> %2 s%3 | "
        "<b>Prefill:</b> %4 tok in %5 s (<b>%6 tok/s</b>) | "
        "<b>Generation:</b> %7 tok in %8 s (<b>%9 tok/s</b>) | "
        "<b>Reason:</b> %10")
        .arg(totalSec, 0, 'f', 1)
        .arg(loadSec, 0, 'f', 1)
        .arg(loadNotice)
        .arg(promptTok)
        .arg(promptSec, 0, 'f', 1)
        .arg(promptTokPerSec, 0, 'f', 1)
        .arg(genTok)
        .arg(genSec, 0, 'f', 1)
        .arg(genTokPerSec, 0, 'f', 1)
        .arg(doneReason);

      appendVerbosityLog(perfLog);

      ui->labelStatus->setText(
        QString("Done in %1s | VRAM Load: %2s | Prefill: %3 tok (%4 t/s) | Gen: %5 tok (%6 t/s)")
          .arg(totalSec, 0, 'f', 1)
          .arg(loadSec, 0, 'f', 1)
          .arg(promptTok)
          .arg(promptTokPerSec, 0, 'f', 1)
          .arg(genTok)
          .arg(genTokPerSec, 0, 'f', 1));
    }
  }
}

void
OllamaQAWidget::onFinished()
{
  if (!m_Reply)
    return;

  // Flush any remaining buffered data
  onReadyRead();

  m_Reply->deleteLater();
  m_Reply = nullptr;
  m_InFlightModel.clear();
  setBusy(false);
}

void
OllamaQAWidget::onErrorOccurred()
{
  if (!m_Reply)
    return;

  QString err = m_Reply->errorString();
  appendMessageToDisplay("assistant",
                         QString("[Error contacting Ollama: %1]").arg(err));
  ui->labelStatus->setText("Error: " + err);
}

void
OllamaQAWidget::onStopClicked()
{
  if (m_Reply)
  {
    m_Reply->abort();
  }
  // Aborting the QNetworkReply only closes the client side of the stream;
  // Ollama's server keeps generating until it hits done, and the model
  // remains resident in VRAM. Force an explicit unload so the GPU is freed
  // as soon as the abort is acknowledged.
  if (!m_InFlightModel.isEmpty())
  {
    requestModelUnload(m_InFlightModel);
    m_InFlightModel.clear();
  }
}

void
OllamaQAWidget::onClearClicked()
{
  m_Messages = QJsonArray();
  m_TurnMetricsHistory = QJsonArray();
  m_CurrentAssistantText.clear();
  m_CurrentThinkingText.clear();
  ui->chatDisplay->clear();
  ui->thinkingDisplay->clear();
  ui->tabWidgetMain->setTabText(1, "🧠 Thinking");
  ui->chatDisplay->setHtml("<i style='color:gray'>Chat history cleared.</i>");
  appendVerbosityLog("<i style='color:gray'>Conversation history and metrics reset.</i>");
}

void
OllamaQAWidget::onRefreshModelsClicked()
{
  fetchModels();
}

void
OllamaQAWidget::onServerUrlChanged()
{
  fetchModels();
}

void
OllamaQAWidget::fetchModels()
{
  QString server = ui->editServer->text().trimmed();
  if (server.isEmpty())
  {
    ui->labelStatus->setText("Error: server URL is empty.");
    return;
  }

  // Cancel any in-flight model list request.
  if (m_ModelsReply)
  {
    m_ModelsReply->disconnect(this);
    m_ModelsReply->abort();
    m_ModelsReply->deleteLater();
    m_ModelsReply = nullptr;
  }

  m_AllModels.clear();
  m_VisionModels.clear();
  m_PendingShowRequests = 0;
  m_TotalShowRequests = 0;

  ui->labelStatus->setText("Contacting server...");
  ui->btnRefreshModels->setEnabled(false);

  if (getBackendType() == BACKEND_VLLM_OPENAI)
  {
    // OpenAI / vLLM standard models endpoint: GET /v1/models
    QString modelsUrl = server;
    if (!modelsUrl.endsWith("/v1") && !modelsUrl.contains("/v1/"))
    {
      if (modelsUrl.endsWith("/"))
        modelsUrl += "v1/models";
      else
        modelsUrl += "/v1/models";
    }
    else if (!modelsUrl.endsWith("/models"))
    {
      if (modelsUrl.endsWith("/"))
        modelsUrl += "models";
      else
        modelsUrl += "/models";
    }

    QUrl url(modelsUrl);
    QNetworkRequest req(url);
    m_ModelsReply = m_Network->get(req);

    connect(m_ModelsReply, &QNetworkReply::finished,
            this, &OllamaQAWidget::onModelsFetched);
    connect(m_ModelsReply, &QNetworkReply::errorOccurred,
            this, &OllamaQAWidget::onModelsFetchError);
  }
  else
  {
    // Ollama native tags endpoint: GET /api/tags
    QUrl url(server + "/api/tags");
    QNetworkRequest req(url);
    m_ModelsReply = m_Network->get(req);

    connect(m_ModelsReply, &QNetworkReply::finished,
            this, &OllamaQAWidget::onModelsFetched);
    connect(m_ModelsReply, &QNetworkReply::errorOccurred,
            this, &OllamaQAWidget::onModelsFetchError);
  }
}

void
OllamaQAWidget::onModelsFetched()
{
  if (!m_ModelsReply)
    return;

  // If the reply errored out, onModelsFetchError already reported it; but
  // this finished() slot still fires. Bail out if there was an error.
  if (m_ModelsReply->error() != QNetworkReply::NoError)
  {
    ui->btnRefreshModels->setEnabled(true);
    m_ModelsReply->deleteLater();
    m_ModelsReply = nullptr;
    return;
  }

  QByteArray data = m_ModelsReply->readAll();
  m_ModelsReply->deleteLater();
  m_ModelsReply = nullptr;

  QJsonParseError err;
  QJsonDocument doc = QJsonDocument::fromJson(data, &err);
  if (err.error != QJsonParseError::NoError || !doc.isObject())
  {
    ui->btnRefreshModels->setEnabled(true);
    ui->labelStatus->setText("Error: invalid response from server.");
    return;
  }

  m_AllModels.clear();
  m_VisionModels.clear();

  if (getBackendType() == BACKEND_VLLM_OPENAI)
  {
    // OpenAI / vLLM response: {"object": "list", "data": [{"id": "model_id"}, ...]}
    QJsonArray dataArr = doc.object().value("data").toArray();
    for (const QJsonValue &v : dataArr)
    {
      QJsonObject o = v.toObject();
      QString id = o.value("id").toString();
      if (!id.isEmpty())
      {
        m_AllModels << id;
      }
    }

    // vLLM / OpenAI backend: do NOT apply any heuristic filtering on model
    // names. The /v1/models endpoint does not expose vision capabilities in
    // a standardised way and any name-based guess is unreliable. All served
    // models are considered "vision-capable" from the widget's point of view
    // so that the "Vision Only" filter behaves identically to "show all".
    m_VisionModels = m_AllModels;

    ui->btnRefreshModels->setEnabled(true);
    updateModelCombo();
    return;
  }

  // Ollama response: {"models": [{"name": "...", "details": ...}]}
  QJsonArray models = doc.object().value("models").toArray();

  for (const QJsonValue &v : models)
  {
    QJsonObject o = v.toObject();
    QString name = o.value("name").toString();
    if (!name.isEmpty())
      m_AllModels << name;
  }

  if (m_AllModels.isEmpty())
  {
    ui->btnRefreshModels->setEnabled(true);
    updateModelCombo();
    return;
  }

  // Inspect each model's capacity via POST /api/show to check for vision support.
  QString server = ui->editServer->text().trimmed();
  m_TotalShowRequests = m_AllModels.size();
  m_PendingShowRequests = m_AllModels.size();

  ui->labelStatus->setText(
    QString("Checking model capabilities (0/%1)...").arg(m_TotalShowRequests));

  for (const QString &modelName : m_AllModels)
  {
    QJsonObject body;
    body["name"] = modelName;

    QUrl showUrl(server + "/api/show");
    QNetworkRequest showReq(showUrl);
    showReq.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    QNetworkReply *reply = m_Network->post(
      showReq, QJsonDocument(body).toJson(QJsonDocument::Compact));
    reply->setProperty("modelName", modelName);

    connect(reply, &QNetworkReply::finished,
            this, &OllamaQAWidget::onShowFetched);
  }
}

void
OllamaQAWidget::onShowFetched()
{
  QNetworkReply *reply = qobject_cast<QNetworkReply *>(sender());
  if (!reply)
    return;

  QString modelName = reply->property("modelName").toString();
  QByteArray data = reply->readAll();
  reply->deleteLater();

  QJsonParseError err;
  QJsonDocument doc = QJsonDocument::fromJson(data, &err);
  if (err.error == QJsonParseError::NoError && doc.isObject())
  {
    if (parseVisionCapability(doc.object()))
    {
      if (!m_VisionModels.contains(modelName))
        m_VisionModels << modelName;
    }
  }

  m_PendingShowRequests--;
  if (m_PendingShowRequests <= 0)
  {
    m_PendingShowRequests = 0;
    ui->btnRefreshModels->setEnabled(true);
    updateModelCombo();
  }
  else
  {
    int done = m_TotalShowRequests - m_PendingShowRequests;
    ui->labelStatus->setText(
      QString("Checking model capabilities (%1/%2)...")
        .arg(done).arg(m_TotalShowRequests));
  }
}

bool
OllamaQAWidget::parseVisionCapability(const QJsonObject &obj) const
{
  // 1. Primary check: Ollama's explicit "capabilities" array
  if (obj.contains("capabilities"))
  {
    QJsonArray caps = obj.value("capabilities").toArray();
    for (const QJsonValue &v : caps)
    {
      if (v.toString().trimmed().compare("vision", Qt::CaseInsensitive) == 0)
        return true;
    }
  }

  // 2. Fallback check: inspect details.family or details.families for known vision architectures
  if (obj.contains("details"))
  {
    QJsonObject details = obj.value("details").toObject();
    QString family = details.value("family").toString().toLower();
    QJsonArray families = details.value("families").toArray();

    QStringList visionFamilies = {
      "clip", "mllama", "llava", "llava-llama", "llava_next",
      "qwen2vl", "qwen2_vl", "minicpmv", "paligemma", "pixtral",
      "glm4v", "moondream", "vision", "vlm"
    };

    for (const QString &vf : visionFamilies)
    {
      if (family.contains(vf))
        return true;
    }

    for (const QJsonValue &fv : families)
    {
      QString fName = fv.toString().toLower();
      for (const QString &vf : visionFamilies)
      {
        if (fName.contains(vf))
          return true;
      }
    }
  }

  // 3. Fallback check: GGUF model_info tensor keys
  if (obj.contains("model_info"))
  {
    QJsonObject info = obj.value("model_info").toObject();
    for (auto it = info.begin(); it != info.end(); ++it)
    {
      if (it.key().contains(".vision.", Qt::CaseInsensitive) ||
          it.key().contains("mm.mm_input_projection", Qt::CaseInsensitive))
        return true;
    }
  }

  return false;
}

void
OllamaQAWidget::onVisionFilterToggled()
{
  updateModelCombo();
}

void
OllamaQAWidget::updateModelCombo()
{
  bool visionOnly = ui->checkVisionOnly->isChecked();
  const QStringList &list = visionOnly ? m_VisionModels : m_AllModels;

  QString previous = ui->comboModel->currentText().trimmed();
  ui->comboModel->clear();

  if (list.isEmpty())
  {
    if (m_AllModels.isEmpty())
    {
      ui->comboModel->addItem("(no models installed)");
      ui->labelStatus->setText(
        "Connected, but no models are installed on the server.");
    }
    else if (visionOnly)
    {
      ui->comboModel->addItem("(no vision models found)");
      ui->labelStatus->setText(
        QString("Connected. %1 total model(s) installed, but none have vision capacity.")
          .arg(m_AllModels.size()));
    }
    return;
  }

  ui->comboModel->addItems(list);

  int idx = list.indexOf(previous);
  if (idx >= 0)
    ui->comboModel->setCurrentIndex(idx);
  else
    ui->comboModel->setCurrentIndex(0);

  if (visionOnly)
  {
    ui->labelStatus->setText(
      QString("Connected. %1 vision model(s) available (%2 total installed).")
        .arg(m_VisionModels.size()).arg(m_AllModels.size()));
  }
  else
  {
    ui->labelStatus->setText(
      QString("Connected. %1 model(s) available.").arg(m_AllModels.size()));
  }
}

void
OllamaQAWidget::onModelsFetchError()
{
  if (!m_ModelsReply)
    return;

  QString err = m_ModelsReply->errorString();
  ui->labelStatus->setText(QString("Error contacting Ollama: %1").arg(err));
  ui->btnRefreshModels->setEnabled(true);

  // finished() will still fire and take care of deleteLater.
}

// ===========================================================================
// Free GPU: query /api/ps to enumerate loaded models, then unload each.
// ===========================================================================

void
OllamaQAWidget::onFreeGpuClicked()
{
  QString server = ui->editServer->text().trimmed();
  if (server.isEmpty())
  {
    ui->labelStatus->setText("Error: server URL is empty.");
    return;
  }

  if (m_PsReply)
  {
    // A previous free-GPU sequence is already probing the server.
    return;
  }

  ui->btnFreeGpu->setEnabled(false);
  ui->labelStatus->setText("Querying loaded models...");

  QUrl url(server + "/api/ps");
  QNetworkRequest req(url);
  m_PsReply = m_Network->get(req);

  connect(m_PsReply, &QNetworkReply::finished,
          this, &OllamaQAWidget::onPsFetched);
}

void
OllamaQAWidget::onPsFetched()
{
  if (!m_PsReply)
    return;

  QNetworkReply::NetworkError netErr = m_PsReply->error();
  QByteArray data = m_PsReply->readAll();
  m_PsReply->deleteLater();
  m_PsReply = nullptr;

  if (netErr != QNetworkReply::NoError)
  {
    ui->labelStatus->setText("Error: could not query /api/ps.");
    ui->btnFreeGpu->setEnabled(true);
    return;
  }

  QJsonParseError err;
  QJsonDocument doc = QJsonDocument::fromJson(data, &err);
  if (err.error != QJsonParseError::NoError || !doc.isObject())
  {
    ui->labelStatus->setText("Error: invalid response from /api/ps.");
    ui->btnFreeGpu->setEnabled(true);
    return;
  }

  QJsonArray models = doc.object().value("models").toArray();
  if (models.isEmpty())
  {
    ui->labelStatus->setText("Nothing to unload — no models resident.");
    ui->btnFreeGpu->setEnabled(true);
    return;
  }

  m_PendingUnloads = 0;
  m_UnloadedSummary.clear();

  for (const QJsonValue &v : models)
  {
    QJsonObject o = v.toObject();
    QString name = o.value("name").toString();
    if (name.isEmpty())
      continue;

    qint64 bytes = static_cast<qint64>(o.value("size_vram").toDouble(0.0));
    QString sizeStr;
    if (bytes >= (1LL << 30))
      sizeStr = QString("%1 GB").arg(bytes / double(1LL << 30), 0, 'f', 1);
    else if (bytes > 0)
      sizeStr = QString("%1 MB").arg(bytes / double(1LL << 20), 0, 'f', 0);

    m_UnloadedSummary << (sizeStr.isEmpty()
                            ? name
                            : QString("%1 (%2)").arg(name, sizeStr));
    m_PendingUnloads += 1;
    requestModelUnload(name);
  }

  if (m_PendingUnloads == 0)
  {
    ui->labelStatus->setText("Nothing to unload.");
    ui->btnFreeGpu->setEnabled(true);
  }
}

void
OllamaQAWidget::requestModelUnload(const QString &modelName)
{
  QString server = ui->editServer->text().trimmed();
  if (server.isEmpty() || modelName.isEmpty())
    return;

  // POST /api/generate with keep_alive=0 and an empty prompt is Ollama's
  // documented way to unload a model without a chat turn. The server
  // responds almost immediately.
  QJsonObject body;
  body["model"]      = modelName;
  body["prompt"]     = "";
  body["keep_alive"] = 0;
  body["stream"]     = false;

  QUrl url(server + "/api/generate");
  QNetworkRequest req(url);
  req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

  QNetworkReply *reply = m_Network->post(
    req, QJsonDocument(body).toJson(QJsonDocument::Compact));
  connect(reply, &QNetworkReply::finished,
          this,  &OllamaQAWidget::onUnloadFinished);
}

void
OllamaQAWidget::onUnloadFinished()
{
  QNetworkReply *reply = qobject_cast<QNetworkReply *>(sender());
  if (reply)
    reply->deleteLater();

  if (m_PendingUnloads > 0)
    m_PendingUnloads -= 1;

  if (m_PendingUnloads <= 0)
  {
    m_PendingUnloads = 0;
    ui->btnFreeGpu->setEnabled(true);
    if (!m_UnloadedSummary.isEmpty())
    {
      ui->labelStatus->setText(
        QString("Freed VRAM: %1").arg(m_UnloadedSummary.join(", ")));
      m_UnloadedSummary.clear();
    }
  }
}

// ===========================================================================
// Manual VRAM Load / Pre-warm with automatic previous model offload
// ===========================================================================

void
OllamaQAWidget::onLoadVramClicked()
{
  QString model = ui->comboModel->currentText().trimmed();
  QString server = ui->editServer->text().trimmed();

  if (model.isEmpty() || server.isEmpty())
  {
    QMessageBox::warning(this, "AI Assistant",
                         "Please select a model and server first.");
    return;
  }

  // If a different model was previously loaded in VRAM, offload it first.
  if (!m_LoadedVramModel.isEmpty() && m_LoadedVramModel != model)
  {
    appendVerbosityLog(
      QString("<b style='color:#c00000;'>[VRAM]</b> Offloading previous model '%1'...")
        .arg(m_LoadedVramModel));
    requestModelUnload(m_LoadedVramModel);
  }

  ui->btnLoadVram->setEnabled(false);
  ui->labelStatus->setText(QString("Pre-warming %1 into VRAM...").arg(model));

  appendVerbosityLog(
    QString("<b style='color:#0060c0;'>[VRAM]</b> Pre-warming model '%1' into GPU memory (keep_alive=-1)...")
      .arg(model));

  // POST /api/generate with keep_alive=-1 (or 10m) and empty prompt pre-loads the model.
  QJsonObject body;
  body["model"]      = model;
  body["prompt"]     = "";
  body["keep_alive"] = -1; // keep in VRAM until explicitly offloaded
  body["stream"]     = false;

  QUrl url(server + "/api/generate");
  QNetworkRequest req(url);
  req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

  if (m_PrewarmReply)
  {
    m_PrewarmReply->abort();
    m_PrewarmReply->deleteLater();
  }

  m_PrewarmReply = m_Network->post(
    req, QJsonDocument(body).toJson(QJsonDocument::Compact));
  m_PrewarmReply->setProperty("modelName", model);

  connect(m_PrewarmReply, &QNetworkReply::finished,
          this, &OllamaQAWidget::onPrewarmFinished);
}

void
OllamaQAWidget::onPrewarmFinished()
{
  ui->btnLoadVram->setEnabled(true);

  if (!m_PrewarmReply)
    return;

  QString modelName = m_PrewarmReply->property("modelName").toString();
  QNetworkReply::NetworkError err = m_PrewarmReply->error();
  QByteArray data = m_PrewarmReply->readAll();
  m_PrewarmReply->deleteLater();
  m_PrewarmReply = nullptr;

  if (err != QNetworkReply::NoError)
  {
    ui->labelStatus->setText(QString("Failed to pre-warm %1.").arg(modelName));
    appendVerbosityLog(
      QString("<b style='color:#c00000;'>[VRAM Error]</b> Could not pre-warm %1.").arg(modelName));
    return;
  }

  m_LoadedVramModel = modelName;

  // Extract load duration if available
  QJsonDocument doc = QJsonDocument::fromJson(data);
  double loadSec = 0.0;
  if (doc.isObject())
    loadSec = doc.object().value("load_duration").toDouble(0) / 1e9;

  ui->labelStatus->setText(
    QString("Model %1 resident in VRAM (%2s load).").arg(modelName).arg(loadSec, 0, 'f', 1));

  appendVerbosityLog(
    QString("<b style='color:#008040;'>[VRAM Ready]</b> Model '%1' is now pre-warmed in GPU memory (Load time: %2s).")
      .arg(modelName).arg(loadSec, 0, 'f', 1));
}

// ===========================================================================
// JSON Conversation Export
// ===========================================================================

void
OllamaQAWidget::onExportJsonClicked()
{
  if (m_Messages.isEmpty())
  {
    QMessageBox::information(this, "Export JSON",
                            "There are no messages in the conversation to export.");
    return;
  }

  QString defaultFileName = QString("ollama_chat_%1.json")
    .arg(QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss"));

  QString filePath = QFileDialog::getSaveFileName(
    this, tr("Export Conversation JSON"), defaultFileName, tr("JSON Files (*.json)"));

  if (filePath.isEmpty())
    return;

  QJsonObject sessionObj;
  sessionObj["export_timestamp"]  = QDateTime::currentDateTime().toString(Qt::ISODate);
  sessionObj["server"]            = ui->editServer->text().trimmed();
  sessionObj["model"]             = ui->comboModel->currentText().trimmed();
  sessionObj["disable_thinking"]  = ui->checkDisableThinking->isChecked();
  sessionObj["auto_release_gpu"]  = ui->checkAutoRelease->isChecked();
  sessionObj["messages"]          = m_Messages;
  sessionObj["thinking_log"]      = m_CurrentThinkingText;
  sessionObj["turn_metrics"]      = m_TurnMetricsHistory;

  QJsonDocument doc(sessionObj);
  QFile file(filePath);
  if (!file.open(QIODevice::WriteOnly))
  {
    QMessageBox::critical(this, "Export JSON Error",
                          QString("Could not open file for writing: %1").arg(file.errorString()));
    return;
  }

  file.write(doc.toJson(QJsonDocument::Indented));
  file.close();

  ui->labelStatus->setText(QString("Exported conversation to %1").arg(QFileInfo(filePath).fileName()));
  appendVerbosityLog(
    QString("<b style='color:#008040;'>[EXPORT]</b> Conversation exported to: %1").arg(filePath));
}
