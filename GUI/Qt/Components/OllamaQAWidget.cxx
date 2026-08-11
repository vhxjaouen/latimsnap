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
#include "QtFrameBufferOpenGLWidget.h"
#include "SNAPCommon.h"
#include "GlobalUIModel.h"
#include "IRISApplication.h"

#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QUrl>
#include <QBuffer>
#include <QPixmap>
#include <QPainter>
#include <QDateTime>
#include <QScrollBar>
#include <QMessageBox>
#include <QTimer>
#include <QSettings>

OllamaQAWidget::OllamaQAWidget(QWidget *parent)
  : QWidget(parent)
  , ui(new Ui::OllamaQAWidget)
  , m_Backend(0)
  , m_IsModelFetch(false)
{
  ui->setupUi(this);

  m_Network = new QNetworkAccessManager(this);

  m_RenderTimer = new QTimer(this);
  m_RenderTimer->setSingleShot(true);
  m_RenderTimer->setInterval(70);
  connect(m_RenderTimer, &QTimer::timeout, this, &OllamaQAWidget::renderConversation);

  connect(ui->btnSend,  &QPushButton::clicked, this, &OllamaQAWidget::onSendClicked);
  connect(ui->btnStop,  &QPushButton::clicked, this, &OllamaQAWidget::onStopClicked);
  connect(ui->btnClear, &QPushButton::clicked, this, &OllamaQAWidget::onClearClicked);
  connect(ui->btnRefreshModels, &QPushButton::clicked, this, &OllamaQAWidget::onRefreshModelsClicked);
  connect(ui->comboBackend, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged),
          this, &OllamaQAWidget::onBackendChanged);

  connect(ui->checkAttachAll, static_cast<void (QCheckBox::*)(int)>(&QCheckBox::stateChanged),
          this, &OllamaQAWidget::onAttachAllToggled);
  connect(ui->checkAxial, static_cast<void (QCheckBox::*)(int)>(&QCheckBox::stateChanged),
          this, &OllamaQAWidget::onSliceViewToggled);
  connect(ui->checkSagittal, static_cast<void (QCheckBox::*)(int)>(&QCheckBox::stateChanged),
          this, &OllamaQAWidget::onSliceViewToggled);
  connect(ui->checkCoronal, static_cast<void (QCheckBox::*)(int)>(&QCheckBox::stateChanged),
          this, &OllamaQAWidget::onSliceViewToggled);

  ui->chatDisplay->setMarkdown(
      "_AI Assistant ready. Type a question and click Send. Check view boxes "
      "to attach screenshots to a Vision-Language Model._");

  // Restore the last used connection (backend, server, model, view
  // checkboxes) from persistent settings.
  loadSettings();

  // Trigger the initial model-list fetch for the restored backend.
  onBackendChanged(ui->comboBackend->currentIndex());
}

OllamaQAWidget::~OllamaQAWidget()
{
  if (m_Reply)
  {
    m_Reply->abort();
    m_Reply->deleteLater();
  }
  delete ui;
}

void
OllamaQAWidget::SetMainImageWindow(MainImageWindow *win)
{
  m_MainWindow = win;
}

void
OllamaQAWidget::setBusy(bool busy)
{
  ui->btnSend->setEnabled(!busy);
  ui->btnStop->setEnabled(busy);
  ui->editPrompt->setReadOnly(busy);
  ui->labelStatus->setText(busy ? "Generating..." : "Ready.");
}

void
OllamaQAWidget::onBackendChanged(int idx)
{
  m_Backend = idx;
  m_Messages = QJsonArray();
  ui->comboModel->clear();

  // Update default server URL hint when switching backends
  if (idx == 0)
  {
    // Ollama
    if (ui->editServer->text().trimmed().isEmpty())
      ui->editServer->setText("http://localhost:11434");
  }
  else
  {
    // llama.cpp
    if (ui->editServer->text().trimmed().isEmpty())
      ui->editServer->setText("http://localhost:8080");
  }

  // Refresh model list for the selected backend
  onRefreshModelsClicked();

  // Persist the backend choice immediately so switching sticks.
  saveSettings();
}

void
OllamaQAWidget::onRefreshModelsClicked()
{
  if (m_Reply)
    return;

  QString server = serverUrl();
  if (server.isEmpty())
  {
    ui->labelStatus->setText("Enter a server URL first.");
    return;
  }

  ui->labelStatus->setText("Loading models...");
  m_IsModelFetch = true;

  if (m_Backend == 0)
    fetchOllamaModels();
  else
    fetchLlamaCppModels();
}

void
OllamaQAWidget::fetchOllamaModels()
{
  QUrl url(serverUrl() + "/api/tags");
  QNetworkRequest req(url);

  m_LineBuffer.clear();
  m_Reply = m_Network->get(req);

  connect(m_Reply, &QNetworkReply::readyRead, this, &OllamaQAWidget::onReadyRead);
  connect(m_Reply, &QNetworkReply::finished,  this, &OllamaQAWidget::onFinished);
  connect(m_Reply, &QNetworkReply::errorOccurred, this, &OllamaQAWidget::onErrorOccurred);
}

void
OllamaQAWidget::fetchLlamaCppModels()
{
  QString base = serverUrl();
  // If user already included /v1 in the URL, don't double it
  QString endpoint;
  if (base.endsWith("/v1"))
    endpoint = base + "/models";
  else
    endpoint = base + "/v1/models";

  QUrl url(endpoint);
  QNetworkRequest req(url);

  m_LineBuffer.clear();
  m_Reply = m_Network->get(req);

  connect(m_Reply, &QNetworkReply::readyRead, this, &OllamaQAWidget::onReadyRead);
  connect(m_Reply, &QNetworkReply::finished,  this, &OllamaQAWidget::onFinished);
  connect(m_Reply, &QNetworkReply::errorOccurred, this, &OllamaQAWidget::onErrorOccurred);
}

void
OllamaQAWidget::onSendClicked()
{
  if (m_Reply)
    return;

  QString prompt = ui->editPrompt->toPlainText().trimmed();
  if (prompt.isEmpty())
    return;

  QString model  = ui->comboModel->currentText().trimmed();
  QString server = serverUrl();
  if (model.isEmpty() || server.isEmpty())
  {
    QMessageBox::warning(this, "AI Assistant",
                          "Please provide both a server URL and a model name.");
    return;
  }

  // Remember the working connection so the user doesn't have to retype it.
  saveSettings();

  QList<QPair<QString, QString>> views = collectAttachedImages();

  QJsonObject userMsg;
  userMsg["role"] = "user";

  // Compose all attached views into a SINGLE labeled image.  Some vision
  // servers/models only accept a limited number of image parts per message
  // (e.g. 1 or 2) and silently drop the rest.  A single montage always
  // carries every view, and the embedded labels tell the model what each
  // panel shows.
  QList<QPair<QString, QString>> images;
  QString compositeB64 = buildCompositeImageBase64(views);
  if (!compositeB64.isEmpty())
    images.append(qMakePair(QString("Composite"), compositeB64));

  // The model only receives raw pixels; describe the montage explicitly.
  QString modelContent = prompt;
  if (!images.isEmpty())
  {
    QStringList labels;
    int idx = 1;
    for (const auto &v : views)
    {
      QString desc;
      if (v.first == "3D View")
        desc = "3-dimensional volume rendering";
      else
        desc = "2D slice (" + v.first + " anatomical view)";
      labels << QString("%1. %2 — %3").arg(idx++).arg(v.first, desc);
    }
    modelContent = QString(
        "I am attaching one medical composite image containing %1 "
        "labeled panel(s), in left-to-right order:\n%2\n\n%3")
                       .arg(views.size())
                       .arg(labels.join("\n"))
                       .arg(prompt);
  }
  userMsg["content"] = modelContent;

  // Show the prompt with a note about attached views.  The note is appended
  // as regular markdown so it renders inline with the user's text.
  QString note;
  if (!views.isEmpty())
  {
    QStringList labels;
    for (const auto &v : views)
      labels << v.first;
    note = QString(" *(attached: %1)*").arg(labels.join(", "));
  }
  appendMessageToDisplay("user", prompt, note);
  ui->editPrompt->clear();

  m_CurrentAssistantText.clear();
  m_AssistantDone = false;

  m_IsModelFetch = false;
  setBusy(true);

  if (m_Backend == 0)
    sendOllamaRequest(userMsg, images);
  else
    sendLlamaCppRequest(userMsg, images);
}

void
OllamaQAWidget::sendOllamaRequest(const QJsonObject &userMsg,
                                  const QList<QPair<QString, QString>> &images)
{
  QString model = ui->comboModel->currentText().trimmed();

  QJsonObject msgWithImages = userMsg;
  if (!images.isEmpty())
  {
    QJsonArray imgArr;
    for (const auto &p : images)
      imgArr.append(p.second);
    msgWithImages["images"] = imgArr;
  }

  QJsonObject body;
  body["model"] = model;
  {
    QJsonArray msgs = m_Messages;
    msgs.append(msgWithImages);
    body["messages"] = msgs;
  }
  body["stream"] = true;

  QJsonDocument doc(body);
  QByteArray payload = doc.toJson(QJsonDocument::Compact);

  QUrl url(serverUrl() + "/api/chat");
  QNetworkRequest req(url);
  req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

  m_LineBuffer.clear();
  m_Reply = m_Network->post(req, payload);

  connect(m_Reply, &QNetworkReply::readyRead, this, &OllamaQAWidget::onReadyRead);
  connect(m_Reply, &QNetworkReply::finished,  this, &OllamaQAWidget::onFinished);
  connect(m_Reply, &QNetworkReply::errorOccurred, this, &OllamaQAWidget::onErrorOccurred);
}

void
OllamaQAWidget::sendLlamaCppRequest(const QJsonObject &userMsg,
                                    const QList<QPair<QString, QString>> &images)
{
  QString model = ui->comboModel->currentText().trimmed();
  QString base = serverUrl();

  QJsonObject msgWithImages = userMsg;
  if (!images.isEmpty())
  {
    QJsonArray contentArr;
    QJsonObject textPart;
    textPart["type"] = "text";
    textPart["text"] = userMsg["content"].toString();
    contentArr.append(textPart);

    for (const auto &p : images)
    {
      QJsonObject imgPart;
      imgPart["type"] = "image_url";
      QJsonObject imgUrl;
      imgUrl["url"] = "data:image/jpeg;base64," + p.second;
      imgPart["image_url"] = imgUrl;
      contentArr.append(imgPart);
    }

    msgWithImages["content"] = contentArr;
  }

  QJsonObject body;
  body["model"] = model;
  {
    QJsonArray msgs = m_Messages;
    msgs.append(msgWithImages);
    body["messages"] = msgs;
  }
  body["stream"] = true;

  QJsonDocument doc(body);
  QByteArray payload = doc.toJson(QJsonDocument::Compact);

  QString endpoint;
  if (base.endsWith("/v1"))
    endpoint = base + "/chat/completions";
  else
    endpoint = base + "/v1/chat/completions";

  QUrl url(endpoint);
  QNetworkRequest req(url);
  req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

  m_LineBuffer.clear();
  m_Reply = m_Network->post(req, payload);

  connect(m_Reply, &QNetworkReply::readyRead, this, &OllamaQAWidget::onReadyRead);
  connect(m_Reply, &QNetworkReply::finished,  this, &OllamaQAWidget::onFinished);
  connect(m_Reply, &QNetworkReply::errorOccurred, this, &OllamaQAWidget::onErrorOccurred);
}

void
OllamaQAWidget::parseOllamaLine(const QByteArray &line)
{
  QJsonParseError err;
  QJsonDocument doc = QJsonDocument::fromJson(line, &err);
  if (err.error != QJsonParseError::NoError || !doc.isObject())
    return;

  QJsonObject obj = doc.object();

  if (obj.contains("message"))
  {
    QJsonObject msg = obj["message"].toObject();
    QString delta = msg["content"].toString();
    if (!delta.isEmpty())
      updateStreamingAssistantText(delta);
  }

  if (obj["done"].toBool(false))
  {
    if (!m_AssistantDone)
    {
      QJsonObject assistantMsg;
      assistantMsg["role"]    = "assistant";
      assistantMsg["content"] = m_CurrentAssistantText;
      m_Messages.append(assistantMsg);
      finalizeAssistantMessage();
    }
  }
}

void
OllamaQAWidget::parseLlamaCppSSELine(const QByteArray &line)
{
  QByteArray json;
  if (line.startsWith("data: "))
  {
    json = line.mid(6); // strip "data: "
  }
  else
  {
    // Non-SSE line: may be a JSON error response from the server
    json = line;
  }

  // Check for server error response: {"error":{"message":"...","code":500}}
  QJsonParseError err;
  QJsonDocument errDoc = QJsonDocument::fromJson(json, &err);
  if (err.error == QJsonParseError::NoError && errDoc.isObject())
  {
    QJsonObject errObj = errDoc.object();
    if (errObj.contains("error") && !errObj["error"].isObject())
    {
      // Not an error, continue normal parsing below
    }
    else if (errObj.contains("error"))
    {
      QJsonObject e = errObj["error"].toObject();
      QString msg = e["message"].toString();
      if (msg.isEmpty())
        msg = e["code"].toString() + " " + e["type"].toString();
      if (!msg.isEmpty())
      {
        m_CurrentAssistantText.clear();
        m_AssistantDone = true;
        appendMessageToDisplay("assistant",
                                QString("[Server error: %1]").arg(msg));
        if (msg.contains("image", Qt::CaseInsensitive) ||
            msg.contains("vision", Qt::CaseInsensitive) ||
            msg.contains("mmproj", Qt::CaseInsensitive))
        {
          ui->checkAttachAll->setCheckState(Qt::Unchecked);
          ui->checkAxial->setChecked(false);
          ui->checkSagittal->setChecked(false);
          ui->checkCoronal->setChecked(false);
          ui->checkAttach3d->setChecked(false);
          appendMessageToDisplay("assistant",
                                  "[Image attachment disabled — model does not support vision input.]");
        }
        return;
      }
    }
  }

  if (json == "[DONE]")
  {
    // Stream finished
    if (!m_AssistantDone)
    {
      QJsonObject assistantMsg;
      assistantMsg["role"]    = "assistant";
      assistantMsg["content"] = m_CurrentAssistantText;
      m_Messages.append(assistantMsg);
      finalizeAssistantMessage();
    }
    return;
  }

  QJsonParseError err2;
  QJsonDocument doc = QJsonDocument::fromJson(json, &err2);
  if (err2.error != QJsonParseError::NoError || !doc.isObject())
    return;

  QJsonObject obj = doc.object();

  // {"choices":[{"delta":{"content":"..."},"finish_reason":null}]}
  if (obj.contains("choices"))
  {
    QJsonArray choices = obj["choices"].toArray();
    if (choices.isEmpty())
      return;

    QJsonObject choice = choices[0].toObject();

    // Check for finish_reason
    if (choice.contains("finish_reason") && !choice["finish_reason"].isNull() && choice["finish_reason"].toString() != "null")
    {
      if (!m_AssistantDone)
      {
        QJsonObject assistantMsg;
        assistantMsg["role"]    = "assistant";
        assistantMsg["content"] = m_CurrentAssistantText;
        m_Messages.append(assistantMsg);
        finalizeAssistantMessage();
      }
      return;
    }

    if (choice.contains("delta"))
    {
      QJsonObject delta = choice["delta"].toObject();
      QString text = delta["content"].toString();
      if (!text.isEmpty())
        updateStreamingAssistantText(text);
    }
  }
}

void
OllamaQAWidget::onReadyRead()
{
  if (!m_Reply)
    return;

  m_LineBuffer += m_Reply->readAll();

  // For llama.cpp: if no newline yet and buffer looks like a JSON error,
  // process it immediately (server may send a single-shot error instead of SSE).
  if (m_Backend != 0 && !m_IsModelFetch && m_LineBuffer.indexOf('\n') < 0)
  {
    QByteArray buf = m_LineBuffer;
    QJsonParseError e;
    QJsonDocument d = QJsonDocument::fromJson(buf, &e);
    if (e.error == QJsonParseError::NoError && d.isObject())
    {
      QJsonObject o = d.object();
      if (o.contains("error"))
      {
        parseLlamaCppSSELine(buf);
        m_LineBuffer.clear();
      }
    }
  }

  while (true)
  {
    int nl = m_LineBuffer.indexOf('\n');
    if (nl < 0)
      break;
    QByteArray line = m_LineBuffer.left(nl).trimmed();
    m_LineBuffer.remove(0, nl + 1);

    if (line.isEmpty())
      continue;

    // Model fetch responses are single-shot JSON, not streaming
    if (m_IsModelFetch)
      continue; // handled in onFinished

    if (m_Backend == 0)
      parseOllamaLine(line);
    else
      parseLlamaCppSSELine(line);
  }
}

void
OllamaQAWidget::onFinished()
{
  if (!m_Reply)
    return;

  // Flush any remaining buffered data
  onReadyRead();

  // Handle model fetch response
  if (m_IsModelFetch)
  {
    QByteArray allData = m_LineBuffer;
    m_LineBuffer.clear();

    QJsonParseError err;
    QJsonDocument doc = QJsonDocument::fromJson(allData, &err);
    if (err.error == QJsonParseError::NoError && doc.isObject())
    {
      QJsonObject resp = doc.object();
      ui->comboModel->clear();

      if (m_Backend == 0)
      {
        // Ollama: {"models": [{"name": "gemma4:latest"}, ...]}
        if (resp.contains("models"))
        {
          QJsonArray models = resp["models"].toArray();
          for (auto it : models)
          {
            QString name = it.toObject()["name"].toString();
            if (!name.isEmpty())
              ui->comboModel->addItem(name);
          }
        }
      }
      else
      {
        // llama.cpp: {"data": [{"id": "model-name"}, ...]}
        if (resp.contains("data"))
        {
          QJsonArray models = resp["data"].toArray();
          for (auto it : models)
          {
            QString id = it.toObject()["id"].toString();
            if (!id.isEmpty())
              ui->comboModel->addItem(id);
          }
        }
      }

      if (ui->comboModel->count() == 0)
      {
        ui->labelStatus->setText("No models found (enter manually).");
      }
      else
      {
        ui->labelStatus->setText(QString("%1 model(s) loaded.").arg(ui->comboModel->count()));
      }

      // Re-select the persisted model for this backend if it is available,
      // otherwise show the saved name as editable text.
      if (!m_PreferredModel.isEmpty())
      {
        int idx = ui->comboModel->findText(m_PreferredModel);
        if (idx >= 0)
          ui->comboModel->setCurrentIndex(idx);
        else
          ui->comboModel->setCurrentText(m_PreferredModel);
      }
    }
    else
    {
      ui->labelStatus->setText("Failed to parse model list.");
    }
    m_IsModelFetch = false;
  }

  m_Reply->deleteLater();
  m_Reply = nullptr;
  setBusy(false);
}

void
OllamaQAWidget::onErrorOccurred()
{
  if (!m_Reply)
    return;

  QString err = m_Reply->errorString();

  if (m_IsModelFetch)
  {
    ui->labelStatus->setText("Error loading models: " + err);
    m_IsModelFetch = false;
  }
  else
  {
    m_AssistantDone = true;
    appendMessageToDisplay("assistant",
                          QString("[Error contacting server: %1]").arg(err));
    ui->labelStatus->setText("Error: " + err);
  }

  m_Reply->deleteLater();
  m_Reply = nullptr;
  setBusy(false);
}

void
OllamaQAWidget::appendMessageToDisplay(const QString &role, const QString &text,
                                       const QString &noteMarkdown)
{
  QString body = text;
  if (!noteMarkdown.isEmpty())
    body += noteMarkdown;
  m_DisplayMessages.append(qMakePair(role, body));
  scheduleRender();
}

void
OllamaQAWidget::updateStreamingAssistantText(const QString &delta)
{
  m_CurrentAssistantText += delta;
  scheduleRender();
}

void
OllamaQAWidget::scheduleRender()
{
  if (!m_RenderTimer->isActive())
    m_RenderTimer->start();
}

void
OllamaQAWidget::renderConversation()
{
  m_RenderTimer->stop();

  // Remember whether the user is pinned to the bottom so we don't yank the
  // view while streaming.
  QScrollBar *sb = ui->chatDisplay->verticalScrollBar();
  const bool stickToBottom = !sb || sb->value() >= sb->maximum() - 5;

  QStringList blocks;
  for (const auto &m : m_DisplayMessages)
  {
    QString label = (m.first == "user") ? "You" : "Assistant";
    blocks << QString("**%1:**\n\n%2").arg(label, m.second);
  }
  if (!m_CurrentAssistantText.isEmpty())
    blocks << QString("**Assistant:**\n\n%1").arg(m_CurrentAssistantText);

  if (blocks.isEmpty())
    ui->chatDisplay->setMarkdown("_AI Assistant ready._");
  else
    ui->chatDisplay->setMarkdown(blocks.join("\n\n---\n\n"));

  if (stickToBottom && sb)
    sb->setValue(sb->maximum());
}

void
OllamaQAWidget::finalizeAssistantMessage()
{
  m_AssistantDone = true;
  m_DisplayMessages.append(qMakePair(QString("assistant"), m_CurrentAssistantText));
  m_CurrentAssistantText.clear();
  scheduleRender();
}

QImage
OllamaQAWidget::grabSlicePanel(int index) const
{
  if (!m_MainWindow || index < 0 || index > 2)
    return QImage();

  SliceViewPanel *panel = m_MainWindow->GetSlicePanel(static_cast<unsigned int>(index));
  if (!panel)
    return QImage();

  // Grab the raw framebuffer of the slice canvas directly.  This is more
  // reliable than QWidget::grab() on the whole panel, which can come back
  // empty for OpenGL-backed content.
  QWidget *canvas = panel->findChild<QWidget *>("sliceViewCanvas");
  if (auto *gl = qobject_cast<QOpenGLWidget *>(canvas))
    return gl->grabFramebuffer();

  QPixmap pix = panel->grab();
  return pix.toImage();
}

QImage
OllamaQAWidget::grab3DView() const
{
  if (!m_MainWindow)
    return QImage();

  QWidget *target = m_MainWindow->centralWidget();
  if (!target)
    return QImage();

  QPixmap pix = target->grab();
  return pix.toImage();
}

QList<QPair<QString, QString>>
OllamaQAWidget::collectAttachedImages()
{
  QList<QPair<QString, QString>> result;

  // Map each anatomical direction to the display window (panel index) that
  // is actually showing it.  This index depends on the image orientation and
  // is NOT fixed (0=Axial, 1=Sagittal, 2=Coronal are only the defaults).
  auto dirToPanel = [this](AnatomicalDirection d) -> int {
    if (!m_MainWindow || !m_MainWindow->GetModel() ||
        !m_MainWindow->GetModel()->GetDriver())
      return -1;
    return m_MainWindow->GetModel()->GetDriver()
        ->GetDisplayWindowForAnatomicalDirection(d);
  };

  struct {
    QCheckBox *cb;
    const char *label;
    AnatomicalDirection dir;
  } slices[] = {
    { ui->checkAxial,    "Axial",    ANATOMY_AXIAL    },
    { ui->checkSagittal, "Sagittal", ANATOMY_SAGITTAL },
    { ui->checkCoronal,  "Coronal",  ANATOMY_CORONAL  },
  };

  for (auto &s : slices)
  {
    if (!s.cb->isChecked())
      continue;
    QImage img = grabSlicePanel(dirToPanel(s.dir));
    QString b64 = encodeImageBase64(img);
    if (!b64.isEmpty())
      result.append(qMakePair(QString(s.label), b64));
  }

  if (ui->checkAttach3d->isChecked())
  {
    QImage img = grab3DView();
    QString b64 = encodeImageBase64(img);
    if (!b64.isEmpty())
      result.append(qMakePair("3D View", b64));
  }

  return result;
}

void
OllamaQAWidget::onAttachAllToggled(int state)
{
  ui->checkAxial->blockSignals(true);
  ui->checkSagittal->blockSignals(true);
  ui->checkCoronal->blockSignals(true);

  bool checked = (state == Qt::Checked);
  ui->checkAxial->setChecked(checked);
  ui->checkSagittal->setChecked(checked);
  ui->checkCoronal->setChecked(checked);

  ui->checkAxial->blockSignals(false);
  ui->checkSagittal->blockSignals(false);
  ui->checkCoronal->blockSignals(false);
}

void
OllamaQAWidget::onSliceViewToggled(int /*state*/)
{
  syncAttachAllState();
}

void
OllamaQAWidget::syncAttachAllState()
{
  bool axial = ui->checkAxial->isChecked();
  bool sagittal = ui->checkSagittal->isChecked();
  bool coronal = ui->checkCoronal->isChecked();

  ui->checkAttachAll->blockSignals(true);
  if (axial && sagittal && coronal)
    ui->checkAttachAll->setCheckState(Qt::Checked);
  else if (!axial && !sagittal && !coronal)
    ui->checkAttachAll->setCheckState(Qt::Unchecked);
  else
    ui->checkAttachAll->setCheckState(Qt::PartiallyChecked);
  ui->checkAttachAll->blockSignals(false);
}

QString
OllamaQAWidget::encodeImageBase64(const QImage &img) const
{
  if (img.isNull())
    return QString();

  QImage scaled = img;
  const int maxDim = 256;
  if (scaled.width() > maxDim || scaled.height() > maxDim)
    scaled = scaled.scaled(maxDim, maxDim, Qt::KeepAspectRatio, Qt::SmoothTransformation);

  QByteArray bytes;
  QBuffer buffer(&bytes);
  buffer.open(QIODevice::WriteOnly);
  scaled.save(&buffer, "JPEG", 85);
  return QString::fromLatin1(bytes.toBase64());
}

QString
OllamaQAWidget::buildCompositeImageBase64(
    const QList<QPair<QString, QString>> &views) const
{
  if (views.isEmpty())
    return QString();

  // Decode each view and render it at 256px max per view.
  const int maxDim = 256;
  const int gap = 8;
  int panelH = 0, totalW = gap * (views.size() - 1);
  QVector<QImage> panels;
  QVector<QString> labels;

  for (const auto &v : views)
  {
    QImage img;
    if (!img.loadFromData(QByteArray::fromBase64(v.second.toLatin1())))
      continue;
    img = img.scaled(maxDim, maxDim, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    panelH = qMax(panelH, img.height());
    totalW += img.width();
    panels.push_back(img);
    labels.push_back(v.first);
  }

  if (panels.isEmpty())
    return QString();

  // Thin label strip across the top of the montage.
  const int labelH = 24;
  QImage composite(totalW, panelH + labelH, QImage::Format_RGB32);
  composite.fill(Qt::black);

  QPainter p(&composite);
  p.setRenderHint(QPainter::Antialiasing);
  QFont f = p.font();
  f.setPointSize(14);
  f.setBold(true);
  p.setFont(f);

  int x = 0;
  for (int i = 0; i < panels.size(); ++i)
  {
    const QImage &panel = panels[i];
    // Draw the label centered above the panel
    QRect labelRect(x, 0, panel.width(), labelH);
    p.setPen(Qt::white);
    p.drawText(labelRect, Qt::AlignCenter, labels[i]);
    // Draw the panel below the label
    p.drawImage(x, labelH, panel);
    x += panel.width() + gap;
  }
  p.end();

  QByteArray bytes;
  QBuffer buffer(&bytes);
  buffer.open(QIODevice::WriteOnly);
  composite.save(&buffer, "JPEG", 85);
  return QString::fromLatin1(bytes.toBase64());
}

QString
OllamaQAWidget::serverUrl() const
{
  QString url = ui->editServer->text().trimmed();
  // Strip trailing slash
  while (url.endsWith('/'))
    url.chop(1);
  return url;
}

void
OllamaQAWidget::saveSettings()
{
  QSettings s;
  s.setValue("llm/backend", ui->comboBackend->currentIndex());
  s.setValue("llm/server", ui->editServer->text().trimmed());
  s.setValue("llm/model", ui->comboModel->currentText().trimmed());
  s.setValue("llm/attach_axial", ui->checkAxial->isChecked());
  s.setValue("llm/attach_sagittal", ui->checkSagittal->isChecked());
  s.setValue("llm/attach_coronal", ui->checkCoronal->isChecked());
  s.setValue("llm/attach_3d", ui->checkAttach3d->isChecked());

  m_PreferredModel = ui->comboModel->currentText().trimmed();
}

void
OllamaQAWidget::loadSettings()
{
  QSettings s;
  m_PreferredModel = s.value("llm/model", QString()).toString();

  QString server = s.value("llm/server", QString()).toString();
  if (!server.isEmpty())
    ui->editServer->setText(server);

  ui->checkAxial->setChecked(s.value("llm/attach_axial", true).toBool());
  ui->checkSagittal->setChecked(s.value("llm/attach_sagittal", true).toBool());
  ui->checkCoronal->setChecked(s.value("llm/attach_coronal", true).toBool());
  ui->checkAttach3d->setChecked(s.value("llm/attach_3d", false).toBool());
  syncAttachAllState();

  int backend = s.value("llm/backend", 0).toInt();
  ui->comboBackend->blockSignals(true);
  ui->comboBackend->setCurrentIndex(backend);
  ui->comboBackend->blockSignals(false);
}

void
OllamaQAWidget::onStopClicked()
{
  if (m_Reply)
  {
    m_Reply->abort();
  }
}

void
OllamaQAWidget::onClearClicked()
{
  m_Messages = QJsonArray();
  m_CurrentAssistantText.clear();
  m_AssistantDone = true;
  m_DisplayMessages.clear();
  m_RenderTimer->stop();
  ui->chatDisplay->setMarkdown("_Chat history cleared._");
}
