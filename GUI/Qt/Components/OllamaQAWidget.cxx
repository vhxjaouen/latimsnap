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

OllamaQAWidget::OllamaQAWidget(QWidget *parent)
  : QWidget(parent)
  , ui(new Ui::OllamaQAWidget)
{
  ui->setupUi(this);

  // Placeholder items shown until we successfully talk to the server.
  ui->comboModel->addItem("(fetching models...)");
  ui->comboModel->setCurrentIndex(0);

  m_Network = new QNetworkAccessManager(this);

  connect(ui->btnSend,  &QPushButton::clicked, this, &OllamaQAWidget::onSendClicked);
  connect(ui->btnStop,  &QPushButton::clicked, this, &OllamaQAWidget::onStopClicked);
  connect(ui->btnClear, &QPushButton::clicked, this, &OllamaQAWidget::onClearClicked);
  connect(ui->btnRefreshModels, &QPushButton::clicked,
          this, &OllamaQAWidget::onRefreshModelsClicked);
  connect(ui->btnFreeGpu, &QPushButton::clicked,
          this, &OllamaQAWidget::onFreeGpuClicked);
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

  // Replace last assistant block if present; otherwise append.
  // Simpler approach: append delta to end of display.
  QTextCursor cursor = ui->chatDisplay->textCursor();
  cursor.movePosition(QTextCursor::End);
  cursor.insertText(delta);
  QScrollBar *sb = ui->chatDisplay->verticalScrollBar();
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

  if (ui->checkAttachView->isChecked())
  {
    QImage img = grabCurrentView();
    QString b64 = encodeImageBase64(img);
    if (!b64.isEmpty())
    {
      QJsonArray imgArr;
      imgArr.append(b64);
      userMsg["images"] = imgArr;
    }
  }

  m_Messages.append(userMsg);

  // Display the user's prompt (without embedding the base64)
  appendMessageToDisplay("user", prompt);
  ui->editPrompt->clear();

  // Prepare a placeholder for the assistant response
  m_CurrentAssistantText.clear();
  QString color = "#008040";
  QString html  = QString("<div style='margin:6px 0;'>"
                          "<b style='color:%1'>Assistant:</b><br></div>")
                    .arg(color);
  ui->chatDisplay->append(html);

  // Build the request body
  QJsonObject body;
  body["model"]    = model;
  body["messages"] = m_Messages;
  body["stream"]   = true;

  // Prevent the model from camping in VRAM once the reply is done. This is
  // the primary mechanism to keep the GPU available for pTVreg and other
  // GPU-heavy workflows. Users can disable via the checkbox for a snappier
  // multi-turn UX at the cost of GPU residency.
  if (ui->checkAutoRelease->isChecked())
    body["keep_alive"] = 0;

  QJsonDocument doc(body);
  QByteArray payload = doc.toJson(QJsonDocument::Compact);

  QUrl url(server + "/api/chat");
  QNetworkRequest req(url);
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
      QString delta = msg["content"].toString();
      if (!delta.isEmpty())
        updateStreamingAssistantText(delta);
    }

    if (obj["done"].toBool(false))
    {
      // Persist assistant message into history
      QJsonObject assistantMsg;
      assistantMsg["role"]    = "assistant";
      assistantMsg["content"] = m_CurrentAssistantText;
      m_Messages.append(assistantMsg);
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
  m_CurrentAssistantText.clear();
  ui->chatDisplay->clear();
  ui->chatDisplay->setHtml("<i style='color:gray'>Chat history cleared.</i>");
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

  ui->labelStatus->setText("Contacting Ollama server...");
  ui->btnRefreshModels->setEnabled(false);

  QUrl url(server + "/api/tags");
  QNetworkRequest req(url);
  m_ModelsReply = m_Network->get(req);

  connect(m_ModelsReply, &QNetworkReply::finished,
          this, &OllamaQAWidget::onModelsFetched);
  connect(m_ModelsReply, &QNetworkReply::errorOccurred,
          this, &OllamaQAWidget::onModelsFetchError);
}

void
OllamaQAWidget::onModelsFetched()
{
  if (!m_ModelsReply)
    return;

  ui->btnRefreshModels->setEnabled(true);

  // If the reply errored out, onModelsFetchError already reported it; but
  // this finished() slot still fires. Bail out if there was an error.
  if (m_ModelsReply->error() != QNetworkReply::NoError)
  {
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
    ui->labelStatus->setText("Error: invalid response from Ollama server.");
    return;
  }

  QJsonArray models = doc.object().value("models").toArray();

  // Preserve the currently selected/edited model name so we can restore it
  // if it still exists on the server.
  QString previous = ui->comboModel->currentText().trimmed();

  ui->comboModel->clear();

  QStringList names;
  for (const QJsonValue &v : models)
  {
    QJsonObject o = v.toObject();
    QString name = o.value("name").toString();
    if (!name.isEmpty())
      names << name;
  }

  if (names.isEmpty())
  {
    ui->comboModel->addItem("(no models installed)");
    ui->labelStatus->setText(
      "Connected, but no models are installed on the server.");
    return;
  }

  ui->comboModel->addItems(names);

  int idx = names.indexOf(previous);
  if (idx >= 0)
    ui->comboModel->setCurrentIndex(idx);
  else
    ui->comboModel->setCurrentIndex(0);

  ui->labelStatus->setText(
    QString("Connected. %1 model(s) available.").arg(names.size()));
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
