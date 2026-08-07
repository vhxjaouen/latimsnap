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

OllamaQAWidget::OllamaQAWidget(QWidget *parent)
  : QWidget(parent)
  , ui(new Ui::OllamaQAWidget)
{
  ui->setupUi(this);

  // Default models list. User can freely edit the combo since it is editable.
  ui->comboModel->addItem("gemma4");
  ui->comboModel->addItem("llava");
  ui->comboModel->addItem("llama3.2-vision");
  ui->comboModel->addItem("llama3");
  ui->comboModel->setCurrentText("gemma4");

  m_Network = new QNetworkAccessManager(this);

  connect(ui->btnSend,  &QPushButton::clicked, this, &OllamaQAWidget::onSendClicked);
  connect(ui->btnStop,  &QPushButton::clicked, this, &OllamaQAWidget::onStopClicked);
  connect(ui->btnClear, &QPushButton::clicked, this, &OllamaQAWidget::onClearClicked);

  ui->chatDisplay->setHtml("<i style='color:gray'>AI Assistant ready. "
                           "Type a question and click Send. "
                           "Enable <b>Attach Current View</b> to send the "
                           "displayed view to a Vision-Language Model.</i>");
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

  QJsonDocument doc(body);
  QByteArray payload = doc.toJson(QJsonDocument::Compact);

  QUrl url(server + "/api/chat");
  QNetworkRequest req(url);
  req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

  m_LineBuffer.clear();
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
}

void
OllamaQAWidget::onClearClicked()
{
  m_Messages = QJsonArray();
  m_CurrentAssistantText.clear();
  ui->chatDisplay->clear();
  ui->chatDisplay->setHtml("<i style='color:gray'>Chat history cleared.</i>");
}
