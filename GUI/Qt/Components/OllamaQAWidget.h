/*=========================================================================

  Program:   ITK-SNAP
  Module:    OllamaQAWidget.h
  Language:  C++

  This file is part of ITK-SNAP

  ITK-SNAP is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

=========================================================================*/
#ifndef OLLAMAQAWIDGET_H
#define OLLAMAQAWIDGET_H

#include <QWidget>
#include <QJsonArray>
#include <QJsonObject>
#include <QImage>
#include <QString>
#include <QByteArray>

class QNetworkAccessManager;
class QNetworkReply;
class MainImageWindow;

namespace Ui
{
class OllamaQAWidget;
}

/**
 * A dockable side panel that provides a chat-style interface to a locally
 * running Ollama server. Supports Vision-Language Models (VLMs) such as
 * gemma / llava by attaching the currently displayed slice or 3D rendering
 * as a Base64-encoded image alongside the prompt.
 *
 * Uses Ollama's native /api/chat endpoint with streaming enabled so tokens
 * appear in real time.
 */
class OllamaQAWidget : public QWidget
{
  Q_OBJECT

public:
  explicit OllamaQAWidget(QWidget *parent = nullptr);
  ~OllamaQAWidget();

  /** Provide access to the main window so we can grab framebuffer contents
   *  from the slice / 3D view widgets. */
  void SetMainImageWindow(MainImageWindow *win);

private slots:
  void onSendClicked();
  void onStopClicked();
  void onClearClicked();
  void onReadyRead();
  void onFinished();
  void onErrorOccurred();

private:
  Ui::OllamaQAWidget *ui;

  QNetworkAccessManager *m_Network = nullptr;
  QNetworkReply         *m_Reply = nullptr;
  MainImageWindow       *m_MainWindow = nullptr;

  // Conversation history stored as an array of {role, content, images?} objects.
  QJsonArray m_Messages;

  // Buffer for a partially received line (JSON is streamed as ND-JSON).
  QByteArray m_LineBuffer;

  // Current assistant response being streamed
  QString m_CurrentAssistantText;

  void appendMessageToDisplay(const QString &role, const QString &text);
  void updateStreamingAssistantText(const QString &delta);
  QImage grabCurrentView() const;
  QString encodeImageBase64(const QImage &img) const;
  void setBusy(bool busy);
};

#endif // OLLAMAQAWIDGET_H
