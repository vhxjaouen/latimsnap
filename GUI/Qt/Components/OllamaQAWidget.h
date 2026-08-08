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
  void onRefreshModelsClicked();
  void onServerUrlChanged();
  void onModelsFetched();
  void onModelsFetchError();
  void onFreeGpuClicked();
  void onPsFetched();
  void onUnloadFinished();

private:
  Ui::OllamaQAWidget *ui;

  QNetworkAccessManager *m_Network = nullptr;
  QNetworkReply         *m_Reply = nullptr;
  QNetworkReply         *m_ModelsReply = nullptr;
  QNetworkReply         *m_PsReply = nullptr;
  MainImageWindow       *m_MainWindow = nullptr;

  // Model name of the request currently in flight; used to force an unload
  // when the user aborts a stream so VRAM does not stay pinned until the
  // Ollama keep_alive TTL expires server-side.
  QString m_InFlightModel;

  // Number of unload requests still pending after a "Free GPU" click.
  int m_PendingUnloads = 0;
  // Human-readable list of models freed during the current unload batch.
  QStringList m_UnloadedSummary;

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
  void fetchModels();

  // Issue POST /api/generate {"model": <name>, "keep_alive": 0} to force
  // Ollama to release VRAM for a specific model. Fire-and-forget: the
  // finished signal is routed to onUnloadFinished so we can update the
  // status label once every pending unload has replied.
  void requestModelUnload(const QString &modelName);
};

#endif // OLLAMAQAWIDGET_H
