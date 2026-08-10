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
  enum BackendType { BACKEND_VLLM_OPENAI = 0, BACKEND_OLLAMA };

  explicit OllamaQAWidget(QWidget *parent = nullptr);
  ~OllamaQAWidget();

  /** Provide access to the main window so we can grab framebuffer contents
   *  from the slice / 3D view widgets. */
  void SetMainImageWindow(MainImageWindow *win);

private slots:
  void onBackendChanged(int index);
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
  void onShowFetched();
  void onVisionFilterToggled();
  void onFreeGpuClicked();
  void onLoadVramClicked();
  void onExportJsonClicked();
  void onPsFetched();
  void onUnloadFinished();
  void onPrewarmFinished();

private:
  Ui::OllamaQAWidget *ui;

  QNetworkAccessManager *m_Network = nullptr;
  QNetworkReply         *m_Reply = nullptr;
  QNetworkReply         *m_ModelsReply = nullptr;
  QNetworkReply         *m_PsReply = nullptr;
  QNetworkReply         *m_PrewarmReply = nullptr;
  MainImageWindow       *m_MainWindow = nullptr;

  // Currently pre-warmed / loaded VRAM model name
  QString m_LoadedVramModel;

  // List of all models returned by /api/tags, and subset verified to have vision capacity.
  QStringList m_AllModels;
  QStringList m_VisionModels;
  int m_PendingShowRequests = 0;
  int m_TotalShowRequests = 0;

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

  // Current assistant thinking / reasoning text being streamed
  QString m_CurrentThinkingText;

  // Start time of active chat request for TTFT / latency tracking
  qint64 m_RequestStartMs = 0;
  qint64 m_FirstTokenMs = 0;

  // History of performance metrics per assistant turn
  QJsonArray m_TurnMetricsHistory;

  BackendType getBackendType() const;
  void appendMessageToDisplay(const QString &role, const QString &text);
  void updateStreamingAssistantText(const QString &delta);
  void updateStreamingThinkingText(const QString &delta);
  void appendVerbosityLog(const QString &htmlOrText);
  QImage grabCurrentView() const;
  QString encodeImageBase64(const QImage &img) const;
  void setBusy(bool busy);
  void fetchModels();
  void updateModelCombo();
  bool parseVisionCapability(const QJsonObject &showObj) const;

  // Issue POST /api/generate {"model": <name>, "keep_alive": 0} to force
  // Ollama to release VRAM for a specific model. Fire-and-forget: the
  // finished signal is routed to onUnloadFinished so we can update the
  // status label once every pending unload has replied.
  void requestModelUnload(const QString &modelName);
};

#endif // OLLAMAQAWIDGET_H
