/*=========================================================================

  Program:   LaTIM-SNAP
  Module:    OllamaQAWidget.h
  Language:  C++

  This file is part of LaTIM-SNAP

  LaTIM-SNAP is free software: you can redistribute it and/or modify
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
class QTimer;
class MainImageWindow;

namespace Ui
{
class OllamaQAWidget;
}

/**
 * A dockable side panel that provides a chat-style interface to an LLM
 * server.  Supports two backends:
 *
 *   Ollama      – native /api/chat (ND-JSON streaming)
 *   llama.cpp   – OpenAI-compatible /v1/chat/completions (SSE streaming)
 *
 * Vision-Language Models are supported by attaching up to 4 views
 * (Axial, Sagittal, Coronal, 3D) as Base64-encoded images alongside
 * the prompt.  Each image is scaled to a maximum dimension of 256px.
 */
class OllamaQAWidget : public QWidget
{
  Q_OBJECT

public:
  explicit OllamaQAWidget(QWidget *parent = nullptr);
  ~OllamaQAWidget();

  void SetMainImageWindow(MainImageWindow *win);

private slots:
  void onSendClicked();
  void onStopClicked();
  void onClearClicked();
  void onRefreshModelsClicked();
  void onReadyRead();
  void onFinished();
  void onErrorOccurred();
  void onBackendChanged(int idx);

  // Multi-view attachment sync
  void onAttachAllToggled(int state);
  void onSliceViewToggled(int state);

private:
  Ui::OllamaQAWidget *ui;

  QNetworkAccessManager *m_Network = nullptr;
  QNetworkReply         *m_Reply = nullptr;
  MainImageWindow       *m_MainWindow = nullptr;

  // Which backend is active: 0=Ollama, 1=llama.cpp
  int m_Backend = 0;

  // Conversation history (OpenAI-style {role, content} objects).
  QJsonArray m_Messages;

  // Buffer for partially received response data.
  QByteArray m_LineBuffer;

  // Current assistant response being streamed
  QString m_CurrentAssistantText;

  // True when the in-progress assistant stream has been finalized into
  // m_DisplayMessages (guards against double-finalization from servers that
  // send both finish_reason and [DONE]).
  bool m_AssistantDone = true;

  // Display conversation: {role, markdown text} for every completed message.
  // Rendered into chatDisplay via renderConversation().  This is separate
  // from m_Messages, which carries the raw API payload sent to the server.
  QList<QPair<QString, QString>> m_DisplayMessages;

  // Debounce timer that coalesces streaming deltas before re-rendering the
  // markdown view.
  QTimer *m_RenderTimer = nullptr;

  // True while we are fetching the model list (not a chat reply).
  bool m_IsModelFetch;

  // Persisted model name requested on startup, applied once the model list
  // for the selected backend has been fetched.
  QString m_PreferredModel;

  void saveSettings();
  void loadSettings();
  void appendMessageToDisplay(const QString &role, const QString &text,
                              const QString &noteMarkdown = QString());
  void updateStreamingAssistantText(const QString &delta);
  void scheduleRender();
  void renderConversation();
  void finalizeAssistantMessage();
  QImage grabSlicePanel(int index) const;
  QImage grab3DView() const;
  QString encodeImageBase64(const QImage &img) const;
  QString buildCompositeImageBase64(
      const QList<QPair<QString, QString>> &views) const;
  void setBusy(bool busy);
  QString serverUrl() const;

  // Collect base64 images for all checked views; returns list of
  // QPair<label, base64> (e.g. ("Axial", "...")).
  QList<QPair<QString, QString>> collectAttachedImages();

  // Sync the "All 3" tristate checkbox to reflect Axial/Sagittal/Coronal.
  void syncAttachAllState();

  // Backend-specific helpers
  void sendOllamaRequest(const QJsonObject &userMsg,
                         const QList<QPair<QString, QString>> &images);
  void sendLlamaCppRequest(const QJsonObject &userMsg,
                           const QList<QPair<QString, QString>> &images);
  void parseOllamaLine(const QByteArray &line);
  void parseLlamaCppSSELine(const QByteArray &line);
  void fetchOllamaModels();
  void fetchLlamaCppModels();
};

#endif // OLLAMAQAWIDGET_H
