#ifndef I2IRUNDIALOG_H
#define I2IRUNDIALOG_H

#include <QDialog>
#include <QStringList>
#include <vector>
#include <string>

class GlobalUIModel;
class ImageToImageModel;
class QTimer;

namespace Ui
{
class I2IRunDialog;
}

/**
 * Dialog for running an image-to-image (I2I) task against a LaTIM-SNAP
 * deep-learning server, e.g. MR -> CT synthesis. The returned image is added
 * as a new overlay on the selected source layer.
 */
class I2IRunDialog : public QDialog
{
  Q_OBJECT

public:
  I2IRunDialog(ImageToImageModel *i2iModel, GlobalUIModel *parentModel, QWidget *parent = nullptr);
  ~I2IRunDialog();

private slots:
  void on_btnRun_clicked();
  void on_btnRefreshModels_clicked();
  void onPoll();
  void on_buttonBox_rejected();

private:
  void PopulateLayers();
  void PopulateModels();
  void RefreshAxisState();
  std::string GetSelectedAxis() const;
  std::string GetSelectedFusion() const;
  void SetBusy(bool busy);
  void ShowError(const QString &message);

  Ui::I2IRunDialog *ui;
  ImageToImageModel *m_Model;
  GlobalUIModel *m_ParentModel;
  QTimer *m_PollTimer;

  QString m_ActiveModelId;
};

#endif // I2IRUNDIALOG_H
