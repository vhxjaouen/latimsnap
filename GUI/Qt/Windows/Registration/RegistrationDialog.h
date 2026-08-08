#ifndef REGISTRATIONDIALOG_H
#define REGISTRATIONDIALOG_H

#include <SNAPComponent.h>
#include <SNAPCommon.h>

#include <QProcess>

class RegistrationModel;
class QAbstractButton;
class QCheckBox;
class QLabel;
class QLineEdit;
class QProgressBar;
class QProcessOutputTextWidget;
class QPushButton;
class QSpinBox;
class QStackedWidget;
class QToolButton;
class QWidget;
class OptimizationProgressRenderer;

namespace Ui {
class RegistrationDialog;
}

class RegistrationDialog : public SNAPComponent
{
  Q_OBJECT

public:
  explicit RegistrationDialog(QWidget *parent = 0);
  ~RegistrationDialog();

  void SetModel(RegistrationModel *model);

signals:

  void wizardFinished();

private slots:
  void on_pushButton_clicked();

  void on_pushButton_2_clicked();

  void on_btnRunRegistration_clicked();

  void on_btnLoad_clicked();

  void on_btnSave_clicked();

  void on_buttonBox_clicked(QAbstractButton *button);

  void on_tabAutoManual_currentChanged(int index);

  void on_btnReslice_clicked();

  void on_actionImage_Centers_triggered();

  void on_actionCenters_of_Mass_triggered();

  void on_actionMoments_of_Inertia_triggered();

  void onFreeRotationModeChange(const EventBucket &);

  // Deformable pTVreg additions
  void onTransformationChanged();
  void onPythonPathBrowseClicked();
  void onPythonPathTextChanged();
  void onDeformableRunClicked();
  void onDeformableCancelClicked();
  void onDeformableAdoptClicked();
  void onDeformableProcessStdout();
  void onDeformableProcessStderr();
  void onDeformableProcessFinished(int exitCode, QProcess::ExitStatus status);
  void onDeformableSettingsClicked();

private:
  Ui::RegistrationDialog *ui;

  RegistrationModel *m_Model;

  int GetTransformFormat(QString &format);

  // Programmatically-built deformable UI. These pointers are owned by the
  // widget hierarchy of the dialog and do not need manual deletion.
  QWidget                  *m_DeformablePanel;
  QLineEdit                *m_DeformablePythonEdit;
  QToolButton              *m_DeformablePythonBrowse;
  QLabel                   *m_DeformablePythonStatus;
  QCheckBox                *m_DeformableLiveCheck;
  QSpinBox                 *m_DeformableLiveEvery;
  QPushButton              *m_DeformableCancelButton;
  QPushButton              *m_DeformableAdoptButton;
  QPushButton              *m_DeformableSettingsButton;
  QLabel                   *m_DeformableSummaryLabel;
  QLabel                   *m_DeformableStatusLabel;
  QProgressBar             *m_DeformableProgressBar;
  QProcessOutputTextWidget *m_DeformableLogWidget;
  QStackedWidget           *m_DeformableStack;

  void BuildDeformableUi();
  void UpdatePythonStatusIndicator();
  void UpdateDeformableSummaryLabel();
  void PopulateReslicerInterpCombo(class QComboBox *combo);

  // This is a bit unfortunate but we need to keep a list of renderers for the
  // plot widgets currently shown
  typedef SmartPtr<OptimizationProgressRenderer> RendererPtr;

  std::vector<RendererPtr> m_PlotRenderers;


};

#endif // REGISTRATIONDIALOG_H
