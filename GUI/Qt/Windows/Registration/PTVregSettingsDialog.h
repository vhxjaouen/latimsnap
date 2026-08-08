#ifndef PTVREGSETTINGSDIALOG_H
#define PTVREGSETTINGSDIALOG_H

#include <QDialog>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QSpinBox;
class QTabWidget;

class GlobalUIModel;
class RegistrationModel;
struct PTVregSettings;

/**
 *  Modal dialog exposing every pTVreg CLI argument surfaced to the user.
 *  Reads the initial values from the RegistrationModel and, on accept,
 *  writes them back and persists them via SystemInterface.
 */
class PTVregSettingsDialog : public QDialog
{
  Q_OBJECT
public:
  explicit PTVregSettingsDialog(RegistrationModel *model, QWidget *parent = nullptr);
  ~PTVregSettingsDialog() override;

private slots:
  void onResetToDefaults();
  void onMetricChanged(int index);
  void onAccept();

private:
  RegistrationModel *m_Model;

  // Grid & Optimization
  QSpinBox       *m_spinGridSpacing;
  QDoubleSpinBox *m_spinScaleFactor;
  QLineEdit      *m_editIterations;
  QLineEdit      *m_editLambdaReg;

  // Similarity
  QComboBox      *m_comboMetric;
  QDoubleSpinBox *m_spinMetricParam;

  // VFC
  QDoubleSpinBox *m_spinVfcRadius;
  QDoubleSpinBox *m_spinVfcBeta;
  QCheckBox      *m_chkVfcSignInvariant;
  QCheckBox      *m_chkVfcNormalize;
  QLabel         *m_lblVfcRadius;
  QLabel         *m_lblVfcBeta;

  // Penalties & boundary
  QDoubleSpinBox *m_spinLambdaJac;
  QDoubleSpinBox *m_spinDvfEpsilon;
  QSpinBox       *m_spinBorderMask;
  QCheckBox      *m_chkUseClip;
  QDoubleSpinBox *m_spinClipMin;
  QDoubleSpinBox *m_spinClipMax;

  // Label-guided
  QDoubleSpinBox *m_spinDiceWeight;
  QComboBox      *m_comboFixedLabels;
  QComboBox      *m_comboMovingLabels;

  QTabWidget     *m_tabs;

  void BuildUi();
  void PopulateLayerCombo(QComboBox *combo);
  void ApplySettingsToUi(const PTVregSettings &s);
  void ReadSettingsFromUi(PTVregSettings &s) const;
  void RefreshVfcVisibility();
};

#endif // PTVREGSETTINGSDIALOG_H
