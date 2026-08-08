#include "RegistrationDialog.h"
#include "ui_RegistrationDialog.h"

#include <QMenu>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QCheckBox>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QProgressBar>
#include <QSpinBox>
#include <QStackedWidget>
#include <QToolButton>
#include <QFileDialog>
#include <QMessageBox>
#include <QDialog>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QDialogButtonBox>

#include "QProcessOutputTextWidget.h"
#include "PTVregSettingsDialog.h"
#include "QtComboBoxCoupling.h"
#include "QtCheckBoxCoupling.h"
#include "QtLineEditCoupling.h"
#include "QtDoubleSpinBoxCoupling.h"
#include "QtSliderCoupling.h"
#include "QtAbstractButtonCoupling.h"
#include "QtPagedWidgetCoupling.h"
#include "QtWidgetArrayCoupling.h"
#include "RegistrationModel.h"
#include "QtWidgetActivator.h"
#include "QtCursorOverride.h"
#include "LoadTransformationDialog.h"
#include "SimpleFileDialogWithHistory.h"
#include "ProcessEventsITKCommand.h"
#include "OptimizationProgressRenderer.h"
#include "QtVTKRenderWindowBox.h"

Q_DECLARE_METATYPE(RegistrationModel::Transformation)
Q_DECLARE_METATYPE(RegistrationModel::SimilarityMetric)

RegistrationDialog::RegistrationDialog(QWidget *parent) :
  SNAPComponent(parent),
  ui(new Ui::RegistrationDialog),
  m_Model(nullptr),
  m_DeformablePanel(nullptr),
  m_DeformablePythonEdit(nullptr),
  m_DeformablePythonBrowse(nullptr),
  m_DeformablePythonStatus(nullptr),
  m_DeformableLiveCheck(nullptr),
  m_DeformableLiveEvery(nullptr),
  m_DeformableCancelButton(nullptr),
  m_DeformableAdoptButton(nullptr),
  m_DeformableSettingsButton(nullptr),
  m_DeformableSummaryLabel(nullptr),
  m_DeformableStatusLabel(nullptr),
  m_DeformableProgressBar(nullptr),
  m_DeformableLogWidget(nullptr),
  m_DeformableStack(nullptr)
{
  ui->setupUi(this);

  // Set up a menu
  QMenu *menuMatch = new QMenu(this);
  menuMatch->addAction(ui->actionImage_Centers);
  menuMatch->addAction(ui->actionCenters_of_Mass);
  menuMatch->addAction(ui->actionMoments_of_Inertia);
  ui->btnMatchCenters->setMenu(menuMatch);

  // Build the (initially hidden) deformable-registration panel.
  this->BuildDeformableUi();
}

RegistrationDialog::~RegistrationDialog()
{
  delete ui;
}



void RegistrationDialog::SetModel(RegistrationModel *model)
{
  m_Model = model;

  // Handling the transition from free rotation to registration is too complex to
  // do through flags and activation. Instead we have a dedicated slot for this
  connectITK(m_Model->GetFreeRotationModeModel(), ValueChangedEvent(),
             SLOT(onFreeRotationModeChange(const EventBucket &)));

  // Couple top page to registration/rotation mode
  std::map<bool, QWidget *> free_rotation_page_map;
  free_rotation_page_map[false] = ui->pgRegistration;
  free_rotation_page_map[true] = ui->pgRotation;
  makePagedWidgetCoupling(ui->stackFreeRotationMode, m_Model->GetFreeRotationModeModel(),
                          free_rotation_page_map);

  makeCoupling(ui->inMovingLayer, m_Model->GetMovingLayerModel());

  // Manual page couplings
  makeCoupling((QAbstractButton *) ui->btnInteractiveTool, m_Model->GetInteractiveToolModel());

  makeArrayCoupling(ui->inRotX, ui->inRotY, ui->inRotZ,
                    m_Model->GetEulerAnglesModel());

  makeArrayCoupling(ui->inRotXSlider, ui->inRotYSlider, ui->inRotZSlider,
                    m_Model->GetEulerAnglesModel());

  makeArrayCoupling(ui->inTranX, ui->inTranY, ui->inTranZ,
                    m_Model->GetTranslationModel());

  makeArrayCoupling(ui->inTranXSlider, ui->inTranYSlider, ui->inTranZSlider,
                    m_Model->GetTranslationModel());

  makeArrayCoupling(ui->inScaleX, ui->inScaleY, ui->inScaleZ,
                    m_Model->GetScalingModel());

  makeArrayCoupling((QAbstractButton *) ui->btnFlipX, (QAbstractButton *) ui->btnFlipY, (QAbstractButton *) ui->btnFlipZ,
                    m_Model->GetFlipModel());

  makeArrayCoupling(ui->inScaleXSlider, ui->inScaleYSlider, ui->inScaleZSlider,
                    m_Model->GetLogScalingModel());

  // Automatic page couplings
  ui->inTransformation->addItem(tr("Rigid"), QVariant(RegistrationModel::RIGID));
  ui->inTransformation->addItem(tr("Affine"), QVariant(RegistrationModel::AFFINE));
  ui->inTransformation->addItem(tr("Deformable (pTVreg)"),
                                QVariant(RegistrationModel::DEFORMABLE_PTVREG));
  makeCoupling(ui->inTransformation, m_Model->GetTransformationModel());

  // React to transformation changes so we can show/hide the deformable UI.
  connect(ui->inTransformation,
          QOverload<int>::of(&QComboBox::currentIndexChanged),
          this, &RegistrationDialog::onTransformationChanged);

  ui->inSimilarityMetric->addItem(tr("Mutual information"), QVariant(RegistrationModel::NMI));
  ui->inSimilarityMetric->addItem(tr("Cross-correlation"), QVariant(RegistrationModel::NCC));
  ui->inSimilarityMetric->addItem(tr("Intensity difference"), QVariant(RegistrationModel::SSD));
  makeCoupling(ui->inSimilarityMetric, m_Model->GetSimilarityMetricModel());
  makeCoupling(ui->inUseMask, m_Model->GetUseSegmentationAsMaskModel());

  makeCoupling(ui->inCoarseLevel, m_Model->GetCoarsestResolutionLevelModel());
  makeCoupling(ui->inFineLevel, m_Model->GetFinestResolutionLevelModel());

  activateOnFlag(ui->inMovingLayer, m_Model,
                 RegistrationModel::UIF_MOVING_SELECTION_AVAILABLE);
  activateOnFlag(ui->pgManual, m_Model,
                 RegistrationModel::UIF_MOVING_SELECTED);
  activateOnFlag(ui->pgAuto, m_Model,
                 RegistrationModel::UIF_MOVING_SELECTED);
  // activateOnAllFlags(ui->grpMovingImage, m_Model,
  //                    RegistrationModel::UIF_MOVING_SELECTED, RegistrationModel::UIF_REGISTRATION_MODE,
  //                    QtWidgetActivator::HideInactive);

  // This command just updates the GUI after each iteration - causing the images to
  // jitter over different iterations
  ProcessEventsITKCommand::Pointer cmdProcEvents = ProcessEventsITKCommand::New();
  m_Model->SetIterationCommand(cmdProcEvents);

  // Populate the persisted Python interpreter path into the deformable panel.
  if(m_DeformablePythonEdit)
    {
    QString stored = m_Model->GetPythonInterpreterPath();
    if(!stored.isEmpty())
      m_DeformablePythonEdit->setText(stored);
    this->UpdatePythonStatusIndicator();
    }

  // Trigger initial visibility update.
  this->onTransformationChanged();
  this->UpdateDeformableSummaryLabel();
}

void RegistrationDialog::on_pushButton_clicked()
{
  m_Model->SetCenterOfRotationToCursor();
}

void RegistrationDialog::on_pushButton_2_clicked()
{
  m_Model->ResetTransformToIdentity();
}

void RegistrationDialog::on_btnRunRegistration_clicked()
{
  // If the user has selected the deformable option, route the click to the
  // Python bridge and return early — the pyramid-plot code below is only for
  // Greedy affine/rigid.
  if(m_Model->GetTransformation() == RegistrationModel::DEFORMABLE_PTVREG)
    {
    this->onDeformableRunClicked();
    return;
    }

  // Create the render panels based on the number of iterations
  int coarsest = m_Model->GetCoarsestResolutionLevel();
  int finest = m_Model->GetFinestResolutionLevel();
  int n_levels = 1 + (coarsest - finest);
  assert(n_levels > 0);

  // Delete all existing render boxes
  QList<QWidget *> bx = ui->grpPlots->findChildren<QWidget*>(QString(), Qt::FindDirectChildrenOnly);
  foreach (QWidget *w, bx)
    delete w;

  // Turn on the wait cursor
  QtCursorOverride cursor(Qt::WaitCursor);

  // Update the user interface
  QCoreApplication::processEvents();

  // Vector of renderers - so they don't disappear
  m_PlotRenderers.clear();
  m_PlotRenderers.resize(n_levels);

  // Background color
  QPalette pMain = ui->pgAuto->palette(), pScroll = ui->scrollPlots->palette();
  pScroll.setBrush(QPalette::Window, pMain.window());
  ui->scrollPlots->setVisible(false);
  ui->scrollPlots->setPalette(pScroll);
  ui->scrollAreaWidgetContents->setPalette(pScroll);

  // Vector of box widgets
  for(int k = 0; k < n_levels; k++)
    {
    // Create a new VTK box
    QtVTKRenderWindowBox *plot = new QtVTKRenderWindowBox(ui->grpPlots);
    m_PlotRenderers[k] = OptimizationProgressRenderer::New();
    m_PlotRenderers[k]->SetModel(m_Model);
    m_PlotRenderers[k]->SetPyramidLevel(k);
    m_PlotRenderers[k]->SetPyramidZoom(1 << (coarsest - k));
    plot->SetRenderer(m_PlotRenderers[k]);
    plot->setMinimumHeight(76);
    plot->setMaximumHeight(76);

    ui->grpPlots->layout()->addWidget(plot);
    }

  // Add a spacer so that the plots behave well
  QVBoxLayout* lo = dynamic_cast<QVBoxLayout *>(ui->grpPlots->layout());
  lo->addStretch(1.0);

  ui->scrollPlots->setVisible(true);

  m_Model->RunAutoRegistration();
}

int RegistrationDialog::GetTransformFormat(QString &format)
{
  if(format == "ITK Transform Files")
    return RegistrationModel::FORMAT_ITK;
  else if(format == "Convert3D Transform Files")
    return RegistrationModel::FORMAT_C3D;
  else
    return RegistrationModel::FORMAT_ITK;
}

void RegistrationDialog::on_btnLoad_clicked()
{
  // Ask for a filename
  LoadTransformationDialog::QueryResult result =
      LoadTransformationDialog::showDialog(this, m_Model->GetParent());

  RegistrationModel::TransformFormat format =
      (RegistrationModel::TransformFormat) this->GetTransformFormat(result.activeFormat);


  // Open
  if(result.filename.length())
    {
    try
      {
      std::string utf = to_utf8(result.filename);
      m_Model->LoadTransform(utf.c_str(), format, result.compose, result.inverse);
      }
    catch(std::exception &exc)
      {
        ReportNonLethalException(
          this, exc, tr("Transform IO Error"), tr("Failed to load transform file"));
      }
    }
}

void RegistrationDialog::on_btnSave_clicked()
{
  // Ask for a filename
  SimpleFileDialogWithHistory::QueryResult result = SimpleFileDialogWithHistory::showSaveDialog(
    this,
    m_Model->GetParent(),
    tr("Save Transform - ITK-SNAP"),
    tr("Transform File"),
    "AffineTransform",
    tr("ITK Transform Files (%1);; Convert3D Transform Files (%2)").arg("*.txt", "*.mat"),
    true);

  RegistrationModel::TransformFormat format =
      (RegistrationModel::TransformFormat) this->GetTransformFormat(result.activeFormat);

  // Save
  if(result.filename.length())
    {
    try
      {
      std::string utf = to_utf8(result.filename);
      m_Model->SaveTransform(utf.c_str(), format);
      }
    catch(std::exception &exc)
      {
        ReportNonLethalException(
          this, exc, tr("Transform IO Error"), tr("Failed to save transform file"));
      }
    }
}

void RegistrationDialog::on_buttonBox_clicked(QAbstractButton *button)
{
  // Tell the model the dialog is closing
  m_Model->OnDialogClosed();

  // The only button is close
  emit wizardFinished();
}

void RegistrationDialog::on_tabAutoManual_currentChanged(int index)
{
  // Activate the interactive tool when the user switches to the manual page
  if(ui->tabAutoManual->currentWidget() == ui->pgManual)
    m_Model->GetInteractiveToolModel()->SetValue(true);
}

#include <QDialog>
#include <QFormLayout>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QDialogButtonBox>

void RegistrationDialog::on_btnReslice_clicked()
{
  // Prompt the user for the kind of reslicing to do
  QDialog *dialog = new QDialog(this);
  QFormLayout *lo = new QFormLayout();

  dialog->setWindowTitle(tr("Reslicing Options - ITK-SNAP"));

  // Set up interpolation options. Per-run default is Linear.
  QComboBox *cbInterp = new QComboBox(dialog);
  this->PopulateReslicerInterpCombo(cbInterp);
  cbInterp->setCurrentIndex(1); // Linear
  lo->addRow(tr("&Interpolation:"), cbInterp);

  // Set up background value (currently unimplemented)
  /*
  QDoubleSpinBox *inBkValue = new QDoubleSpinBox(dialog);
  inBkValue->setValue(0.0);
  inBkValue->setToolTip(
        "Intensity value that will be assigned to voxels that are "
        "outside of the image domain.");
  lo->addRow("&Background intensity:", inBkValue);
  */

  QLabel *label = new QLabel;
  label->setText(tr("The resliced image will be created as an addtional\n"
                 "image layer. You can save the resliced image using\n"
                 "the context menu."));
  lo->addRow(label);

  QDialogButtonBox *bbox =
      new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  connect(bbox, SIGNAL(accepted()), dialog, SLOT(accept()));
  connect(bbox, SIGNAL(rejected()), dialog, SLOT(reject()));
  lo->addRow(bbox);

  dialog->setLayout(lo);


  // Get the selected values
  if(dialog->exec() == QDialog::Accepted)
    {
    int chosen = cbInterp->currentData().toInt();
    InterpolationMethod method = TRILINEAR;
    switch(chosen)
      {
      case NEAREST_NEIGHBOR: method = NEAREST_NEIGHBOR; break;
      case TRILINEAR:        method = TRILINEAR;        break;
      case TRICUBIC:         method = TRICUBIC;         break;
      default:               method = TRILINEAR;        break;
      }
    m_Model->ResliceMovingImage(method);
    }

  delete dialog;
}

void RegistrationDialog::on_actionImage_Centers_triggered()
{
  m_Model->MatchImageCenters();
}

void RegistrationDialog::on_actionCenters_of_Mass_triggered()
{
  // Turn on the wait cursor
  QtCursorOverride cursor(Qt::WaitCursor);
  m_Model->MatchByMoments(1);
}

void RegistrationDialog::on_actionMoments_of_Inertia_triggered()
{
  // Turn on the wait cursor
  QtCursorOverride cursor(Qt::WaitCursor);
  m_Model->MatchByMoments(2);
  }

void RegistrationDialog::onFreeRotationModeChange(const EventBucket &)
{
  std::cout << "ROTATION MODE CHANGE" << std::endl;
  if(m_Model->GetFreeRotationMode())
    {
    ui->tabAutoManual->setTabEnabled(ui->tabAutoManual->indexOf(ui->pgAuto), false);
    ui->grpMovingImage->setVisible(false);
    ui->grpScaling->setVisible(false);
    ui->btnMatchCenters->setVisible(false);
    }
  else
    {
    ui->tabAutoManual->setTabEnabled(ui->tabAutoManual->indexOf(ui->pgAuto), true);
    ui->grpMovingImage->setVisible(true);
    ui->grpScaling->setVisible(true);
    ui->btnMatchCenters->setVisible(true);
    }
}


// ==========================================================================
// Deformable pTVreg UI
// ==========================================================================

void RegistrationDialog::PopulateReslicerInterpCombo(QComboBox *combo)
{
  combo->clear();
  combo->addItem(tr("Nearest Neighbor"), QVariant(NEAREST_NEIGHBOR));
  combo->addItem(tr("Linear"),           QVariant(TRILINEAR));
  combo->addItem(tr("B-Spline (order 3)"), QVariant(TRICUBIC));
}

void RegistrationDialog::BuildDeformableUi()
{
  // Panel that will live below the Run button on the Auto page, hidden by
  // default. It hosts Python-path selection, the live-preview checkbox,
  // status/log/progress widgets and Cancel/Adopt buttons.
  m_DeformablePanel = new QWidget(this);
  m_DeformablePanel->setObjectName("deformablePanel");
  m_DeformablePanel->setVisible(false);

  QVBoxLayout *root = new QVBoxLayout(m_DeformablePanel);
  root->setContentsMargins(0, 4, 0, 0);
  root->setSpacing(4);

  // Python interpreter row.
  QHBoxLayout *pyRow = new QHBoxLayout;
  pyRow->setSpacing(4);
  QLabel *pyLabel = new QLabel(tr("Python (venv):"), m_DeformablePanel);
  pyLabel->setStyleSheet("font-size:11px;");
  m_DeformablePythonEdit = new QLineEdit(m_DeformablePanel);
  m_DeformablePythonEdit->setPlaceholderText(
    tr("/path/to/venv/bin/python"));
  m_DeformablePythonBrowse = new QToolButton(m_DeformablePanel);
  m_DeformablePythonBrowse->setText("...");
  m_DeformablePythonStatus = new QLabel(m_DeformablePanel);
  m_DeformablePythonStatus->setText("●");
  m_DeformablePythonStatus->setStyleSheet("color:gray;");
  m_DeformablePythonStatus->setToolTip(
    tr("pTVreg import status. Green = OK, red = not importable, grey = untested."));

  pyRow->addWidget(pyLabel);
  pyRow->addWidget(m_DeformablePythonEdit, 1);
  pyRow->addWidget(m_DeformablePythonBrowse);
  pyRow->addWidget(m_DeformablePythonStatus);
  root->addLayout(pyRow);

  // Live-preview row.
  QHBoxLayout *liveRow = new QHBoxLayout;
  liveRow->setSpacing(4);
  m_DeformableLiveCheck = new QCheckBox(
    tr("Live preview every"), m_DeformablePanel);
  m_DeformableLiveCheck->setStyleSheet("font-size:11px;");
  m_DeformableLiveEvery = new QSpinBox(m_DeformablePanel);
  m_DeformableLiveEvery->setRange(1, 100);
  m_DeformableLiveEvery->setValue(10);
  m_DeformableLiveEvery->setSuffix(tr(" iterations (slower)"));
  m_DeformableLiveEvery->setEnabled(false);
  connect(m_DeformableLiveCheck, &QCheckBox::toggled,
          m_DeformableLiveEvery, &QSpinBox::setEnabled);
  liveRow->addWidget(m_DeformableLiveCheck);
  liveRow->addWidget(m_DeformableLiveEvery, 1);
  root->addLayout(liveRow);

  // Settings summary row.
  QHBoxLayout *setRow = new QHBoxLayout;
  setRow->setSpacing(4);
  m_DeformableSummaryLabel = new QLabel(m_DeformablePanel);
  m_DeformableSummaryLabel->setStyleSheet(
    "font-size:11px; color:#333;");
  m_DeformableSummaryLabel->setWordWrap(true);
  m_DeformableSettingsButton = new QPushButton(
    tr("pTVreg Settings..."), m_DeformablePanel);
  m_DeformableSettingsButton->setToolTip(
    tr("Open the full pTVreg parameter dialog."));
  setRow->addWidget(m_DeformableSummaryLabel, 1);
  setRow->addWidget(m_DeformableSettingsButton);
  root->addLayout(setRow);
  connect(m_DeformableSettingsButton, &QPushButton::clicked,
          this, &RegistrationDialog::onDeformableSettingsClicked);

  // Status label + progress bar.
  m_DeformableStatusLabel = new QLabel(
    tr("Deformable registration idle."), m_DeformablePanel);
  m_DeformableStatusLabel->setStyleSheet("font-size:11px;");
  m_DeformableStatusLabel->setWordWrap(true);
  root->addWidget(m_DeformableStatusLabel);

  m_DeformableProgressBar = new QProgressBar(m_DeformablePanel);
  m_DeformableProgressBar->setRange(0, 1);
  m_DeformableProgressBar->setValue(0);
  m_DeformableProgressBar->setTextVisible(true);
  root->addWidget(m_DeformableProgressBar);

  // Log widget.
  m_DeformableLogWidget = new QProcessOutputTextWidget(m_DeformablePanel);
  m_DeformableLogWidget->setMinimumHeight(160);
  root->addWidget(m_DeformableLogWidget, 1);

  // Button row.
  QHBoxLayout *btnRow = new QHBoxLayout;
  m_DeformableCancelButton = new QPushButton(tr("Cancel"), m_DeformablePanel);
  m_DeformableCancelButton->setEnabled(false);
  m_DeformableAdoptButton = new QPushButton(
    tr("Adopt preview as moving image"), m_DeformablePanel);
  m_DeformableAdoptButton->setEnabled(false);
  btnRow->addWidget(m_DeformableCancelButton);
  btnRow->addStretch(1);
  btnRow->addWidget(m_DeformableAdoptButton);
  root->addLayout(btnRow);

  // Slot connections.
  connect(m_DeformablePythonBrowse, &QToolButton::clicked,
          this, &RegistrationDialog::onPythonPathBrowseClicked);
  connect(m_DeformablePythonEdit, &QLineEdit::editingFinished,
          this, &RegistrationDialog::onPythonPathTextChanged);
  connect(m_DeformableCancelButton, &QPushButton::clicked,
          this, &RegistrationDialog::onDeformableCancelClicked);
  connect(m_DeformableAdoptButton, &QPushButton::clicked,
          this, &RegistrationDialog::onDeformableAdoptClicked);

  // Insert the panel into the same containing layout as the plot scroll area
  // so it lives right beside it. We locate that parent layout dynamically —
  // this is safer than editing the .ui XML.
  if(ui->scrollPlots && ui->scrollPlots->parentWidget())
    {
    QWidget *host = ui->scrollPlots->parentWidget();
    if(QVBoxLayout *hostLo = qobject_cast<QVBoxLayout *>(host->layout()))
      {
      hostLo->addWidget(m_DeformablePanel, 1);
      }
    else if(host->layout())
      {
      host->layout()->addWidget(m_DeformablePanel);
      }
    }
}

void RegistrationDialog::onTransformationChanged()
{
  if(!m_Model || !m_DeformablePanel) return;

  const bool deformable =
    (m_Model->GetTransformation() == RegistrationModel::DEFORMABLE_PTVREG);

  m_DeformablePanel->setVisible(deformable);
  if(ui->scrollPlots)
    ui->scrollPlots->setVisible(!deformable);

  if(deformable)
    this->UpdateDeformableSummaryLabel();
}

void RegistrationDialog::UpdateDeformableSummaryLabel()
{
  if(!m_Model || !m_DeformableSummaryLabel) return;
  m_DeformableSummaryLabel->setText(
    m_Model->GetPTVregSettings().ToSummaryString());
}

void RegistrationDialog::onDeformableSettingsClicked()
{
  if(!m_Model) return;
  PTVregSettingsDialog dlg(m_Model, this);
  if(dlg.exec() == QDialog::Accepted)
    this->UpdateDeformableSummaryLabel();
}

void RegistrationDialog::onPythonPathBrowseClicked()
{
  QString start = m_DeformablePythonEdit->text();
  if(start.isEmpty()) start = QDir::homePath();
  QString file = QFileDialog::getOpenFileName(
    this, tr("Select Python interpreter (venv)"), start);
  if(!file.isEmpty())
    {
    m_DeformablePythonEdit->setText(file);
    this->onPythonPathTextChanged();
    }
}

void RegistrationDialog::onPythonPathTextChanged()
{
  if(!m_Model) return;
  m_Model->SetPythonInterpreterPath(m_DeformablePythonEdit->text().trimmed());
  this->UpdatePythonStatusIndicator();
}

void RegistrationDialog::UpdatePythonStatusIndicator()
{
  if(!m_Model || !m_DeformablePythonStatus) return;

  QString err;
  bool ok = m_Model->VerifyPythonEnvironment(err);
  if(ok)
    {
    m_DeformablePythonStatus->setStyleSheet("color:green; font-weight:bold;");
    m_DeformablePythonStatus->setToolTip(tr("pTVreg is importable."));
    }
  else
    {
    m_DeformablePythonStatus->setStyleSheet("color:red; font-weight:bold;");
    m_DeformablePythonStatus->setToolTip(err);
    }
}

void RegistrationDialog::onDeformableRunClicked()
{
  if(!m_Model) return;

  // Push the currently-entered path into the model before starting.
  m_Model->SetPythonInterpreterPath(m_DeformablePythonEdit->text().trimmed());

  QString err;
  if(!m_Model->VerifyPythonEnvironment(err))
    {
    QMessageBox::warning(this, tr("pTVreg not available"), err);
    return;
    }

  // Clear the log and start.
  if(m_DeformableLogWidget) m_DeformableLogWidget->clear();
  if(m_DeformableStatusLabel)
    m_DeformableStatusLabel->setText(tr("Starting pTVreg..."));

  const bool live = m_DeformableLiveCheck && m_DeformableLiveCheck->isChecked();
  const int  every = m_DeformableLiveEvery ? m_DeformableLiveEvery->value() : 0;

  m_Model->RunDeformableRegistration(
    m_DeformableLogWidget,
    m_DeformableStatusLabel,
    m_DeformableProgressBar,
    live,
    every);

  // Connect the process signals to our forwarding slots.
  QProcess *proc = m_Model->GetDeformableProcess();
  if(proc)
    {
    connect(proc, &QProcess::readyReadStandardOutput,
            this, &RegistrationDialog::onDeformableProcessStdout);
    connect(proc, &QProcess::readyReadStandardError,
            this, &RegistrationDialog::onDeformableProcessStderr);
    connect(proc,
            QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this,
            &RegistrationDialog::onDeformableProcessFinished);
    }

  m_DeformableCancelButton->setEnabled(true);
  m_DeformableAdoptButton->setEnabled(false);
}

void RegistrationDialog::onDeformableCancelClicked()
{
  if(!m_Model) return;
  m_Model->CancelDeformableRegistration();
  if(m_DeformableCancelButton) m_DeformableCancelButton->setEnabled(false);
}

void RegistrationDialog::onDeformableAdoptClicked()
{
  if(!m_Model) return;
  m_Model->AdoptPreviewAsMovingImage();
  if(m_DeformableAdoptButton) m_DeformableAdoptButton->setEnabled(false);
}

void RegistrationDialog::onDeformableProcessStdout()
{
  if(!m_Model) return;
  QProcess *proc = m_Model->GetDeformableProcess();
  if(!proc) return;
  m_Model->HandleDeformableStdout(proc->readAllStandardOutput());
}

void RegistrationDialog::onDeformableProcessStderr()
{
  if(!m_Model) return;
  QProcess *proc = m_Model->GetDeformableProcess();
  if(!proc) return;
  m_Model->HandleDeformableStderr(proc->readAllStandardError());
}

void RegistrationDialog::onDeformableProcessFinished(int exitCode,
                                                     QProcess::ExitStatus status)
{
  if(!m_Model) return;
  m_Model->HandleDeformableFinished(exitCode, (int)status);
  if(m_DeformableCancelButton) m_DeformableCancelButton->setEnabled(false);
  if(m_DeformableAdoptButton)
    m_DeformableAdoptButton->setEnabled(m_Model->HasDeformablePreview());
}
