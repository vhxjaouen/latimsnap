#include "PTVregSettingsDialog.h"

#include "RegistrationModel.h"
#include "GlobalUIModel.h"
#include "IRISApplication.h"
#include "GenericImageData.h"
#include "LayerIterator.h"
#include "ImageWrapperBase.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QTabWidget>
#include <QVBoxLayout>

PTVregSettingsDialog::PTVregSettingsDialog(RegistrationModel *model,
                                            QWidget *parent)
  : QDialog(parent), m_Model(model)
{
  setWindowTitle(tr("pTVreg Settings"));
  setModal(true);
  this->BuildUi();
  if(m_Model)
    this->ApplySettingsToUi(m_Model->GetPTVregSettings());
  this->RefreshVfcVisibility();
}

PTVregSettingsDialog::~PTVregSettingsDialog() = default;

void PTVregSettingsDialog::BuildUi()
{
  m_tabs = new QTabWidget(this);

  // -----------------------------------------------------------------------
  // Tab 1 — Grid & Optimization
  // -----------------------------------------------------------------------
  QWidget *pageGrid = new QWidget;
  QFormLayout *lgrid = new QFormLayout(pageGrid);

  m_spinGridSpacing = new QSpinBox;
  m_spinGridSpacing->setRange(1, 128);
  m_spinGridSpacing->setSuffix(tr(" voxels"));
  m_spinGridSpacing->setToolTip(
    tr("Control-point grid spacing at the finest level. "
       "Typical: 4/8/16. Smaller = finer deformation but slower."));
  lgrid->addRow(tr("Grid spacing (--spacing):"), m_spinGridSpacing);

  m_spinScaleFactor = new QDoubleSpinBox;
  m_spinScaleFactor->setRange(0.05, 1.0);
  m_spinScaleFactor->setSingleStep(0.05);
  m_spinScaleFactor->setDecimals(3);
  m_spinScaleFactor->setToolTip(
    tr("Downsample input images by this factor before registration."
       " 0.5 halves each axis (≈1/8 memory)."));
  lgrid->addRow(tr("Downsample factor (--scale_factor):"), m_spinScaleFactor);

  m_editIterations = new QLineEdit;
  m_editIterations->setPlaceholderText(tr("e.g. 100  or  50 100 100"));
  m_editIterations->setToolTip(
    tr("LBFGS iterations per pyramid level (coarsest→finest). "
       "Enter one integer to reuse for all levels, or a whitespace-separated "
       "list to specify per level."));
  lgrid->addRow(tr("Iterations (--iterations):"), m_editIterations);

  m_editLambdaReg = new QLineEdit;
  m_editLambdaReg->setPlaceholderText(tr("e.g. 0.15  or  0.05 0.15 0.30"));
  m_editLambdaReg->setToolTip(
    tr("TV regularisation weight. Enter one float for all levels or a list."));
  lgrid->addRow(tr("Regularisation λ (--lambda_reg):"), m_editLambdaReg);

  m_tabs->addTab(pageGrid, tr("Grid && Optimization"));

  // -----------------------------------------------------------------------
  // Tab 2 — Similarity Metric & VFC
  // -----------------------------------------------------------------------
  QWidget *pageMetric = new QWidget;
  QFormLayout *lmet = new QFormLayout(pageMetric);

  m_comboMetric = new QComboBox;
  m_comboMetric->addItem(tr("Local Cross-Correlation (LCC)"), "lcc");
  m_comboMetric->addItem(tr("Sum of Squared Differences (SSD)"), "ssd");
  m_comboMetric->addItem(tr("Nuclear Norm"), "nuclear");
  m_comboMetric->addItem(tr("Expected Mean Squared Error (EMSE)"), "emse");
  m_comboMetric->addItem(tr("Vector Field Convolution (VFC)"), "vfc");
  m_comboMetric->setToolTip(
    tr("Similarity metric optimised by pTVreg."));
  connect(m_comboMetric, QOverload<int>::of(&QComboBox::currentIndexChanged),
          this, &PTVregSettingsDialog::onMetricChanged);
  lmet->addRow(tr("Metric (--metric):"), m_comboMetric);

  m_spinMetricParam = new QDoubleSpinBox;
  m_spinMetricParam->setRange(0.1, 20.0);
  m_spinMetricParam->setSingleStep(0.1);
  m_spinMetricParam->setDecimals(3);
  m_spinMetricParam->setSuffix(tr(" mm"));
  m_spinMetricParam->setToolTip(
    tr("LCC Gaussian sigma in millimetres. Typical range 1.5–4."));
  lmet->addRow(tr("Metric parameter (--metric_param):"), m_spinMetricParam);

  m_GrpVfc = new QGroupBox(tr("Vector Field Convolution (VFC)"));
  QFormLayout *lvfc = new QFormLayout(m_GrpVfc);

  m_spinVfcRadius = new QDoubleSpinBox;
  m_spinVfcRadius->setRange(0.1, 500.0);
  m_spinVfcRadius->setDecimals(3);
  m_spinVfcRadius->setSuffix(tr(" mm"));
  m_spinVfcRadius->setToolTip(
    tr("Characteristic damping distance. Larger = more diffuse field."));
  m_lblVfcRadius = new QLabel(tr("VFC radius R (--vfc-radius):"));
  lvfc->addRow(m_lblVfcRadius, m_spinVfcRadius);

  m_spinVfcBeta = new QDoubleSpinBox;
  m_spinVfcBeta->setRange(0.01, 10.0);
  m_spinVfcBeta->setSingleStep(0.1);
  m_spinVfcBeta->setDecimals(3);
  m_spinVfcBeta->setToolTip(
    tr("Weibull shape exponent. β=2 Gaussian, β=1 exponential."));
  m_lblVfcBeta = new QLabel(tr("VFC beta β (--vfc-beta):"));
  lvfc->addRow(m_lblVfcBeta, m_spinVfcBeta);

  m_chkVfcSignInvariant = new QCheckBox(
    tr("Sign-invariant (cos² alignment) (--vfc-sign-invariant)"));
  m_chkVfcSignInvariant->setToolTip(
    tr("Enable when edge maps are extracted from label boundaries."));
  lvfc->addRow(m_chkVfcSignInvariant);

  m_chkVfcNormalize = new QCheckBox(
    tr("L₂-normalise VFC fields to unit vectors (--vfc-normalize)"));
  m_chkVfcNormalize->setToolTip(
    tr("Makes the metric purely directional."));
  lvfc->addRow(m_chkVfcNormalize);

  lmet->addRow(m_GrpVfc);

  m_tabs->addTab(pageMetric, tr("Metric && VFC"));

  // -----------------------------------------------------------------------
  // Tab 3 — Penalties & Boundary
  // -----------------------------------------------------------------------
  QWidget *pagePen = new QWidget;
  QFormLayout *lpen = new QFormLayout(pagePen);

  m_spinLambdaJac = new QDoubleSpinBox;
  m_spinLambdaJac->setRange(0.0, 100.0);
  m_spinLambdaJac->setSingleStep(0.1);
  m_spinLambdaJac->setDecimals(3);
  m_spinLambdaJac->setToolTip(
    tr("Weight for the Jacobian-determinant penalty against folding. "
       "0 disables. Typical: 0.5–5."));
  lpen->addRow(tr("Jacobian penalty λ_jac (--lambda_jac):"), m_spinLambdaJac);

  m_spinDvfEpsilon = new QDoubleSpinBox;
  m_spinDvfEpsilon->setRange(0.0, 10.0);
  m_spinDvfEpsilon->setSingleStep(0.05);
  m_spinDvfEpsilon->setDecimals(3);
  m_spinDvfEpsilon->setSuffix(tr(" mm"));
  m_spinDvfEpsilon->setToolTip(
    tr("Zero out displacement vectors below this physical magnitude."
       " 0 disables the sparsification."));
  lpen->addRow(tr("DVF sparsification ε (--dvf_epsilon):"), m_spinDvfEpsilon);

  m_spinBorderMask = new QSpinBox;
  m_spinBorderMask->setRange(0, 200);
  m_spinBorderMask->setSuffix(tr(" voxels"));
  m_spinBorderMask->setToolTip(
    tr("Number of boundary voxels excluded from the metric to avoid "
       "padding artefacts."));
  lpen->addRow(tr("Border margin (--border_mask):"), m_spinBorderMask);

  m_chkUseClip = new QCheckBox(tr("Clip intensities before normalisation (--clip)"));
  lpen->addRow(m_chkUseClip);

  m_spinClipMin = new QDoubleSpinBox;
  m_spinClipMin->setRange(-1e6, 1e6);
  m_spinClipMin->setDecimals(3);
  lpen->addRow(tr("Clip min:"), m_spinClipMin);
  m_spinClipMax = new QDoubleSpinBox;
  m_spinClipMax->setRange(-1e6, 1e6);
  m_spinClipMax->setDecimals(3);
  lpen->addRow(tr("Clip max:"), m_spinClipMax);
  connect(m_chkUseClip, &QCheckBox::toggled,
          m_spinClipMin, &QDoubleSpinBox::setEnabled);
  connect(m_chkUseClip, &QCheckBox::toggled,
          m_spinClipMax, &QDoubleSpinBox::setEnabled);

  m_tabs->addTab(pagePen, tr("Penalties && Boundary"));

  // -----------------------------------------------------------------------
  // Tab 4 — Label-guided soft-Dice
  // -----------------------------------------------------------------------
  QWidget *pageLabels = new QWidget;
  QFormLayout *llab = new QFormLayout(pageLabels);

  m_spinDiceWeight = new QDoubleSpinBox;
  m_spinDiceWeight->setRange(0.0, 100.0);
  m_spinDiceWeight->setDecimals(3);
  m_spinDiceWeight->setSingleStep(0.05);
  m_spinDiceWeight->setToolTip(
    tr("Weight for the soft-Dice label loss term. 0 disables."));
  llab->addRow(tr("Soft-Dice weight (--dice-weight):"), m_spinDiceWeight);

  m_comboFixedLabels = new QComboBox;
  m_comboMovingLabels = new QComboBox;
  this->PopulateLayerCombo(m_comboFixedLabels);
  this->PopulateLayerCombo(m_comboMovingLabels);
  m_comboFixedLabels->setToolTip(
    tr("Segmentation / label overlay aligned to the fixed image."));
  m_comboMovingLabels->setToolTip(
    tr("Segmentation / label overlay aligned to the moving image."));
  llab->addRow(tr("Fixed labels (--fixed-labels):"), m_comboFixedLabels);
  llab->addRow(tr("Moving labels (--moving-labels):"), m_comboMovingLabels);

  m_tabs->addTab(pageLabels, tr("Labels"));

  // Root layout
  QVBoxLayout *root = new QVBoxLayout(this);
  root->addWidget(m_tabs, 1);

  QDialogButtonBox *bb = new QDialogButtonBox(
    QDialogButtonBox::Ok | QDialogButtonBox::Cancel |
    QDialogButtonBox::Reset);
  root->addWidget(bb);

  connect(bb, &QDialogButtonBox::accepted, this, &PTVregSettingsDialog::onAccept);
  connect(bb, &QDialogButtonBox::rejected, this, &QDialog::reject);
  QPushButton *btnReset = bb->button(QDialogButtonBox::Reset);
  btnReset->setText(tr("Reset to Defaults"));
  connect(btnReset, &QPushButton::clicked,
          this, &PTVregSettingsDialog::onResetToDefaults);

  resize(560, 480);
}

void PTVregSettingsDialog::PopulateLayerCombo(QComboBox *combo)
{
  combo->clear();
  combo->addItem(tr("(none)"), (qulonglong)0);
  if(!m_Model || !m_Model->GetParent()) return;
  IRISApplication *app = m_Model->GetParent()->GetDriver();
  if(!app) return;
  GenericImageData *gid = app->GetCurrentImageData();
  if(!gid) return;
  // List every non-mesh layer as a candidate.
  for(LayerIterator it = gid->GetLayers(
        MAIN_ROLE | OVERLAY_ROLE | LABEL_ROLE);
      !it.IsAtEnd(); ++it)
    {
    ImageWrapperBase *w = it.GetLayer();
    if(!w) continue;
    combo->addItem(
      QString::fromStdString(w->GetNickname()),
      (qulonglong)w->GetUniqueId());
    }
}

void PTVregSettingsDialog::ApplySettingsToUi(const PTVregSettings &s)
{
  m_spinGridSpacing->setValue(s.gridSpacing);
  m_spinScaleFactor->setValue(s.scaleFactor);
  m_editIterations->setText(s.iterations);
  m_editLambdaReg->setText(s.lambdaReg);

  int idx = m_comboMetric->findData(s.metric);
  m_comboMetric->setCurrentIndex(idx >= 0 ? idx : 0);

  m_spinMetricParam->setValue(s.metricParam);
  m_spinVfcRadius->setValue(s.vfcRadius);
  m_spinVfcBeta->setValue(s.vfcBeta);
  m_chkVfcSignInvariant->setChecked(s.vfcSignInvariant);
  m_chkVfcNormalize->setChecked(s.vfcNormalize);

  m_spinLambdaJac->setValue(s.lambdaJac);
  m_spinDvfEpsilon->setValue(s.dvfEpsilon);
  m_spinBorderMask->setValue(s.borderMask);
  m_chkUseClip->setChecked(s.useClip);
  m_spinClipMin->setEnabled(s.useClip);
  m_spinClipMax->setEnabled(s.useClip);
  m_spinClipMin->setValue(s.clipMin);
  m_spinClipMax->setValue(s.clipMax);

  m_spinDiceWeight->setValue(s.diceWeight);
  int fi = m_comboFixedLabels->findData((qulonglong)s.fixedLabelsLayerId);
  m_comboFixedLabels->setCurrentIndex(fi >= 0 ? fi : 0);
  int mi = m_comboMovingLabels->findData((qulonglong)s.movingLabelsLayerId);
  m_comboMovingLabels->setCurrentIndex(mi >= 0 ? mi : 0);
}

void PTVregSettingsDialog::ReadSettingsFromUi(PTVregSettings &s) const
{
  s.gridSpacing         = m_spinGridSpacing->value();
  s.scaleFactor         = m_spinScaleFactor->value();
  s.iterations          = m_editIterations->text().trimmed();
  s.lambdaReg           = m_editLambdaReg->text().trimmed();
  s.metric              = m_comboMetric->currentData().toString();
  s.metricParam         = m_spinMetricParam->value();
  s.vfcRadius           = m_spinVfcRadius->value();
  s.vfcBeta             = m_spinVfcBeta->value();
  s.vfcSignInvariant    = m_chkVfcSignInvariant->isChecked();
  s.vfcNormalize        = m_chkVfcNormalize->isChecked();
  s.lambdaJac           = m_spinLambdaJac->value();
  s.dvfEpsilon          = m_spinDvfEpsilon->value();
  s.borderMask          = m_spinBorderMask->value();
  s.useClip             = m_chkUseClip->isChecked();
  s.clipMin             = m_spinClipMin->value();
  s.clipMax             = m_spinClipMax->value();
  s.diceWeight          = m_spinDiceWeight->value();
  s.fixedLabelsLayerId  = (unsigned long)
                          m_comboFixedLabels->currentData().toULongLong();
  s.movingLabelsLayerId = (unsigned long)
                          m_comboMovingLabels->currentData().toULongLong();
}

void PTVregSettingsDialog::onResetToDefaults()
{
  PTVregSettings defaults;
  this->ApplySettingsToUi(defaults);
  this->RefreshVfcVisibility();
}

void PTVregSettingsDialog::onMetricChanged(int)
{
  this->RefreshVfcVisibility();
}

void PTVregSettingsDialog::RefreshVfcVisibility()
{
  const bool isVfc = (m_comboMetric->currentData().toString() == "vfc");
  // Hide the whole VFC group when a non-VFC metric is selected so the tab is
  // not cluttered with widgets that have no effect.
  if(m_GrpVfc) m_GrpVfc->setVisible(isVfc);
  m_spinVfcRadius->setVisible(isVfc);
  m_spinVfcBeta->setVisible(isVfc);
  m_chkVfcSignInvariant->setVisible(isVfc);
  m_chkVfcNormalize->setVisible(isVfc);
  if(m_lblVfcRadius) m_lblVfcRadius->setVisible(isVfc);
  if(m_lblVfcBeta)   m_lblVfcBeta->setVisible(isVfc);
}

void PTVregSettingsDialog::onAccept()
{
  if(m_Model)
    {
    this->ReadSettingsFromUi(m_Model->GetPTVregSettings());
    m_Model->SavePTVregSettings();
    }
  accept();
}
