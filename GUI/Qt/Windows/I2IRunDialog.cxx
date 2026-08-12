#include "I2IRunDialog.h"
#include "ui_I2IRunDialog.h"

#include "ImageToImageModel.h"
#include "GlobalUIModel.h"
#include "IRISApplication.h"
#include "GenericImageData.h"
#include "ImageWrapperBase.h"

#include <QTimer>
#include <QMessageBox>
#include <QPushButton>
#include <QVariant>

I2IRunDialog::I2IRunDialog(ImageToImageModel *i2iModel, GlobalUIModel *parentModel, QWidget *parent)
  : QDialog(parent), ui(new Ui::I2IRunDialog), m_Model(i2iModel), m_ParentModel(parentModel)
{
  ui->setupUi(this);

  m_PollTimer = new QTimer(this);
  m_PollTimer->setInterval(400);
  connect(m_PollTimer, &QTimer::timeout, this, &I2IRunDialog::onPoll);

  PopulateLayers();
  PopulateModels();
  SetBusy(false);
}

I2IRunDialog::~I2IRunDialog()
{
  m_PollTimer->stop();
  delete ui;
}

void
I2IRunDialog::PopulateLayers()
{
  ui->inLayer->clear();
  auto *driver = m_ParentModel->GetDriver();
  if(!driver || !driver->IsMainImageLoaded())
    return;

  auto *gid = driver->GetCurrentImageData();
  ImageWrapperBase *main = gid->GetMain();
  if(!main)
    return;
  ui->inLayer->addItem(QString::fromStdString(main->GetNickname()),
                       static_cast<unsigned long long>(main->GetUniqueId()));

  // Then overlays
  for(auto it = gid->GetLayers(OVERLAY_ROLE); !it.IsAtEnd(); ++it)
    {
    ImageWrapperBase *ov = it.GetLayer();
    ui->inLayer->addItem(QString::fromStdString(ov->GetNickname()),
                         static_cast<unsigned long long>(ov->GetUniqueId()));
    }

  ui->inLayer->setCurrentIndex(0);
}

void
I2IRunDialog::PopulateModels()
{
  ui->inModel->clear();
  std::vector<std::string> ids, names;
  if(m_Model->FetchAvailableModels(ids, names))
    {
    for(size_t i = 0; i < ids.size(); i++)
      {
      QString label = QString::fromStdString(names[i]);
      if(label.isEmpty())
        label = QString::fromStdString(ids[i]);
      ui->inModel->addItem(label, QString::fromStdString(ids[i]));
      }
    }
  else
    {
    ui->inModel->addItem("(model list unavailable - is the server connected?)");
    }

  if(ui->inModel->count())
    ui->inModel->setCurrentIndex(0);
}

void
I2IRunDialog::on_btnRefreshModels_clicked()
{
  PopulateModels();
}

void
I2IRunDialog::SetBusy(bool busy)
{
  ui->btnRun->setEnabled(!busy);
  ui->inLayer->setEnabled(!busy);
  ui->inModel->setEnabled(!busy);
  if(busy == false)
    m_PollTimer->stop();
}

void
I2IRunDialog::on_btnRun_clicked()
{
  if(ui->inLayer->currentData().isNull() || ui->inModel->currentData().isNull())
    {
    ShowError("Please select a source image and a model.");
    return;
    }

  // Resolve the selected layer
  unsigned long layerId = ui->inLayer->currentData().toULongLong();
  ImageWrapperBase *layer =
      m_ParentModel->GetDriver()->GetCurrentImageData()->FindLayer(layerId, false,
                                                                   MAIN_ROLE | OVERLAY_ROLE);
  if(!layer)
    {
    ShowError("Selected source image is no longer available.");
    return;
    }

  m_ActiveModelId = ui->inModel->currentData().toString();
  m_Model->SetSourceImage(layer);

  std::string error;
  if(!m_Model->StartTransfer(m_ActiveModelId.toStdString(), error))
    {
    ShowError(QString::fromStdString(error));
    return;
    }

  ui->progressTransfer->setValue(0);
  SetBusy(true);
  ui->statusLabel->setText(QString::fromStdString(m_Model->GetStatusMessage()));
  m_PollTimer->start();
}

void
I2IRunDialog::onPoll()
{
  int result = m_Model->PollTransfer();

  if(result == ImageToImageModel::POLL_NETWORK_ERROR)
    {
    m_PollTimer->stop();
    SetBusy(false);
    ui->statusLabel->setText(QString::fromStdString(m_Model->GetStatusMessage()));
    ShowError("Network error while communicating with the server:\n"
              + QString::fromStdString(m_Model->GetStatusMessage()));
    m_Model->Reset();
    return;
    }

  if(result == ImageToImageModel::POLL_ERROR)
    {
    m_PollTimer->stop();
    SetBusy(false);
    ui->statusLabel->setText(QString::fromStdString(m_Model->GetStatusMessage()));
    ShowError(QString::fromStdString(m_Model->GetStatusMessage()));
    m_Model->Reset();
    return;
    }

  if(result == ImageToImageModel::POLL_DONE)
    {
    m_PollTimer->stop();
    ui->progressTransfer->setValue(100);
    ui->statusLabel->setText("Fetching result...");

    std::string error;
    if(m_Model->FetchResult(error))
      {
      ui->statusLabel->setText("Result added as a new overlay image.");
      QMessageBox::information(this, "Image-to-Image",
                               "The transferred image was added as a new "
                               "overlay layer. You can save it from the layer "
                               "context menu.");
      }
    else
      {
      ui->statusLabel->setText("Failed to fetch result: " + QString::fromStdString(error));
      ShowError(QString::fromStdString(error));
      }
    m_Model->Reset();
    SetBusy(false);
    return;
    }

  // POLL_RUNNING - update progress from the model
  ui->progressTransfer->setValue((int)(m_Model->GetTransferProgress() * 100.0));
  ui->statusLabel->setText(QString::fromStdString(m_Model->GetStatusMessage()));
}

void
I2IRunDialog::on_buttonBox_rejected()
{
  m_PollTimer->stop();
  m_Model->Reset();
  reject();
}

void
I2IRunDialog::ShowError(const QString &message)
{
  QMessageBox::warning(this, "Image-to-Image", message);
}
