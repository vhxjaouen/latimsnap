#include "ImageToImageModel.h"
#include "GlobalUIModel.h"
#include "IRISApplication.h"
#include "GenericImageData.h"
#include "IRISException.h"
#include "RESTClient.h"
#include "base64.h"
#include "DLSUtility.h"
#include "DeepLearningSegmentationModel.h"
#include "UIReporterDelegates.h"
#include "AllPurposeProgressAccumulator.h"
#include "QtCursorOverride.h"

#include "itkVectorImage.h"
#include "ImageWrapperTraits.h"
#include "VectorImageWrapper.h"
#include <chrono>
#include <cstring>

ImageToImageModel::ImageToImageModel()
{
  m_RESTSharedData = new RESTSharedDataType();
  m_TransferProgressModel = NewRangedConcreteProperty(0.0, 0.0, 1.0, 0.01);
  m_StatusMessageModel = NewSimpleConcreteProperty(std::string("Idle"));
  m_IsRunningModel = NewSimpleConcreteProperty(false);
  m_ParentModel = nullptr;
}

ImageToImageModel::~ImageToImageModel()
{
  delete m_RESTSharedData;
}

void
ImageToImageModel::SetParentModel(GlobalUIModel *parent)
{
  m_ParentModel = parent;
}

void
ImageToImageModel::SetSourceImage(ImageWrapperBase *layer)
{
  m_SourceImage = layer;
}

std::string
ImageToImageModel::GetActiveServerURL()
{
  auto *dls = m_ParentModel->GetDeepLearningSegmentationModel();
  if(!dls || !dls->GetServerProperties())
    return std::string();
  return dls->GetActiveServerURL();
}

bool
ImageToImageModel::EnsureSessionAndUpload(RESTClientType &cli, std::string &error_out)
{
  // Ensure a session with the I2I server
  if(m_ActiveSession.size() == 0)
    {
    if(!cli.Get("start_session"))
      {
      error_out = std::string("Error creating session on I2I server: ") + cli.GetErrorString();
      return false;
      }

    std::string out = cli.GetOutput() ? cli.GetOutput() : "";
    Json::CharReaderBuilder rbuilder;
    std::unique_ptr<Json::CharReader> reader(rbuilder.newCharReader());
    Json::Value root;
    std::string errs;
    if(!reader->parse(out.data(), out.data() + out.size(),
                      &root, &errs) || !root.isMember("session_id"))
      {
      error_out = "I2I server returned an unexpected session response";
      return false;
      }
    m_ActiveSession = root["session_id"].asString();
    }

  // Upload the source image only if it changed
  ImageWrapperBase *layer = m_SourceImage;
  auto *driver = m_ParentModel->GetDriver();
  unsigned int tp = driver->GetCursorTimePoint();
  LayerSelection sel(layer->GetUniqueId(), (int) tp);
  if(m_UploadedLayer != sel)
    {
    using FloatImageType = ImageWrapperBase::FloatImageType;
    FloatImageType *src = layer->GetDefaultScalarRepresentation()
                              ->CreateCastToFloatPipeline("I2IExport");
    src->GetSource()->Update();

    RESTMultipartData mpd;
    std::string gzip_buffer;
    dls_utility::EncodeImage(mpd, src, gzip_buffer);

    auto *pdel = m_ParentModel->GetProgressReporterDelegate();
    pdel->Show("Uploading image to server...");
    SmartPtr<itk::Command> cmd_progress = pdel->CreateCommand();
    SmartPtr<AllPurposeProgressAccumulator> accum = AllPurposeProgressAccumulator::New();
    accum->AddObserver(itk::ProgressEvent(), cmd_progress);
    void *transfer_progress_src = accum->RegisterGenericSource(1, 1.0);
    cli.SetProgressCallback(transfer_progress_src,
                            AllPurposeProgressAccumulator::GenericProgressCallback);

    if(!cli.PostMultipart("upload_raw/%s?filename=upload.nii.gz", &mpd, m_ActiveSession.c_str()))
      {
      pdel->Hide();
      layer->ReleaseInternalPipeline("I2IExport");
      accum->UnregisterAllSources();
      error_out = std::string("Error uploading source image to I2I server: ")
                  + cli.GetErrorString();
      return false;
      }

    pdel->Hide();
    layer->ReleaseInternalPipeline("I2IExport");
    accum->UnregisterAllSources();
    m_UploadedLayer = sel;
    }

  return true;
}

bool
ImageToImageModel::StartTransfer(const std::string &model_id, std::string &error_out)
{
  std::lock_guard<std::mutex> guard(m_Mutex);
  if(!m_SourceImage)
    {
    error_out = "No source image selected";
    return false;
    }

  QtCursorOverride cursor(Qt::WaitCursor);

  RESTClientType cli(m_RESTSharedData);
  cli.SetServerURL(GetActiveServerURL().c_str());
  if(GetActiveServerURL().empty())
    {
    error_out = "No deep-learning server is configured/connected";
    return false;
    }

  try
    {
    if(!EnsureSessionAndUpload(cli, error_out))
      return false;

    if(!cli.Get("run_transfer/%s?model=%s", m_ActiveSession.c_str(), model_id.c_str()))
      {
      // The server may have restarted, invalidating the cached session id.
      // Retry once with a fresh session before giving up.
      std::string first_err = cli.GetErrorString();
      m_ActiveSession.clear();
      m_UploadedLayer = std::make_tuple(-1, -1);
      if(!EnsureSessionAndUpload(cli, error_out))
        return false;
      if(!cli.Get("run_transfer/%s?model=%s", m_ActiveSession.c_str(), model_id.c_str()))
        {
        error_out = cli.GetErrorString();
        this->Reset();
        return false;
        }
      }

    std::string out = cli.GetOutput() ? cli.GetOutput() : "";
    Json::CharReaderBuilder rbuilder;
    std::unique_ptr<Json::CharReader> reader(rbuilder.newCharReader());
    Json::Value root;
    std::string errs;
    if(!reader->parse(out.data(), out.data() + out.size(),
                      &root, &errs) || !root.isMember("job_id"))
      {
      error_out = "I2I server did not start a transfer job";
      this->Reset();
      return false;
      }

    m_ActiveJobId = root["job_id"].asString();
    this->SetIsRunning(true);
    this->SetTransferProgress(0.0);
    this->SetStatusMessage("Transfer started");
    this->InvokeEvent(TransferStateChangedEvent());
    return true;
    }
  catch(IRISException &exc)
    {
    error_out = exc.what();
    this->Reset();
    return false;
    }
}

int
ImageToImageModel::PollTransfer()
{
  std::lock_guard<std::mutex> guard(m_Mutex);
  if(!this->GetIsRunning() || m_ActiveJobId.empty() || m_ActiveSession.empty())
    return POLL_NETWORK_ERROR;

  RESTClientType cli(m_RESTSharedData);
  cli.SetServerURL(GetActiveServerURL().c_str());
  try
    {
    if(!cli.Get("transfer_progress/%s?job=%s", m_ActiveSession.c_str(), m_ActiveJobId.c_str()))
      {
      this->SetStatusMessage(std::string("Network error while polling transfer: ") + (cli.GetErrorString() ? cli.GetErrorString() : ""));
      return POLL_NETWORK_ERROR;
      }

    std::string out = cli.GetOutput() ? cli.GetOutput() : "";
    Json::CharReaderBuilder rbuilder;
    std::unique_ptr<Json::CharReader> reader(rbuilder.newCharReader());
    Json::Value root;
    std::string errs;
    if(!reader->parse(out.data(), out.data() + out.size(),
                      &root, &errs) || !root.isObject())
      return POLL_NETWORK_ERROR;

    this->SetTransferProgress(root.get("progress", 0.0).asDouble());

    std::string status = root.get("status", "").asString();
    if(status == "done")
      {
      this->SetIsRunning(false);
      this->SetStatusMessage("Transfer complete");
      this->InvokeEvent(TransferStateChangedEvent());
      return POLL_DONE;
      }
    else if(status == "error")
      {
      this->SetIsRunning(false);
      std::string msg = root.get("message", "unknown error").asString();
      this->SetStatusMessage("Transfer failed: " + msg);
      this->InvokeEvent(TransferStateChangedEvent());
      return POLL_ERROR;
      }

    this->SetStatusMessage(root.get("message", "running").asString());
    this->InvokeEvent(TransferStateChangedEvent());
    return POLL_RUNNING;
    }
  catch(IRISException &exc)
    {
    this->SetStatusMessage(exc.what());
    return POLL_NETWORK_ERROR;
    }
}

bool
ImageToImageModel::FetchResult(std::string &error_out)
{
  std::lock_guard<std::mutex> guard(m_Mutex);
  if(m_ActiveJobId.empty() || m_ActiveSession.empty())
    {
    error_out = "No transfer job is active";
    return false;
    }

  RESTClientType cli(m_RESTSharedData);
  cli.SetServerURL(GetActiveServerURL().c_str());
  try
    {
    if(!cli.Get("transfer_result/%s?job=%s", m_ActiveSession.c_str(), m_ActiveJobId.c_str()))
      {
      error_out = cli.GetErrorString();
      return false;
      }
    AddOverlayFromResult(cli.GetOutput(), m_SourceImage, error_out);
    return true;
    }
  catch(IRISException &exc)
    {
    error_out = exc.what();
    return false;
    }
}

void
ImageToImageModel::AddOverlayFromResult(const std::string &json, ImageWrapperBase *source,
                                        std::string &error_out)
{
  // Decode the base64(gzip(raw)) payload into raw voxel bytes
  std::string raw;
  if(!dls_utility::DecodeResultToRaw(json, raw) || raw.empty())
    {
    error_out = "Failed to decode the transfer result payload";
    return;
    }

  // Parse the result metadata
  Json::CharReaderBuilder rbuilder;
  std::unique_ptr<Json::CharReader> reader(rbuilder.newCharReader());
  Json::Value root, meta;
  std::string errs;
  if(!reader->parse(json.data(), json.data() + json.size(), &root, &errs) || !root.isObject()
     || !root.isMember("metadata"))
    {
    error_out = "Transfer result is missing metadata";
    return;
    }
  meta = root["metadata"];

  long dims[3] = {1, 1, 1};
  double spacing[3] = {1.0, 1.0, 1.0};
  double origin[3] = {0.0, 0.0, 0.0};
  double direction[9] = {1,0,0, 0,1,0, 0,0,1};
  int components = 1;

  if(meta.isMember("dimensions"))
    for(int i = 0; i < 3 && i < (int) meta["dimensions"].size(); i++)
      dims[i] = meta["dimensions"][i].asInt64();
  if(meta.isMember("spacing"))
    for(int i = 0; i < 3 && i < (int) meta["spacing"].size(); i++)
      spacing[i] = meta["spacing"][i].asDouble();
  if(meta.isMember("origin"))
    for(int i = 0; i < 3 && i < (int) meta["origin"].size(); i++)
      origin[i] = meta["origin"][i].asDouble();
  if(meta.isMember("direction"))
    for(int i = 0; i < 9 && i < (int) meta["direction"].size(); i++)
      direction[i] = meta["direction"][i].asDouble();
  if(meta.isMember("components_per_pixel"))
    components = meta["components_per_pixel"].asInt();
  if(meta.isMember("component_type") && meta["component_type"].asString() != "float32")
    {
    error_out = "Image-to-image result has unsupported component type "
                + meta["component_type"].asString();
    return;
    }

  size_t n_vox = (size_t) dims[0] * (size_t) dims[1] * (size_t) dims[2];
  size_t n_floats = n_vox * (size_t) components;
  if(raw.size() < n_floats * sizeof(float))
    {
    error_out = "Transfer result payload is smaller than expected";
    return;
    }

  // Build a 4D vector image (single time point) that holds the raw result.
  typedef itk::VectorImage<float, 4> Img4DType;
  Img4DType::Pointer img4 = Img4DType::New();
  {
  auto region = img4->GetLargestPossibleRegion();
  region.SetSize(0, dims[0]);
  region.SetSize(1, dims[1]);
  region.SetSize(2, dims[2]);
  region.SetSize(3, 1);
  region.SetIndex(3, 0);
  img4->SetRegions(region);

  typename Img4DType::SpacingType spc4;
  typename Img4DType::PointType org4;
  typename Img4DType::DirectionType dir4;
  for(int i = 0; i < 4; i++)
    {
    for(int j = 0; j < 4; j++)
      dir4(i, j) = (i < 3 && j < 3) ? direction[i * 3 + j] : (i == j ? 1.0 : 0.0);
    spc4[i] = i < 3 ? spacing[i] : 1.0;
    org4[i] = i < 3 ? origin[i] : 0.0;
    }
  img4->SetSpacing(spc4);
  img4->SetOrigin(org4);
  img4->SetDirection(dir4);
  }
  img4->SetNumberOfComponentsPerPixel(components);
  img4->Allocate();
  memcpy(img4->GetPixelContainer()->GetBufferPointer(), raw.data(), n_floats * sizeof(float));

  // Wrap the result as a new overlay derived from the source layer.
  typedef AnatomicImageWrapperTraits<float> ResultTraits;
  typedef ResultTraits::WrapperType WrapperType;
  SmartPtr<WrapperType> wrapper = WrapperType::New();
  wrapper->InitializeToWrapper(source, img4);
  wrapper->SetDefaultNickname("transferred");
  m_ParentModel->GetDriver()->AddDerivedOverlayImage(source, wrapper, false);
}

bool
ImageToImageModel::FetchAvailableModels(std::vector<std::string> &out_ids,
                                        std::vector<std::string> &out_names)
{
  out_ids.clear();
  out_names.clear();
  if(GetActiveServerURL().empty())
    return false;

  RESTClientType cli(m_RESTSharedData);
  cli.SetServerURL(GetActiveServerURL().c_str());
  try
    {
    if(!cli.Get("transfer_models"))
      return false;
    std::string out = cli.GetOutput() ? cli.GetOutput() : "";
    Json::CharReaderBuilder rbuilder;
    std::unique_ptr<Json::CharReader> reader(rbuilder.newCharReader());
    Json::Value root;
    std::string errs;
    if(!reader->parse(out.data(), out.data() + out.size(),
                      &root, &errs) || !root.isObject() || !root.isMember("models"))
      return false;
    for(auto &m : root["models"])
      {
      out_ids.push_back(m.get("id", "").asString());
      out_names.push_back(m.get("name", m.get("id", "").asString()).asString());
      }
    return true;
    }
  catch(IRISException &)
    {
    return false;
    }
}

void
ImageToImageModel::Reset()
{
  {
  std::lock_guard<std::mutex> guard(m_Mutex);
  m_ActiveJobId.clear();
  // Keep the session (and uploaded source) for reuse on the same source.
  }
  this->SetIsRunning(false);
  this->SetStatusMessage("Idle");
  this->InvokeEvent(TransferStateChangedEvent());
}
