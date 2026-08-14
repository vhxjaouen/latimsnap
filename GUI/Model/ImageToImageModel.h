#ifndef IMAGETOIMAGEMODEL_H
#define IMAGETOIMAGEMODEL_H

#include "AbstractModel.h"
#include "PropertyModel.h"
#include "Registry.h"
#include <mutex>
#include <map>

class GlobalUIModel;
class ImageWrapperBase;
template <class ServerTraits> class RESTClient;
template <class ServerTraits> class RESTSharedData;
class DLSServerTraits;

namespace itk {
template <typename TPixel, unsigned int Dimension> class VectorImage;
}

/**
 * Model driving the LaTIM-SNAP image-to-image (I2I) pipeline.
 *
 * This reuses the same REST server framework as interactive segmentation
 * (RESTClient<DLSServerTraits> + DeepLearningServerPropertiesModel) but
 * against the vendored `latimsnap_i2i` server, producing a NEW image (e.g.
 * MR -> CT synthesis) that becomes an overlay - not a segmentation label.
 *
 * Flow:
 *   StartTransfer(model_id, axis) -> ensure session + upload source, start job
 *   PollTransfer()              -> update progress; returns RUNNING/DONE/ERROR
 *   FetchResult()               -> download decoded image, add as overlay
 */
class ImageToImageModel : public AbstractModel
{
public:
  irisITKObjectMacro(ImageToImageModel, AbstractModel)

  itkEventMacro(TransferStateChangedEvent, IRISEvent)
  FIRES(TransferStateChangedEvent)

  /** Result of a single poll of the server-side job. */
  enum PollResult
  {
    POLL_NETWORK_ERROR = -1,
    POLL_RUNNING = 0,
    POLL_DONE = 1,
    POLL_ERROR = 2
  };

  void SetParentModel(GlobalUIModel *parent);

  irisRangedPropertyAccessMacro(TransferProgress, double)
  irisSimplePropertyAccessMacro(StatusMessage, std::string)
  irisSimplePropertyAccessMacro(IsRunning, bool)

  /** The image to be transformed. */
  void SetSourceImage(ImageWrapperBase *layer);
  ImageWrapperBase *GetSourceImage() const { return m_SourceImage; }

  /** Start a transfer for the given model spec id.
   *
   *  ``axis`` selects which plane(s) a 2D model is applied to: "axial",
   *  "sagittal", "frontal" (coronal), "all" (3-axis ensemble), or a
   *  comma-separated list. Pass an empty string to use the model's default.
   *
   *  ``fusion`` selects how multiple planes are combined ("average"/"median"/
   *  "fba"); ignored for a single axis or 3D models. Pass empty for default. */
  bool StartTransfer(const std::string &model_id, const std::string &axis,
                     const std::string &fusion, std::string &error_out);

  /** Poll the running job once; update progress. See PollResult. */
  int PollTransfer();

  /** Fetch a finished transfer and add it as a new overlay on the source. */
  bool FetchResult(std::string &error_out);

  /** Cancel tracking of the current transfer (network op is not cancelled). */
  void Reset();

  /** List of available model specs, fetched from the server. May be empty. */
  bool FetchAvailableModels(std::vector<std::string> &out_ids,
                            std::vector<std::string> &out_names);

  /** True if the given model is a 2D model (which supports axis selection).
   *  Only meaningful after FetchAvailableModels() succeeded. */
  bool IsModel2D(const std::string &model_id) const;

  /** True if a transfer is currently in progress. */
  bool IsTransferInProgress() const { return this->GetIsRunning(); }

protected:
  ImageToImageModel();
  virtual ~ImageToImageModel();

  using RESTClientType = RESTClient<DLSServerTraits>;
  using RESTSharedDataType = RESTSharedData<DLSServerTraits>;

  GlobalUIModel *m_ParentModel;

  // model id -> spatial dim (2 or 3), populated by FetchAvailableModels()
  std::map<std::string, int> m_ModelDimension;

  ImageWrapperBase *m_SourceImage = nullptr;

  RESTSharedDataType *m_RESTSharedData;

  std::string m_ActiveSession;
  std::string m_ActiveJobId;
  int m_ActiveModelIndex = -1;

  // (layer unique id, timepoint) of the image already uploaded to the server
  using LayerSelection = std::tuple<int, int>;
  LayerSelection m_UploadedLayer = std::make_tuple(-1, -1);

  // Property models
  SmartPtr<ConcreteRangedDoubleProperty> m_TransferProgressModel;
  SmartPtr<ConcreteSimpleStringProperty> m_StatusMessageModel;
  SmartPtr<ConcreteSimpleBooleanProperty> m_IsRunningModel;

  std::string GetActiveServerURL();

  bool EnsureSessionAndUpload(RESTClientType &cli, std::string &error_out);

  void AddOverlayFromResult(const std::string &json, ImageWrapperBase *source,
                            std::string &error_out);

  std::mutex m_Mutex;
};

#endif // IMAGETOIMAGEMODEL_H
