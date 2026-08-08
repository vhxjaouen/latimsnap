#ifndef REGISTRATIONMODEL_H
#define REGISTRATIONMODEL_H

#include "AbstractModel.h"
#include "PropertyModel.h"
#include "itkMatrix.h"
#include "itkVector.h"
#include "MultiComponentMetricReport.h"

#include <QString>
#include <QStringList>

#include <functional>

class GlobalUIModel;
class IRISApplication;
class ImageWrapperBase;
class OptimizationProgressRenderer;
class QLabel;
class QProgressBar;
class QProcess;
class QProcessOutputTextWidget;
class Registry;

// -----------------------------------------------------------------------------
// PTVregSettings — POD holding every pTVreg CLI knob exposed to the user.
// Persisted via SystemInterface (SNAP Registry), converted to argv by
// ``ToCliArguments()``.
// -----------------------------------------------------------------------------
struct PTVregSettings
{
  // ----- Grid & Optimization -----
  int    gridSpacing;          // --spacing (voxels)
  double scaleFactor;          // --scale_factor
  QString iterations;          // --iterations, comma/space separated ints
  QString lambdaReg;           // --lambda_reg, comma/space separated floats

  // ----- Similarity metric -----
  QString metric;              // --metric: lcc|ssd|nuclear|emse|vfc
  double  metricParam;         // --metric_param (LCC sigma mm)

  // ----- VFC-specific -----
  double vfcRadius;            // --vfc-radius (mm)
  double vfcBeta;              // --vfc-beta
  bool   vfcSignInvariant;     // --vfc-sign-invariant
  bool   vfcNormalize;         // --vfc-normalize

  // ----- Penalties & boundary -----
  double lambdaJac;            // --lambda_jac
  double dvfEpsilon;           // --dvf_epsilon (mm)
  int    borderMask;           // --border_mask (voxels)
  bool   useClip;              // whether --clip is passed
  double clipMin;
  double clipMax;

  // ----- Label-guided soft-Dice -----
  double diceWeight;                // --dice-weight
  unsigned long fixedLabelsLayerId; // 0 = none; SNAP layer wrapper unique id
  unsigned long movingLabelsLayerId;

  PTVregSettings();
  void ResetToDefaults();

  /** Convert this settings block to CLI arguments (excluding SNAP-only
   *  markers, input/output paths, or fixed-mask). ``labelPathResolver`` is a
   *  callback used to export a SNAP layer id to a temporary NIfTI file. If a
   *  label id is 0 or the callback returns empty, that label argument is
   *  skipped. */
  typedef std::function<QString(unsigned long)> LabelPathResolver;
  QStringList ToCliArguments(const LabelPathResolver &resolver) const;

  void SaveToRegistry(Registry &out) const;
  void LoadFromRegistry(Registry &in);

  /** Compact one-line description used in the RegistrationDialog summary. */
  QString ToSummaryString() const;
};

template <unsigned int VDim, class TReal> class GreedyApproach;

namespace itk
{
  class Command;
  template <typename TParametersValueType, unsigned int N1, unsigned int N2> class MatrixOffsetTransformBase;
}

class RegistrationModel : public AbstractModel
{
public:
  irisITKObjectMacro(RegistrationModel, AbstractModel)

  // Sometimes the vnl types are easier to work with
  typedef vnl_matrix_fixed<double, 3, 3> Mat3;
  typedef vnl_matrix_fixed<double, 4, 4> Mat4;
  typedef vnl_vector_fixed<double, 3> Vec3;

  /**
    States in which the model can be, which allow the activation and
    deactivation of various widgets in the interface
    */
  enum UIState {
    UIF_MOVING_SELECTION_AVAILABLE,
    UIF_MOVING_SELECTED,
    UIF_FREE_ROTATION_MODE,
    UIF_REGISTRATION_MODE
  };

  /** Allowed transformation models - to be expanded in the future */
  enum Transformation { RIGID = 0, AFFINE, DEFORMABLE_PTVREG, INVALID_MODE };

  /** Image similarity metrics */
  enum SimilarityMetric { NMI = 0, NCC, SSD, INVALID_METRIC };

  /** Types of transform file */
  enum TransformFormat { FORMAT_ITK = 0, FORMAT_C3D };

  /**
    Check the state flags above
    */
  bool CheckState(UIState state);

  void SetParentModel(GlobalUIModel *model);
  irisGetMacro(Parent, GlobalUIModel *)

  virtual void OnUpdate() override;

  typedef SimpleItemSetDomain<unsigned long, std::string> LayerSelectionDomain;
  typedef AbstractPropertyModel<unsigned long, LayerSelectionDomain> AbstractLayerSelectionModel;

  /** Model for selecting the layer for registration */
  irisGetMacro(MovingLayerModel, AbstractLayerSelectionModel *)

  /** Euler angles */
  irisGetMacro(EulerAnglesModel, AbstractRangedDoubleVec3Property *)

  /** Translation */
  irisGetMacro(TranslationModel, AbstractRangedDoubleVec3Property *)

  /** Scaling factor */
  irisGetMacro(ScalingModel, AbstractRangedDoubleVec3Property *)

  /** Scaling factor */
  irisGetMacro(FlipModel, AbstractSimpleIntVec3Property *)

  /** Logarithm of the scaling factor - for the slider */
  irisGetMacro(LogScalingModel, AbstractRangedDoubleVec3Property *)

  /** Interactive registration tool button */
  irisGetMacro(InteractiveToolModel, AbstractSimpleBooleanProperty *)

  /** Set the center of rotation to current cross-hairs position */
  void SetCenterOfRotationToCursor();

  /** Reset the rotation to identity */
  void ResetTransformToIdentity();

  /** Match image centers */
  void MatchImageCenters();

  /** Match moments of inertia */
  void MatchByMoments(int order);

  /** Apply a rotation around a fixed angle */
  void ApplyRotation(const Vector3d &axis, double theta);

  /** Apply a translation (specified in physical ITK space) */
  void ApplyTranslation(const Vector3d &tran);

  /** Get a pointer to the selected moving wrapper, or NULL if none selected */
  ImageWrapperBase *GetMovingLayerWrapper() const;

  /** Get the center of rotation, in voxel units of the main image */
  irisGetMacro(RotationCenter, Vector3ui)

  // Automatic registration parameter domains
  typedef SimpleItemSetDomain<int, std::string> ResolutionLevelDomain;

  // Access to registration models
  irisSimplePropertyAccessMacro(Transformation, Transformation)
  irisSimplePropertyAccessMacro(SimilarityMetric, SimilarityMetric)
  irisSimplePropertyAccessMacro(UseSegmentationAsMask, bool)
  irisGenericPropertyAccessMacro(CoarsestResolutionLevel, int, ResolutionLevelDomain)
  irisGenericPropertyAccessMacro(FinestResolutionLevel, int, ResolutionLevelDomain)
  irisSimplePropertyAccessMacro(FreeRotationMode, bool)

  void SetIterationCommand(itk::Command *command);

  void RunAutoRegistration();

  void LoadTransform(const char *filename, TransformFormat format,
                     bool compose = false, bool inverse = false);

  void SaveTransform(const char *filename, TransformFormat format);

  /** Metric log data structure */
  typedef std::vector<std::vector<MultiComponentMetricReport> > MetricLog;

  /** Return the metric log from the registration */
  const MetricLog &GetRegistrationMetricLog() const;

  irisSimplePropertyAccessMacro(LastMetricValue, double)

  /** Get the progress renderer object */
  irisGetMacro(RegistrationProgressRenderer, OptimizationProgressRenderer *)

  /** Cleanup on dialog closed (but not necessarily destroyed) */
  void OnDialogClosed();

  /** Reslice moving image */
  void ResliceMovingImage(InterpolationMethod method);

  // ---------------------------------------------------------------------
  // Deformable registration (pTVreg) support
  // ---------------------------------------------------------------------

  /** Persisted user preference: absolute path to the Python interpreter that
   *  has access to the pTVreg package (typically a venv). */
  QString GetPythonInterpreterPath() const;
  void SetPythonInterpreterPath(const QString &path);

  /** Access the settings object used to configure pTVreg. Modifying the
   *  returned reference does not persist automatically — call
   *  ``SavePTVregSettings()`` after edits. */
  PTVregSettings &GetPTVregSettings();
  const PTVregSettings &GetPTVregSettings() const;

  /** Load the persisted settings from SystemInterface into
   *  ``GetPTVregSettings()``. Called automatically when the model receives
   *  its parent. */
  void LoadPTVregSettings();

  /** Persist the current ``GetPTVregSettings()`` to SystemInterface so they
   *  survive across sessions. */
  void SavePTVregSettings();

  /** Try to import pTVreg using the configured interpreter. Runs synchronously
   *  with a short timeout. On failure ``errMsg`` receives a human-readable
   *  explanation. */
  bool VerifyPythonEnvironment(QString &errMsg) const;

  /** Kick off a deformable registration run. Non-blocking: the QProcess and
   *  the widgets are wired up and control returns to the caller immediately.
   *  ``log``, ``statusLine`` and ``bar`` are populated as the process
   *  produces output. */
  void RunDeformableRegistration(QProcessOutputTextWidget *log,
                                 QLabel *statusLine,
                                 QProgressBar *bar,
                                 bool livePreviewPerIter,
                                 int  livePreviewIterStep);

  /** Access the QProcess of the currently running deformable job, or NULL if
   *  no job is active. The caller can use it to connect Qt signals such as
   *  readyReadStandardOutput to its own slots (typically forwarders that
   *  call the model's Handle* methods). */
  QProcess *GetDeformableProcess() const;

  /** True while a deformable process is in flight. */
  bool IsDeformableRegistrationRunning() const;

  /** Ask the current deformable process to terminate. Preview overlay stays
   *  in place. */
  void CancelDeformableRegistration();

  /** Returns true when the preview overlay produced by the last (or current)
   *  deformable registration is still available in the workspace. */
  bool HasDeformablePreview() const;

  /** Replace the moving layer's image content with the preview overlay and
   *  remove the preview layer. Requires ``HasDeformablePreview()``. */
  void AdoptPreviewAsMovingImage();

  // Map parameters to an affine transform
  Mat4 MapParametersToAffineTransform(
      const Vec3 &euler_angles, const Vec3 &translation,
      const Vec3 &scales, const Vec3 &shear_euler_angles) const;



protected:
  RegistrationModel();
  ~RegistrationModel();

  typedef itk::Matrix<double, 3, 3> ITKMatrixType;
  typedef itk::Vector<double, 3> ITKVectorType;
  typedef GreedyApproach<3, float> GreedyAPI;
  typedef itk::MatrixOffsetTransformBase<double, 3, 3> AffineTransform;

  // A little function to make homogeneous matrices from matrix/offset
  static Mat4 make_homog(const Mat3 &A, const Vec3 &b) ;

  GlobalUIModel *m_Parent;
  IRISApplication *m_Driver;

  // Pointer to the GreedyAPI. This is only non-null during RunAutoRegistration();
  GreedyAPI *m_GreedyAPI;

  // Shorthand to generate an ITK affine transform from a matrix and a vector
  SmartPtr<AffineTransform> MakeTransform(const ITKMatrixType &matrix, const ITKVectorType &offset) const;
  SmartPtr<AffineTransform> MakeIdentityTransform() const;

  void ResetOnMainImageChange();

  // This method is used to updated the cached matrix/offset and the parameters such
  // as Euler angles from the information in the moving image wrapper
  void UpdateManualParametersFromWrapper(bool reset_flips = false, bool force_update = false);

  // This method is called to recompute the transform in the moving image wrapper from
  // parameters including scaling, euler angles, and translation
  void UpdateWrapperFromManualParameters();

  void SetRotationCenter(const Vector3ui &pos);

  // Get the transform currently stored in the moving layer
  void GetMovingTransform(ITKMatrixType &matrix, ITKVectorType &offset);

  // Set the transform in the moving layer
  void SetMovingTransform(const ITKMatrixType &matrix, const ITKVectorType &offset, bool skipParameterUpdate = false);

  SmartPtr<AbstractLayerSelectionModel> m_MovingLayerModel;
  bool GetMovingLayerValueAndRange(unsigned long &value, LayerSelectionDomain *range);
  void SetMovingLayerValue(unsigned long value);

  SmartPtr<AbstractSimpleBooleanProperty> m_InteractiveToolModel;
  bool GetInteractiveToolValue(bool &value);
  void SetInteractiveToolValue(bool value);

  SmartPtr<AbstractRangedDoubleVec3Property> m_EulerAnglesModel;
  bool GetEulerAnglesValueAndRange(Vector3d &value, NumericValueRange<Vector3d> *range);
  void SetEulerAnglesValue(Vector3d value);

  SmartPtr<AbstractRangedDoubleVec3Property> m_TranslationModel;
  bool GetTranslationValueAndRange(Vector3d &value, NumericValueRange<Vector3d> *range);
  void SetTranslationValue(Vector3d value);

  SmartPtr<AbstractRangedDoubleVec3Property> m_ScalingModel;
  bool GetScalingValueAndRange(Vector3d &value, NumericValueRange<Vector3d> *range);
  void SetScalingValue(Vector3d value);

  SmartPtr<AbstractSimpleIntVec3Property> m_FlipModel;
  bool GetFlipValue(Vector3i &value);
  void SetFlipValue(Vector3i value);

  SmartPtr<AbstractRangedDoubleVec3Property> m_LogScalingModel;
  bool GetLogScalingValueAndRange(Vector3d &value, NumericValueRange<Vector3d> *range);
  void SetLogScalingValue(Vector3d value);

  typedef ConcretePropertyModel<Transformation, TrivialDomain> TransformationModel;
  SmartPtr<TransformationModel> m_TransformationModel;

  typedef ConcretePropertyModel<SimilarityMetric, TrivialDomain> SimilarityMetricModel;
  SmartPtr<SimilarityMetricModel> m_SimilarityMetricModel;

  SmartPtr<ConcreteSimpleBooleanProperty> m_UseSegmentationAsMaskModel;

  SmartPtr<ConcreteSimpleDoubleProperty> m_LastMetricValueModel;

  SmartPtr<ConcreteSimpleBooleanProperty> m_FreeRotationModeModel;

  // Multi-resolution schedule - coarsest and finest levels
  int m_CoarsestResolutionLevel, m_FinestResolutionLevel;
  ResolutionLevelDomain m_ResolutionLevelDomain;

  typedef AbstractPropertyModel<int, ResolutionLevelDomain> ResolutionLevelModel;
  SmartPtr<ResolutionLevelModel> m_CoarsestResolutionLevelModel;
  SmartPtr<ResolutionLevelModel> m_FinestResolutionLevelModel;

  bool GetCoarsestResolutionLevelValueAndRange(int &value, ResolutionLevelDomain *domain);
  void SetCoarsestResolutionLevelValue(int value);

  bool GetFinestResolutionLevelValueAndRange(int &value, ResolutionLevelDomain *domain);
  void SetFinestResolutionLevelValue(int value);

  // Value corresponding to no layer selected
  static const unsigned long NOID;

  // The active layer for the segmentation
  unsigned long m_MovingLayerId;

  // The components of the transform that are presented to the user by this widget
  struct TransformManualParameters
  {
    // The affine matrix/offset from which the parameters were generated
    ITKMatrixType AffineMatrix;
    ITKVectorType AffineOffset;

    // Euler angles
    Vector3d EulerAngles;

    // Translation
    Vector3d Translation;

    // Scaling
    Vector3d Scaling;

    // Shearing
    Vector3d ShearingEulerAngles;

    // Range of translation
    NumericValueRange<Vector3d> TranslationRange;

    // The unique layer ID for which this data was computed
    unsigned long LayerID;

    // Time stamp of the last update
    itk::TimeStamp UpdateTime;

    // Flipping
    Vector3i Flip;

    TransformManualParameters() : LayerID(NOID) {}
  };

  // The current cached manual parameters
  TransformManualParameters m_ManualParam;

  // Current center of rotation - should be initialized to the center when new image is loaded
  Vector3ui m_RotationCenter;

  // Callback for when the transform being computed by auto-registration is modified
  void IterationCallback(const itk::Object *object, const itk::EventObject &event);

  // The number of iterations per registration level
  // TODO: make this a model
  std::vector<int> m_IterationPyramid;

  // Command used for responding to intermediate data generated by registration
  SmartPtr<itk::Command> m_IterationCommand;

  // Renderer used to plot the metric
  SmartPtr<OptimizationProgressRenderer> m_RegistrationProgressRenderer;

  // Euler angles to a rotation matrix
  Mat3 MapEulerAnglesToRotationMatrix(const Vec3 &euler_angles) const;
  Vec3 MapRotationMatrixToEulerAngles(const Mat3 &rotation) const;

  // ---------------------------------------------------------------------
  // Deformable registration state
  // ---------------------------------------------------------------------

  // The QProcess is deliberately not owned by this ITK-based model; we keep
  // a raw pointer that we allocate with ``this`` as a QObject parent-less
  // pointer, and delete when the process finishes or when the model is
  // destroyed. Because RegistrationModel is not a QObject we cannot connect
  // signals directly; the RegistrationDialog is responsible for connecting
  // widget-facing slots and forwarding lines back to this model.
  QProcess *m_DeformableProcess;

  // Cached widget pointers, only valid while a deformable run is active.
  QProcessOutputTextWidget *m_DeformableLogWidget;
  QLabel                   *m_DeformableStatusLabel;
  QProgressBar             *m_DeformableProgressBar;

  // Working directory for the current or last deformable run. Kept on disk
  // even after a failure so the user can inspect intermediate previews.
  QString m_DeformableTempDir;

  // Path to the preview NIfTI most recently applied to the overlay.
  QString m_DeformableLastPreviewPath;

  // Pointer to the "pTVreg preview" overlay layer for in-place buffer swaps.
  ImageWrapperBase *m_DeformablePreviewWrapper;

  // Number of pyramid levels reported by the Python bridge (from SNAP_META).
  int m_DeformableNumLevels;

  // Reference wrapper for the fixed image at the start of a run — used to
  // clone geometry for the preview overlay.
  ImageWrapperBase *m_DeformableFixedWrapper;

  // Cached moving wrapper reference — used at Adopt time.
  ImageWrapperBase *m_DeformableMovingWrapper;

  // Persisted user preferences.
  QString m_PythonInterpreterPath;

  // pTVreg CLI configuration exposed via the settings dialog.
  PTVregSettings m_PTVregSettings;

  // Line buffer for QProcess output parsing (fed by the dialog).
public:
  /** Called by the dialog whenever the QProcess has produced a chunk of
   *  stdout. Splits into lines, dispatches SNAP_* markers, forwards the rest
   *  to the log widget. */
  void HandleDeformableStdout(const QByteArray &chunk);

  /** Called by the dialog when the QProcess produces stderr. Forwarded to
   *  the log widget in red. */
  void HandleDeformableStderr(const QByteArray &chunk);

  /** Called by the dialog when the QProcess has terminated. */
  void HandleDeformableFinished(int exitCode, int exitStatus);

protected:
  QByteArray m_DeformableStdoutBuffer;

  // Parses a single line for SNAP_* markers. Returns true if the line was a
  // marker (and therefore should not be echoed to the log widget).
  bool ProcessDeformableMarkerLine(const QString &line);

  // Extract inputs from ImageWrapperBase to on-disk NIfTI files. Returns true
  // on success. Populates the passed paths with the resulting file names.
  bool ExportImagesForDeformable(const QString &tempDir,
                                 QString &fixedPath,
                                 QString &movingPath,
                                 QString &maskPath);

  // Locate cli_snap.py. Order: bundle path, source-tree path,
  // $SNAP_PTVREG_DIR override. Returns empty string on failure.
  QString ResolvePTVregScriptPath() const;

  // Load a NIfTI at ``path`` and copy its buffer into the preview overlay
  // wrapper. Creates the overlay on first call.
  void ApplyPreviewFromFile(const QString &path);

  // Convert a metric enum to the pTVreg CLI string.
  QString GetDeformableMetricCliName() const;
};

#endif // REGISTRATIONMODEL_H
