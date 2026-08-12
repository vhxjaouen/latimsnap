#ifndef DLSUtility_h
#define DLSUtility_h

/* Shared helpers for the LaTIM-SNAP deep-learning servers (interactive
 * segmentation and image-to-image).
 *
 * These mirror the wire format used by java's itksnap_dls and our vendored
 * latimsnap_i2i server:
 *   - image upload: gzip(raw little-endian voxels) in a multipart "file" part
 *                   plus a "metadata" JSON part
 *   - result image: base64(gzip(raw little-endian voxels)) plus metadata JSON
 */

#include <cstdint>
#include <string>
#include <vector>
#include <typeinfo>
#include <sstream>

#include "json/json.h"
#include "RESTClient.h"
#include "base64.h"

#include "itk_zlib.h"
#include "zlib.h"

namespace dls_utility
{

inline bool gzipInflate(const std::string &compressedBytes, std::string &uncompressedBytes)
{
  if (compressedBytes.size() == 0)
  {
    uncompressedBytes = compressedBytes;
    return true;
  }

  uncompressedBytes.clear();
  constexpr unsigned buffer_size = 1024 * 1024;
  std::vector<unsigned char> buffer(buffer_size);

  z_stream strm;
  strm.next_in = (Bytef *) compressedBytes.c_str();
  strm.avail_in = compressedBytes.size();
  strm.total_out = 0;
  strm.zalloc = Z_NULL;
  strm.zfree = Z_NULL;

  if (inflateInit2(&strm, (16 + MAX_WBITS)) != Z_OK)
    return false;

  bool done = false;
  while (!done)
  {
    strm.next_out = buffer.data();
    strm.avail_out = buffer_size;
    size_t n0 = strm.total_out;
    int err = inflate(&strm, Z_SYNC_FLUSH);
    size_t n = strm.total_out - n0;
    uncompressedBytes.append((char *) buffer.data(), n);
    if (err == Z_STREAM_END)
      done = true;
    else if (err != Z_OK)
      break;
  }
  if (inflateEnd(&strm) != Z_OK)
    return false;
  return true;
}

inline bool gzipDeflate(const char *uncompressedBytes, size_t n_bytes, std::string &compressedBytes)
{
  compressedBytes.clear();
  if (n_bytes == 0)
    return true;

  constexpr unsigned buffer_size = 1024 * 1024;
  std::vector<unsigned char> buffer(buffer_size);

  z_stream strm;
  strm.next_in = (Bytef *) uncompressedBytes;
  strm.avail_in = n_bytes;
  strm.total_out = 0;
  strm.zalloc = Z_NULL;
  strm.zfree = Z_NULL;

  if (deflateInit2(&strm, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 15 | 16, 8, Z_DEFAULT_STRATEGY) != Z_OK)
    return false;

  bool done = false;
  while (!done)
  {
    strm.next_out = buffer.data();
    strm.avail_out = buffer_size;
    size_t n0 = strm.total_out;
    int err = deflate(&strm, Z_FINISH);
    size_t n = strm.total_out - n0;
    compressedBytes.append((char *) buffer.data(), n);
    if (err == Z_STREAM_END)
      done = true;
    else if (err != Z_OK)
      break;
  }
  if (deflateEnd(&strm) != Z_OK)
    return false;
  return true;
}

template <class T>
inline std::string get_numpy_type()
{
  auto &type = typeid(T);
  if (type == typeid(int8_t))     return "int8";
  if (type == typeid(uint8_t))    return "uint8";
  if (type == typeid(int16_t))    return "int16";
  if (type == typeid(uint16_t))   return "uint16";
  if (type == typeid(int32_t))    return "int32";
  if (type == typeid(uint32_t))   return "uint32";
  if (type == typeid(int64_t))    return "int64";
  if (type == typeid(uint64_t))   return "uint64";
  if (type == typeid(float))      return "float32";
  if (type == typeid(double))     return "float64";
  return "unknown";
}

template <class TImage>
inline void EncodeImage(RESTMultipartData &mpd, TImage *image, std::string &buffer_storage,
                        double reserve_ratio = 1.0)
{
  size_t n_bytes_raw = image->GetPixelContainer()->Size() * sizeof(typename TImage::InternalPixelType);
  buffer_storage.reserve((size_t) (n_bytes_raw * reserve_ratio));
  gzipDeflate((char *) image->GetBufferPointer(), n_bytes_raw, buffer_storage);
  mpd.addBytes("file", "application/gzip", "image.gz", buffer_storage.c_str(), buffer_storage.size());

  Json::Value root(Json::objectValue);
  root["dimensions"] = Json::Value(Json::arrayValue);
  root["spacing"] = Json::Value(Json::arrayValue);
  root["origin"] = Json::Value(Json::arrayValue);
  root["direction"] = Json::Value(Json::arrayValue);
  root["components_per_pixel"] = Json::Value(image->GetNumberOfComponentsPerPixel());
  root["component_type"] = get_numpy_type<typename TImage::InternalPixelType>();
  for (unsigned int i = 0; i < TImage::ImageDimension; i++)
  {
    root["dimensions"].append((int) image->GetBufferedRegion().GetSize()[i]);
    root["origin"].append((double) image->GetOrigin()[i]);
    root["spacing"].append((double) image->GetSpacing()[i]);
    for (unsigned int j = 0; j < TImage::ImageDimension; j++)
      root["direction"].append((double) image->GetDirection()(i, j));
  }
  std::ostringstream oss;
  oss << root;
  mpd.addString("metadata", "application/json", oss.str());
}

/** Decode a server JSON result payload into raw (inflated) voxel bytes. */
inline bool DecodeResultToRaw(const std::string &json, std::string &raw)
{
  Json::CharReaderBuilder reader_builder;
  std::unique_ptr<Json::CharReader> reader(reader_builder.newCharReader());
  Json::Value root;
  std::string errs;
  if (!reader->parse(json.data(), json.data() + json.size(), &root, &errs) || !root.isObject())
    return false;
  if (!root.isMember("result") || !root["result"].isString())
    return false;

  std::string result_b64 = root["result"].asString();
  std::string result_gzip;
  try
  {
    result_gzip = base64::from_base64(result_b64);
  }
  catch (...)
  {
    return false;
  }
  return gzipInflate(result_gzip, raw);
}

} // namespace dls_utility

#endif // DLSUtility_h
