// Copyright 2014 BVLC and contributors.

#include <cstdio> 
#include <string>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>

#include <google/protobuf/text_format.h>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)


int main(int argc, char** argv) {

  typedef float Dtype;

  const int num_required_args = 5;
  if (argc < num_required_args) {
    LOG(ERROR)<< "feature_saved_check_point feature_extraction_proto_file  extract_feature_blob_name image-list save_feature_leveldb_name" << std::endl;
    return 1;
  }

  int arg_pos = 1;
  const string arg_saved_net_p(argv[arg_pos++]);
  const string arg_proto_file(argv[arg_pos++]);
  const string arg_extract_layer_name(argv[arg_pos++]);
  const string arg_input_images_p(argv[arg_pos++]);
  const string arg_save_feature_root_p(argv[arg_pos++]);

  Caffe::set_mode(Caffe::CPU);
  Caffe::set_phase(Caffe::TEST);

  string pretrained_binary_proto(arg_saved_net_p);
  string feature_extraction_proto(arg_proto_file);

  std::ifstream ifs_images(arg_input_images_p.c_str());
  if (!ifs_images.is_open()) {
      LOG(ERROR) << "Could not open or find file " << arg_input_images_p;
      return -1;
  }
  vector<shared_ptr<cv::Mat> > imagePtrs;
  string image_p;
  while (ifs_images >> image_p) {
    shared_ptr<cv::Mat> pMat(new cv::Mat(cv::imread(image_p, CV_LOAD_IMAGE_COLOR)));
    imagePtrs.push_back(pMat);
  }
  shared_ptr<Net<Dtype> > feature_extraction_net(new Net<Dtype>(feature_extraction_proto, imagePtrs));

 //const int batch_size = feature_extraction_net->params().front()->batch_size();
  const int num_mini_batches = std::ceil((float)imagePtrs.size()/static_cast<ImageDataLayer<Dtype>*>(feature_extraction_net->layers().front().get())->batch_size());
  std::cout << "num_mini_batches: " << num_mini_batches << std::endl;

  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

  string extract_feature_blob_name(arg_extract_layer_name);
  CHECK(feature_extraction_net->has_blob(extract_feature_blob_name))
      << "Unknown feature blob name " << extract_feature_blob_name
      << " in the network " << feature_extraction_proto;

  std::ofstream ofs(arg_save_feature_root_p.c_str());
  if (!ofs.is_open()) {
      LOG(ERROR) << "Can not open file " << arg_saved_net_p << std::endl;
  }

  vector<Blob<float>*> input_vec;
  int image_index = 0;
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
      feature_extraction_net->Forward(input_vec);
      const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net
          ->blob_by_name(extract_feature_blob_name);
      int num_features = feature_blob->num();
      int dim_features = feature_blob->count() / num_features;
      Dtype* feature_blob_data;
      for (int n = 0; n < num_features; ++n) {
          feature_blob_data = feature_blob->mutable_cpu_data() + feature_blob->offset(n);
          ofs << image_index << " : ";
          for (int d = 0; d < dim_features; ++d) {
              ofs << feature_blob_data[d] << " ";
          }
          ofs << std::endl;
          ++image_index;
          if (image_index % 1000 == 0) {
              LOG(ERROR)<< "Extracted features of " << image_index << " query images.";
          }
      }
  }
  ofs.close();

  LOG(ERROR)<< "Successfully extracted the features!";
  return 0;
}

