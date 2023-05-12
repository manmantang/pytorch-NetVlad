#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <cuda_runtime_api.h>
using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;

class Logger : public ILogger
{
void log(Severity severity, const char* msg)  noexcept
{
// suppress info-level messages
if (severity != Severity::kINFO)
std::cout << msg << std::endl;
}
} gLogger;

int main(int argc, char* argv[]) {
  // Load the ONNX model
  const char* model_path = "vgg16.onnx";
  unique_ptr<IBuilder> builder(createInferBuilder(gLogger));
  unique_ptr<INetworkDefinition> network(builder->createNetworkV2());
  unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, gLogger));
  parser->parseFromFile(model_path, static_cast<int>(ILogger::Severity::kINFO));
  
  // Set input and output names
  const char* input_name = "input";
  const char* output_name = "output";
  
  // Set input and output dimensions
  const int batch_size = 1;
  const int input_channels = 3;
  const int input_height = 224;
  const int input_width = 224;
  const int output_classes = 1000;
  Dims input_dims{4, {batch_size, input_channels, input_height, input_width}};
  Dims output_dims{2, {batch_size, output_classes}};
  
  // Set input and output types
  DataType input_type = DataType::kFLOAT;
  DataType output_type = DataType::kFLOAT;
  
  // Set input and output tensors
  ITensor* input_tensor = network->addInput(input_name, input_type, input_dims);
  ITensor* output_tensor = network->addOutput(output_name, output_type, output_dims);
  
  // Build the engine
  unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
  config->setMaxWorkspaceSize(1 << 30);
  unique_ptr<ICudaEngine> engine(builder->buildEngineWithConfig(*network, *config));
  
  // Create execution context
  unique_ptr<IExecutionContext> context(engine->createExecutionContext());
  
  // Allocate memory for input and output tensors
  const int input_size = batch_size * input_channels * input_height * input_width * sizeof(float);
  const int output_size = batch_size * output_classes * sizeof(float);
  void* input_data = malloc(input_size);
  void* output_data = malloc(output_size);
  
  // Run inference
  context->execute(batch_size, { {0, input_data} }, { {0, output_data} });
  
  // Print the results
  float* output_data_ptr = static_cast<float*>(output_data);
  for (int i = 0; i < output_classes; i++) {
    cout << output_data_ptr[i] << " ";
  }
  
  // Free memory
  free(input_data);
  free(output_data);
  
  return 0;
}
