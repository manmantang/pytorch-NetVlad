/*
 * @Description: Joyson Intelligent Automobile Research Institute Documents
 * @Author: manman
 * @Date: 2023-05-12 17:14:24
 * @LastEditTime: 2023-05-23 18:05:45
 * @LastEditors:  
 */
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "logging.h"
#include "NvOnnxParser.h"
#include <cstring>

#include <opencv2/opencv.hpp>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 480;
static const int INPUT_W = 640;
static const int OUTPUT_H = 30;
static const int OUTPUT_W = 40;
static const int VLAD_OUTPUT = 32768;

static float middle_out[512 * OUTPUT_H * OUTPUT_W];

const char* INPUT1_BLOB_NAME = "data";
const char* OUTPUT1_BLOB_NAME = "prob";
const char* INPUT2_BLOB_NAME = "vgg16_tensor";
const char* OUTPUT2_BLOB_NAME = "vlad_encoding";

using namespace nvinfer1;
using namespace nvonnxparser;

static Logger gLogger;

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    // Create input tensor of shape { 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT1_BLOB_NAME, dt, Dims4{maxBatchSize, 3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("/workspace/workspace/pytorch-NetVlad/vgg16_weights.wts");
    // Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{3, 3}, weightMap["0.weight"], weightMap["0.bias"]);  //0 layer : convolution
    assert(conv1);
    conv1->setPaddingNd(DimsHW{1, 1});
    IActivationLayer* relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);                              // 1 layer : activation
    assert(relu1);
    conv1 = network->addConvolutionNd(*relu1->getOutput(0), 64, DimsHW{3, 3}, weightMap["2.weight"], weightMap["2.bias"]);      // 2 layer : convolution
    conv1->setPaddingNd(DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);                                                // 3 layer : activation
    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});                        // 4 layer : pooling
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});

    conv1 = network->addConvolutionNd(*pool1->getOutput(0), 128, DimsHW{3, 3}, weightMap["5.weight"], weightMap["5.bias"]);     // 5 layer : convolution
    conv1->setPaddingNd(DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);                                                // 6 layer : activation
    conv1 = network->addConvolutionNd(*relu1->getOutput(0), 128, DimsHW{3, 3}, weightMap["7.weight"], weightMap["7.bias"]);     // 7 layer : convolution
    conv1->setPaddingNd(DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);                                                // 8 layer : activation
    pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});                                       // 9 layer : pooling
    pool1->setStrideNd(DimsHW{2, 2});

    conv1 = network->addConvolutionNd(*pool1->getOutput(0), 256, DimsHW{3, 3}, weightMap["10.weight"], weightMap["10.bias"]);   // 10 layer : convolution
    conv1->setPaddingNd(DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);                                                // 11 layer : activation
    conv1 = network->addConvolutionNd(*relu1->getOutput(0), 256, DimsHW{3, 3}, weightMap["12.weight"], weightMap["12.bias"]);   // 12 layer : convolution
    conv1->setPaddingNd(DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);                                                // 13 layer : activation
    conv1 = network->addConvolutionNd(*relu1->getOutput(0), 256, DimsHW{3, 3}, weightMap["14.weight"], weightMap["14.bias"]);   // 14 layer : convolution
    conv1->setPaddingNd(DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);                                                // 15 layer : activation
    pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});                                       // 16 layer : pooling
    pool1->setStrideNd(DimsHW{2, 2});

    conv1 = network->addConvolutionNd(*pool1->getOutput(0), 512, DimsHW{3, 3}, weightMap["17.weight"], weightMap["17.bias"]);   // 17 layer : convolution
    conv1->setPaddingNd(DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);                                                // 18 layer : activation
    conv1 = network->addConvolutionNd(*relu1->getOutput(0), 512, DimsHW{3, 3}, weightMap["19.weight"], weightMap["19.bias"]);   // 19 layer : convolution
    conv1->setPaddingNd(DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);                                                // 20 layer : activation
    conv1 = network->addConvolutionNd(*relu1->getOutput(0), 512, DimsHW{3, 3}, weightMap["21.weight"], weightMap["21.bias"]);   // 21 layer : convolution
    conv1->setPaddingNd(DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);                                                // 22 layer : activation
    pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});                                       // 23 layer : pooling
    pool1->setStrideNd(DimsHW{2, 2});

    conv1 = network->addConvolutionNd(*pool1->getOutput(0), 512, DimsHW{3, 3}, weightMap["24.weight"], weightMap["24.bias"]);   // 24 layer : convolution
    conv1->setPaddingNd(DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);                                                // 25 layer : activation
    conv1 = network->addConvolutionNd(*relu1->getOutput(0), 512, DimsHW{3, 3}, weightMap["26.weight"], weightMap["26.bias"]);   // 26 layer : convolution
    conv1->setPaddingNd(DimsHW{1, 1});
    relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);                                                // 27 layer : activation
    conv1 = network->addConvolutionNd(*relu1->getOutput(0), 512, DimsHW{3, 3}, weightMap["28.weight"], weightMap["28.bias"]);   // 28 layer : convolution
    conv1->setPaddingNd(DimsHW{1, 1});
    
    conv1->getOutput(0)->setName(OUTPUT1_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*conv1->getOutput(0));

    // Build engine
    // builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

ICudaEngine* createEngine(){
    // Load the ONNX model
    const char* model_path = "/workspace/workspace/pytorch-NetVlad/vladNet.onnx";
    std::unique_ptr<IBuilder> builder(createInferBuilder(gLogger));  
    std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, gLogger));
    parser->parseFromFile(model_path, static_cast<int>(ILogger::Severity::kINFO));
    for (int i = 0; i < parser->getNbErrors(); ++i)
    {
            std::cout << parser->getError(i)->desc() << std::endl;
    }
    std::cout << "successfully load the onnx model" << std::endl;

    // 设置输入输出张量
    nvinfer1::ITensor* input_tensor = network->getInput(0);
    std::cerr << "netvlad onnx input tensor num : " << network->getNbInputs() << std::endl;
    std::cerr << "netvlad onnx input tensor : " << input_tensor << std::endl;
    nvinfer1::ITensor* output_tensor = network->getOutput(0);
    nvinfer1::Dims input_dims{4, {1, 512, OUTPUT_H, OUTPUT_W}};
    input_tensor->setDimensions(input_dims);

    // 创建TensorRT引擎
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    // 释放资源
    config->destroy();

    return engine;
}

void vgg16ToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void netVladToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine();
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void vggInference(IExecutionContext& context, float* input, float* output, int batchSize){
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT1_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT1_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * 512 * OUTPUT_H * OUTPUT_W * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueueV2(buffers, stream, nullptr);

    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * 512 * OUTPUT_H * OUTPUT_W * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

void netVladInference(IExecutionContext& context, float* input, float* output, int batchSize){
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT2_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT2_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 512 * OUTPUT_H * OUTPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * VLAD_OUTPUT * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 512 * OUTPUT_H * OUTPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    // context.enqueue(batchSize, buffers, stream, nullptr);
    context.enqueueV2(buffers, stream, nullptr);

    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * VLAD_OUTPUT * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

void doInference(IExecutionContext& context_vgg, IExecutionContext& context_netvlad, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine_vgg = context_vgg.getEngine();
    assert(engine_vgg.getNbBindings() == 2);
    void* buffers_vgg[2];
    std::cout << "vgg engin IO sensor size : " << engine_vgg.getNbBindings() << std::endl;
    std::cout << "vgg engin I sensor name : " << engine_vgg.getBindingName(0) << std::endl;
    std::cout << "vgg engin O sensor name : " << engine_vgg.getBindingName(1) << std::endl;

    const ICudaEngine& engine_netvlad = context_netvlad.getEngine();
    assert(engine_netvlad.getNbBindings() == 2);
    void* buffers_netvlad[2];
    std::cout << "netvlad engin IO sensor size : " << engine_netvlad.getNbBindings() << std::endl;
    std::cout << "netvlad engin I sensor name : " << engine_netvlad.getBindingName(0) << std::endl;
    std::cout << "netvlad engin O sensor name : " << engine_netvlad.getBindingName(1) << std::endl;

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine_vgg.getBindingIndex(INPUT1_BLOB_NAME);
    const int encoder_outputIndex = engine_vgg.getBindingIndex(OUTPUT1_BLOB_NAME);
    const int pool_inputIndex = engine_netvlad.getBindingIndex(INPUT2_BLOB_NAME);
    const int outputIndex = engine_netvlad.getBindingIndex(OUTPUT2_BLOB_NAME);
    std::cout << "inputIndex : " << inputIndex << std::endl;
    std::cout << "encoder_outputIndex : " << encoder_outputIndex << std::endl;
    std::cout << "pool_inputIndex : " << pool_inputIndex << std::endl;
    std::cout << "outputIndex : " << outputIndex << std::endl;

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers_vgg[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers_vgg[encoder_outputIndex], batchSize * 512 * 30 * 40 * sizeof(float)));
    CHECK(cudaMalloc(&buffers_netvlad[pool_inputIndex], batchSize * 512 * 30 * 40 * sizeof(float)));
    CHECK(cudaMalloc(&buffers_netvlad[outputIndex], batchSize * 32768 * sizeof(float)));
    std::cout << "construct GPU buffers !" << std::endl;

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    std::cout << "construct stream !" << std::endl;

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers_vgg[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    std::cout << "copy input to buffers_vgg[inputIndex] !" << std::endl;
    context_vgg.enqueue(batchSize, buffers_vgg, stream, nullptr);
    std::cout << "context_vgg excutable finish !" << std::endl;
    CHECK(cudaMemcpyAsync(middle_out, buffers_vgg[outputIndex], batchSize * 512 * 30 * 40 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(buffers_netvlad[pool_inputIndex], middle_out, batchSize * 512 * 30 * 40 * sizeof(float), cudaMemcpyHostToDevice, stream));
    std::cout << "copy buffers_vgg[1] to buffer_netvlad[0] !" << std::endl;

    context_netvlad.enqueue(batchSize, buffers_netvlad, stream, nullptr);
    std::cout << "context_netvlad excutable finish !" << std::endl;

    CHECK(cudaMemcpyAsync(output, buffers_netvlad[outputIndex], batchSize * 32768 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    std::cout << "copy buffer_netvlad[1] to host output !" << std::endl;

    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers_vgg[inputIndex]));
    CHECK(cudaFree(buffers_vgg[encoder_outputIndex]));
    CHECK(cudaFree(buffers_netvlad[pool_inputIndex]));
    CHECK(cudaFree(buffers_netvlad[outputIndex]));
}

void loadImage(const std::string& file, float* data){
    cv::Mat img = cv::imread("/workspace/workspace/pytorch-NetVlad/data/pittsburgh/000/000000_pitch1_yaw1.jpg",cv::IMREAD_COLOR); //BGR
    if(img.empty()){
        std::cout << "can not read image !" << std::endl;
        return;
    }

    cv::Mat rgb_image;
    cv::cvtColor(img, rgb_image, cv::COLOR_BGR2RGB); //HWC need to transform CHW
    std::vector<cv::Mat> rgb_channels(3);
    cv::split(rgb_image, rgb_channels);
    
    float mean_rgb[3] = {0.485, 0.456, 0.406};
    float std_rgb[3] = {0.229, 0.224, 0.225};
    for(int i = 0; i < 3; ++i){
        std::uint8_t* image_ptr = rgb_channels[i].data;
        for(int h = 0; h < rgb_image.rows; ++h){
            for(int w = 0; w < rgb_image.cols; ++w){
                data[i * rgb_image.total() + h * rgb_image.cols + w] = 
                (static_cast<float>((image_ptr[h * rgb_image.cols + w] * 1.0f/255.0f)) - mean_rgb[i])/std_rgb[i];
            }
        }
    }
    return;
}


int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./vgg -s   // serialize model to plan file" << std::endl;
        std::cerr << "./vgg -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream_vgg{nullptr};
    char *trtModelStream_netvlad{nullptr};
    size_t size1{0};
    size_t size2{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream1{nullptr};
        IHostMemory* modelStream2{nullptr};
        vgg16ToModel(1,&modelStream1);
        assert(modelStream1 != nullptr);
        netVladToModel(1, &modelStream2);
        assert(modelStream2 != nullptr);

        std::ofstream p1("correct_vgg16.engine", std::ios::binary);
        std::ofstream p2("correct_netvlad.engine", std::ios::binary);
        if (!p1) {
            std::cerr << "could not open vgg16.engin" << std::endl;
            return -1;
        }
        if (!p2) {
            std::cerr << "could not open netvlad.engin" << std::endl;
            return -1;
        }
        p1.write(reinterpret_cast<const char*>(modelStream1->data()), modelStream1->size());
        p2.write(reinterpret_cast<const char*>(modelStream2->data()), modelStream2->size());

        modelStream1->destroy();
        modelStream2->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file1("correct_vgg16.engine", std::ios::binary);
        std::ifstream file2("correct_netvlad.engine", std::ios::binary);

        if (file1.good() && file2.good()) {
            file1.seekg(0, file1.end);
            size1 = file1.tellg();
            file1.seekg(0, file1.beg);
            trtModelStream_vgg = new char[size1];
            assert(trtModelStream_vgg);
            file1.read(trtModelStream_vgg, size1);
            file1.close();

            file2.seekg(0, file2.end);
            size2 = file2.tellg();
            file2.seekg(0, file2.beg);
            trtModelStream_netvlad = new char[size2];
            assert(trtModelStream_netvlad);
            file2.read(trtModelStream_netvlad, size2);
            file2.close();
        }
    } else {
        return -1;
    }


    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine_vgg = runtime->deserializeCudaEngine(trtModelStream_vgg, size1);
    assert(engine_vgg != nullptr);
    ICudaEngine* engine_netvlad = runtime->deserializeCudaEngine(trtModelStream_netvlad, size2);
    assert(engine_netvlad != nullptr);
    IExecutionContext* context_vgg = engine_vgg->createExecutionContext();
    assert(context_vgg != nullptr);
    IExecutionContext* context_netvlad = engine_netvlad->createExecutionContext();
    assert(context_netvlad != nullptr);
    delete[] trtModelStream_vgg;
    delete[] trtModelStream_netvlad;

    // Run inference
    static float data[3 * INPUT_H * INPUT_W];
    // for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //     data[i] = 1.;
    loadImage("", data);
    static float prob[32768];

    auto start = std::chrono::system_clock::now();
    vggInference(*context_vgg, data, middle_out, 1);
    netVladInference(*context_netvlad, middle_out, prob, 1);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    std::ofstream file;
    file.open("tensorrt_netvlad.txt", std::ios::out | std::ios::trunc);
    for(int i = 0;i < 32768; i++){
        file << prob[i] << " ";
    }
    std::cout << std::endl;

    // Destroy the engine
    context_vgg->destroy();
    context_netvlad->destroy();
    engine_vgg->destroy();
    engine_netvlad->destroy();
    runtime->destroy();

    return 0;
}
