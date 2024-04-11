#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>  ///如果使用cuda加速，需要取消注释
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;

class C2PNet
{
public:
	C2PNet(string modelpath);
	Mat predict(Mat srcimg);
private:
	void preprocess(Mat img);
	vector<float> input_image;
	int inpWidth;
	int inpHeight;
	bool dynamic_shape;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "Image Dehazing");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
	Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
};

C2PNet::C2PNet(string model_path)
{
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());  ////windows写法
	///OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);   ///如果使用cuda加速，需要取消注释

	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions); ////windows写法
	////ort_session = new Session(env, model_path.c_str(), sessionOptions); ////linux写法

	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
		
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
	if (this->inpHeight == -1 || this->inpWidth == -1)
	{
		this->dynamic_shape = true;
	}
	else
	{
		this->dynamic_shape = false;
	}
}

void C2PNet::preprocess(Mat img)
{
	Mat rgbimg;
	cvtColor(img, rgbimg, COLOR_BGR2RGB);
	if (!dynamic_shape)
	{
		resize(rgbimg, rgbimg, cv::Size(this->inpWidth, this->inpHeight));
	}
	else
	{
		this->inpHeight = rgbimg.rows;
		this->inpWidth = rgbimg.cols;
	}
	
	vector<cv::Mat> rgbChannels(3);
	split(rgbimg, rgbChannels);
	for (int c = 0; c < 3; c++)
	{
		rgbChannels[c].convertTo(rgbChannels[c], CV_32FC1, 1 / 255.0);
	}

	const int image_area = this->inpHeight * this->inpWidth;
	this->input_image.resize(3 * image_area);
	size_t single_chn_size = image_area * sizeof(float);
	memcpy(this->input_image.data(), (float*)rgbChannels[0].data, single_chn_size);
	memcpy(this->input_image.data() + image_area, (float*)rgbChannels[1].data, single_chn_size);
	memcpy(this->input_image.data() + image_area * 2, (float*)rgbChannels[2].data, single_chn_size);
}

Mat C2PNet::predict(Mat srcimg)
{
	const int srch = srcimg.rows;
	const int srcw = srcimg.cols;
	this->preprocess(srcimg);

	std::vector<int64_t> input_img_shape = { 1, 3, this->inpHeight, this->inpWidth };
	Value input_tensor_ = Value::CreateTensor<float>(memory_info_handler, this->input_image.data(), this->input_image.size(), input_img_shape.data(), input_img_shape.size());

	Ort::RunOptions runOptions;
	vector<Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names.data(), &input_tensor_, 1, this->output_names.data(), output_names.size());
	
	float* pdata = ort_outputs[0].GetTensorMutableData<float>();
	std::vector<int64_t> outs_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	const int out_h = outs_shape[2];
	const int out_w = outs_shape[3];
	const int channel_step = out_h * out_w;
	Mat rmat(out_h, out_w, CV_32FC1, pdata);
	Mat gmat(out_h, out_w, CV_32FC1, pdata + channel_step);
	Mat bmat(out_h, out_w, CV_32FC1, pdata + 2 * channel_step);
	rmat *= 255.f;
	gmat *= 255.f;
	bmat *= 255.f;
	///output_image = 等价np.clip(output_image, 0, 255)
	rmat.setTo(0, rmat < 0);
	rmat.setTo(255, rmat > 255);
	gmat.setTo(0, gmat < 0);
	gmat.setTo(255, gmat > 255);
	bmat.setTo(0, bmat < 0);
	bmat.setTo(255, bmat > 255);

	vector<Mat> channel_mats(3);
	channel_mats[0] = bmat;
	channel_mats[1] = gmat;
	channel_mats[2] = rmat;

	Mat dstimg;
	merge(channel_mats, dstimg);
	dstimg.convertTo(dstimg, CV_8UC3);
	resize(dstimg, dstimg, cv::Size(srcw, srch));
	return dstimg;
}

int main()
{
	C2PNet mynet("weights/c2pnet_indoor_HxW.onnx");
	
	string imgpath = "testimgs/indoor/1420_10.png";
	Mat srcimg = imread(imgpath);
	Mat dstimg = mynet.predict(srcimg);

	namedWindow("srcimg", WINDOW_NORMAL);
	imshow("srcimg", srcimg);
	static const string kWinName = "Deep learning Image Dehaze use ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, dstimg);
	waitKey(0);
	destroyAllWindows();

	return 0;
}
