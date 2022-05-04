#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace cv;
using namespace Ort;

struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	float objThreshold;  //Object Confidence threshold
};

typedef struct BoxInfo
{
	vector<Point> pts;
	float score;
	int label;
} BoxInfo;

class YOLO
{
public:
	YOLO(Net_config config);
	void detect(Mat& frame);
private:
	const float anchors[3][6] = { {31, 30, 28, 49, 50, 31},
								 {46, 45, 58, 58, 74, 74},
								 {94, 94, 115, 115, 151, 151} };
	const float stride[3] = { 8.0, 16.0, 32.0 };
	const int inpWidth = 1024;
	const int inpHeight = 1024;
	int nout;
	int num_proposal;
	vector<string> class_names;
	int num_class;
	
	float confThreshold;
	float nmsThreshold;
	float objThreshold;
	const bool keep_ratio = true;
	vector<float> input_image_;
	void normalize_(Mat img);
	
	Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left);
	void nms(vector<BoxInfo>& input_boxes, int max_w, int max_h);
	float IoU(BoxInfo polya, BoxInfo polyb, int max_w, int max_h);

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "yolov5");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

YOLO::YOLO(Net_config config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->objThreshold = config.objThreshold;

	string classesFile = "class.names";
	string model_path = "best.onnx";
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
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
	/*this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];*/
	this->nout = output_node_dims[0][2];
	this->num_proposal = output_node_dims[0][1];

	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) this->class_names.push_back(line);
	this->num_class = class_names.size();
}

Mat YOLO::resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 114);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*top = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 114);
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

void YOLO::normalize_(Mat img)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];
				this->input_image_[c * row * col + i * col + j] = pix / 255.0;

			}
		}
	}
}

float YOLO::IoU(BoxInfo polya, BoxInfo polyb, int max_w, int max_h)
{
	vector<vector<Point>> poly_array0 = { polya.pts };
	vector<vector<Point>> poly_array1 = { polyb.pts };


	Mat _poly0 = Mat::zeros(max_h, max_w, CV_8UC1);
	Mat _poly1 = Mat::zeros(max_h, max_w, CV_8UC1);
	Mat _result;

	vector<Point *> _pts0;
	vector<int> _npts0;

	for (auto &_v : poly_array0) {

		if (_v.size() < 3)//invalid poly
			return -1.f;

		_pts0.push_back((Point *)&_v[0]);
		_npts0.push_back((int)_v.size());
	}

	vector<Point *> _pts1;
	vector<int> _npts1;
	for (auto &_v : poly_array1) {

		if (_v.size() < 3)//invalid poly
			return -1.f;

		_pts1.push_back((Point *)&_v[0]);
		_npts1.push_back((int)_v.size());
	}

	fillPoly(_poly0, (const Point **)&_pts0[0], &_npts0[0], _npts0.size(), Scalar(1));
	fillPoly(_poly1, (const Point **)&_pts1[0], &_npts1[0], _npts1.size(), Scalar(1));
	bitwise_and(_poly0, _poly1, _result);

	int _area0 = countNonZero(_poly0);
	int _area1 = countNonZero(_poly1);
	int _intersection_area = countNonZero(_result);
	float _iou = (float)_intersection_area / (float)(_area0 + _area1 - _intersection_area);
	return _iou;
}


void YOLO::nms(vector<BoxInfo>& input_boxes, int max_w, int max_h)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
	
	vector<bool> isSuppressed(input_boxes.size(), false);
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < int(input_boxes.size()); ++j)
		{
			if (isSuppressed[j]) { continue; }
			float ovr = this->IoU(input_boxes[i], input_boxes[j], max_w, max_h);
			//float ovr = inter / (vArea[i] + vArea[j] - inter);

			if (ovr >= this->nmsThreshold)
			{
				isSuppressed[j] = true;
			}
		}
	}
	// return post_nms;
	int idx_t = 0;
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
}

inline float sigmoid(float x)
{
	return 1.0 / (1 + expf(-x));
}

void YOLO::detect(Mat& frame)
{
	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw);
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	/////generate proposals
	vector<BoxInfo> generate_boxes;
	float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;
	int n = 0, q = 0, i = 0, j = 0, k = 0, row_ind = 0; ///x1,y1,x2,y2,x3,y3,x4,y4,box_score,class_score
	float* pdata = ort_outputs[0].GetTensorMutableData<float>();
	for (n = 0; n < 3; n++)   ///特征图尺度
	{
		int num_grid_x = (int)(this->inpWidth / this->stride[n]);
		int num_grid_y = (int)(this->inpHeight / this->stride[n]);
		for (q = 0; q < 3; q++)    ///anchor
		{
			const float anchor_w = this->anchors[n][q * 2];
			const float anchor_h = this->anchors[n][q * 2 + 1];
			for (i = 0; i < num_grid_y; i++)
			{
				for (j = 0; j < num_grid_x; j++)
				{
					float box_score = sigmoid(pdata[8]);
					if (box_score > this->objThreshold)
					{
						int class_idx = 0;
						float max_class_socre = 0;
						for (k = 0; k < num_class; k++)
						{
							float score = sigmoid(pdata[k + 9]);
							if (score > max_class_socre)
							{
								max_class_socre = score;
								class_idx = k;
							}
						}
						max_class_socre = max_class_socre * box_score;
						if (max_class_socre > this->confThreshold)
						{ 
							vector<Point> pts;
							for (k = 0; k < 8; k += 2)
							{
								float x = (pdata[k] + j) * this->stride[n];  ///x
								float y = (pdata[k + 1] + i) * this->stride[n];   ///y
								x = (x - padw)*ratiow;
								y = (y - padh)*ratioh;
								pts.push_back(Point(x, y));
							}
							generate_boxes.push_back(BoxInfo{ pts, max_class_socre, class_idx });
						}
					}
					row_ind++;
					pdata += nout;
				}
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	nms(generate_boxes, int(frame.cols), int(frame.rows));
	for (size_t i = 0; i < generate_boxes.size(); ++i)
	{
		for (int j = 0; j < 4; j++)
		{
			line(frame, generate_boxes[i].pts[j], generate_boxes[i].pts[(j + 1) % 4], Scalar(0, 0, 255), 2);
		}

		int xmin = (int)generate_boxes[i].pts[0].x;
		int ymin = (int)generate_boxes[i].pts[0].y - 10;
		string label = format("%.2f", generate_boxes[i].score);
		label = this->class_names[generate_boxes[i].label] + ":" + label;
		putText(frame, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
	}
}

int main()
{
	Net_config yolo_nets = { 0.5, 0.5, 0.5 };
	YOLO yolo_model(yolo_nets);
	string imgpath = "images/1103.png";
	Mat srcimg = imread(imgpath);
	yolo_model.detect(srcimg);
	//imwrite("result.jpg", srcimg);
	static const string kWinName = "Deep learning object detection in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}