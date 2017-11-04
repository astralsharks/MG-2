#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include "matrix.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);
    
    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels, 
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}

// Exatract features from dataset.
// You should implement this function by yourself =)

Matrix<double> kernel_=1; //convolution kernel
class Convolution
{
public:
    uint radius = 1; //size of kernel
    uint &vert_radius = radius, &hor_radius = radius;
    Convolution(const Matrix<double> &kernel){
         kernel_ = kernel;
         radius = (kernel.n_rows - 1) / 2;
    }
    double operator()(const Matrix<double> &area) const
    {
      uint size = 2 * radius + 1;
      assert(area.n_cols == area.n_rows);
      assert(radius == (area.n_cols - 1) / 2);

    double sum = 0.0;
    for (uint i = 0; i < size ; i++) {
        for (uint j = 0; j < size; j++) {
            sum += area(i, j) * kernel_(i, j);
        }
    }
    if (sum < 0) sum = 0.0;
    else if (sum > 255) sum = 255;
    return sum;
    }
};
Matrix<double> custom(Matrix<double> src_image, Matrix<double> kernel)
{
    Matrix<double> fin = src_image.unary_map(Convolution{kernel});
    return fin;
}

Matrix<double> sobel_x(const Matrix<double> &src_image) {
    Matrix<double> kernel = {{-1, 0, 1},
                             {-2, 0, 2},
                             {-1, 0, 1}};
    return custom(src_image, kernel);
}

Matrix<double> sobel_y(const Matrix<double> &src_image) {
    Matrix<double> kernel = {{ 1,  2,  1},
                             { 0,  0,  0},
                             {-1, -2, -1}};
    return custom(src_image, kernel);
}

Matrix<double> GrayScale(BMP& bmp){
    Matrix<double> res(bmp.TellHeight(), bmp.TellWidth());
    for (uint i = 0; i < res.n_cols; i++) {
        for (uint j = 0; j < res.n_rows; j++) {
            RGBApixel *pixel = bmp(i,j);
            res(j,i) =  0.299*pixel->Red+0.587*pixel->Green+0.0144*pixel->Blue;
        }
    }
//    cout << "grayscale is ok" << endl;
    return res;
}

vector<double> normalize(vector<double> hist)
{
    double norm = 0.0;
    for (uint i=0; i<hist.size(); i++){
        norm += pow(hist[i], 2);
    }
    norm = sqrt(norm);
    for (uint i=0; i<hist.size(); i++){
      //  cout << hist[i] << " ";
        if (norm == 0) norm = 1;
        hist[i] /= norm;
      //  cout << hist[i] << endl;
    }
  //  cout << "normalize is ok" << endl;
    return hist;
}

vector<double> histogram(Matrix<double> module, Matrix<double> direction)
{
    uint count = 0;
    double step = M_PI / 4;
    vector<double> this_hist(8);
    for (uint i=0; i < module.n_rows; i++) {
        for (uint j=0; j < module.n_cols; j++) {
            for (double k = -M_PI; k <= M_PI-step; k+=step){
                if (direction(i,j) >= k && direction(i,j) < k+step){
                    this_hist[count]+=module(i,j);
                    count++;
                }
                else if (direction(i,j) == M_PI-step){
                    this_hist[count]+=module(i,j);
                    count++;
                } 
                else
                    count++;
            }
            count=0;
        }
    }
  //  cout << "histogram is ok" << endl;
    return this_hist;
}

vector<double> LBP(Matrix<double> img)
{
    double norm = 0.0;
    for (uint i=0; i<hist.size(); i++){
        norm += pow(hist[i], 2);
    }
    norm = sqrt(norm);
    for (uint i=0; i<hist.size(); i++){
      //  cout << hist[i] << " ";
        if (norm == 0) norm = 1;
        hist[i] /= norm;
      //  cout << hist[i] << endl;
    }
  //  cout << "normalize is ok" << endl;
    return hist;
}

/*
Matrix<double> origin(BMP &img)
{
    Matrix<std::tuple<uint, uint, uint>> imgMatrix(static_cast<uint>(img.TellHeight()),
                                                   static_cast<uint>(img.TellWidth()));
    for (uint i = 0; i < imgMatrix.n_rows; ++i) {
        for (uint j = 0; j < imgMatrix.n_cols; ++j) {
            RGBApixel *p = img(j, i);
            imgMatrix(i, j) = std::make_tuple(p->Red, p->Green, p->Blue);
        }
    }
    return imgMatrix;
}*/

Matrix<double> make_img(BMP& bmp){
    Matrix<double> res(bmp.TellHeight(), bmp.TellWidth());
    for (uint i = 0; i < res.n_cols; i++) {
        for (uint j = 0; j < res.n_rows; j++) {
            RGBApixel *pixel = bmp(i,j);
            res(j,i) =  pixel->Red+pixel->Green+pixel->Blue;
        }
    }
//    cout << "grayscale is ok" << endl;
    return res;
}



template <typename T>
Matrix<T> increase(Matrix<T> src, uint n, uint m)
{
    Matrix<T> res(n, m);
    for (uint i = 0; i < n; i++) {
        for (uint j = 0; j < m; j++) {
            if (i < src.n_rows && j < src.n_cols) {
                res(i, j) = src(i, j);
            } 
            else {
                res(i, j) = 0.0;
            }
        }
    }
    return res;
}

vector<float> hog(BMP& bmp){
    uint n = bmp.TellHeight();
    uint m = bmp.TellWidth();
    if (n % 8 != 0)
    n = (n / 8 + 1) * 8;
    if (m % 8 != 0)
    m = (m / 8 + 1) * 8;

    uint cage_height = n / 8;
    uint cage_width = m / 8;
    vector<float> descriptor;
    vector<double> cage_histogram, normal;
    Matrix<double> res(n, m), x_sob(n, m), y_sob(n, m), direction(n, m), module(n, m), sublbp, submod, subdir;
    Matrix<double> lbp(n,m);

    //1. apply grayscale

    res = increase(GrayScale(bmp), n, m);

    //2. apply x and y sobel filters

    x_sob = sobel_x(res);
    y_sob = sobel_y(res);

    //3. calculate abs and directions of gradients

    for (uint i=0; i<n; i++){
        for (uint j=0; j<m; j++){
            module(i, j) = std::sqrt(std::pow(x_sob(i, j), 2) + std::pow(y_sob(i, j), 2));
            direction(i,j) = atan2(y_sob(i,j), x_sob(i,j));
        }
    }

    //4. calculate histogram
    
    lbp = increase(make_img(bmp), n, m);

    for (uint i = 0; i <= n - cage_height; i+=cage_height){
        for (uint j = 0; j <= m - cage_width; j+=cage_width){

            submod = module.submatrix(i, j, cage_height, cage_width);
            subdir = direction.submatrix(i, j, cage_height, cage_width);

            sublbp = lbp.submatrix(i, j, cage_height, cage_width);
            
            cage_histogram = histogram(submod, subdir);
            normal = normalize(cage_histogram);

            descriptor.insert(descriptor.end(), normal.begin(), normal.end());
        }
    }
    return descriptor;
}

void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
        
        auto &bmp = *(data_set[image_idx].first);
        vector<float> descriptor = hog(bmp);
        features->push_back(make_pair(descriptor, data_set[image_idx].second));
        }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////end of code////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;
    
        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here
    params.C = 0.5;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}