/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>

using namespace cv;
using namespace std;

string annotation_path = "/home/danny/Documents/emo.csv";

// k-fold parameter
const int k_max = 5;
const int test_size = 20;

enum descriptor {
    EIGEN, FISHER, LBP
};
descriptor dp = LBP;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if (!path.empty() && !classlabel.empty()) {

            imshow("Display window", imread(path, 0)); // Show our image inside it.
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

void trainAndPredict(Ptr<FaceRecognizer>  model, vector<Mat> train_images, vector<Mat> test_images, vector<int> train_labels, vector<int> test_labels) {
    clock_t begin1 = clock();

    model->train(train_images, train_labels);
    // The following line predicts the label of a given
    // test image:

    clock_t end1 = clock();
    double elapsed_secs1 = double(end1 - begin1) / CLOCKS_PER_SEC;
    int right_case = 0;
    clock_t begin2 = clock();


    vector<Mat>::iterator iti = test_images.begin();
    vector<int>::iterator itl = test_labels.begin();
    for (; iti != test_images.end(); ++iti, ++itl) {
        int predictedLabel = model->predict(*iti);
        if (predictedLabel == *itl)
            right_case++;
    }

    clock_t end2 = clock();
    double elapsed_secs2 = double(end2 - begin2) / CLOCKS_PER_SEC;

    cout << (1.0 * right_case / test_size) << ";" << elapsed_secs1 << ";" << elapsed_secs2 << endl;
}

int main(int argc, const char *argv[]) {
    // Check for valid command line arguments, print usage
    // if no arguments were given.

    // Get the path to your CSV.
    // string fn_csv = string(argv[1]);

    // These vectors hold the images and corresponding labels.
    vector<Mat> images, images_bak, test_images;
    vector<int> labels, labels_bak, test_labels;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        read_csv(annotation_path, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << annotation_path << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Quit if there are not enough images for this demo.
    if (images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }

    // The following lines simply get the last several images from
    // your dataset and remove it from the vector. This is
    // done, so that the training data (which we learn the
    // cv::FaceRecognizer on) and the test data we test
    // the model with, do not overlap.

    // bak vars to hold original var
    images_bak = images;
    labels_bak = labels;
    
    // initial model
    Ptr<FaceRecognizer> model = NULL;
    string model_type = "";
    switch (dp) {
        case EIGEN:
        {
            model = createEigenFaceRecognizer();
            cout << "Eigen Face" << endl;
            break;
        }
        case FISHER:
        {
            model = createFisherFaceRecognizer();
            cout << "Fisher Face" << endl;
            break;
        }
        case LBP:
        {
            model = createLBPHFaceRecognizer();
            cout << "LBP" << endl;
            break;
        }
        default:
            model = createLBPHFaceRecognizer();
            cout << "LBP" << endl;
            break;
    }
    
    // k-fold test
    // every time sample m instance as test dataset, the others as training set
    for (int k = 0; k < k_max; k++) {
        // recover from bak
        images = images_bak;
        labels = labels_bak;

        test_images.clear();
        test_labels.clear();

        vector<Mat>::iterator iti = images.begin();
        vector<int>::iterator itl = labels.begin();

        // sampling
        int left_size = images.size();
        int sample_size = 0;
        while (sample_size < test_size) {
            int rnd_index = rand() % left_size;
            test_images.push_back(*(iti + rnd_index));
            test_labels.push_back(*(itl + rnd_index));
            images.erase(iti + rnd_index);
            labels.erase(itl + rnd_index);
            sample_size++;
            left_size = images.size();
        }

        // The following lines create an LBPH model for
        // face recognition and train it with the images and
        // labels read from the given CSV file.
        //
        // The LBPHFaceRecognizer uses Extended Local Binary Patterns
        // (it's probably configurable with other operators at a later
        // point), and has the following default values
        //
        //      radius = 1
        //      neighbors = 8
        //      grid_x = 8
        //      grid_y = 8
        //
        // So if you want a LBPH FaceRecognizer using a radius of
        // 2 and 16 neighbors, call the factory method with:
        //
        //      cv::createLBPHFaceRecognizer(2, 16);
        //
        // And if you want a threshold (e.g. 123.0) call it with its default values:
        //
        //      cv::createLBPHFaceRecognizer(1,8,8,8,123.0)
        //

        trainAndPredict(model, images, test_images, labels, test_labels);

    }

    // To get the confidence of a prediction call the model with:
    //
    //      int predictedLabel = -1;
    //      double confidence = 0.0;
    //      model->predict(testSample, predictedLabel, confidence);
    //
    // Sometimes you'll need to get/set internal model data,
    // which isn't exposed by the public cv::FaceRecognizer.
    // Since each cv::FaceRecognizer is derived from a
    // cv::Algorithm, you can query the data.
    //
    // First we'll use it to set the threshold of the FaceRecognizer
    // to 0.0 without retraining the model. This can be useful if
    // you are evaluating the model:
    //
    //    model->set("threshold", 0.0);
    // Now the threshold of this model is set to 0.0. A prediction
    // now returns -1, as it's impossible to have a distance below
    // it
    //    predictedLabel = model->predict(testSample);
    //    cout << "Predicted class = " << predictedLabel << endl;
    // Show some informations about the model, as there's no cool
    // Model data to display as in Eigenfaces/Fisherfaces.
    // Due to efficiency reasons the LBP images are not stored
    // within the model:
    if(dp == LBP){
    cout << "Model Information:" << endl;
        string model_info = format("\tLBPH(radius=%i, neighbors=%i, grid_x=%i, grid_y=%i, threshold=%.2f)",
                model->getInt("radius"),
                model->getInt("neighbors"),
                model->getInt("grid_x"),
                model->getInt("grid_y"),
                model->getDouble("threshold"));
        cout << model_info << endl;
    // We could get the histograms for example:
        vector<Mat> histograms = model->getMatVector("histograms");
    // But should I really visualize it? Probably the length is interesting:
        cout << "Size of the histograms: " << histograms[0].total() << endl;
    }
    return 0;
}