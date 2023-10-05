// I have used a constant velocity model to forecast object locations. 
// While I have looked into into the fundamentals of the Kalman filter to incorporate confidence score, 
// I believe that the constant velocity model aligns well with the requirements of my Full Stack Co-op position application.


// Initially I am considering 4 objects at t=0. Then at t=5, I have put ground truth labels for these 4 objects with their
// dimensions. 
// Later, I predicted the locations of these 4 objects at t=5 using data from t=0. 
// Then I compared generated data with ground truth and achived the precision and recall scores.
// The ground truths are placed in such a way that object 1 and 2 gets a hit, but 3 and 4 doesn't.



// Why IoU?
// - when using IoU, we make bounding boxes, scale invariant. So, even if one box is very large when compared to other, model performs well
// - In bird's eye view, alignment of bounding boxes with axes is important. IoU accounts for it as most of the vehicles are aligned with road.
// - Computationally inexpensive


// Also, this was mentioned in the question "FP: False Positive - A prediction is not matched to any label". My understanding is that false positive occurs when we 
// identify prediction A to label C. But here, definition says when A is not matched to any label. I'm going with written definition

// Again: TP: True Positive - A correct match between a prediction and a label
// Here I am sligtly confused as line emphasizes "correct match" which indicates we have to compare same object's prediction and label.
// Not just any prediction and label. But, prediction structure doesn't have ID data. 
// And FP says "A prediction is not matched to any label". So, I'm conflicted, but going with TP "when any prediction matches with any label" 

// FN: When a lable doesn't get matched with any prediction.

#include <iostream>
#include <vector>
#include <tuple>
#include <cstdint>
#include <cassert>
#include <cmath>
using namespace std;

struct Center {
    double x;
    double y;
    double z;
};

struct Dimensions {
    double w;
    double l;
    double h;
};

struct Velocity {
    double vx;
    double vy;
};

struct BoundingBox {
    Center center;
    Dimensions dimensions;
};

struct Label {
    BoundingBox bounding_box;
    uint64_t id;
};

struct Labels{
    vector<Label> labels;
    uint64_t timestamp;
};


struct Prediction {
    BoundingBox bounding_box;
    Velocity velocity;
    uint64_t timestamp;
    double confidence;
};

double ComputeIoU(const BoundingBox& box1, const BoundingBox& box2) {
    // Compute the coordinates of the intersection box
    double x1 = std::max(box1.center.x - box1.dimensions.w / 2, box2.center.x - box2.dimensions.w / 2);
    double y1 = std::max(box1.center.y - box1.dimensions.l / 2, box2.center.y - box2.dimensions.l / 2);


    double x2 = std::min(box1.center.x + box1.dimensions.w / 2, box2.center.x + box2.dimensions.w / 2);
    double y2 = std::min(box1.center.y + box1.dimensions.l / 2, box2.center.y + box2.dimensions.l / 2);

    // Compute the volume of the intersection box
    double intersection_volume = std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);

    // Compute the volume of both boxes
    double volume_box1 = box1.dimensions.w * box1.dimensions.l;
    double volume_box2 = box2.dimensions.w * box2.dimensions.l;

    // Compute IoU
    double iou = intersection_volume / (volume_box1 + volume_box2 - intersection_volume);

    return iou;
}


class Predictions {
public:
    Predictions(const std::vector<Prediction>& predictions) : predictions_{ predictions } {}

    std::vector<Prediction> predictions_;

    void ExtrapolatePredictions(uint64_t timestamp) {
        // Write code here to extrapolate the predictions to the given timestamp

        for (auto& prediction : predictions_) {
            double time_diff = timestamp - prediction.timestamp;
            prediction.bounding_box.center.x += prediction.velocity.vx * time_diff;
            prediction.bounding_box.center.y += prediction.velocity.vy * time_diff;
            
            // updating timestamp
            prediction.timestamp = timestamp;
        }
    }
};

std::tuple<double, double> ComputePrecisionRecall(const std::vector<Label>& labels, const Predictions& predictions, double threshold = 0.5) {
    // Implement the computation of precision and recall here
    
    int TP = 0;
    int FP = 0;
    int FN = 0;

    int matchedWithdifferent = 0;

    size_t i = 0;
    for (const auto& prediction : predictions.predictions_) {
        bool match_found = false;
        size_t j = 0;
        for (const auto& label : labels) {
            if (ComputeIoU(prediction.bounding_box, label.bounding_box) > threshold) {
                if (i == j){
                    match_found = true;
                    break;
                }else{
                    matchedWithdifferent++;
                }
            }
            j++;
        }
        if (match_found) TP++;
        else FP++;
        i++;
    }

    FP -= matchedWithdifferent;

    for (const auto& label : labels) {
        bool match_found = false;
        for (const auto& prediction : predictions.predictions_) {
            if (ComputeIoU(prediction.bounding_box, label.bounding_box) > threshold) {
                match_found = true;
                break;
            }
        }
        if (!match_found) FN++;
    }

    double precision = static_cast<double>(TP) / (TP + FP);
    double recall = static_cast<double>(TP) / (TP + FN);

    return std::make_tuple(precision, recall);
}

int main() {
    // At t = 0
    uint64_t initialTimestamp{ 0 };
    // prediction-1
    Center center1{ 0, 0, 0 }; Dimensions dim1{ 2, 4, 1.5 }; Velocity vel1{ 1, 6 }; double confidence1{ 0.8 };
    BoundingBox bounding_box1{ center1, dim1 };
    Prediction prediction1{ bounding_box1, vel1, initialTimestamp, confidence1 };

    // prediction-2
    Center center2{ 3, 4, 0 }; Dimensions dim2{ 1, 4, 8 }; Velocity vel2{ 0.2, 0.4 }; double confidence2{ 0.8 };
    BoundingBox bounding_box2{ center2, dim2 };
    Prediction prediction2{ bounding_box2, vel2, initialTimestamp, confidence2 };

    // prediction-3
    Center center3{ 4, 0, 0 }; Dimensions dim3{ 4, 2, 6 }; Velocity vel3{ 2, 0 }; double confidence3{ 0.8 };
    BoundingBox bounding_box3{ center3, dim3 };
    Prediction prediction3{ bounding_box3, vel3, initialTimestamp, confidence3 };

    // prediction-4
    Center center4{ 8, 2, 0 }; Dimensions dim4{ 4, 2, 1 }; Velocity vel4{ 1, 0.2 }; double confidence4{ 0.8 };
    BoundingBox bounding_box4{ center4, dim4 };
    Prediction prediction4{ bounding_box4, vel4, initialTimestamp, confidence4 };
    
    //prediction vector
    vector<Prediction> predictions{ prediction1, prediction2, prediction3, prediction4 };


    // At t = 5
    uint64_t finalTimestamp{ 5 };

    // label-1
    center1 = { 5, 31, 0 }; dim1 = { 2, 4, 1.5 }; 
    bounding_box1 = { center1, dim1 };
    Label label1{ bounding_box1, 0 };

    // label-2
    center2 = { 6, 6, 0 }; dim2 = { 1, 4, 8 };
    bounding_box2 = { center2, dim2 };
    Label label2{ bounding_box2, 1 };
    
    // label-3
    center3 = { 50, 80, 0 }; dim3 = { 4, 2, 6 }; 
    bounding_box3 = { center3, dim3 };
    Label label3{ bounding_box3, 2 };

    // label-4
    center4 = { 14, 0, 0 }; dim4 = { 4, 2, 1 };
    bounding_box4 = { center4, dim4 };
    Label label4{ bounding_box4, 4 };


    std::vector<Label> labels{ label1, label2, label3, label4 };

    // Create Predictions object
    Predictions predictionData{ predictions };

    // Extrapolate predictions to a specific timestamp (e.g., 1)
    predictionData.ExtrapolatePredictions(5);

    // Compute precision and recall for the same
    tuple<double, double> output;
    output = ComputePrecisionRecall(labels, predictionData);

    // scenarios that happen:
    //  - with prediction 1, only label 1 and gets a match and they are same object. (TP++) 
    //  - with prediction 2, best possible match is label 2 which is less than threshold, I have considered (FP++)
    //  - with prediction 3, only label 4 gets a match but since they are different objects. (No change in values)
    //  - with prediction 4, no label gets a match. (FP++)

    //  - label 2 doesn't get matched with any prediction. (FN++) 
    //  - label 3 doesn't get matched with any prediction. (FN++)
    
    //So, precision -> 0.3333
    //    recall    -> 0.3333


    double precision = get<0>(output);
    double recall = get<1>(output);

	assert (abs(precision - 0.333333) < 0.000001);
	assert (abs(recall - 0.333333) < 0.000001);
	
    return 0;
}
