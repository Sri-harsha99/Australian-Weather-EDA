#include <iostream>
#include <vector>
#include <tuple>
#include <cstdint>
#include <cassert>

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
        prediction.timestamp = timestamp;
    }
    }
};

std::tuple<double, double> ComputePrecisionRecall(const std::vector<Label>& labels, const Predictions& predictions, double threshold = 0.5) {
    // Implement the computation of precision and recall here
    
    int TP = 0;
    int FP = 0;
    int FN = 0;

    for (const auto& prediction : predictions.predictions_) {
        bool match_found = false;
        for (const auto& label : labels) {
            if (ComputeIoU(prediction.bounding_box, label.bounding_box) > threshold) {
                match_found = true;
                break;
            }
        }
        if (match_found) TP++;
        else FP++;
    }

    // Compute FN
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
    // Test 1: Extrapolate predictions and compute precision-recall
    Center center1{ 0, 0, 0 };
    Dimensions dim1{ 2, 4, 1.5 };
    Velocity vel1{ 0.5, 0.2 };
    uint64_t timestamp1{ 0 };
    double confidence1{ 0.8 };

    BoundingBox bounding_box1{ center1, dim1 };
    Label label1{ bounding_box1, 0 };
    std::vector<Label> labels1{ label1 };

    Prediction prediction1{ bounding_box1, vel1, timestamp1, confidence1 };
    std::vector<Prediction> predictions1{ prediction1 };

    // Create Predictions object
    Predictions predictionObj1{ predictions1 };

    // Extrapolate predictions to a specific timestamp (e.g., 1)
    predictionObj1.ExtrapolatePredictions(1);

    // Compute precision and recall for the same
    tuple<double, double> output;
    output = ComputePrecisionRecall(labels1, predictionObj1);

    std::cout << "Test 1: Precision = " << get<0>(output) << ", Recall = " << get<1>(output) << std::endl;

    return 0;
}
