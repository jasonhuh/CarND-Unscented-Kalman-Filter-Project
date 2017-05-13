#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd rmse(4);
    rmse << 0,0,0,0;

    if (estimations.size() == 0) {
        cout << "the estimation vector size should not be zero\n";
        return rmse;
    }
    if (estimations.size() != ground_truth.size()) {
        cout << "the estimation vector size should equal ground truth vector size\n";
        return rmse;
    }

    //accumulate squared residuals
    for(size_t i = 0; i < estimations.size(); ++i){
        VectorXd residual = (estimations[i] - ground_truth[i]).array().square();
        rmse += residual;
    }
    rmse = rmse / estimations.size();
    rmse = rmse.array().sqrt();

    return rmse;
}

double Tools::NormalizeAngle(double angle) {
    //double val = atan2(sin(angle), cos(angle));
    return fmod (angle, 3.142);
}