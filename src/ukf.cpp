#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
using std::pair;

static const bool DEBUG_MODE = true;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Set state dimension
  n_x_ = 5;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ =  0.8; // original value was 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.6; // original value was 30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

    // Set augmented dimension
    n_aug_ = 7;

    // Define spreading parameter
    lambda_ = 3 - n_x_;

    // Init vector for weights
    weights_ = VectorXd(2 * n_aug_ + 1);

    // Init matrix with predicted sigma points as columns
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

    // Set NIS (Normalized Innovation Squared) for radar
    NIS_radar_ = 0.0;

    // Set NIS for laser
    NIS_laser_ = 0.0;

    // Set previous timestamp
    previous_timestamp_ = 0.0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(const MeasurementPackage& meas_package) {

    // Step1: Initialize
    if (!is_initialized_) {
        ConfigureInitialMeasurement(meas_package);
        previous_timestamp_ = meas_package.timestamp_;
        is_initialized_ = true;
        return;
    }

    // Step2: Predict
    //compute the time elapsed between the current and previous measurements
    auto dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
    previous_timestamp_ = meas_package.timestamp_;
    Predict(dt);

    // Step3: Update
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) { // Radar updates
        UpdateRadar(meas_package);
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) { // Laser updates
        UpdateLidar(meas_package);
    }
}

void UKF::ConfigureInitialMeasurement(const MeasurementPackage& meas_package) {

    x_ << 1, 1, 0, 0, 0;
    P_ << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;
    //set weights
    weights_(0) = lambda_/(lambda_ + n_aug_);
    for (auto i = 1; i < 2 * n_aug_ + 1; ++i) {
        weights_(i) = 0.5/(lambda_ + n_aug_);
    }
    Xsig_pred_.fill(0.0);

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        /**
        Convert radar from polar to cartesian coordinates and initialize state.
        */
        auto ro = meas_package.raw_measurements_[0];
        auto phi = meas_package.raw_measurements_[1];
        //auto ro_dot = meas_package.raw_measurements_[2];
        x_ << ro * cos(phi), ro * sin(phi), 0.0, 0.0, 0.0; //ro_dot * cos(phi), ro_dot * sin(phi);
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0.0, 0.0, 0.0;
    }
    return;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Predict(double delta_t) {
    // Step1: Generate sigma points
    MatrixXd Xsig_aug = GenerateAugmentedSigmaPoints(x_, P_);

    // Step2: Predict sigma points
    Xsig_pred_ = PredictSigmaPoints(Xsig_aug, delta_t);

    // Step3: Predict Mean and Covariance
    auto predction = PredictMeanAndCovariance(Xsig_pred_);
    x_ = predction.first;
    P_ = predction.second;
}

MatrixXd UKF::GenerateAugmentedSigmaPoints(const VectorXd& x, const MatrixXd& P) {

    //create augmented mean vector
    VectorXd x_aug = VectorXd(n_aug_);

    //create augmented state covariance
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    //create augmented mean state
    x_aug.fill(0.0);
    x_aug << x, 0, 0;

    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(n_x_, n_x_) << P;
    P_aug.bottomRightCorner(2, 2) << (std_a_ * std_a_), 0, 0, (std_yawdd_ * std_yawdd_);

    //create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    //create augmented sigma points
    Xsig_aug.col(0)  = x_aug;
    for (int i = 0; i< n_aug_; ++i)
    {
        Xsig_aug.col(i+1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
        Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
    }

    if (DEBUG_MODE) {
        std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;
    }

    return Xsig_aug;
}

MatrixXd UKF::PredictSigmaPoints(const MatrixXd& Xsig_aug, const double delta_t) {

    //create matrix with predicted sigma points as columns
    MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

    //predict sigma points
    for (auto i = 0; i < 2 * n_aug_ + 1; ++i)
    {
        //extract values for better readability
        double p_x = Xsig_aug(0,i);
        double p_y = Xsig_aug(1,i);
        double v = Xsig_aug(2,i);
        double yaw = Xsig_aug(3,i);
        double yawd = Xsig_aug(4,i);
        double nu_a = Xsig_aug(5,i);
        double nu_yawdd = Xsig_aug(6,i);

        //predicted state values
        double px_p, py_p;

        //avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v/yawd * ( sin (yaw + yawd * delta_t) - sin(yaw));
            py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd * delta_t) );
        }
        else {
            px_p = p_x + v * delta_t * cos(yaw);
            py_p = p_y + v * delta_t * sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd*delta_t;
        double yawd_p = yawd;

        //add noise
        px_p += 0.5 * nu_a * delta_t * delta_t * cos(yaw);
        py_p += + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
        v_p += + nu_a * delta_t;

        yaw_p += 0.5 * nu_yawdd * delta_t * delta_t;
        yawd_p += nu_yawdd * delta_t;

        //write predicted sigma point into right column
        Xsig_pred(0,i) = px_p;
        Xsig_pred(1,i) = py_p;
        Xsig_pred(2,i) = v_p;
        Xsig_pred(3,i) = yaw_p;
        Xsig_pred(4,i) = yawd_p;
    }

    if (DEBUG_MODE) {
        std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;
    }

    return Xsig_pred;
}

pair<VectorXd, MatrixXd> UKF::PredictMeanAndCovariance(const MatrixXd& Xsig_pred) {

    //create vector for predicted state
    VectorXd x = VectorXd(n_x_);
    x.fill(0.0);

    //create covariance matrix for prediction
    MatrixXd P = MatrixXd(n_x_, n_x_);
    P.fill(0.0);

    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        x += weights_(i) * Xsig_pred_.col(i);
    }

    //predicted state covariance matrix
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  //iterate over sigma points
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x;
        x_diff(3) = Tools::NormalizeAngle(x_diff(3));
        P += weights_(i) * x_diff * x_diff.transpose() ;
    }

    if (DEBUG_MODE) {
        std::cout << "Predicted state" << std::endl;
        std::cout << x << std::endl;
        std::cout << "Predicted covariance matrix" << std::endl;
        std::cout << P << std::endl;
    }

    return {x, P};
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(const MeasurementPackage& meas_package) {
    if(x_(0) == 0) { return; }

    const int n_z = 2; // measurement dimension for lidar
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1); // measurement model

    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        Zsig(0,i) = Xsig_pred_(0,i);    //px
        Zsig(1,i) = Xsig_pred_(1,i);    //py
    }
    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z,n_z);
    R <<    std_laspx_*std_laspx_, 0,
            0, std_laspx_*std_laspx_;

    VectorXd z = VectorXd(n_z);
    z << meas_package.raw_measurements_(0),
            meas_package.raw_measurements_(1);

    NIS_laser_ = UpdateCommon(n_z, R, z, Zsig);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(const MeasurementPackage& meas_package) {
    if(x_(0) == 0) { return; }

    const int n_z = 3; // measurement dimension, radar can measure r, phi, and r_dot
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1); // measurement model

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        // extract values for better readibility
        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);
        double v  = Xsig_pred_(2,i);
        double yaw = Xsig_pred_(3,i);

        double v1 = cos(yaw)*v;
        double v2 = sin(yaw)*v;

        Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
        Zsig(1,i) = atan2(p_y,p_x);                                 //phi
        Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
    }
    MatrixXd R = MatrixXd(n_z,n_z);
    R <<    std_radr_* std_radr_, 0, 0,
            0, std_radphi_ * std_radphi_, 0,
            0, 0, std_radrd_ * std_radrd_;

    VectorXd z = VectorXd(n_z);
    z << meas_package.raw_measurements_(0),
            meas_package.raw_measurements_(1),
            meas_package.raw_measurements_(2);

    NIS_radar_ = UpdateCommon(n_z, R, z, Zsig);
}

double UKF::UpdateCommon(const int n_z, const MatrixXd R,
                         const VectorXd& z, const MatrixXd& Zsig) {

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i=0; i < 2 * n_aug_ + 1; ++i) {
        z_pred += weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  //2n+1 simga points
        VectorXd risidual = Zsig.col(i) - z_pred;
        risidual(1) = Tools::NormalizeAngle(risidual(1));
        S += weights_(i) * risidual * risidual.transpose();
    }

    S += R;

    if (DEBUG_MODE) {
        std::cout << "z_pred: " << std::endl << z_pred << std::endl;
        std::cout << "S: " << std::endl << S << std::endl;
    }

    //calculate cross correlation matrix
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
        VectorXd z_diff = Zsig.col(i) - z_pred;
        z_diff(1) = Tools::NormalizeAngle(z_diff(1));

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        x_diff(3) = Tools::NormalizeAngle(x_diff(3));

        Tc += weights_(i) * x_diff * z_diff.transpose();
    }
    //calculate Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    VectorXd z_diff = z - z_pred;

    z_diff(1) = Tools::NormalizeAngle(z_diff(1));

    //update state mean and covariance matrix
    x_ += K * z_diff;
    P_ -= K * S * K.transpose();

    double nis = z_diff.transpose() * S.inverse() * z_diff;
    if (DEBUG_MODE) {
        std::cout << "Updated state x: " << std::endl << x_ << std::endl;
        std::cout << "Updated state covariance P: " << std::endl << P_ << std::endl;
        cout << "NIS: " << NIS_laser_ << '\n';
    }
    return nis;
}
