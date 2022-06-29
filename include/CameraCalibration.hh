#include <iostream>
#include <cstdint>
#include <string>
#include <filesystem>
#include <functional>

#include "opencv2/core/types.hpp"
#include "opencv2/core/persistence.hpp"

namespace FS = std::filesystem;

typedef enum {
    /*
     * selects the source on which calibration should be performed
     */
    CAPTURED_IMAGES = 0,
	CAPTURED_VIDEO = 1,
    LIVE_STREAM = 2
} InputType;

typedef enum {
    CHESS_BOARD = 0,
    CIRCLE_GRID = 1,
    ASYMMETRIC_CIRCLE_GRID = 2
} PatternType;

typedef struct CalibrationSettings {
    /*
     * Settings used when performing camera calibration
     */
	// General variables
    cv::Size2i board_size; // number of columns in pattern
    float square_size; // size of the squares in chessboard pattern
    float dist_between_centers; // distance between centers in circle grid pattern
    InputType input; // source of the input
    PatternType pattern; // pattern chosen for calibration
    bool write_extrinsic_params; // output extrinsic params if true
    bool write_detected_feature_points; //output feature points detected if true
	bool write_3d_grid_points;
	bool show_undistorted_image;
	bool flip_around_horizontal_axis;
	bool no_tangential_distortion;
	bool fix_principal_point;
	bool fix_k1, fix_k2, fix_k3, fix_k4, fix_k5;

	// Variables related to CAPTURED_IMAGES input type
    std::string image_folder; // the absolute folder path where images are stored

	// Variables related to LIVE_VIDEO_STREAM input type
    std::string capture_store_path; // path for storing the captured images  
    std::string video_folder;
	bool fix_aspect_ratio;
	uint16_t delay;
	
    // Depending on InputType it either refers to 
    // number of frames at fixed intervals to be used from the input video
    // or number of images to be captured from the live video feed
	uint16_t num_frames; // number of frames to use for calibration
    
    CalibrationSettings() = delete;
    CalibrationSettings(const cv::FileNode& node);
    void validateSettings();
    ~CalibrationSettings();

} CalibrationSettings;

class CameraCalibration {
    /*
     * This class can be used to capture images and store them in 
     * a user-defined folder which can be later used for 
     * camera calibration
     */

    bool verbose;
    // currently only integer device ids are supported
    // TODO: Figure out a way to properly support 
    // different ways of video capture 
    int device_id;
    bool device_id_is_set;
    CalibrationSettings calib_settings;
    
    // object points on the given pattern. It must be of form 
    // [0, 0, 0], [1, 0, 0] ... [rows-1, 0, 0]
    // ...                                 ...
    // ...                                 ...    
    // [6, 0, 0], [6, 1, 0] ... [rows-1, colums-1, 0]
    // The z values are zero. The rows and columns 
    // are variables in CalibrationSettings struct
    
    void fillObjectPoints(std::vector<cv::Point3f>& object_points_buffer);
    void findSalientPoints(const cv::Mat& image, 
            std::vector<cv::Point2f>& salientPoints,
            FS::path& image_path);

    public:
    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;
    cv::Mat camera_matrix; // instrinsic camera matrix
    std::vector<cv::Mat> rvecs; // rotation vectors calculated
    std::vector<cv::Mat> tvecs; // translation vectors calculated
    std::vector<float> dist_coeffs;
    std::vector<float> per_capture_projection_error;
    
    CameraCalibration() = delete;
    CameraCalibration(bool verbose,
            CalibrationSettings calib_settings);
    // This function provides a way for user to capture
    // images from a given SDK of a camera.
    // NOTE: The output of getImageFromSDK() should be 
    // cv::Mat
    void calibrateCamera(std::function<cv::Mat(void*)> getImageFromSDK, 
            void* handle);
    double computeReprojectionErrors();
    ~CameraCalibration();
};
