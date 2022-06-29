#include <cmath>

#include "CameraCalibration.hh"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"

CalibrationSettings::CalibrationSettings(const cv::FileNode& node) {
	node["BoardSize_Width" ] >> board_size.width;
	node["BoardSize_Height"] >> board_size.height;
	node["Calibrate_Pattern"] >> pattern;
	node["Square_Size"]  >> square_size;
	
	node["Input"] >> input;
	node["Input_FlipAroundHorizontalAxis"] >> flip_around_horizontal_axis;
	node["Input_Delay"] >> delay;
	
	node["Show_UndistortedImage"] >> show_undistorted_image;
	
	node["Write_DetectedFeaturePoints"] >> write_detected_feature_points;
	node["Write_extrinsicParameters"] >> write_extrinsic_params;
	node["Write_gridPoints"] >> write_3d_grid_points;
	node["Write_capturedImagesPath"] >> capture_store_path;
	
	node["Calibrate_AssumeZeroTangentialDistortion"] >> no_tangential_distortion;
	node["Calibrate_NrOfFrameToUse"] >> num_frames;
	node["Calibrate_FixAspectRatio"] >> fix_aspect_ratio;
	node["Calibrate_FixPrincipalPointAtTheCenter"] >> fix_principal_point;
	
	node["Fix_K1"] >> fix_k1;
	node["Fix_K2"] >> fix_k2;
	node["Fix_K3"] >> fix_k3;
	node["Fix_K4"] >> fix_k4;
	node["Fix_K5"] >> fix_k5;
    
	validateSettings(); // terminates if the settings are not valid
}

void CalibrationSettings::validateSettings() {
	if(board_size.width < 0 or board_size.height < 0)
		std::cerr << "Invalid board size!\n";
	if(pattern < 0 or pattern > 2)
		std::cerr << "Invalid pattern!\n";
	if(square_size <= 10e-4)
		std::cerr << "Invalid square sizes!\n";
	if(input < 0 and input > 2)
		std::cerr << "Invalid Input Type chosen!\n";
}

void modifyImagePath(FS::path& image_path, std::string str_to_append) {
    std::string file_name_with_ext = image_path.filename();
    std::string file_name = file_name_with_ext.substr(0, file_name_with_ext.find_last_of('.'));
    file_name_with_ext = file_name + (str_to_append + image_path.extension().string());
    image_path.replace_filename(file_name_with_ext);
}

CameraCalibration::CameraCalibration(bool verbose_,
        CalibrationSettings calib_settings_) : 
    verbose(verbose_) , calib_settings(calib_settings_){
    device_id = 0;
    device_id_is_set = false;
}

CameraCalibration::~CameraCalibration() {

}

void CameraCalibration::fillObjectPoints(std::vector<cv::Point3f>& object_points_buffer) {
    switch(calib_settings.pattern) {
        case(PatternType::CHESS_BOARD): {
            for(auto i = 0; i < calib_settings.board_size.height; i++)
                for(auto j = 0; j < calib_settings.board_size.width; j++)
                    object_points_buffer.push_back(cv::Point3f(
                                j * calib_settings.square_size, 
                                i * calib_settings.square_size, 0));
            break;
        }
        case(PatternType::CIRCLE_GRID): {
            for(auto i = 0; i < calib_settings.board_size.height; i++)
                for(auto j = 0; j < calib_settings.board_size.width; j++)
                    object_points_buffer.push_back(cv::Point3f(
                                j * calib_settings.dist_between_centers, 
                                i * calib_settings.dist_between_centers, 0));
            break;
        }
        case(PatternType::ASYMMETRIC_CIRCLE_GRID): {
            for(auto i = 0; i < calib_settings.board_size.height; i++)
                for(auto j = 0; j < calib_settings.board_size.width; j++)
                    object_points_buffer.push_back(cv::Point3f(
                                ((2 * j) + (0.5f * i)) * calib_settings.dist_between_centers, 
                                i * calib_settings.dist_between_centers, 0));
            break;
        }
        default: {
            std::cerr << "Pattern not supported\n";
            break;
        }
    }
}

void CameraCalibration::findSalientPoints(const cv::Mat& image, 
        std::vector<cv::Point2f>& salientPoints,
        FS::path& image_path) {
    cv::Mat image_copy;
    image.copyTo(image_copy);
    if(calib_settings.pattern == CHESS_BOARD) {
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(
                        cv::InputArray(image),
                        calib_settings.board_size,
                        cv::OutputArray(corners),
                        cv::CALIB_CB_ADAPTIVE_THRESH +
                        cv::CALIB_CB_NORMALIZE_IMAGE +
                        cv::CALIB_CB_FAST_CHECK);
        if(found) {
            cv::cornerSubPix(
                    cv::InputArray(image), 
                    cv::InputOutputArray(salientPoints),
                    cv::Size2i(11,11), 
                    cv::Size(-1,-1),
                    cv::TermCriteria(cv::TermCriteria::Type::EPS + 
                                    cv::TermCriteria::Type::COUNT,
                                    30,
                                    0.0001));
            cv::drawChessboardCorners(
                    cv::InputOutputArray(image_copy),
                    calib_settings.board_size,
                    cv::InputArray(salientPoints),
                    found);
            if(verbose) {
                modifyImagePath(image_path, "_corners");
                cv::imwrite(image_path.string(), image_copy);
            }
            cv::imshow("Chessboard with detected Corners", image_copy);
            cv::waitKey(1);

        }
        else
            std::cout << "No corner points found for " << image_path.string() << "\n";
    }
    else if((calib_settings.pattern == CIRCLE_GRID) or 
            (calib_settings.pattern == ASYMMETRIC_CIRCLE_GRID)) {
        std::vector<cv::Point2f> centers; 
        bool found = false;
        if(calib_settings.pattern == CIRCLE_GRID)
            found = cv::findCirclesGrid(cv::InputArray(image),
                        calib_settings.board_size,
                        cv::OutputArray(salientPoints),
                        cv::CALIB_CB_SYMMETRIC_GRID);
        else
            found = cv::findCirclesGrid(cv::InputArray(image),
                        calib_settings.board_size,
                        cv::OutputArray(salientPoints),
                        cv::CALIB_CB_ASYMMETRIC_GRID);
        if(found) {
            cv::drawChessboardCorners(
                    cv::InputOutputArray(image_copy),
                    calib_settings.board_size,
                    cv::InputArray(salientPoints),
                    found);
            if(verbose) {
                modifyImagePath(image_path, "_centers");
                cv::imwrite(image_path.string(), image_copy);
            }
            cv::imshow("Circle grid with detected centers", image_copy);
            cv::waitKey(1);
        }
        else
            std::cout << "No center points found for " << image_path.string() << "\n";
    }
    else
        std::cerr << " Unknown Pattern Type defined";
}

void CameraCalibration::calibrateCamera( 
        std::function<cv::Mat(void*)> getImageFromSDK,
        void* handle) {
    if(calib_settings.input == CAPTURED_IMAGES) {
        FS::path image_folder = FS::path(calib_settings.image_folder);
        if(not FS::is_directory(image_folder))
            std::cerr << "Image path should be a valid directory";
        else {
            for(const auto& p : FS::directory_iterator(image_folder)) {
                FS::path image_path = p.path();
                if(FS::is_regular_file(image_path)) {
                    cv::Mat image = cv::imread(image_path.string());
                    if(image.data == nullptr)
                        continue;
                    std::vector<cv::Point2f> salientPoints;
                    findSalientPoints(image, salientPoints, image_path);
                    image_points.push_back(salientPoints);

                    std::vector<cv::Point3f> object_points_buffer;
                    fillObjectPoints(object_points_buffer);
                    object_points.push_back(object_points_buffer);
                }
            }
            auto flags = 0;
            cv::calibrateCamera(
                    cv::InputArrayOfArrays(object_points),
                    cv::InputArrayOfArrays(image_points),
                    calib_settings.board_size,
                    cv::InputOutputArray(camera_matrix),
                    cv::InputOutputArray(dist_coeffs),
                    cv::OutputArrayOfArrays(rvecs),
                    cv::OutputArrayOfArrays(tvecs),
                    flags);
        }
    }
    else if(calib_settings.input == CAPTURED_VIDEO) {
        FS::path video_folder = FS::path(calib_settings.video_folder);
        if(not FS::is_directory(video_folder))
            std::cerr << "Video path should be a valid directory";
        else {
            for(const auto& p : FS::directory_iterator(video_folder)) {
                FS::path video_path = p.path();
                if(FS::is_regular_file(video_path)) {
                    // if on windows system use the DirectShow api
                    // else use the FFMPEG library in linux
                    #if (defined(_WIN64) or defined(_WIN32))
                        cv::VideoCapture capture{video_path.string(),
                                cv::CAP_DSHOW};
                    #else
                        cv::VideoCapture capture{video_path.string(),
                                cv::CAP_FFMPEG};
                    #endif
                    if(not capture.isOpened()) {
                        std::cout << "File found is not a valid video file";
                        continue;
                    }
                    
                    uint32_t frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);
                    uint32_t frame_index = (frame_count - 1) / calib_settings.num_frames;
                    capture.set(cv::CAP_PROP_POS_FRAMES, frame_index);
                    
                    FS::path image_path{video_path};
                    image_path.replace_filename(image_path.filename().string() + "i");
                    image_path.replace_extension(".jpg");
                    
                    for(uint16_t i = 0; i < calib_settings.num_frames; i++) {
                        cv::Mat image;
                        if(capture.read(image))
                            std::cerr << "Frame obtained from Video is invalid";
                        std::vector<cv::Point2f> salientPoints;
                        findSalientPoints(image, salientPoints, video_path);
                        image_points.push_back(salientPoints);

                        std::vector<cv::Point3f> object_points_buffer;
                        fillObjectPoints(object_points_buffer);
                        object_points.push_back(object_points_buffer);
                    }
                    auto flags = 0;
                    cv::calibrateCamera(
                            cv::InputArrayOfArrays(object_points),
                            cv::InputArrayOfArrays(image_points),
                            calib_settings.board_size,
                            cv::InputOutputArray(camera_matrix),
                            cv::InputOutputArray(dist_coeffs),
                            cv::OutputArrayOfArrays(rvecs),
                            cv::OutputArrayOfArrays(tvecs),
                            flags);
                }
            }
        }
    }
    else if (calib_settings.input == LIVE_STREAM){
        FS::path capture_path{calib_settings.capture_store_path};
        if(not getImageFromSDK) {
            if(not device_id_is_set)
                std::cout << "device_id variable not set. Default value of 0 will be used";
            cv::VideoCapture capture(device_id, cv::CAP_ANY);
            std::cout << "Capturing " << calib_settings.num_frames << " frames!\n";
            for(uint16_t i = 0; i < calib_settings.num_frames; i++) {
                cv::Mat image;
                if(not capture.read(image)) {
                    std::cout << "Image wasn't captured. Trying again ...";
                    --i;
                    continue;
                }
                std::vector<cv::Point2f> salientPoints;
                findSalientPoints(image, salientPoints, capture_path);
                image_points.push_back(salientPoints);

                std::vector<cv::Point3f> object_points_buffer;
                fillObjectPoints(object_points_buffer);
                object_points.push_back(object_points_buffer);
            }
        }
        else {
            for(uint16_t i = 0; i < calib_settings.num_frames; i++) {
                cv::Mat image = getImageFromSDK(handle);
                if(image.data == nullptr) {
                    std::cout << "Image wasn't captured. Trying again ...";
                    --i;
                    continue;
                }
                std::vector<cv::Point2f> salientPoints;
                findSalientPoints(image, salientPoints, capture_path);
                image_points.push_back(salientPoints);

                std::vector<cv::Point3f> object_points_buffer;
                fillObjectPoints(object_points_buffer);
                object_points.push_back(object_points_buffer);
            }
        }
        auto flags = 0;
        cv::calibrateCamera(
                cv::InputArrayOfArrays(object_points),
                cv::InputArrayOfArrays(image_points),
                calib_settings.board_size,
                cv::InputOutputArray(camera_matrix),
                cv::InputOutputArray(dist_coeffs),
                cv::OutputArrayOfArrays(rvecs),
                cv::OutputArrayOfArrays(tvecs),
                flags);
    }
    else{
        std::cerr << "Input type not supported\n";
    }
}

double CameraCalibration::computeReprojectionErrors() {
    std::vector<cv::Point2f> projected_image_points;
    per_capture_projection_error.resize(object_points.size());
    double total_error = 0.0;
    uint64_t total_points = 0;

    for(std::size_t i = 0; i < object_points.size(); i++) {
        std::size_t points_per_capture = object_points[i].size();
        projectPoints(cv::InputArray(object_points[i]),
                cv::InputArray(rvecs[i]),
                cv::InputArray(tvecs[i]),
                cv::InputArray(camera_matrix),
                cv::InputArray(dist_coeffs),
                cv::OutputArray(projected_image_points));
        double error = cv::norm(
                cv::InputArray(image_points[i]), 
                cv::InputArray(projected_image_points), 
                cv::NORM_L2);
        per_capture_projection_error[i] = std::sqrt((error * error) / points_per_capture);
        total_error += (error * error);
        total_points += points_per_capture;
    }
    return(std::sqrt(total_error / total_points));
}
