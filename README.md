# Camera Calibration using OpenCV

This library facilitates calibration of cameras using [OpenCV](https://opencv.org/) library.

## Installation
Use the following commands to install the library. Make sure that [OpenCV](https://opencv.org/) is installed
```
git clone https://github.com/unbeschwert/opencv-camera-calibration.git camera-calibration
cd camera-calibration
mkdir build && cd build
cmake ..
make
sudo make install
```

## Important

1. The parameters for the ```calibrateCamera((std::function<cv::Mat(void *)> getImageFromSDK, void *handle)``` were used based on the [Microsoft Azure Kinect SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK) library. There, a device handle is used to capture images and video. Here is an example code for ```getImageFromSDK()``` :
```
cv::Mat getImageFromKamera(void* handle) {
    /*
     * This function is mainly designed to be used 
     * as a callback inside camera calibration library 
     * to get images from Kinect Kamera. Note that you can
     * rewrite this function to obtain images from any other camera. 
     * It is supposed to be a drop in replacement to provide a means to 
     * get images from any camera.
     */
    k4a::capture capture_handle;

    k4a::device *device_handle = reinterpret_cast<k4a::device*>(handle);
	
    if(not device_handle->get_capture(&capture_handle)) {
		std::cout << "Couldn't create handle for capture!\n";
		exit(1);
	}
    k4a::image rgb_image_handle = capture_handle.get_color_image();
    if(not rgb_image_handle.is_valid()) {
		std::cout << "Couldn't get an image!\n";
		exit(1);
	}
    
    uint32_t width = rgb_image_handle.get_width_pixels();
    uint32_t height = rgb_image_handle.get_height_pixels();

    uint8_t channels = rgb_image_handle.get_size() / 
        (width * height * sizeof(uint8_t));

    cv::Mat image(height, width, CV_MAKETYPE(
            cv::DataType<uint8_t>::type, channels));
    std::memcpy(mat.data, rgb_image_handle.get_buffer(),
            width * height * channels * sizeof(uint8_t));

    rgb_image_handle.reset();
    capture_handle.reset();
	
    return(image);
}
```
2. This library was only tested on linux. I would be glad if someone tests this on Windows and Mac OS.


## Issues and Pull Requests
If you have any ideas about improvements or changes that might make this library better, please first create an issue so that we can discuss first. 
