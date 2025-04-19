# TopDon_Thermal
This repository demonstrates how to use TopDon Thermal Camera.

## Getting Started
Topdon Camera can be read like a normal USB camera using OpenCV if you are on linux and Rpi (again linux). However, on Mac you need to fetch the frames using other means. I have used FFmpeg to fetch the frames and then used OpenCV to process them. Once the frames are fetched, you can use OpenCV to process them as you would with any other camera. 

## Notes for Rpi.
* Try not to instal OpenCV using pip. Whenever I tried it always had some dependency issue with either numpy or torch. Better to use apt-get to install OpenCV.
* Performance is limilted on Rpi. The camera is capable of 25FPS but the given code can process only at around 11FPS. However, this could be increased by using a better buffers I believe.

## Notes for Mac.
* The camera is not detected as a normal USB camera. You need to use FFmpeg to fetch the frames. The given code uses FFmpeg to fetch the frames and then uses OpenCV to process them.

## Notes for Linux
* No notes. Things should work out of box without hassle.


## Files
* AccessThermalCam.ipynb : Provides a minial example of how to access the camera using OpenCV (FFmpeg if on Mac).
* ThermalRecord.py : Provides a minimal example of how to record the camera using OpenCV (FFmpeg if on Mac).
* NasalRespiration.py (ipynb) : Provides a minimal example of how to use the camera to detect nasal respiration. The code is not optimized and is just a proof of concept. The code uses OpenCV to process the frames and then uses a simple algorithm to detect nasal respiration. 
* models folder : Contains a few YOLO models used to localize Chest, Face and Nostrils. 
