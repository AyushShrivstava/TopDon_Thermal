import cv2 as cv
import numpy as np
import argparse
import os
import time

# Function to record thermal video from a specified device and save it to a directory
def thermal_record(dev, dir):
    # Open video capture on the given device using the V4L backend
    cap = cv.VideoCapture('/dev/video' + str(dev), cv.CAP_V4L)

    # Disable RGB conversion (captures raw YUYV)
    cap.set(cv.CAP_PROP_CONVERT_RGB, 0.0)

    # Get original width and half the height (assuming image is vertically stacked)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) // 2

    # Generate timestamp for the output file name
    start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    file_path = os.path.join(dir, 'thermal_rec_' + start_time + '.avi')

    # Create a VideoWriter to save the thermal video at 25 FPS
    videowriter = cv.VideoWriter(file_path, cv.VideoWriter_fourcc(*'XVID'), 25, (width, height))

    print("Starting thermal recording at", start_time)
    frames = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit if frame not captured
        

            # Split YUYV data (top and bottom halves), and keep the first (thermal) part
            imdata, _ = np.array_split(frame, 2)
            
            # Flip vertically (adjust orientation)
            imdata = cv.flip(imdata, 0)

            
            # Convert YUYV format to BGR for visualization and saving
            imdata = cv.cvtColor(imdata, cv.COLOR_YUV2BGR_YUYV)

            # Resize for display (optional, not used for saving)
            bgr = cv.resize(imdata, (width * 3, height * 3), interpolation=cv.INTER_CUBIC)
            cv.imshow('frame', bgr)

            # Write the processed thermal frame to video file
            videowriter.write(imdata)
            frames += 1

            # Stop if 'q' is pressed
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        pass

    # Release all resources and close windows
    cv.destroyAllWindows()
    cap.release()
    videowriter.release()

    # Print summary
    print("\nRecording stopped at", time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    print("Thermal recording saved in", file_path)
    print("Total frames recorded:", frames)
    print("Recording duration:", time.strftime("%H:%M:%S", time.gmtime(frames // 25)))

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Thermal Recording')
    parser.add_argument('--dev', type=int, default=0, help='Device number')
    parser.add_argument('--dir', type=str, default='Rpi/thermal-rec', help='Directory to save the recording')
    args = parser.parse_args()

    # Validate device and directory inputs
    if args.dev < 0:
        print("Invalid device number. Exiting Program")
        exit()

    if not os.path.exists(args.dir) or not os.path.isdir(args.dir):
        print("Invalid directory. Exiting Program")
        exit()

    # Start recording
    thermal_record(args.dev, args.dir)
    cv.destroyAllWindows()
