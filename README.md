# yolo-rtsp-security-cam
Python RTSP security camera app with motion detection features that are based on image processing instead of a dedicated sensor. Also includes YOLO object detection so you can set it to record only specific things such as people, dogs, other animals or particular objects. All that's required is an RTSP camera and a PC with a GPU capable of running YOLO.

## Getting Started

To get started, first make sure your system has the required software. If using a Debian/Ubuntu based distro, you can install the required software with the following:

```bash
sudo apt update
sudo apt install gcc python3-dev python3-pip git ffmpeg
```

YOLO object detection also requires CUDA for Nvidia GPUs or ROCm for AMD GPUs. I've created step by step guides on how to install both of these.

CUDA Guide: https://phazertech.com/tutorials/cuda.html

ROCm Guide: https://phazertech.com/tutorials/rocm.html

It might be possible to run YOLO on the CPU, but it will be slow and I didn't test it, so you're highly encouraged to use GPU acceleration.

Next, clone this repository:

```bash
git clone https://github.com/PhazerTech/yolo-rtsp-security-cam
```

Now install the required dependencies with pip:

```bash
cd yolo-rtsp-security-cam
pip3 install -r requirements.txt
```

## Running the App

The only arguments required to run the app are --stream followed by the RTSP address of your video stream, and --yolo followed by a comma separated list of objects you'd like the app to detect. The list of valid objects can be found in the coco.names file.

To run it with default settings, enter the following and replace 'ip:port/stream-name' with your stream's address.  Feel free to modify 'person,dog,cat' to which ever objects you'd like the app to detect.

```bash
python3 yolo-rtsp-security-cam.py --stream rtsp://ip:port/stream-name --yolo person,dog,cat
```

To open a window where you can view the stream while the program is running, include the --monitor argument. The YOLO bounding boxes and class IDs will be drawn on the stream in this window, however the recordings will not include these bounding boxes, only the raw stream will be recorded.

```bash
python3 yolo-rtsp-security-cam.py --stream rtsp://ip:port/stream-name --yolo person,dog,cat --monitor
```

The program will print a message whenever it starts a recording and ends a recording, and also provide a timestamp.
For example:

```bash
$ python3 yolo-rtsp-security-cam.py --stream rtsp://192.168.0.156:8554/frontdoor --yolo person,dog,cat
13-05-09 recording started
13-05-30 recording stopped
14-01-01 recording started
14-01-09 recording stopped
```

It will create a folder with the current date for storing that day's recordings. A new folder will be created each day with the current date so that it can be left to run indefinitely. Press the 'Q' key to quit the program.

By default, YOLO will run the nano sized model, but you can change this by using the --model argument and specify which sized model you want to run. For example, enter '--model yolov8s' to run the small model, or '--model yolov8m' to run the medium model, and so on. See the official YOLOv8 repo for more info on these models: https://github.com/ultralytics/ultralytics

AMD GPUs might stall for a minute the first time they run a new model, in which case you might see errors about dropped frames, but this should only happen during the first run.

Larger models will provide better detection results, but will also require more memory and processing power. I recommend sticking with the default nano model or the small model, because anything larger will have major diminishing returns.  If you'd like to run the program without YOLO so that it only records based on motion detection, simply omit the --yolo argument. Doing this will make the app extremely lightweight and able to run on low power devices such as a Raspberry Pi 4, exactly like the previous version of the app I made here: https://github.com/PhazerTech/rtsp-security-cam

## Advanced Settings

The program doesn't constantly run YOLO object detection, instead it constantly detects motion and only starts the YOLO object detection if it detects motion first. It works this way in order to be more power efficient.  If the default motion detection settings are providing poor results, additional arguments can be provided to tweak the sensitivity of the motion detection algorithm and to enable testing mode which helps to find the optimal threshold value, but in most cases this shouldn't be necessary.

--threshold - Threshold value determines the amount of motion required to trigger a recording. Higher values decrease sensitivity to help reduce false positives. If using YOLO detection, false positives are less of a concern so the default value should be fine in most cases. Default is 350. Max is 10000.

--start_frames - The number of consecutive frames with motion activity required to start a recording. Raising this value might help if there's too many false positive recordings, especially when using a high frame rate stream greater than 30 FPS. But again, if using YOLO then the default value should be fine. Default is 3. Max is 30.

--tail_length - The number of seconds without motion activity required to stop a recording. Raising this value might help if the recordings are stopping too early. Default is 8. Max is 30.

--auto_delete - Entering this argument enables the auto-delete feature. Recordings that have a total length equal to the tail_length value are assumed to be false positives and are auto-deleted.

--testing - Testing mode disables recordings and prints out the motion value for each frame if greater than threshold. Helpful when fine tuning the threshold value.

--frame_click - Allows the user to advance frames one by one by pressing any key. For use with testing mode on video files, not live streams, so make sure to provide a video file instead of an RTSP address for the --stream argument if using this feature.

Check out my video about this app on my YouTube channel for more details: https://youtu.be/m8dIJN6ePKA

## Contact & Support

If you found value in this software then consider supporting me: https://phazertech.com/funding.html

If you have any questions feel free to contact me: https://phazertech.com/contact.html

## Copyright

Copyright (c) 2023, Phazer Tech

This source code is licensed under the Affero GPL. See the LICENSE file for details.
