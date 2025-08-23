# Driver drowsiness detection (DDD) training and evaluation workspace

Workspace for testing and evaluating a <b>Driver Drowsiness Detection</b> (DDD) system
using various models, features and metrics.

Currently, the system is based only on the [UTA-RLDD](https://sites.google.com/view/utarldd/home) dataset.

## Setup

> Setup instructions are specific for <b>Windows 11</b>, since the core version was developed on a Windows machine.
> Due to this, some old packages are required (e.g. tensorflow, as seen below).

> For optimal experience development on Linux is recommended.
> Future releases may or may not include an updated/non-deprecated setup. 

> NVIDIA GPU is required due to CUDA (some fallbacks to use CPU exist - may or may not be fixed in the future).

### Tensorflow

- [Windows Native](https://www.tensorflow.org/install/pip#windows-native)
 
### [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)

- [Miniconda3-py310_23.11.0-1-Windows-x86_64.exe](https://repo.anaconda.com/miniconda/)

```
conda info --envs       
conda activate <ENV_NAME>
```

### Pip install

```
pip install <PACKAGE_NAME>
```
```
"numpy<2.0"
scipy
opencv-contrib-python
scikit-learn
ffmpeg-python-0.2.0
ffmpeg
"facenet-pytorch<2.6.0" --no-deps  
ultralytics
"mediapipe<0.10.9" 
"protobuf<3.20"
numpy
imbalanced-learn
numbda
decord
```

#### Pytorch

Find a compatible CUDA version (run `nvidia-smi` in terminal).

Follow [the official pytorch docs](https://pytorch.org/get-started/locally/) to get the install command. 
For example:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Conda install

```
conda install <PACKAGE_NAME> 
```

```
openpyxl
-c conda-forge ffmpeg
```

### Manual requirements

#### Misc
Need to add 
[builder.py](https://github.com/protocolbuffers/protobuf/blob/main/python/google/protobuf/internal/builder.py)
manually to miniconda environment folder

``C:\\Users\\gvese\\miniconda3\\envs\\drows_d_d\\lib\\site-packages\\google\\protobuf\\internal``

<br>

#### Models for face detection and alignment

Download models into `/models`:
###### Face detection
- dlib (DLIB)
- MTCNN (MTCNN)
- [haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) (OPENCV)
- [yolov11n-face.pt](https://github.com/akanametov/yolo-face?tab=readme-ov-file) (YOLO)
- [face_detection_yunet_2023mar.onnx](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (YUNET)
- [blaze_face_short_range.tflite](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector) (MEDIAPIPE)
- [mmod_human_face_detector.dat](https://github.com/davisking/dlib-models/blob/master/mmod_human_face_detector.dat.bz2) (DLIB_CNN)
###### Landmark
- [face_landmarker.task](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models) (MEDIAPIPE landmarker)
- [lbfmodel.yaml](https://github.com/php-opencv/php-opencv-examples/blob/master/models/opencv-facemark-lbf/lbfmodel.yaml) (LBF landmarker)

<br>

#### CUDA + CUDNN (NVIDIA GPU support)

- [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11)

- [CUDNN](https://developer.nvidia.com/cudnn-downloads) ([CUDNN Windows tutorial](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/windows.html))

- [NVIDIA VIDEO CODEC SDK](https://developer.nvidia.com/nvidia-video-codec-sdk/download)


<br>


#### Decord using CMake

- after installing nvidia codec sdk, get [DFFMPEG](https://github.com/GyanD/codexffmpeg/releases) (`ffmpeg-{version}-full_build-shared.zip`) and extract dependencies to a folder of your choosing e.g. `C:/Users/gvese/ffmpeg` (last checked working version was `ffmpeg-4.4-full_build-shared.zip`)

> `*.lib` files need to be renamed so that they contain the lib prefix: `lib*.lib`.
> For example `avfilter.lib` becomes `libavfilter.lib` etc. 

> `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\cudart64_12.dll` needs to be copied into conda decord site packages `C:\Users\gvese\miniconda3\envs\drows_d_d\Lib\site-packages\decord\`

> Make sure to include the correct `CMAKE_CUDA_ARCHITECTURES` version(s) (see [CMAKE_CUDA_ARCHITECTURES](https://developer.nvidia.com/cuda-gpus); e.g. for RTX 3060 it's `86`, as provided below).

```
git clone --recursive https://github.com/dmlc/decord

cd decord

mkdir build

cd build

cmake -G "Visual Studio 17 2022" -A x64 `
      -D CMAKE_BUILD_TYPE=RELEASE `
      -D CMAKE_INSTALL_PREFIX="C:/Users/gvese/miniconda3/envs/drows_d_d" `
      -D CMAKE_CXX_FLAGS="/DDECORD_EXPORTS" `
      -D USE_CUDA=ON `
      -D FFMPEG_DIR="C:/Users/gvese/ffmpeg" `
      -D CMAKE_CUDA_ARCHITECTURES="86" `
      -D CMAKE_CONFIGURATION_TYPES="Release" `
      -D NV_VIDEO_CODEC_SDK_DIR="C:/Users/gvese/Video_Codec_SDK_12.2.72" `
      ..
      
cmake --build . --config Release --target INSTALL

cd ..

cd python

pip install .
```

<br>

#### Dlib CUDA using CMake

###### Miniconda terminal

```
git clone https://github.com/davisking/dlib.git

cd dlib

mkdir build

cd build

$env:DLIB_USE_CUDA = 1 
$env:DLIB_ENABLE_AVX = 1 
$env:DLIB_CUDA_COMPUTE_CAPABILITY = "86" 

pip install . --verbose --no-cache-dir

```
> [$env:DLIB_CUDA_COMPUTE_CAPABILITY](https://developer.nvidia.com/cuda-gpus)

<br>

#### OpenCV CUDA using CMake 

- [opencv](https://github.com/opencv/opencv)
- [opencv_contrib](https://github.com/opencv/opencv_contrib)
- [CMake](https://cmake.org/download/)

###### A. Miniconda terminal

- run in target environment:

> Python paths need to be modified

```
conda install cmake 

Invoke-WebRequest -Uri "https://github.com/opencv/opencv/archive/4.x.zip" -OutFile "opencv-4.x.zip"

Invoke-WebRequest -Uri "https://github.com/opencv/opencv_contrib/archive/4.x.zip" -OutFile "opencv_contrib-4.x.zip"

Expand-Archive -Path opencv-4.x.zip -DestinationPath .

Expand-Archive -Path opencv_contrib-4.x.zip -DestinationPath .

cd opencv-4.x

mkdir build

cd build
      
cmake -G "Visual Studio 17 2022" -A x64 `
      -D CMAKE_BUILD_TYPE=RELEASE `
      -D CMAKE_INSTALL_PREFIX="C:/Users/gvese/miniconda3/envs/drows_d_d" `
      -D OPENCV_EXTRA_MODULES_PATH="../../opencv_contrib-4.x/modules" `
      -D PYTHON3_EXECUTABLE="C:/Users/gvese/miniconda3/envs/drows_d_d/python.exe" `
      -D PYTHON3_LIBRARY="C:/Users/gvese/miniconda3/envs/drows_d_d/libs/python310.lib" `
      -D PYTHON3_INCLUDE_DIR="C:/Users/gvese/miniconda3/envs/drows_d_d/include" `
      -D PYTHON3_PACKAGES_PATH="C:/Users/gvese/miniconda3/envs/drows_d_d/Lib/site-packages" `
      -D WITH_CUDA=ON `
      -D CUDA_FAST_MATH=ON `
      -D BUILD_opencv_world=ON `
      -D OPENCV_DNN_CUDA=ON `
      -D BUILD_opencv_python3=ON `
      -D INSTALL_PYTHON_EXAMPLES=OFF `
      -D INSTALL_C_EXAMPLES=OFF `
      -D BUILD_EXAMPLES=OFF `
      -D OPENCV_ENABLE_NONFREE=ON `
      ..

cmake --build . --config Release --target INSTALL
```

###### B. GUI

> This approach is  not recommended, especially if using multiple miniconda environments. Prefer the above `A. Miniconda terminal` tutorial.  

- setup CMake config (e.g. [Youtube tutorial](https://www.youtube.com/watch?v=d8Jx6zO1yw0)) with config params:

```
WITH_CUDA=ON 
CUDA_FAST_MATH=ON 
BUILD_opencv_world=ON 
OPENCV_DNN_CUDA=ON 
BUILD_opencv_python3=ON 
INSTALL_PYTHON_EXAMPLES=OFF 
INSTALL_C_EXAMPLES=OFF 
BUILD_EXAMPLES=OFF 
OPENCV_ENABLE_NONFREE=ON 
```

- paths below <b>may or may not need configuring</b>, depending on how CMake detects these and whether a new or default miniconda environment is used (check nevertheless, otherwise build files won't be installed to the desired environment)
```
CMAKE_INSTALL_PREFIX="C:/Users/gvese/miniconda3/envs/drows_d_d" `
OPENCV_EXTRA_MODULES_PATH="../../opencv_contrib-4.x/modules" `
PYTHON3_EXECUTABLE="C:/Users/gvese/miniconda3/envs/drows_d_d/python.exe" `
PYTHON3_LIBRARY="C:/Users/gvese/miniconda3/envs/drows_d_d/libs/python310.lib" `
PYTHON3_INCLUDE_DIR="C:/Users/gvese/miniconda3/envs/drows_d_d/include" `
PYTHON3_PACKAGES_PATH="C:/Users/gvese/miniconda3/envs/drows_d_d/Lib/site-packages" `
```
- after configuring, generate build files and run the following command in target miniconda environment:

```
cmake --build "C:/Users/gvese/Python/opencv_GPU/build" --target INSTALL --config Release
```