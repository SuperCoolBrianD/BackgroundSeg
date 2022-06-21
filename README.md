Code for background segmentation

python==3.8
cuda==10.2

using conda to install pytorch or whatever you like, just make sure that pytorch is installed with cuda

conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch

pip install -r requirement.txt

first get a video, running the notebook background_reconstruction will create an image of the background

The background_reconstruction_gmm does background subtraction and foreground segmentation using gmm and the constructed background