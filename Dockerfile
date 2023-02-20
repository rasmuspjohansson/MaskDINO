FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
# updated to 20.04 : https://github.com/facebookresearch/detectron2/issues/4335
# use an older system (18.04) to avoid opencv incompatibility (issue#3524)

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
        python3-opencv ca-certificates python3-dev git wget sudo ninja-build
RUN ln -sv /usr/bin/python3 /usr/bin/python




#in order to make things simple(compativle wiht the mask-dino instalation) I install everything as root in home/nouser folder
RUN mkdir /home/nouser
WORKDIR /home/nouser

ENV PATH="/home/nouser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py && \
        python3 get-pip.py && \
        rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install tensorboard cmake onnx   # cmake from apt-get is too old
RUN pip install torch==1.10 torchvision==0.11.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN pip install 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
#RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
RUN apt install nano
RUN git clone https://github.com/rasmuspjohansson/detectron2 detectron2_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN pip install -e detectron2_repo

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
WORKDIR /home/nouser/detectron2_repo


#FROM detectron2:nouser


#based on  https://github.com/IDEA-Research/MaskDINO.git
#includes some minor bug fixes to allow the code to run on a more recent versions of numpy and pytorch
WORKDIR "/home/nouser"
RUN git clone https://github.com/rasmuspjohansson/MaskDINO.git
RUN pwd
WORKDIR "/home/nouser/MaskDINO"
RUN pip install -r requirements.txt
WORKDIR "/home/nouser/MaskDINO/maskdino/modeling/pixel_decoder/ops"
#USER root
#RUN mkdir /usr/lib/python3.8/site-packages/
#RUN chmod a+r -R /usr/lib/python3.8/site-packages/
#USER 1000
RUN sh make.sh
RUN pip3 install --editable .
WORKDIR "/home/nouser/MaskDINO/maskdino/modeling/pixel_decoder/ops"
ENV PYTHONPATH "${PYTHONPATH}:/home/nouser/detectron2_repo"
ENV PYTHONPATH "${PYTHONPATH}:/home/nouser/MaskDINO/maskdino/modeling/pixel_decoder/ops"
RUN pip install git+https://github.com/mcordts/cityscapesScripts.git
RUN apt install nano
RUN pip install git+https://github.com/cocodataset/panopticapi.git
WORKDIR /home/nouser/MaskDINO
#RUN python datasets/prepare_ade20k_sem_seg.py
RUN pip install setuptools==59.5.0
#setting up the ADE dataset for training
WORKDIR /home/nouser/MaskDINO/datasets/ADEChallengeData2016
RUN wget http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar
RUN tar -xf annotations_instance.tar
WORKDIR /home/nouser/MaskDINO/
#RUN python datasets/prepare_ade20k_pan_seg.py
#RUN python datasets/prepare_ade20k_ins_seg.py
#replaced np.int with int (recomended by pytorch as they are the same and np.int is depricated)
#replaced np.float with float for same reason (TODO:automate this with sed or fix in my own branch (pull request? together with updated Dockerfile?))
