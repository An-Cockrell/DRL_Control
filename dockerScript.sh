# sudo docker run --gpus all -it -v "$PWD":/home/dale/DRL_Control --rm tensorflow/tensorflow:latest-gpu sh /home/dale/DRL_Control/dockerScript.sh

FROM tensorflow/tensorflow:latest-gpu

# Dependencies for glvnd and X11.
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libxext6 \
    libx11-6 \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    freeglut3-dev \
  && rm -rf /var/lib/apt/lists/*
# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

cd /home/dale/DRL_Control
pip3 install --upgrade pip
pip3 install gym
pip3 install matplotlib
pip3 install keras
pip3 install tensorflow_probability
pip3 install tensorflow_addons
pip3 install sklearn
pip3 install psutil
g++ -fPIC -Wall -shared -o Simulation.so ./CPP/*.cpp -std=c++11 -O30
python3 ./DDPG.py
