# sudo docker run --gpus all -it -v "$PWD":/home/dale/DRL_Control --rm tensorflow/tensorflow:latest-gpu sh /home/dale/DRL_Control/dockerScript.sh


cd /home/dale/DRL_Control
pip3 install --upgrade pip
pip3 install gym
pip3 install matplotlib
pip3 install keras
pip3 install tensorflow_probability
pip3 install tensorflow_addons
pip3 install sklearn
g++ -fPIC -Wall -shared -o Simulation.so ./CPP/*.cpp -std=c++11 -O30
python3 ./DDPG.py
