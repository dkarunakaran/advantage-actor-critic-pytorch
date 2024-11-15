#!/bin/bash
# Run below step one by one in terminal
sudo docker rmi -f $(sudo docker images -f "dangling=true" -q)
docker build -t actor_critic .
docker run --net host --gpus all -it -v /home/$(whoami)/Documents/projects/advantage-actor-critic-pytorch:/app actor_critic


