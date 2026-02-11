xhost +local:root  # X11 쓸 때만

docker run -it --rm \
  --net=host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $HOME:/host_home \
  osrf/ros:noetic-desktop-full
