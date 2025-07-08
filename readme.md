To build Image:
```bash
docker build -t genesis -f docker/Dockerfile docker
```
To enter the container:

```bash
docker run --gpus all --rm -it \
-e DISPLAY=$DISPLAY \
-v /dev/dri:/dev/dri \
-v /tmp/.X11-unix/:/tmp/.X11-unix \
-v $PWD:/workspace \
genesis
```