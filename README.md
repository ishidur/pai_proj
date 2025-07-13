# Physical AI講座 最終課題プロジェクト

Genesis で遊ぶ

## WSLで環境構築

ubuntu: 22.04  
CUDA toolkit: 12.8  

genesisのビルドにPython.hが必要になるので、`python3-dev`をインストールしておく
```bash
sudo apt install python3-dev
```

## Docker

genesis公式リポジトリより拝借

To build Image:
```bash
docker build -t genesis -f docker/Dockerfile docker
```
To enter the container:

```bash
xhost +local:root # Allow the container to access the display

docker run --gpus all --rm -it \
-e DISPLAY=$DISPLAY \
-v /dev/dri:/dev/dri \
-v /tmp/.X11-unix/:/tmp/.X11-unix \
-v $PWD:/workspace \
genesis
```