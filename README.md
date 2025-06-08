Prerequisites:
1. Docker

Functionalities:
1. ```uv``` over ```pip``` because it is more modern, hundreds of times faster (built on Rust instead of Python)
2. Deprecated use of Git LFS due to insufficient bandwidth, ```wget``` to retrieve data from GitHub instead (refer to Dockerfile)

How to run:
1. Install .zip
2.  Extract contents into a folder e.g. "my_project"
3.  Open terminal, ```cd my_project```

Kedro pipelines:
1.  To build kedro pipeline image: ```docker build -t <image-name> .```
2.  To run kedro pipeline: ```docker run --rm <image-name>``` (--rm argument destroys container after stopping)

Jupyter notebook server: (Download from Brightspace submissions, unavailable on GitHub last updated 8 June 22:40)
1. ```cd notebooks```
2. ```docker build -t jupyter .```
3. ```docker run -p 8888:8888 -v $(pwd):/notebook jupyter```
