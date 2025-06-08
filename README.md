Prerequisites:
1. Docker


How to run:
1. Install .zip
2.  Extract contents into a folder e.g. "my_project"
3.  Open terminal, ```cd my_project```
4.  To build kedro pipeline image: ```docker build -t <image-name> .```
5.  To run kedro pipeline: ```docker run --rm <image-name>``` (--rm argument destroys container after stopping)
