# Dockerfile
FROM debian:stable-slim
RUN apt-get -y update
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

# Rally part.
WORKDIR /app
RUN apt-get update && apt-get install -y git && apt-get clean
RUN git clone https://gitlab.hevs.ch/louis.lettry/rallyrobopilot.git .
RUN git checkout cedric

RUN set -xe \
    && apt-get update \
    && apt-get install -y python3-pip  \
    && apt-get install -y python3-venv

# Set env path. 
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
RUN pip install python3_xlib

CMD ["python", "main.py"]
