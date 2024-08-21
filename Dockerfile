FROM visualbehaviorofficial/aloception-oss:cuda-11.3-pytorch1.13.1-lightning1.9.3
RUN apt-get update
RUN apt-get install -y libglib2.0-0 python3-tk libqt5gui5

COPY requirements.txt /install/requirements.txt
RUN pip install -r /install/requirements.txt

ENV PYTHONPATH=/home/aloception/nndepth:/home/aloception/nndepth/aloception-oss
