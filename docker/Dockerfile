FROM centos:7
WORKDIR /falling
COPY . .
RUN yum install -y wget \
    && wget -P /falling https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh --no-check-certificate \
    && bash Miniconda3-py39_4.12.0-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py39_4.12.0-Linux-x86_64.sh
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch \
    && conda config --set show_channel_urls yes\
    && conda create --name falling-pytorch python=3.9 \
    && source activate falling-pytorch \
    && conda install pytorch torchvision torchaudio cpuonly -c pytorch \
    && conda install matplotlib