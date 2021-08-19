USER_GID=$(id -g)
USER_UID=$(id -u)
USERNAME=nvidian

cat > Dockerfile <<EOF
FROM nvidia/cuda:11.1-devel-ubuntu20.04

RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y wget python3-dev python3-pip build-essential cmake git
# RUN mkdir /workspace
# Create the user
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    apt-get update && \
    apt-get install -y sudo && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME
USER $USERNAME
WORKDIR /home/$USERNAME
ENV PATH="/home/$USERNAME/miniconda3/bin:\${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh && \
    conda init && \
    pip config set global.cache-dir false

RUN conda install -y -c rapidsai -c nvidia -c conda-forge \
    rapids-blazing=21.06 cudatoolkit=11.0

WORKDIR /workspace
ENV PATH="\$PATH:/home/$USERNAME/.local/bin"
ENV CUDA_HOME="/usr/local/cuda"
RUN pip3 install pudb ipython autopep8 yapf bandit flake8 mypy pycodestyle pydocstyle pylint pytest scipy
EOF

docker build -f Dockerfile -t custat .
