FROM bryankp/pycuda:latest
ENV AM_I_IN_A_DOCKER_CONTAINER Yes


RUN apt-get update && apt-get install -y curl

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda



RUN conda install pytorch=1.12.0 torchvision -c pytorch

RUN pip3 install pycuda pydicom scikit-image

#RUN mkdir /opt/configFiles
#RUN mkdir /opt/data
##RUN mkdir /opt/data/inputCTs
##RUN mkdir /opt/data/inferencedCTs
##RUN mkdir /opt/data/output
##RUN mkdir /opt/data/matrices
#
#
#COPY ./src_/model_scatter.pth ./model_scatter.pth
#COPY ./src_/model_segmentation.pth.tar ./model_segmentation.pth.tar
#COPY ./src_/ /opt/source-code
#
#CMD python3 /opt/source-code/example_projectorModified.py /opt/configFiles/inference.txt
