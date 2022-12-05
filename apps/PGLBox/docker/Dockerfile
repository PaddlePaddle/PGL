ARG BASE_BUILD_IMAGE=registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda11.0-cudnn8

FROM ${BASE_BUILD_IMAGE}
USER root

ENV HOME_WORK_DIR='/root/pglbox'
ENV PYTHON_HOME='/opt/_internal/cpython-3.7.0'
ENV PATH=${PYTHON_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${PYTHON_HOME}/lib:${LD_LIBRARY_PATH}

WORKDIR ${HOME_WORK_DIR}

# copy conf
COPY pip.conf ${HOME_WORK_DIR}/
RUN mv ${HOME_WORK_DIR}/etc/pip.conf

ARG PADDLE_WHL=paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
COPY ${PADDLE_WHL} libjemalloc.so ${HOME_WORK_DIR}/
RUN cd ${HOME_WORK_DIR} && \
    mkdir -p ${HOME_WORK_DIR}/dependency && \
    mv ${HOME_WORK_DIR}/libjemalloc.so ${HOME_WORK_DIR}/dependency && \
    pip3 install paddlenlp==2.0.0rc16 && \
    pip3 install ./${PADDLE_WHL} -U --force-reinstall && \
    pip3 install protobuf==3.20.0 -U && \
    pip3 install pgl -U && \
    pip3 install gpustat==1.0.0 -U && \
    rm -rf ${PADDLE_WHL}

# clean
RUN rm -rf ~/.cache/pip && yum clean all && \
    rm -rf /opt/_internal/cpython-2.7.15-ucs2 && \
    rm -rf /opt/_internal/cpython-3.5.1 && \
    rm -rf /opt/_internal/cpython-3.6.0 && \
    rm -rf /opt/_internal/cpython-3.8.0 && \
    rm -rf /gcc-8.2.0.tar.xz  && \
    rm -rf /binutils-2.27.tar.gz

