ARG BASE_BUILD_IMAGE=registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda11.0-cudnn8

FROM ${BASE_BUILD_IMAGE}
USER root

ENV HOME_WORK_DIR='/pglbox_dependency'
ENV PYTHON_HOME='/opt/_internal/cpython-3.7.0'
ENV PATH=${PYTHON_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${PYTHON_HOME}/lib:${LD_LIBRARY_PATH}

WORKDIR /

# copy conf
ARG PIP_CONF=pip.conf
COPY ${PIP_CONF} ${HOME_WORK_DIR}/
RUN mv ${HOME_WORK_DIR}/${PIP_CONF} /etc/pip.conf

# download dependency
ARG PADDLE_WHL=paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
ARG JEMALLOC_SO=libjemalloc.so
ARG PADDLE_WHL_SITE=https://baidu-pgl.gz.bcebos.com/pglbox/dependency/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl.3966e89/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl.3966e89.1
ARG JEMALLOC_SO_SITE=https://baidu-pgl.gz.bcebos.com/pglbox/dependency/libjemalloc.so/libjemalloc.so.3
RUN cd ${HOME_WORK_DIR} && \
    wget -q -O ${PADDLE_WHL} ${PADDLE_WHL_SITE} && \
    wget -q -O ${JEMALLOC_SO} ${JEMALLOC_SO_SITE} && \
    pip3 install paddlenlp==2.0.0rc16 && \
    pip3 install ./${PADDLE_WHL} -U --force-reinstall && \
    pip3 install protobuf==3.20.0 -U && \
    pip3 install pgl -U && \
    pip3 install gpustat==1.0.0 -U && \
    pip3 install requests==2.28.1 -U && \
    pip3 install joblib==1.2.0 -U && \
    rm -rf ${PADDLE_WHL}


# clean
RUN rm -rf ~/.cache/pip && yum clean all && \
    rm -rf /opt/_internal/cpython-2.7.15-ucs2 && \
    rm -rf /opt/_internal/cpython-3.5.1 && \
    rm -rf /opt/_internal/cpython-3.6.0 && \
    rm -rf /opt/_internal/cpython-3.8.0 && \
    rm -rf /gcc-8.2.0.tar.xz  && \
    rm -rf /binutils-2.27.tar.gz
