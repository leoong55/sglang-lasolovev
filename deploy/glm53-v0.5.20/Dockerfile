ARG BASE_IMAGE=lmsysorg/sglang@sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.source="https://github.com/leoong55/sglang-lasolovev"
USER root
COPY . /opt/glm53-v0.5.20/
RUN python3 /opt/glm53-v0.5.20/install.py \
    && python3 /opt/glm53-v0.5.20/install.py --verify-only \
    && python3 /opt/glm53-v0.5.20/check_humming_dependency.py \
    && ln -s /opt/glm53-v0.5.20 /opt/glm53-cp8-dcp4-v1
ENTRYPOINT ["python3", "/opt/glm53-v0.5.20/launch.py"]
CMD []
