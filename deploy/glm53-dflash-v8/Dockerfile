ARG BASE_IMAGE=i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:v0.5.19-latest-0bcd822377da
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.source="https://github.com/leoong55/sglang-lasolovev"
USER root
COPY . /opt/glm53-cp8-dcp4-v1/
RUN python3 /opt/glm53-cp8-dcp4-v1/install.py \
    && python3 /opt/glm53-cp8-dcp4-v1/install.py --verify-only
ENTRYPOINT ["python3", "/opt/glm53-cp8-dcp4-v1/launch.py"]
CMD []
