ARG BASE_IMAGE=i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:glm53-hicache-v9.10-0bcd822377da
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.source="https://github.com/leoong55/sglang-lasolovev"
USER root
COPY . /opt/glm53-cp8-dcp4-v1/
RUN python3 /opt/glm53-cp8-dcp4-v1/install.py \
    && python3 /opt/glm53-cp8-dcp4-v1/install.py --verify-only \
    && python3 /opt/glm53-cp8-dcp4-v1/check_humming_dependency.py
ENTRYPOINT ["python3", "/opt/glm53-cp8-dcp4-v1/launch.py"]
CMD []
