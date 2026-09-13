ARG BASE_IMAGE=lmsysorg/sglang:v0.5.19-cu130@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9
FROM ${BASE_IMAGE}
USER root
ARG SOURCE_COMMIT=unknown
LABEL org.opencontainers.image.source="https://github.com/leoong55/sglang-lasolovev" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.opencontainers.image.title="GLM-5.3 STG PP2 and DP-attention experiments" \
      ai.sglang.upstream.revision="0bcd822377da7b5718e674eaf9c870d349424dd1"
COPY deploy/glm53-stg-topology/ /opt/glm53-stg-topology/
RUN printf '%s\n' "${SOURCE_COMMIT}" > /opt/glm53-stg-topology/image-revision && \
    python3 /opt/glm53-stg-topology/install.py --verify-only && \
    python3 /opt/glm53-stg-topology/launch.py --profile pp2 --model-path /models/GLM-5.3-W4AFP8 --print-command
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
ENTRYPOINT ["python3", "/opt/glm53-stg-topology/launch.py"]
