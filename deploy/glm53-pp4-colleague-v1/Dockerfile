# The installer verifies the exact upstream source, even if this tag is moved.
ARG BASE_IMAGE=lmsysorg/sglang:v0.5.19-latest-0bcd822377da
FROM ${BASE_IMAGE}
USER root
ARG SOURCE_COMMIT=unknown
LABEL org.opencontainers.image.source="https://github.com/leoong55/sglang-lasolovev" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.opencontainers.image.title="GLM-5.3 H200 TP2 PP4 colleague experiment"
COPY deploy/glm53-pp4-colleague-v1/ /opt/glm53-pp4-colleague-v1/
RUN python3 /opt/glm53-pp4-colleague-v1/install.py && \
    python3 /opt/glm53-pp4-colleague-v1/install.py --verify-only && \
    python3 /opt/glm53-pp4-colleague-v1/launch.py --print-command
ENV SGLANG_PP_FULL_NEED=1 \
    SGLANG_PP_SHORT_BYPASS_TOKENS=512 \
    SGLANG_PP_LAYER_PARTITION=21,20,20,17 \
    SGLANG_DSA_FUSE_TOPK=0 \
    PYTHONUNBUFFERED=1
EXPOSE 8080
ENTRYPOINT ["python3", "/opt/glm53-pp4-colleague-v1/launch.py"]
