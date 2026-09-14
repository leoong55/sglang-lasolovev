ARG BASE_IMAGE=lmsysorg/sglang:v0.5.19-cu130@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9
FROM ${BASE_IMAGE}
USER root
ARG SOURCE_COMMIT=unknown
LABEL org.opencontainers.image.source="https://github.com/leoong55/sglang-lasolovev" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.opencontainers.image.title="GLM-5.3 TP4 PP2 EP4 Humming and native CUDA graphs" \
      ai.sglang.upstream.revision="0bcd822377da7b5718e674eaf9c870d349424dd1"
COPY deploy/glm53-pp2-humming-v1/ /opt/glm53-pp2-humming-v1/
COPY python/sglang/srt/layers/quantization/w4afp8.py python/sglang/srt/layers/quantization/w4afp8_humming.py /opt/glm53-pp2-humming-v1/runtime/srt/layers/quantization/
COPY python/sglang/srt/layers/moe/moe_runner/humming.py /opt/glm53-pp2-humming-v1/runtime/srt/layers/moe/moe_runner/humming.py
RUN python3 /opt/glm53-pp2-humming-v1/install.py \
    && python3 /opt/glm53-pp2-humming-v1/install.py --verify-only \
    && python3 /opt/glm53-pp2-humming-v1/check_humming_dependency.py \
    && python3 /opt/glm53-pp2-humming-v1/tests/check_humming_cpu.py -v
RUN SOURCE_COMMIT="${SOURCE_COMMIT}" python3 /opt/glm53-pp2-humming-v1/record_revision.py
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
ENTRYPOINT ["python3", "/opt/glm53-pp2-humming-v1/launch.py"]
CMD []
