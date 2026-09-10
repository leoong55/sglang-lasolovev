ARG BASE_IMAGE=i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang:v0.5.19-latest-0bcd822377da
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="SGLang GLM-5.3 CP8 DCP4 prefill-only DeepEP v4" \
      org.opencontainers.image.source="https://github.com/leoong55/sglang-lasolovev" \
      io.sglang.base.commit="0bcd822377da7b5718e674eaf9c870d349424dd1" \
      io.sglang.backport.pr="36990" \
      io.sglang.backport.head="c6aeb8b9d9128b816777e2b64cbf6603344e8fe1"
# Retain the v1 launch path so existing Deployment.command stays valid.
USER root
ENV SGLANG_GLM53_DEEPEP_PREFILL=1
COPY . /opt/glm53-cp8-dcp4-v1/
RUN python3 /opt/glm53-cp8-dcp4-v1/install.py \
    && python3 /opt/glm53-cp8-dcp4-v1/install.py --verify-only \
    && python3 /opt/glm53-cp8-dcp4-v1/check_dependencies.py
ENTRYPOINT ["python3", "/opt/glm53-cp8-dcp4-v1/launch.py"]
CMD []
