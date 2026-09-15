ARG BASE_IMAGE=lmsysorg/sglang:v0.5.19-cu130@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9
FROM ${BASE_IMAGE}
USER root
ARG SOURCE_COMMIT=unknown
LABEL org.opencontainers.image.source="https://github.com/leoong55/sglang-lasolovev" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.opencontainers.image.title="GLM53 H200 A/B fork of v9.14"
COPY . /opt/glm53-h200-ab/
RUN python3 /opt/glm53-h200-ab/install.py \
    && python3 /opt/glm53-h200-ab/install.py --verify-only \
    && python3 /opt/glm53-h200-ab/check_humming_dependency.py \
    && GLM53_CANCEL_TEST_INSTALLED=1 python3 -m unittest discover \
       -s /opt/glm53-h200-ab/tests -p 'test_host_cancel*.py' -v \
    && python3 /opt/glm53-h200-ab/h200/check_cli.py \
    && H200_INSTALLED=1 python3 -m unittest discover \
       -s /opt/glm53-h200-ab/h200/tests -p 'test_runtime.py' -v \
    && H200_INSTALLED=1 python3 -m unittest discover \
       -s /opt/glm53-h200-ab/h200/tests -p 'test_tilelang_cpu.py' -v
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
ENTRYPOINT ["python3", "/opt/glm53-h200-ab/h200/launch.py"]
CMD []
