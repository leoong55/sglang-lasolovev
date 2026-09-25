ARG BASE_IMAGE=i501-harbor-infra.ai.turbocloud.ru/images/lmsysorg/sglang@sha256:cbe5d638e2c95cefb49a6c5cd3c562a247fae67f731b877f5a824fa604179155
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.source="https://github.com/leoong55/sglang-lasolovev"
USER root
COPY . /opt/glm53-cp8-dcp4-v1/
RUN python3 /opt/glm53-cp8-dcp4-v1/install.py \
    && python3 /opt/glm53-cp8-dcp4-v1/install.py --verify-only \
    && python3 /opt/glm53-cp8-dcp4-v1/check_humming_dependency.py \
    && GLM53_CANCEL_TEST_INSTALLED=1 python3 -m unittest discover \
       -s /opt/glm53-cp8-dcp4-v1/tests -p 'test_host_cancel*.py' -v
ENTRYPOINT ["python3", "/opt/glm53-cp8-dcp4-v1/launch.py"]
CMD []
