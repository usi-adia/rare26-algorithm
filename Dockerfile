FROM --platform=linux/amd64 pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
ENV PYTHONUNBUFFERED=1
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user
WORKDIR /opt/app
COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --user --no-cache-dir --no-color --requirement /opt/app/requirements.txt
# resources/ holds the trained fold checkpoints + ensemble.json (written by train/train.py)
COPY --chown=user:user resources /opt/app/resources
COPY --chown=user:user inference.py /opt/app/
ENTRYPOINT ["python", "inference.py"]
