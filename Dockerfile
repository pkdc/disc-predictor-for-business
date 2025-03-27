FROM python:3.12-slim

WORKDIR /app

COPY ./requirements.txt /tmp/requirements.txt

RUN python -m venv /py && \
    /py/bin/pip install --upgrade pip && \
    /py/bin/pip install -r /tmp/requirements.txt && \
    rm -rf /tmp/requirements.txt

COPY . /app

ENV PATH="/py/bin:$PATH"

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
