FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./app.py /code/app.py
COPY ./results /code/results
COPY ./utils /code/utils



RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["streamlit", "run", "/code/app.py", "--server.address", "0.0.0.0", "--server.port", "7860"]
