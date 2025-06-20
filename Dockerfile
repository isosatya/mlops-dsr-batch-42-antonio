# python version for the docker image
FROM python:3.12-slim

# set the working directory in the container
WORKDIR /app

# copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# install the requirements.txt file
RUN pip install -r  /code/requirements.txt

# copy the content of the current directory to /code in the container
COPY ./app /code/app

# set the environment variables for wandb
ENV WANDB_API_KEY=""
ENV MODEL_PATH=""
# ENV PORT=8080

EXPOSE 8080

# fastapi run app/main.py --port 9090 --reload
# CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
