FROM python:3.9

RUN pip install --upgrade pip

RUN pip install -U setuptools

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install -e . -v
