FROM python:3.9

RUN pip install --upgrade pip
RUN pip install -U setuptools

WORKDIR /orbit
COPY . .

RUN python -m pip install --no-cache-dir -r requirements.txt
RUN python -m pip install --no-cache-dir -r requirements-test.txt

RUN python -m pip install -e . -v

