FROM python:alpine as runner
WORKDIR /flask
RUN pip3 install flask flask-cors psycopg2-binary gunicorn --upgrade
COPY . .
EXPOSE 80
CMD ["gunicorn", "--bind", "0.0.0.0:80", "app:app"]