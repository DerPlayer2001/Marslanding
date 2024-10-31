FROM python:latest as builder
WORKDIR /app
RUN apt update
RUN apt install python3-pygame ffmpeg -y
RUN pip3 install pygbag numpy --upgrade
COPY . .
RUN pygbag --build /app

FROM nginx:alpine as runner
COPY --from=builder /app/build/web /usr/share/nginx/html
