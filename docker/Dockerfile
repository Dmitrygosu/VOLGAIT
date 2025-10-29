FROM golang:1.21-bullseye AS builder

RUN apt-get update && apt-get install -y \
    libopencv-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-rus \
    pkg-config \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=1 go build -o grz-recognition main.go

FROM debian:bullseye-slim

RUN apt-get update && apt-get install -y \
    libopencv-core4.5 \
    libopencv-imgproc4.5 \
    libopencv-imgcodecs4.5 \
    libopencv-videoio4.5 \
    libopencv-dnn4.5 \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/grz-recognition .
COPY --from=builder /app/models ./models

ENTRYPOINT ["./grz-recognition"]

