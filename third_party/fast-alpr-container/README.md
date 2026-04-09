# fast-alpr container

This folder contains a small HTTP wrapper around `fast-alpr`.

## Build

```bash
docker build -t fast-alpr-local ./third_party/fast-alpr-container
```

## Run

```bash
docker run --rm -p 8090:8090 fast-alpr-local
```

## API

- `GET /health`
- `POST /recognize` with multipart field `image`
