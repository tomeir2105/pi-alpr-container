# Standalone fast-alpr deploy

This folder is a standalone `fast-alpr` deployment for a second Raspberry Pi.

## Files

- `Dockerfile`: image for the local `fast-alpr` HTTP service
- `docker-compose.yml`: standalone compose stack for this service only
- `.env.example`: deploy-time variables for the second Pi
- `app.py`: small HTTP wrapper around `fast-alpr`
- `requirements.txt`: Python dependencies

## Deploy on the second Pi

```bash
cd deploy/fast-alpr
cp .env.example .env
docker compose up -d --build
```

The service listens on:

- `http://<second-pi-ip>:8090/health`
- `http://<second-pi-ip>:8090/recognize`

## Watcher configuration on the main Pi

Set `FAST_ALPR_URL` in the watcher `.env` to the second Pi IP, for example:

```dotenv
FAST_ALPR_URL=http://192.168.1.50:8090
```

Then recreate the watcher container from the repo root:

```bash
docker compose up -d --build watcher
```
