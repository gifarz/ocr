// PM2 process definition for running this service alongside CompliFi's
// existing PM2-managed Node/Express processes (see app/main.py's module
// docstring - this is the "production" deployment path it already
// references; the one-off `pm2 start "uvicorn ..." --name complifi-ocr`
// command there does the same thing ad hoc, without a saveable config).
//
// Usage, from the repo root:
//   pm2 start deploy/pm2/ecosystem.config.js
//   pm2 save            # persist across a `pm2 resurrect` / reboot
//
// Requires the venv to already exist (see "Local setup"/scripts/run.sh)
// and .env to be filled in - this does not create either for you, unlike
// scripts/run.sh, since a production host generally shouldn't have
// requirements.txt silently re-installed by a process manager.
module.exports = {
  apps: [
    {
      name: "complifi-ocr",
      // Directly invoke the venv's own uvicorn rather than relying on
      // an activated shell - PM2 doesn't run this inside a login shell,
      // so a bare `uvicorn` on PATH can't be assumed.
      script: ".venv/bin/uvicorn",
      args: "app.main:app --host 0.0.0.0 --port 8088",
      cwd: __dirname + "/../..",
      interpreter: "none",
      env: {
        // PM2's own `env` here is a supplement to, not a replacement
        // for, a real .env file - app/config.py's load_dotenv() reads
        // .env directly, so secrets (OCR_SERVICE_API_KEY) belong there,
        // not hardcoded in this checked-in file.
        PYTHONUNBUFFERED: "1",
      },
      autorestart: true,
      max_restarts: 10,
      restart_delay: 3000,
    },
  ],
};
