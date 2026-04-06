// PM2 ecosystem config for Fathom Scan API v2
// Usage: pm2 start ecosystem.config.js
module.exports = {
  apps: [
    {
      name: "fathom-api",
      script: "uvicorn",
      args: "fathom_api_v2:app --host 0.0.0.0 --port 8500 --workers 1",
      cwd: "C:\\Users\\heyzo\\clawd\\sae-reasoning-depth",
      interpreter: "none",
      env: {
        FATHOM_MODEL: "google/gemma-2-2b-it",
        PYTHONIOENCODING: "utf-8",
        CUDA_VISIBLE_DEVICES: "0",
      },
      max_memory_restart: "8G",
      error_file: "C:\\Users\\heyzo\\clawd\\sae-reasoning-depth\\logs\\fathom-api-error.log",
      out_file: "C:\\Users\\heyzo\\clawd\\sae-reasoning-depth\\logs\\fathom-api-out.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      restart_delay: 10000,
      max_restarts: 3,
      autorestart: true,
    },
  ],
};
