#!/usr/bin/env bash
set -e

echo "🟡 Waiting for Postgres to be ready..."
until pg_isready -h postgres -p 5432 -U admin; do
  echo "postgres:5432 - no response"
  sleep 2
done
echo "🟢 Postgres is ready!"

echo "🌱 Running DB seed script..."
python3 /app/seed_data.py || echo "⚠️ Seeding failed or already seeded."

echo "🚀 Starting DB API..."
exec uvicorn app:app --host 0.0.0.0 --port 9200 --reload
