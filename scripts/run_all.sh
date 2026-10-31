#!/usr/bin/env bash
# Digital Twin automation (Linux/macOS). Usage: ./scripts/run_all.sh [train|test|up|down|logs|replay|all]
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-all}" in
  train)  python scripts/eda.py && python src/train.py ;;
  test)   python -m pytest -v ;;
  up)     docker compose up --build -d
          echo "Streamlit http://localhost:8501 | Grafana http://localhost:3000 (admin / \$GRAFANA_ADMIN_PASSWORD)" ;;
  down)   docker compose down ;;
  logs)   docker compose logs -f --tail=50 ;;
  replay) python src/replay_publisher.py --rate 1.0 ;;
  all)    python scripts/eda.py && python src/train.py && python -m pytest -v ;;
  *)      echo "unknown task"; exit 1 ;;
esac
