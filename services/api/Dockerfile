FROM python:3.12-slim
WORKDIR /srv
COPY pyproject.toml uv.lock ./
COPY packages ./packages
COPY services/api ./services/api
COPY config ./config
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev --package sage-api
# Warm the JIT with a tiny game so the first request is not 20s.
ENV SAGE_API_MAX_ACTIONS_PER_PLAYER=12
EXPOSE 8000
CMD ["sh", "-c", ".venv/bin/python -c 'import strataq, jax.numpy as jnp; strataq.logit_qre(strataq.DenseTensorGame((jnp.eye(2), jnp.eye(2))), 1.0)' && .venv/bin/uvicorn sage_api.main:app --host 0.0.0.0 --port 8000"]
