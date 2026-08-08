# Findings — the anomaly log

Anomalies are the product. Every unexpected meter reading lands here with the config that produced it and whether it was chased. Negative results are written plainly, not dressed up.

Entry format:

```
## F-NNNN — <one-line description>
- Date:
- Instrument / experiment:
- Config: <path to resolved config or gate artifact>
- Expected / observed:
- Chase status: chasing | parked (why) | resolved (what it was)
- Resolution / follow-up:
```

*(No findings yet — instruments arrive in Stage 1.)*
