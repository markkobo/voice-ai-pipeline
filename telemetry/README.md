# Telemetry Module

Monitoring and metrics collection for Voice AI Pipeline.

## Quick Start

### 1. Install Dependencies

```bash
pip install prometheus-client
```

### 2. Start Metrics Server

In your application code:

```python
from telemetry import TelemetryCollector

# Start the metrics server
collector = TelemetryCollector(port=9090)
collector.start_server()
```

Or use the decorator approach:

```python
from telemetry.collector import get_collector

collector = get_collector()
collector.start_server()
```

### 3. View Metrics

```
http://localhost:9090/metrics
```

### 4. Run Grafana Dashboard

```bash
cd telemetry
docker-compose up -d
```

Then open http://localhost:3000 (admin/admin) and import `grafana/dashboard.json`.

## Usage Examples

### Basic Latency Tracking

```python
from telemetry import metrics

# Using context manager
with metrics.asr_latency.labels(component="asr", model="qwen3-asr").time():
    result = asr.process(audio)
```

### Using Decorators

```python
from telemetry.decorators import track_asr_latency

@track_asr_latency(model="qwen3-asr")
def transcribe(audio_data):
    # Your ASR logic
    return text
```

### Streaming Rate Tracking

```python
from telemetry.collector import StreamingRateTracker
from telemetry import metrics

# Track LLM token streaming rate
with StreamingRateTracker(metrics.llm_tokens_per_second, model="llama3") as tracker:
    for token in stream_tokens():
        tracker.add(1)  # Add one token
        yield token
```

### Recording Errors

```python
from telemetry import get_collector

collector = get_collector()
collector.record_error(component="asr", error_type="timeout", model="qwen3-asr")
```

## Metrics Reference

| Metric | Type | Description |
|--------|------|-------------|
| `vad_latency_seconds` | Histogram | VAD detection latency |
| `asr_latency_seconds` | Histogram | ASR processing latency |
| `llm_ttft_seconds` | Histogram | LLM Time to First Token |
| `tts_ttfb_seconds` | Histogram | TTS Time to First Byte |
| `e2e_latency_seconds` | Histogram | End-to-end latency |
| `utterances_total` | Counter | Total utterances processed |
| `llm_tokens_total` | Counter | Total tokens generated |
| `active_sessions` | Gauge | Active sessions |
| `errors_total` | Counter | Total errors |

## Integration with Existing Code

Add to `app/main.py`:

```python
from telemetry import TelemetryCollector

collector = TelemetryCollector(enable_at_start=True)
```

Add to your ASR service (`app/services/asr_engine.py`):

```python
from telemetry import metrics

class Qwen3ASR:
    def process(self, audio):
        with metrics.asr_latency.labels(
            component="asr",
            model="qwen3-asr"
        ).time():
            result = self._process(audio)
            return result
```

## Grafana Dashboard

The dashboard includes:

- Active Sessions (gauge)
- Latency by Component (p50/p95)
- End-to-End Latency (p50/p95/p99)
- Throughput (utterances/sec, tokens/sec)
- Error Rate
- Streaming Rates (tokens/sec, bytes/sec)

Import: `grafana/dashboard.json`
