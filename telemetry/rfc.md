# Telemetry RFC - Voice AI Pipeline Monitoring

## 1. Overview

本 RFC 定義 Voice AI Pipeline 的監控策略，支援本地開發調試與小型部署 (1-3 users)。為未來擴展到多個 local LLM instances 做準備。

## 2. Goals

- 收集關鍵 latency metrics 分析效能瓶頸
- 收集 throughput metrics 了解負載能力
- Streaming 即時 metrics 監控
- 未來可擴展到多個 LLM instances
- Minimal overhead，不影響即時語音處理

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Voice AI Pipeline                         │
│  VAD → ASR → LLM → TTS                                      │
│       │       │      │                                      │
│       └───────┴──────┴──→ Telemetry Module                  │
│                               │                              │
│                               ▼                              │
│                    Prometheus Client                        │
│                         :9090                                │
│                               │                              │
│                               ▼                              │
│                       Grafana                                │
│                    (Dashboard)                               │
└─────────────────────────────────────────────────────────────┘
```

## 4. Metrics Definition

### 4.1 Pipeline Latency Metrics

| Metric Name | Type | Description | Unit |
|-------------|------|-------------|------|
| `vad_latency_seconds` | Histogram | VAD 偵測到語音停止的時間 | seconds |
| `asr_latency_seconds` | Histogram | 音訊結束到文本產出的時間 | seconds |
| `llm_ttft_seconds` | Histogram | LLM 產出第一個 token 的時間 | seconds |
| `tts_ttfb_seconds` | Histogram | 收到文本到產出第一段音訊的時間 | seconds |
| `e2e_latency_seconds` | Histogram | 說完話到聽到回應的總時間 | seconds |

**Histograms Buckets** (for sub-second precision):
```
0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0
```

### 4.2 Throughput Metrics

| Metric Name | Type | Description |
|-------------|------|-------------|
| `audio_chunks_total` | Counter | 收到的音訊 chunks 總數 |
| `utterances_total` | Counter | 處理的utterances總數 |
| `asr_results_total` | Counter | ASR 結果總數 (partial + final) |
| `llm_tokens_total` | Counter | LLM 產出的 tokens 總數 |
| `tts_chunks_total` | Counter | TTS 產出的音訊 chunks 總數 |
| `ws_connections_total` | Counter | WebSocket 連線總數 |
| `ws_messages_total` | Counter | WebSocket 訊息總數 |

### 4.3 Streaming Metrics (Real-time)

| Metric Name | Type | Description |
|-------------|------|-------------|
| `llm_tokens_per_second` | Gauge | LLM streaming tokens/秒 |
| `tts_audio_bytes_per_second` | Gauge | TTS 產出速率 bytes/秒 |
| `streaming_chunk_latency_seconds` | Histogram | 每個 streaming chunk 的延遲 |
| `audio_buffer_size_bytes` | Gauge | 當前音訊 buffer 大小 |
| `active_sessions` | Gauge | 當前 active sessions 數量 |

### 4.4 Error Metrics

| Metric Name | Type | Description |
|-------------|------|-------------|
| `errors_total` | Counter | 各類錯誤總數 (by type, by component) |
| `asr_failures_total` | Counter | ASR 失敗次數 |
| `llm_failures_total` | Counter | LLM 失敗次數 |
| `tts_failures_total` | Counter | TTS 失敗次數 |

### 4.5 Labels Strategy

所有 metrics 支援以下 labels 以後擴展：

```python
labels = {
    "component": "vad|asr|llm|tts",      # 組件名稱
    "model": "qwen3-asr|llama3|...",    # 模型名稱
    "session_id": "xxx",                # 會話 ID
    "error_type": "timeout|invalid|...", # 錯誤類型 (optional)
}
```

## 5. Implementation

### 5.1 Dependencies

```bash
pip install prometheus-client
```

### 5.2 Directory Structure

```
telemetry/
├── rfc.md              # This file
├── __init__.py
├── metrics.py          # Prometheus metrics definitions
├── collector.py        # Metrics collector utilities
├── decorators.py       # decorators for auto-measuring
├── grafana/
│   └── dashboard.json  # Grafana dashboard JSON
└── docker-compose.yml  # Optional: local Grafana stack
```

### 5.3 Usage Example

```python
from telemetry.metrics import (
    vad_latency,
    asr_latency,
    llm_ttft,
    e2e_latency,
    utterances_total,
    errors_total,
)

# Manual measurement
with vad_latency.time():
    result = vad.detect(audio)

# Or use decorator
@vad_latency.time()
def detect_voice(audio):
    ...

# With labels
asr_latency.labels(model="qwen3-asr").observe(0.5)
```

## 6. Grafana Dashboard

### 6.1 Required Panels

1. **Latency Overview** (Latency by component)
   - Line chart: p50, p95, p99 of all latency metrics

2. **E2E Latency Distribution**
   - Heatmap: e2e_latency_seconds

3. **Throughput**
   - Counter charts: utterances/sec, tokens/sec

4. **Active Sessions**
   - Gauge: active_sessions

5. **Error Rate**
   - Rate chart: errors_total by type

6. **Streaming Quality** (for LLM/TTS)
   - Tokens/sec gauge
   - TTFT distribution

### 6.2 Dashboard Export

使用 Grafana HTTP API 或 export JSON:

```bash
# Export
curl -s "http://localhost:3000/api/dashboards/uid/voice-ai" | jq '.dashboard' > grafana/dashboard.json

# Import
curl -X POST "http://localhost:3000/api/dashboards" \
  -H "Content-Type: application/json" \
  -d '{"dashboard": $(cat grafana/dashboard.json)}'
```

## 7. Future Considerations

### 7.2 Multi-LLM Instance Support

當有多個 LLM instances 時：

```python
labels = {
    "component": "llm",
    "model": "llama3-70b",
    "instance": "gpu-server-1",  # 新增
    "region": "local",           # 新增
}
```

### 7.3 OpenTelemetry Migration Path

未來需要完整 tracing 時：

1. 添加 `opentelemetry-sdk` + `opentelemetry-exporter-otlp`
2. 替換 prometheus client 為 OTel metrics
3. 添加 trace context propagation
4. 添加 Tempo backend for traces

### 7.4 Alerting

Prometheus alerting rules:

```yaml
groups:
  - name: voice-ai
    rules:
      - alert: HighE2ELatency
        expr: histogram_quantile(0.95, rate(e2e_latency_seconds_bucket[5m])) > 2
        for: 5m

      - alert: HighErrorRate
        expr: rate(errors_total[5m]) > 0.1
```

## 8. Quick Start

```bash
# 1. Start Prometheus + Grafana
docker-compose up -d

# 2. Run your app (metrics exposed on :9090)
python -m app.main

# 3. Open Grafana
#    http://localhost:3000
#    Add Prometheus datasource: http://localhost:9090
#    Import grafana/dashboard.json
```

## 9. Success Criteria

- [ ] All latency metrics captured with proper histograms
- [ ] Streaming metrics visible in real-time
- [ ] Dashboard shows p50/p95/p99 latencies
- [ ] Error tracking by component
- [ ] < 1ms overhead per metric observation
- [ ] Support 100+ concurrent sessions
