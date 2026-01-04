# Deploying AWF with Grafana

This guide provides step-by-step instructions for deploying the AWF observability stack using Grafana OSS and the LGTM stack (Loki, Grafana, Tempo, Mimir).

## Architecture Overview

AWF integrates with the Grafana ecosystem through OpenTelemetry (OTel). The orchestration engine emits events that are bridged to OTel metrics and traces, which are then collected by Grafana Alloy and routed to the appropriate backends.

```text
                                  ┌─────────────────┐
                                  │   Grafana UI    │
                                  └────────┬────────┘
                                           │
          AWF API / SDK                    ▼
    ┌─────────────────────────┐    ┌─────────────────┐
    │    Orchestration        │    │  Watcher Agent  │
    │  ┌───────────────────┐  │    │ (Remediation)   │
    │  │   EventEmitter    │  │    └────────┬────────┘
    │  └─────────┬─────────┘  │             │
    │            ▼            │             │ 1. Alert Webhook
    │  ┌───────────────────┐  │             │ 2. MCP Queries
    │  │   OTel Exporter   │  │             │
    │  └─────────┬─────────┘  │             │
    └────────────┼────────────┘             │
                 │                          │
                 │ OTLP (4317/4318)         ▼
                 │                 ┌─────────────────┐
                 └────────────────►│  Grafana Alloy  │
                                   └────────┬────────┘
                                            │
                 ┌──────────────────────────┼──────────────────────────┐
                 ▼                          ▼                          ▼
        ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
        │ Mimir (Metrics) │        │   Loki (Logs)   │        │  Tempo (Traces) │
        └─────────────────┘        └─────────────────┘        └─────────────────┘
```

## Prerequisites

- **Docker & Docker Compose**: Required for running the observability stack.
- **Python 3.10+**: Required for AWF.
- **AWF with API/OTel support**:
  ```bash
  pip install "awf[api,otel]"
  ```

## Quick Start (5 Minutes)

1. **Navigate to the Grafana directory**:
   ```bash
   cd docker/grafana
   ```

2. **Start the stack**:
   ```bash
   docker compose up -d
   ```

3. **Verify running containers**:
   ```bash
   docker compose ps
   ```
   You should see `awf-grafana`, `awf-alloy`, `awf-mimir`, `awf-loki`, and `awf-tempo` running.

4. **Access Grafana**:
   Open [http://localhost:3000](http://localhost:3000) in your browser.
   - **User**: `admin`
   - **Password**: `admin`

## AWF API Configuration

To enable telemetry export in your AWF application, you need to initialize the `OTelExporter`.

### Python SDK Setup

```python
from awf.orchestration.otel_exporter import setup_otel_exporter

# Initialize the exporter (connects to Alloy at http://localhost:4317 by default)
exporter = setup_otel_exporter(
    endpoint="http://localhost:4317",
    service_name="my-agent-service",
    deployment_environment="production"
)

# Your AWF orchestration code here...
```

### Environment Variables

Alternatively, you can configure the exporter via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `AWF_OTEL_ENDPOINT` | OTLP gRPC endpoint | `http://localhost:4317` |
| `AWF_SERVICE_NAME` | Service name for traces | `awf-orchestration` |
| `AWF_ENVIRONMENT` | Deployment environment | `development` |

## Connecting Grafana Alerts to AWF Webhooks

The **Watcher Agent** can autonomously investigate and remediate issues triggered by Grafana alerts. To enable this, configure an Alert Contact Point in Grafana.

1. In Grafana, go to **Alerting > Contact points**.
2. Click **New contact point**.
3. Set **Name** to `AWF Watcher`.
4. Set **Integration** to `Webhook`.
5. Set **URL** to `http://<awf-api-host>:8000/webhooks/grafana-alerts`.
6. Click **Save contact point**.
7. Go to **Notification policies** and route the relevant alerts to this contact point.

## Verifying the Setup

1. **Check Alloy**: Visit [http://localhost:12345](http://localhost:12345) to see the Alloy UI and verify it's receiving telemetry.
2. **Run a Workflow**: Execute an AWF workflow to generate telemetry.
3. **View Dashboards**: In Grafana, go to **Dashboards** and open the **AWF Overview** dashboard. You should see metrics and traces appearing within seconds.
4. **Test Traces**: Click on a workflow duration in the dashboard to jump directly to its distributed trace in Tempo.

## Troubleshooting

### Connectivity Issues
- **Alloy not receiving data**: Ensure your AWF application can reach the Alloy container on port 4317 (gRPC) or 4318 (HTTP). If running AWF in a separate Docker network, ensure they are connected.
- **Grafana cannot reach backends**: Check the logs of the `awf-grafana` container. Ensure the datasource UIDs in the JSON dashboards match your provisioning (`mimir`, `loki`, `tempo`).

### Missing Metrics/Traces
- **OTel dependency missing**: Ensure you installed AWF with the `otel` extra: `pip install "awf[otel]"`.
- **Exporter not started**: Verify that `setup_otel_exporter()` is called early in your application lifecycle.

## Production Considerations

- **Persistence**: The default docker-compose uses local volumes. For production, consider using S3/GCS/Azure Blob Storage for Loki, Tempo, and Mimir backends.
- **Security**:
  - Change the default Grafana admin password.
  - Enable TLS for OTLP endpoints.
  - Implement authentication for the AWF webhook endpoint.
- **Resource Limits**: Monitor the CPU and memory usage of Mimir and Tempo, as they can be resource-intensive under high load.
