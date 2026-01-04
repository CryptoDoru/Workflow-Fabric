# Grafana Dashboards Guide

AWF provides pre-built Grafana dashboards for monitoring system health, agent performance, and resource consumption. This guide explains the available dashboards, key metrics, and how to set up SLO monitoring.

## Available Dashboards

AWF provisions two primary dashboards to Grafana by default:

1. **AWF Overview**: High-level system health, workflow statistics, and costs.
2. **Agent Fleet Health**: Detailed performance metrics per agent and framework.

### 1. AWF Overview Dashboard

The **AWF Overview** dashboard is the primary entry point for monitoring your AWF deployment.

- **System Health Row**:
  - **Total Workflows**: Aggregate count of all workflows executed.
  - **Success Rate**: Percentage of successfully completed workflows.
  - **Avg Duration**: Mean execution time for workflows.
  - **Active Workflows**: Number of workflows currently in-progress.
- **Workflow Trends**:
  - **Workflow Completions**: Time-series chart of completed vs failed workflows.
  - **Step Duration Percentiles**: Heatmap/line chart showing P50, P95, and P99 latency.
- **Resource Consumption**:
  - **Token Usage**: Stacked area chart showing input vs output tokens.
  - **Cost by Provider**: USD cost breakdown per LLM provider (OpenAI, Anthropic, etc.).
- **Error Analysis**:
  - **Retries & Fallbacks**: Bar chart showing how many steps required retries or fell back to alternative agents.
  - **Recent Errors**: A table of the latest error logs retrieved from Loki.

### 2. Agent Fleet Health Dashboard

This dashboard focuses on the performance of individual agents within your fleet.

- **Fleet Overview**:
  - **Total Agents**: Count of active agents in the registry.
  - **Fleet Success Rate**: Aggregate success rate across all agents.
  - **Fleet P99 Latency**: Tail latency for the entire fleet.
- **Per-Agent Performance**:
  - A table showing **Success Rate**, **P99 Latency**, **Request Count**, and **Trust Score** for each agent ID.
- **Framework Distribution**:
  - Pie chart showing the distribution of agents by framework (LangGraph, CrewAI, AutoGen, etc.).
- **Trust Score Distribution**:
  - Histogram showing how many agents fall into different trust/sandbox tiers.

## Key Metrics Reference

| Metric Name | Type | Description |
|-------------|------|-------------|
| `awf_workflow_total` | Counter | Total workflow completions (labeled by `workflow_id`, `status`). |
| `awf_step_duration_ms` | Histogram | Execution time for individual steps (labeled by `agent_id`, `step_id`). |
| `awf_tokens_total` | Counter | Cumulative tokens consumed (labeled by `agent_id`, `direction`). |
| `awf_cost_usd_total` | Counter | Cumulative cost in USD (labeled by `agent_id`, `provider`). |
| `awf_retry_total` | Counter | Number of retry attempts (labeled by `step_id`). |
| `awf_workflow_active` | Gauge | Number of workflows currently running. |

## SLO/SLI Monitoring

AWF includes pre-configured Service Level Objectives (SLOs) defined as Prometheus recording rules.

### Service Level Indicators (SLIs)

- **Workflow Success Rate**: `awf:workflow_success_rate:ratio_rate5m`
- **Step Latency (P99)**: `awf:step_latency_p99:5m`
- **Agent Availability**: `awf:agent_available:bool`

### Error Budgets

The system tracks error budgets for a **99.9% success rate** target:
- **Budget Remaining**: `awf:error_budget_remaining:ratio`
- **Burn Rate**: `awf:error_budget_burn_rate:ratio` (Detects if you are consuming your error budget faster than expected).

## Setting Up Alerts

AWF includes pre-defined alerting rules that trigger the **Watcher Agent**'s autonomous remediation.

| Alert Name | Condition | Severity |
|------------|-----------|----------|
| `AWFHighErrorRate` | Success Rate < 95% | Warning |
| `AWFCriticalErrorRate` | Success Rate < 90% | Critical |
| `AWFHighLatency` | P99 Latency > 30s | Warning |
| `AWFErrorBudgetExhausted` | Budget < 0 | Critical |
| `AWFAgentDegraded` | Agent Success Rate < 90% | Warning |

## Customizing Dashboards

1. **Add Custom Panels**: You can add panels to the pre-built dashboards by clicking the **Add** button in Grafana.
2. **Querying Data**: Use PromQL for metrics (`mimir`), LogQL for logs (`loki`), and TraceQL for traces (`tempo`).
3. **Variables**: Dashboards use variables like `$workflow` and `$agent` to allow filtering the entire view by a specific component.
