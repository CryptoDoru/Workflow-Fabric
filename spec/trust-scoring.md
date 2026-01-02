# ASP Trust Scoring Algorithm

**Version:** 1.0.0-draft  
**Status:** Draft  

---

## 1. Overview

The Trust Scoring Algorithm computes a trust score (0.0-1.0) for each agent,
determining sandbox tier and permission grants. The algorithm is:

- **Deterministic**: Same inputs always produce same score
- **Transparent**: Users can see exactly why an agent has a given score
- **Configurable**: Organizations can adjust weights and thresholds

---

## 2. Score Components

Trust is computed from five weighted components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Publisher Trust | 0.25 | Is the publisher verified and reputable? |
| Audit Status | 0.25 | Has the agent been security audited? |
| Community Trust | 0.20 | What do other users think? |
| Permission Analysis | 0.15 | Does it request dangerous permissions? |
| Historical Behavior | 0.15 | How has it performed in practice? |

---

## 3. Component Calculations

### 3.1 Publisher Trust (0.0 - 1.0)

```python
def calculate_publisher_trust(manifest, registry):
    publisher = manifest.trust.publisher
    
    if publisher in registry.certified_publishers:
        base_score = 1.0
    elif publisher in registry.verified_publishers:
        base_score = 0.8
    elif manifest.trust.signature and verify_signature(manifest):
        base_score = 0.6
    else:
        base_score = 0.2
    
    # Publisher reputation modifier
    pub_stats = registry.get_publisher_stats(publisher)
    if pub_stats:
        reputation_modifier = (
            pub_stats.total_agents_published * 0.01 +
            pub_stats.avg_agent_rating * 0.1 +
            (1.0 if pub_stats.security_incidents == 0 else -0.3)
        )
        reputation_modifier = clamp(reputation_modifier, -0.2, 0.2)
    else:
        reputation_modifier = 0.0
    
    return clamp(base_score + reputation_modifier, 0.0, 1.0)
```

**Scoring Table:**

| Condition | Score |
|-----------|-------|
| Certified publisher | 1.0 |
| Verified publisher | 0.8 |
| Valid cryptographic signature | 0.6 |
| No verification | 0.2 |
| Publisher with security incident | -0.3 modifier |

### 3.2 Audit Status (0.0 - 1.0)

| Audit Status | Score | Notes |
|--------------|-------|-------|
| Certified | 1.0 | Professional audit <6 months |
| Certified (stale) | 0.8 | Professional audit >1 year |
| Verified | 0.75 | AWF team review |
| Community | 0.5 | Community review |
| Unaudited | 0.2 | No review |

### 3.3 Community Trust (0.0 - 1.0)

```python
def calculate_community_trust(manifest, registry):
    stats = registry.get_agent_stats(manifest.id)
    
    if not stats or stats.total_runs < 10:
        return 0.3  # Insufficient data
    
    # Rating component (0.0 - 0.4)
    rating_score = (stats.average_rating / 5.0) * 0.4 if stats.average_rating else 0.2
    
    # Usage component (0.0 - 0.3)
    usage_score = min(0.3, 0.05 * math.log10(max(1, stats.total_runs)))
    
    # Abuse reports component (0.0 - 0.3)
    if stats.abuse_reports == 0:
        abuse_score = 0.3
    elif stats.abuse_reports <= 2:
        abuse_score = 0.15
    elif stats.abuse_reports <= 5:
        abuse_score = 0.05
    else:
        abuse_score = 0.0
    
    return clamp(rating_score + usage_score + abuse_score, 0.0, 1.0)
```

### 3.4 Permission Analysis (0.0 - 1.0)

**Permission Penalties:**

| Permission | Penalty | Rationale |
|------------|---------|-----------|
| NETWORK_UNRESTRICTED | -0.30 | Can exfiltrate data anywhere |
| EXEC_SHELL | -0.30 | Full system command access |
| FS_READ_SYSTEM | -0.25 | Can read sensitive files |
| EXEC_SUBPROCESS | -0.20 | Can spawn processes |
| EXEC_CODE | -0.15 | Code execution risks |
| NETWORK_ALLOW_LIST | -0.10 | Limited but expanded network |
| FS_WRITE_WORKSPACE | -0.05 | Can modify files |

### 3.5 Historical Behavior (0.0 - 1.0)

```python
def calculate_history_trust(manifest, registry):
    history = registry.get_execution_history(manifest.id, days=30)
    
    if not history or history.total_runs < 10:
        return 0.5  # Neutral - insufficient data
    
    # Success rate (0.0 - 0.5)
    success_score = (history.successful_runs / history.total_runs) * 0.5
    
    # Security incidents (0.0 - 0.3)
    security_score = 0.3 if history.security_incidents == 0 else 0.0
    
    # Consistency (0.0 - 0.2)
    cv = history.latency_stddev / history.latency_mean if history.latency_mean else 1
    consistency_score = 0.2 if cv < 0.2 else (0.1 if cv < 0.5 else 0.0)
    
    return clamp(success_score + security_score + consistency_score, 0.0, 1.0)
```

---

## 4. Overall Score Calculation

```python
def calculate_trust_score(manifest, registry, weights=DEFAULT_WEIGHTS):
    components = {
        "publisher": calculate_publisher_trust(manifest, registry),
        "audit": calculate_audit_trust(manifest, registry),
        "community": calculate_community_trust(manifest, registry),
        "permissions": calculate_permission_trust(manifest),
        "history": calculate_history_trust(manifest, registry)
    }
    
    overall = sum(components[k] * weights[k] for k in weights)
    
    # Determine sandbox tier
    if overall >= 0.90:
        sandbox_tier = "wasm"
    elif overall >= 0.70:
        sandbox_tier = "gvisor"
    elif overall >= 0.40:
        sandbox_tier = "gvisor_strict"
    else:
        sandbox_tier = "blocked"
    
    return TrustScore(overall, components, sandbox_tier, warnings)
```

---

## 5. Sandbox Tier Mapping

| Trust Score | Sandbox Tier | Startup Latency | Security Level |
|-------------|--------------|-----------------|----------------|
| 0.90 - 1.00 | WASM | ~10ms | Strong isolation |
| 0.70 - 0.89 | gVisor | ~100ms | Strong isolation |
| 0.40 - 0.69 | gVisor Strict | ~150ms | Maximum isolation |
| 0.00 - 0.39 | BLOCKED | N/A | Will not execute |

**gVisor Strict** additional restrictions:
- No network access except explicit allow-list
- Read-only filesystem
- Reduced memory limit (256MB max)
- Reduced timeout (30s max)

---

## 6. Organization Policy Overrides

```yaml
trust_policy:
  publisher_overrides:
    "acme-corp": 1.0
    "untrusted-vendor": 0.0
  
  weights:
    publisher: 0.30
    audit: 0.30
    community: 0.15
    permissions: 0.15
    history: 0.10
  
  thresholds:
    wasm: 0.95
    gvisor: 0.80
    gvisor_strict: 0.50
    blocked: 0.50
  
  blocked_permissions:
    - EXEC_SHELL
    - NETWORK_UNRESTRICTED
```

---

## 7. Trust Score Caching

Trust scores are computed on:
- Agent registration
- Agent version update
- Daily recalculation
- On-demand with `?refresh=true`
