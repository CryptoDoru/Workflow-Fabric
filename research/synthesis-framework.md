# AI Workflow Fabric - Research Synthesis Framework

## Overview

This document provides a structured approach to analyzing user research data from interviews. Use this framework to identify patterns, validate personas, and prioritize features.

---

## Data Collection Protocol

### During Interview

1. **Record** (with consent): Audio/video for later reference
2. **Live Notes**: Capture key quotes, observations, emotional reactions
3. **Timestamps**: Mark significant moments for easy retrieval

### Immediately After Interview (Within 30 min)

1. Complete the **Post-Interview Debrief Template** (in interview-guide.md)
2. Rate confidence in key findings (high/medium/low)
3. Note questions for follow-up or next interview

### Within 24 Hours

1. Review recording, add missed insights
2. Extract verbatim quotes for key themes
3. Tag insights by category (see taxonomy below)

---

## Insight Taxonomy

Tag every insight with one primary category and optional secondary categories:

### Pain Point Categories

| Code | Category | Description |
|------|----------|-------------|
| `PP-INT` | Interoperability | Framework compatibility, cross-agent communication |
| `PP-OBS` | Observability | Debugging, tracing, logging, visibility |
| `PP-SEC` | Security | Permissions, sandboxing, access control |
| `PP-REL` | Reliability | Errors, failures, unexpected behavior |
| `PP-DX` | Developer Experience | Onboarding, documentation, APIs |
| `PP-OPS` | Operations | Deployment, scaling, monitoring |
| `PP-GOV` | Governance | Compliance, audit, approval processes |
| `PP-COST` | Cost | Pricing, budget constraints, ROI |

### Need Categories

| Code | Category | Description |
|------|----------|-------------|
| `ND-DISC` | Discovery | Finding agents, understanding capabilities |
| `ND-ORCH` | Orchestration | Coordinating multiple agents |
| `ND-TRUST` | Trust | Evaluating agent safety, reliability |
| `ND-CTRL` | Control | Policies, limits, guardrails |
| `ND-VIS` | Visibility | Understanding what agents are doing |
| `ND-PORT` | Portability | Moving between frameworks, avoiding lock-in |

### Behavior Categories

| Code | Category | Description |
|------|----------|-------------|
| `BH-EVAL` | Evaluation | How they assess new tools |
| `BH-ADOPT` | Adoption | How they introduce new tools to team |
| `BH-DEBUG` | Debugging | How they troubleshoot issues |
| `BH-BUILD` | Building | How they create agents |
| `BH-DEPLOY` | Deployment | How they get agents to production |

---

## Affinity Mapping Process

### Step 1: Extract Atomic Insights

From each interview, extract discrete insights as separate items:

```
[P001] [PP-INT] "Making LangGraph and CrewAI agents talk to each other required building a custom REST API between them."

[P002] [PP-OBS] "When an agent fails, I have no idea which step failed. I end up adding print statements everywhere."

[P003] [ND-TRUST] "I would never run a third-party agent without understanding exactly what tools it has access to."
```

### Step 2: Physical or Digital Clustering

**Physical Method**: Write each insight on a sticky note, cluster on wall
**Digital Method**: Use Miro, FigJam, or Notion with tags

### Step 3: Name the Clusters

After grouping related insights, name each cluster with a theme:

- "Debugging is a nightmare"
- "Framework choice creates lock-in anxiety"
- "Trust requires transparency"

### Step 4: Quantify Clusters

| Theme | # of Participants | # of Mentions | Severity (Avg) |
|-------|-------------------|---------------|----------------|
| Debugging difficulty | 8/10 | 23 | 4.2/5 |
| Framework lock-in | 7/10 | 15 | 3.8/5 |
| ... | ... | ... | ... |

---

## Pattern Analysis Templates

### Pain Point Analysis

For each significant pain point:

```markdown
## Pain Point: [Name]

### Evidence
- **Frequency**: X of Y participants mentioned this
- **Severity**: Average rating X/5
- **Personas Affected**: [List]

### Representative Quotes
> "[Quote 1]" - P001

> "[Quote 2]" - P004

### Current Workarounds
1. [How users currently address this]
2. [Alternative approaches]

### Root Cause
[Underlying reason this pain exists]

### Opportunity Size
[Impact if we solve this]
```

### Feature Prioritization Matrix

Based on research, score potential features:

| Feature | Pain Severity (1-5) | Frequency (%) | Willingness to Pay | Effort (S/M/L) | Priority Score |
|---------|---------------------|---------------|-------------------|----------------|----------------|
| Unified tracing | 4.5 | 80% | High | M | 18 |
| Cross-framework adapters | 4.0 | 70% | Medium | L | 14 |
| Trust scoring | 3.5 | 40% | High | M | 10 |
| ... | ... | ... | ... | ... | ... |

**Priority Score Formula**: (Severity × Frequency × WTP Multiplier) / Effort
- WTP Multiplier: High=2, Medium=1.5, Low=1
- Effort: S=1, M=2, L=3

---

## Persona Validation Framework

### Persona Fit Assessment

After each interview, assess fit with defined personas:

| Persona | Signals Present | Signals Absent | Fit Score (1-5) | Notes |
|---------|-----------------|----------------|-----------------|-------|
| Vibecoder | Fast iteration, budget-conscious | - | 4 | Matches well except more experienced than expected |
| Orchestrator | Production focus, team lead | Enterprise compliance | 3 | Hybrid with some Governor traits |
| Governor | - | Governance focus, security primary | 1 | Not this persona |

### Persona Refinement

After 5+ interviews, revisit personas:

1. **Validate Assumptions**: Which assumed characteristics were confirmed?
2. **Update Incorrect Assumptions**: What did we get wrong?
3. **Identify Gaps**: Are there user types we missed?
4. **Refine Demographics**: Adjust ranges based on actual participants

---

## Jobs-to-be-Done Analysis

Frame findings as jobs users are trying to accomplish:

### Job Template

```markdown
## Job: [Job Name]

**When** [situation/trigger]
**I want to** [action/capability]
**So that** [desired outcome]

### Context
- Frequency: How often does this job arise?
- Importance: How critical is completing this job?
- Current Solution: How do they do it today?
- Satisfaction: How satisfied are they with current solution?

### Pain Points in Current Solution
1. [Pain point 1]
2. [Pain point 2]

### Success Criteria
What would "done well" look like?
```

### Example

```markdown
## Job: Debug Multi-Agent Workflow Failure

**When** a workflow involving multiple AI agents fails in production
**I want to** quickly trace the failure to the specific agent and step that caused it
**So that** I can fix the issue before it impacts more users

### Context
- Frequency: 2-3 times per week during active development
- Importance: Critical (blocks production, causes user impact)
- Current Solution: Manual log searching, adding print statements, rerunning locally
- Satisfaction: 2/5 (very unsatisfied)

### Pain Points
1. Logs from different agents are in different formats/locations
2. No correlation between agent actions and original request
3. Have to reproduce issue locally, which isn't always possible

### Success Criteria
- Single dashboard showing entire workflow execution
- Click on any step to see inputs, outputs, and errors
- Trace from user request to final response
```

---

## Weekly Synthesis Ritual

### After Each Week of Interviews

1. **Update Affinity Map**: Add new insights, look for emerging themes
2. **Refresh Quantitative Summary**: Update frequency and severity counts
3. **Identify Gaps**: What questions remain unanswered?
4. **Adjust Interview Guide**: Modify questions based on learnings
5. **Share with Team**: Brief stakeholders on emerging findings

### Synthesis Meeting Agenda (30 min)

1. **Quick Stats** (5 min): How many interviews, participant breakdown
2. **Top 3 Insights** (10 min): Most significant learnings this week
3. **Surprise of the Week** (5 min): Something unexpected
4. **Open Questions** (5 min): What we still need to learn
5. **Next Week Focus** (5 min): Adjustments to research plan

---

## Final Research Report Structure

After completing research phase, compile findings into a report:

### Executive Summary
- 3-5 key findings
- Top recommendation

### Research Methodology
- Participants: N, demographics, recruitment
- Methods: Interviews, duration, approach
- Limitations: What we couldn't answer

### Persona Validation
- Confirmed personas with evidence
- Refinements made
- New personas discovered

### Key Findings by Theme
For each major theme:
- Evidence summary
- Representative quotes
- Implications for product

### Jobs to Be Done
- Prioritized list of jobs
- Current solutions and gaps

### Feature Prioritization
- Recommended priority order
- Rationale

### Recommendations
- Product recommendations
- Go-to-market recommendations
- Further research needs

### Appendix
- Interview transcripts/summaries
- Raw affinity map
- All quotes by category

---

## Templates and Tools

### Recommended Tools

| Purpose | Tools |
|---------|-------|
| Recording | Zoom, Google Meet, Otter.ai |
| Note-taking | Notion, Google Docs |
| Affinity Mapping | Miro, FigJam, physical sticky notes |
| Quote Database | Airtable, Notion |
| Analysis | Spreadsheet for quantitative, Miro for qualitative |

### Quick Links

- [Interview Guide](./interview-guide.md)
- [Personas](./personas.md)
- [Post-Interview Debrief Template](./interview-guide.md#post-interview-debrief-template)
