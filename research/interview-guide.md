# AI Workflow Fabric - User Research Interview Guide

## Overview

**Duration**: 60 minutes  
**Format**: Semi-structured interview  
**Target**: Developers and teams who build or orchestrate AI agents  

### Interview Objectives

1. Understand current pain points in multi-agent orchestration
2. Discover unmet needs and workarounds
3. Validate assumptions about security, trust, and governance
4. Identify willingness to pay and value drivers
5. Map the decision-making process for adopting new tools

---

## Pre-Interview Checklist

- [ ] Send calendar invite with Zoom/Meet link
- [ ] Confirm participant's role and experience level
- [ ] Prepare screen recording (with consent)
- [ ] Have note-taking document ready
- [ ] Test audio/video
- [ ] Send reminder 24 hours before

---

## Interview Structure

| Section | Duration | Focus |
|---------|----------|-------|
| 1. Warm-up & Context | 5 min | Build rapport, understand background |
| 2. Current State | 15 min | How they work today |
| 3. Pain Points | 15 min | Deep dive into frustrations |
| 4. Ideal State | 10 min | What they wish existed |
| 5. Security & Trust | 10 min | Governance and safety concerns |
| 6. Wrap-up | 5 min | Prioritization and closing |

---

## Section 1: Warm-up & Context (5 min)

### Opening Script

> "Thank you for taking the time to speak with me today. I'm researching how developers and teams work with AI agents - specifically how they build, deploy, and orchestrate them. There are no right or wrong answers; I'm just trying to understand your experience."

### Questions

**Q1.1**: Tell me about your role. What do you do day-to-day?

**Q1.2**: How long have you been working with AI/LLM-based systems?

**Q1.3**: Can you give me a quick overview of the AI agent projects you're currently working on or have recently completed?

---

## Section 2: Current State (15 min)

### Process Mapping

**Q2.1**: Walk me through how you currently build an AI agent, from idea to production.
- *Probe*: What tools do you use?
- *Probe*: How long does this typically take?
- *Probe*: Who else is involved?

**Q2.2**: What frameworks or platforms are you using today?
- *Listen for*: LangChain, LangGraph, CrewAI, AutoGen, OpenAI Assistants, custom solutions

**Q2.3**: If you're using multiple frameworks, why? What led to that decision?

**Q2.4**: How do your agents communicate with each other today?
- *Probe*: Shared memory? Message passing? API calls?
- *Probe*: What about agents built with different frameworks?

**Q2.5**: Tell me about a recent project where you had multiple agents working together. What did that look like?

### Team & Organization

**Q2.6**: How many people on your team work with AI agents?

**Q2.7**: Who decides which tools/frameworks to use?
- *Probe*: Is there an approval process for new dependencies?

---

## Section 3: Pain Points (15 min)

### Friction Discovery

**Q3.1**: What's the most frustrating part of your current workflow with AI agents?
- *Silence*: Let them think. Don't fill the gap.

**Q3.2**: Tell me about a time when things went wrong with an AI agent in production.
- *Probe*: What happened? How did you find out?
- *Probe*: How long did it take to fix?
- *Probe*: What was the impact?

**Q3.3**: What do you spend too much time on that you wish was automated or simpler?

**Q3.4**: If you could wave a magic wand and fix one thing about how you work with AI agents, what would it be?

### Interoperability Deep Dive

**Q3.5**: Have you ever needed to make agents from different frameworks work together?
- *If yes*: How did you do it? What was hard about it?
- *If no*: Why not? Have you wanted to?

**Q3.6**: What would it mean for your project if you could easily swap out one agent framework for another?

### Observability & Debugging

**Q3.7**: How do you debug issues with your agents today?
- *Probe*: What visibility do you have into what agents are doing?
- *Probe*: How do you trace problems across multiple agents?

**Q3.8**: When an agent behaves unexpectedly, how do you figure out why?

---

## Section 4: Ideal State (10 min)

### Vision Elicitation

**Q4.1**: If you were designing the perfect tool for orchestrating AI agents, what would it look like?
- *Probe*: What features would be must-haves?
- *Probe*: What would it integrate with?

**Q4.2**: How would you know if a new tool for AI orchestration was worth adopting?
- *Listen for*: Time savings, reliability, cost, team adoption

**Q4.3**: What would make you NOT adopt a new tool, even if it was technically good?
- *Listen for*: Vendor lock-in, learning curve, security concerns, cost

### Value & Willingness to Pay

**Q4.4**: If a tool could save you X hours per week on agent orchestration, what would that be worth to you/your company?

**Q4.5**: What's your budget for developer tools and infrastructure today?
- *Probe*: Who controls that budget?

---

## Section 5: Security & Trust (10 min)

### Governance Discovery

**Q5.1**: How do you ensure AI agents behave safely and as expected?
- *Probe*: Any formal review or approval process?
- *Probe*: Guardrails, rate limits, human-in-the-loop?

**Q5.2**: What concerns do you have about agents having access to tools, data, or external services?

**Q5.3**: If you were to use a third-party agent (one you didn't build), what would you need to trust it?
- *Listen for*: Audits, reputation, sandboxing, permissions model

**Q5.4**: Tell me about a time when security or compliance requirements impacted how you built or deployed an agent.

**Q5.5**: How do you handle agent permissions today? Who decides what an agent can access?

---

## Section 6: Wrap-up (5 min)

### Prioritization Exercise

**Q6.1**: I'm going to list some capabilities. Can you rank them from most to least important for your work?

Present these on screen or read aloud:
1. **Universal Discovery**: Find and use agents from any framework in one place
2. **Cross-Framework Orchestration**: Make agents from different frameworks work together seamlessly
3. **Security & Sandboxing**: Run agents in isolated, controlled environments
4. **Observability**: Deep visibility into what agents are doing, with unified tracing
5. **Trust Scoring**: Automated assessment of agent safety and reliability
6. **Governance Controls**: Define and enforce policies about agent behavior

**Q6.2**: Is there anything we didn't cover that you think is important?

**Q6.3**: Would you be interested in trying an early version of a tool that addresses these needs?
- *If yes*: Get email, add to beta list

### Closing Script

> "Thank you so much for your time and insights. This has been incredibly valuable. We'll be in touch as we develop our solution, and we'd love to have you as an early tester if you're interested."

---

## Post-Interview Debrief Template

Complete within 30 minutes of interview:

### Participant Profile
- **ID**: [P001, P002, etc.]
- **Role**: 
- **Company Size**: 
- **AI Experience**: [months/years]
- **Primary Framework(s)**:

### Key Insights (Top 3-5)
1. 
2. 
3. 
4. 
5. 

### Pain Points Identified
| Pain Point | Severity (1-5) | Frequency | Quote |
|------------|----------------|-----------|-------|
|            |                |           |       |

### Surprising Findings

### Quotes Worth Remembering

### Feature Prioritization (from Q6.1)
1. 
2. 
3. 
4. 
5. 
6. 

### Persona Fit
- [ ] Vibecoder (solo dev, ships fast)
- [ ] Orchestrator (tech lead, needs reliability)
- [ ] Governor (enterprise, needs governance)

### Follow-up Actions

---

## Recruitment Screener Survey

Use this to qualify participants before scheduling:

### Screener Questions

1. **What best describes your role?**
   - [ ] Individual developer/engineer
   - [ ] Tech lead / Senior developer
   - [ ] Engineering manager
   - [ ] DevOps / Platform engineer
   - [ ] Security / Compliance
   - [ ] Other: ________

2. **Have you built or deployed AI agents (LLM-based) in the last 6 months?**
   - [ ] Yes, multiple projects
   - [ ] Yes, one project
   - [ ] No, but planning to
   - [ ] No, and no plans (DISQUALIFY)

3. **Which frameworks/platforms have you used?** (Select all that apply)
   - [ ] LangChain
   - [ ] LangGraph
   - [ ] CrewAI
   - [ ] AutoGen
   - [ ] OpenAI Assistants API
   - [ ] Custom solution
   - [ ] Other: ________

4. **Have you ever needed to make agents from different frameworks work together?**
   - [ ] Yes (PRIORITIZE)
   - [ ] No, but wanted to
   - [ ] No, single framework is fine

5. **What's your company size?**
   - [ ] Just me (solo developer)
   - [ ] 2-10 employees (small startup)
   - [ ] 11-50 employees
   - [ ] 51-200 employees
   - [ ] 201-1000 employees
   - [ ] 1000+ employees (enterprise)

6. **Are you willing to participate in a 60-minute interview about your experience?**
   - [ ] Yes
   - [ ] No (DISQUALIFY)

### Scoring Criteria

| Criteria | Points |
|----------|--------|
| Built multiple AI agent projects | +3 |
| Used 2+ different frameworks | +3 |
| Needed cross-framework integration | +5 |
| Enterprise (200+ employees) | +2 |
| Solo developer | +2 |
| Willing to join beta | +1 |

**Interview Priority**:
- 10+ points: High priority, schedule immediately
- 5-9 points: Medium priority
- <5 points: Lower priority, schedule if slots available

---

## Appendix: Empathy Probes

Use these when you want to go deeper:

- "Tell me more about that..."
- "What did that feel like?"
- "Why do you think that happened?"
- "What did you do next?"
- "What were you hoping would happen?"
- "How did you work around that?"
- "What would you have done differently?"
- "Who else was affected by that?"
- "What was at stake?"
- "If you had to explain that to a new team member, how would you describe it?"

---

## Interview Cadence Recommendation

| Week | Interviews | Focus |
|------|------------|-------|
| 1 | 3-4 | Diverse roles, broad discovery |
| 2 | 3-4 | Deep dive on emerging themes |
| 3 | 2-3 | Validate findings, edge cases |

**Target**: 8-12 interviews for initial research phase

After each week, synthesize findings and adjust questions based on patterns emerging.
