# FATHOM for AI Agents

## The Agentic AI Problem

Autonomous AI agents are making high-stakes decisions without human verification:
- **AutoGPT / AgentGPT:** Autonomous task execution chains
- **LangChain / LangGraph:** Agent frameworks with tool use
- **CrewAI:** Multi-agent collaborative systems
- **OpenAI Assistants API:** Function-calling agents in production
- **Anthropic Claude Computer Use:** Agents operating computers

The failure mode is catastrophic: an agent makes a chain of 20 decisions, each building on the last. If decision #3 was shallow pattern-matching instead of genuine reasoning, every subsequent action is built on a brittle foundation. By the time the output reaches the user, the damage is compounded.

Current agent frameworks have no mechanism to detect *when the agent stopped reasoning and started guessing.*

## Fathom as Pre-Action Verification Layer

### Architecture
```
User Request
    ↓
Agent Plan (step 1, 2, 3, ... N)
    ↓
For each step:
    Agent generates action rationale
        ↓
    Fathom scores the rationale
        ↓
    If depth >= threshold: execute action
    If depth < threshold: pause, request human review
        ↓
    Next step
```

### Use Cases

**1. High-Stakes Decision Gate**
Before an agent executes an irreversible action (sending an email, making a purchase, deploying code), Fathom scores the reasoning chain that led to that decision. Shallow reasoning = halt and ask.

**2. Agent Quality Monitoring**
Continuous depth scoring across an agent's decision chain. If average depth drops below baseline, the agent is struggling and should escalate.

**3. Multi-Agent Consensus**
When multiple agents collaborate (CrewAI), score each agent's contribution. If one agent's reasoning is consistently shallow, replace it or weight its outputs less.

**4. Agent Evaluation & Selection**
Compare different AI agents on the same task by depth profile. The agent with deeper reasoning is more likely to handle edge cases correctly.

### Integration Examples

**LangChain:**
```python
from fathom_sdk import FathomClient

fathom = FathomClient(api_key="...")

class FathomVerifiedTool(BaseTool):
    def _run(self, query: str) -> str:
        # Agent generates reasoning
        reasoning = self.agent.generate_reasoning(query)

        # Score reasoning depth
        score = fathom.score(reasoning)

        if score.grade in ["A+", "A", "B+", "B"]:
            return self.execute(query)
        else:
            return f"LOW CONFIDENCE (Fathom {score.grade}). Escalating to human."
```

**CrewAI:**
```python
from crewai import Agent, Task
from fathom_sdk import FathomClient

fathom = FathomClient(api_key="...")

# Score each agent's output before passing to next agent
@task.before_execute
def verify_reasoning(task_output):
    score = fathom.score(task_output.raw)
    if score.fathom_score < 60:
        raise ReasoningDepthError(f"Agent output too shallow: {score.grade}")
```

### Sample API Call
```bash
# Score an agent's reasoning chain
curl -X POST https://api.fathom.darkflobi.com/v2/batch \
  -H "x-api-key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      "Step 1: The user wants to refund the order. I should check the order status first because the refund policy depends on whether the item has",
      "Step 2: The order was delivered 3 days ago. The refund window is 30 days. I should proceed with the refund because the timeline is within",
      "Step 3: Processing the refund to the original payment method because the user did not specify an alternative and the default policy states"
    ]
  }'
```

## Pricing: Agent Tier

| Tier | Price | Includes |
|------|-------|---------|
| Developer | $29/month | 10k scores, Python SDK, basic monitoring |
| Startup | $199/month | 100k scores, real-time scoring, alerting, dashboard |
| Enterprise Agent | $999/month | 1M scores, sub-100ms scoring, multi-agent support |
| Platform (LangChain/CrewAI) | Custom | Unlimited, native integration, co-marketing |

## Target Customers

- AI agent companies (AutoGPT, AgentGPT, Adept, Cognition/Devin)
- Agent framework providers (LangChain, CrewAI, AutoGen)
- Enterprises deploying autonomous AI (customer service, ops, coding)
- AI safety companies building agent guardrails
- Venture-backed startups building on agent frameworks
