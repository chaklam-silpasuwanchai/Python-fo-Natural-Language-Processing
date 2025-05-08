# Agentic AI (tentative)

Welcome to the **Agentic AI** module.  This directory groups practical, reusable components that help you turn large-language-models into *agents*—software entities that can plan, act, reflect, and improve over time.

> *“An agent is an LLM with a purpose, a plan, the ability to call tools, **and the capacity to critique itself.**”*

Status: 🚧 Work in progress — first public launch planned for 2026.

The sub-folders follow the canonical 2025 agent stack:

| Folder            | Role                                    |
| ----------------- | --------------------------------------- |
| **01-Workflows**  | Orchestrate thinking & acting           |
| **02-Tools**      | Interfaces to the external world        |
| **03-Memory**     | Short- & long-term context store        |
| **04-Reflection** | Self-evaluation & iterative improvement |
| **05-Evaluation** | Automated regression & benchmarking     |

---

## ⚙️ 01 – Workflows

Orchestrates *how* an agent thinks and acts.

* Goal → Plan → Act → Observe → Refine templates
* LangGraph / state-machine examples
* Retry, timeout, and tracing utilities

**Start here:** [Building Effective Agents – Anthropic (2025)](https://www.anthropic.com/engineering/building-effective-agents) & [AgentRecipes](https://www.agentrecipes.com/)

---

## 🛠 02 – Tools

APIs and functions the agent can invoke to affect the outside world.

* Tool-spec schema (`tool.json`) with signatures & rate-limit hints
* Wrappers for search, code-exec, and domain APIs
* Guard-rails: allow-lists, parameter validation, safe-exec

---

## 🧠 03 – Memory

Keeps relevant context accessible across multiple turns or sessions.

* Conversation buffer, vector store, summarisation & forgetting policies
* **Reference:** *MemGPT: Towards a Scalable Memory-Augmented Language Model* (An et al., 2023)

---

## 🔄 04 – Reflection

Gives the agent meta-cognition: the ability to critique, debug, and refine its own outputs.

* *Reflexion* and *RAG-as-critic* patterns
* Rubric-based self-grading prompts
* Caching of reflection results to control cost

> **TODO:** Provide a `reflection_runner.py` demo that re-asks the LLM to rate its answer and propose an improvement.

---

## 🧪 05 – Evaluation

Ensures changes don’t silently break agent behaviour.

* Smoke tests executed in CI (`pytest` + `langsmith` traces)
* Golden-conversation dataset for regression
* Automatic pass/fail based on rubric scores or reference answers

> **TODO:** Add `eval/` with sample YAML spec and Harness script.

---
