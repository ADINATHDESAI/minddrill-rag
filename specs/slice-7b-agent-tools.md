# Slice 7b — Agent loop + tools

Tools (registry: name + Pydantic input schema + function). Validate tool INPUTS
with Pydantic — this is the machine-consumed path, so validation applies here.
- calculator(expression): safe arithmetic only (no eval of arbitrary code).
- weather(latitude, longitude): Open-Meteo free API, no key.
- knowledge_base_search(query): wraps the slice 4/5 hybrid_search + rerank.
  The LLM-visible schema has ONLY `query`. user_id is NOT a tool parameter —
  it is injected server-side (see Isolation).

Agent loop:
- Use create_agent (LangChain 1.0, runs on LangGraph). Confirm the exact import
  in the installed version.
- Middleware guardrails: max tool steps (config AGENT_MAX_STEPS, e.g. 6) and
  history trimming. On step-limit, stop and return the best answer so far.
- /chat routes through the agent. /query stays direct RAG (unchanged).

Streaming with tools (extend the slice-6 protocol):
- emit tool_call {tool, args} when the agent calls a tool.
- emit tool_result {tool, result} when it returns.
- interleave with token events; end with done.

Isolation (critical): user_id is a trust boundary. The LLM must NEVER supply it.
Inject it server-side — bind it per request via a closure, or use LangChain's
InjectedToolArg / InjectedState so it's hidden from the model's tool schema. The
tool then filters by that injected user_id. This applies to EVERY tool that
touches user data, not just search. A tool must never query across users.

Tests:
- calculator tool returns correct results; rejects non-arithmetic input.
- knowledge_base_search only returns the current user's chunks (isolation).
- agent calls a tool then answers; stream shows tool_call then tool_result.
- step-limit guardrail stops a runaway loop.
- tool input failing its Pydantic schema is handled, not crashed.

What can go wrong: infinite tool loop (step limit), a tool leaking another
user's data (pass user_id explicitly), unvalidated tool input, a slow tool
(timeout).