"""Workaround for a broken transitive import in ragas 0.4.3.

`ragas.llms.base` does `from langchain_community.chat_models.vertexai import
ChatVertexAI` purely to use it in an `isinstance` check inside its (deprecated)
`LangchainLLMWrapper`. That submodule was removed from `langchain-community`
ahead of ragas shipping a fix (tracked upstream; no compatible ragas release
exists yet). We never use Vertex AI — only Gemini via `langchain-google-genai`
— so a stub with the same class names satisfies the import without pulling in
real Vertex AI support.

Call `patch()` before the first `import ragas` (or anything importing it).
Idempotent and a no-op once the real submodule exists upstream.
"""

import sys
import types


def patch() -> None:
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401

        return  # real module exists; nothing to patch
    except ModuleNotFoundError:
        pass

    module = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # placeholder; never instantiated in our eval path
        pass

    module.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = module
