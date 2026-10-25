"""LangGraph pipeline: classify -> route -> (RAG: retrieve -> generate -> score) | canned reply."""
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

UNCLEAR_MESSAGE = (
    "I didn’t understand your message clearly. Please retype your health question in your preferred language, "
    "or tell me the language name you want me to use."
)
OFF_TOPIC_MESSAGE = "I can only help with health and medical questions."
HIGH_RISK_NOTE = "\n\n Note: I recommend confirming this with a healthcare provider"


class State(TypedDict, total=False):
    user_input: str
    country: str
    user_key: Optional[str]  # opaque id (never the raw phone number)
    classification: Dict[str, Any]
    chunks: List[Dict]
    answer: str
    score_result: Dict[str, Any]


def route_after_classify(state: State) -> str:
    c = state["classification"]
    if c.get("intent") == "set_language":
        return "set_language"
    if c.get("is_emergency"):
        return "emergency"
    if not c.get("is_understandable", True) and c.get("language_confidence", 1.0) < 0.7:
        return "unclear"
    if not c.get("is_health"):
        return "refuse"
    return "retrieve"


def build_graph(
    classify: Callable[..., Dict],
    emergency: Callable[[str, str], str],
    rag,
    scorer,
    lang_pref: Dict[str, str],
):
    """Compile the graph. Dependencies are injected so the app (and tests) own them."""

    def classify_node(state: State):
        return {"classification": classify(state["user_input"], country=state.get("country", "UNKNOWN"))}

    def set_language_node(state: State):
        requested = (state["classification"].get("requested_language") or "").strip()
        key = state.get("user_key")
        if requested and key:
            lang_pref[key] = requested
            return {"answer": f"Okay. I will reply in {requested}. Now ask your health question."}
        return {"answer": "Okay. Tell me your health question, and I will reply in that language."}

    def emergency_node(state: State):
        name = state["classification"].get("language_name", "Unknown")
        return {"answer": emergency(state["user_input"], name)}

    def unclear_node(state: State):
        return {"answer": UNCLEAR_MESSAGE}

    def refuse_node(state: State):
        return {"answer": state["classification"].get("refusal_message") or OFF_TOPIC_MESSAGE}

    def retrieve_node(state: State):
        return {"chunks": rag.retrieve(state["user_input"], top_k=5)}

    def generate_node(state: State):
        key = state.get("user_key")
        language = lang_pref.get(key) if key else None
        return {"answer": rag.generate(state["user_input"], state["chunks"], language=language)}

    def score_node(state: State):
        result = scorer.score(state["answer"], state["chunks"])
        answer = state["answer"]
        if result["label"] == "HIGH":
            answer += HIGH_RISK_NOTE
        return {"score_result": result, "answer": answer}

    g = StateGraph(State)
    g.add_node("classify", classify_node)
    g.add_node("set_language", set_language_node)
    g.add_node("emergency", emergency_node)
    g.add_node("unclear", unclear_node)
    g.add_node("refuse", refuse_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("score", score_node)

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify",
        route_after_classify,
        ["set_language", "emergency", "unclear", "refuse", "retrieve"],
    )
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "score")
    for terminal in ("set_language", "emergency", "unclear", "refuse", "score"):
        g.add_edge(terminal, END)
    return g.compile()
