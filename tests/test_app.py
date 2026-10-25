CLASS = {
    "is_health": True, "is_emergency": False, "is_understandable": True,
    "intent": "ask", "requested_language": None,
    "language_code": "en", "language_name": "English",
    "language_confidence": 0.9, "refusal_message": None,
}


def patch_classify(appmod, monkeypatch, **over):
    monkeypatch.setattr(appmod, "classify_message", lambda *a, **k: {**CLASS, **over})


def test_voice_end_accepts_post(appmod):
    c = appmod.app.test_client()
    assert c.post("/voice/end").status_code == 200
    assert c.get("/voice/end").status_code == 200


def test_voice_passes_country_to_process(appmod):
    r = appmod.app.test_client().post("/voice", data={"From": "+2348012345678"})
    assert b"/voice/process?country=NG" in r.data


def test_set_language_stores_preference(appmod, monkeypatch):
    patch_classify(appmod, monkeypatch, intent="set_language", requested_language="Swahili")
    out = appmod.generate_ai_response("reply in Swahili", user_id="+254700000000")
    assert "Swahili" in out
    assert appmod.USER_LANG_PREF[appmod.user_key("+254700000000")] == "Swahili"


def test_preference_is_used_for_rag(appmod, monkeypatch):
    patch_classify(appmod, monkeypatch)
    appmod.USER_LANG_PREF[appmod.user_key("u1")] = "Hausa"
    appmod.rag.retrieve.return_value = [{"text": "x"}]
    appmod.rag.generate.return_value = "answer"
    appmod.scorer.score.return_value = {"label": "LOW", "score": 0.1}
    assert appmod.generate_ai_response("q", user_id="u1") == "answer"
    assert appmod.rag.generate.call_args.kwargs["language"] == "Hausa"


def test_high_risk_adds_note(appmod, monkeypatch):
    patch_classify(appmod, monkeypatch)
    appmod.rag.retrieve.return_value = []
    appmod.rag.generate.return_value = "answer"
    appmod.scorer.score.return_value = {"label": "HIGH", "score": 0.9}
    assert "healthcare provider" in appmod.generate_ai_response("q")


def test_off_topic_refused(appmod, monkeypatch):
    patch_classify(appmod, monkeypatch, is_health=False, refusal_message="No.")
    assert appmod.generate_ai_response("write code") == "No."


def test_api_ask_validation_and_user_id(appmod, monkeypatch):
    c = appmod.app.test_client()
    assert c.post("/api/ask", json={}).status_code == 400
    seen = {}
    monkeypatch.setattr(appmod, "generate_ai_response",
                        lambda q, country="UNKNOWN", user_id=None: seen.update(u=user_id) or "ok")
    assert c.post("/api/ask", json={"question": "hi", "user_id": "abc"}).json["answer"] == "ok"
    assert seen["u"] == "abc"


def test_user_key_hides_phone_number(appmod):
    key = appmod.user_key("+254700000000")
    assert key and "254" not in key
    assert appmod.user_key(None) is None


def test_emergency_and_unclear_routes(appmod, monkeypatch):
    patch_classify(appmod, monkeypatch, is_emergency=True)
    monkeypatch.setattr(appmod, "generate_emergency_response", lambda *a, **k: "EMERGENCY")
    assert appmod.generate_ai_response("chest pain") == "EMERGENCY"
    patch_classify(appmod, monkeypatch, is_understandable=False, language_confidence=0.2)
    assert "didn" in appmod.generate_ai_response("asdf")
    appmod.rag.retrieve.assert_not_called()


def test_graph_error_returns_fallback(appmod, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(appmod, "classify_message", boom)
    assert "trouble" in appmod.generate_ai_response("hi")
