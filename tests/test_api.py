from tests.conftest import DOCS, make_settings

DEMO = {"X-API-Key": "demo-key"}
ADMIN = {"X-API-Key": "admin-key"}
HR = {"X-API-Key": "hr-key"}


def _upload(client, collection, names, headers, **form):
    files = [("files", (n, (DOCS / n).read_bytes())) for n in names]
    return client.post("/ingest", data={"collection": collection, **form}, files=files, headers=headers)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["version"] == "0.1.0" and body["auth_enabled"] is True


def test_ingest_query_with_citations(client):
    r = _upload(
        client,
        "demo",
        ["security-policy.pdf", "employee-handbook.md", "product-faq.html", "onboarding-guide.docx"],
        DEMO,
    )
    assert r.status_code == 200, r.text
    docs = {d["document"]: d for d in r.json()["documents"]}
    assert docs["security-policy.pdf"]["pages"] == 2
    assert all(d["chunks"] > 0 for d in docs.values())

    r = client.post(
        "/query",
        json={"collection": "demo", "question": "How long are security logs retained?"},
        headers=DEMO,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "400 days" in body["answer"]
    cite = body["citations"][0]
    assert cite["document"] == "security-policy.pdf"
    assert cite["page"] == 2
    assert {"doc_id", "chunk_id", "chunk_index", "score", "snippet", "ref"} <= cite.keys()
    assert f"[{cite['ref']}]" in body["answer"]
    assert len(body["retrieved"]) <= 5

    r = client.post(
        "/query", json={"collection": "demo", "question": "parental leave", "top_k": 2}, headers=DEMO
    )
    assert len(r.json()["retrieved"]) == 2


def test_auth_required_and_collection_scoping(client):
    assert client.get("/collections").status_code == 401
    assert client.get("/collections", headers={"X-API-Key": "wrong"}).status_code == 401

    assert _upload(client, "demo", ["employee-handbook.md"], DEMO).status_code == 200
    assert _upload(client, "hr", ["employee-handbook.md"], DEMO).status_code == 403
    assert _upload(client, "hr", ["onboarding-guide.docx"], HR).status_code == 200

    q = {"collection": "hr", "question": "probation period"}
    assert client.post("/query", json=q, headers=DEMO).status_code == 403
    assert client.post("/query", json=q, headers=HR).status_code == 200

    names = lambda h: [c["name"] for c in client.get("/collections", headers=h).json()["collections"]]  # noqa: E731
    assert names(DEMO) == ["demo"]
    assert names(HR) == ["hr"]
    assert names(ADMIN) == ["demo", "hr"]


def test_reingest_replaces_document(client):
    first = _upload(client, "demo", ["employee-handbook.md"], DEMO).json()["documents"][0]
    again = _upload(client, "demo", ["employee-handbook.md"], DEMO, chunk_size="200", chunk_overlap="20")
    doc = again.json()["documents"][0]
    assert doc["chunks"] > first["chunks"]
    info = client.get("/collections", headers=DEMO).json()["collections"][0]
    assert info == {"name": "demo", "documents": 1, "chunks": doc["chunks"]}


def test_ingest_validation_errors(client):
    r = client.post("/ingest", data={"collection": "demo"}, files=[("files", ("a.exe", b"MZ"))], headers=DEMO)
    assert r.status_code == 415
    r = client.post(
        "/ingest", data={"collection": "bad name!"}, files=[("files", ("a.md", b"x"))], headers=ADMIN
    )
    assert r.status_code == 422
    r = client.post(
        "/ingest",
        data={"collection": "demo", "chunk_size": "100", "chunk_overlap": "200"},
        files=[("files", ("a.md", b"x"))],
        headers=DEMO,
    )
    assert r.status_code == 422
    r = client.post("/query", json={"collection": "demo", "question": ""}, headers=DEMO)
    assert r.status_code == 422


def test_upload_size_limit():
    from fastapi.testclient import TestClient

    from rag.api import create_app

    app = create_app(make_settings(max_upload_mb=1))
    with TestClient(app) as c:
        big = b"a " * (600 * 1024)
        r = c.post("/ingest", data={"collection": "demo"}, files=[("files", ("big.md", big))])
        assert r.status_code == 413


def test_open_mode_and_empty_collection(open_client):
    assert open_client.get("/healthz").json()["auth_enabled"] is False
    r = open_client.post("/query", json={"collection": "empty", "question": "anything?"})
    assert r.status_code == 200
    assert r.json()["citations"] == [] and "could not find" in r.json()["answer"]
