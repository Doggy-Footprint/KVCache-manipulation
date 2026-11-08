import pytest

from kvmeaning import ambig as A


@pytest.fixture(scope="module")
def items():
    return A.load_items()


@pytest.fixture(scope="module")
def prompts(tok, items):
    return A.build(tok, items)


def _get(prompts, item, cond, cue="A", order="AB"):
    return next(p for p in prompts if (p.item, p.cond, p.cue, p.order) == (item, cond, cue, order))


def _text(tok, p, span):
    return tok.decode(p.enc.ids[slice(*p.enc.spans[span])])


def test_items_cover_four_types(items):
    assert sorted({it["type"] for it in items}) == ["entity", "lexical", "pragmatic", "structural"]
    assert len(items) == 60


def test_alignment_holds_for_all_items(tok, items, prompts):
    assert A.check(prompts) == []
    assert A.canonical_mismatches(tok, items) == []


def test_spans_decode_to_item_text(tok, items, prompts):
    it = next(i for i in items if i["id"] == "lex-01")
    p = _get(prompts, "lex-01", "B_pre")
    assert _text(tok, p, "span").strip() == it["span"]
    assert _text(tok, p, "cue").strip() == it["cues"]["A"]
    assert _text(tok, p, "expl") == A.PRE + it["expl"]["B"]
    assert _text(tok, _get(prompts, "lex-01", "U_pre"), "expl") == A.PRE + it["unrelated"]
    assert "cue" not in _get(prompts, "lex-01", "none", cue=None).enc.spans


def test_relation_labels(prompts):
    assert _get(prompts, "str-01", "A_pre", cue="A").relation == "correct"
    assert _get(prompts, "str-01", "B_post", cue="A").relation == "wrong"
    assert _get(prompts, "str-01", "A_pre", cue=None).relation == "cueless"
    assert _get(prompts, "str-01", "U_pre", cue="B").relation == "unrelated"
    assert _get(prompts, "str-01", "none", cue="B").relation == "none"


def test_order_swap_moves_gold_letter(tok, items, prompts):
    it = next(i for i in items if i["id"] == "ent-01")
    ab, ba = _get(prompts, "ent-01", "none", "A", "AB"), _get(prompts, "ent-01", "none", "A", "BA")
    assert (ab.gold, ba.gold) == ("A", "B")
    q = _text(tok, ba, "question")
    assert f"A. {it['options']['B']}" in q and f"B. {it['options']['A']}" in q


def test_pre_conditions_share_document_positions_and_post_shares_prefix_with_none(prompts):
    pre = [_get(prompts, "prag-01", c).enc.spans["doc"] for c in ("A_pre", "B_pre", "U_pre", "pA_pre", "pB_pre")]
    assert len(set(pre)) == 1
    none, post = _get(prompts, "prag-01", "none").enc, _get(prompts, "prag-01", "A_post").enc
    end = none.spans["doc"][1]
    assert post.ids[:end] == none.ids[:end] and none.spans["doc"] == post.spans["doc"]


def test_paraphrase_conditions_only_for_items_with_paraphrase(items, prompts):
    has = {i["id"] for i in items if "expl_para" in i}
    with_para = {p.item for p in prompts if p.cond.startswith("p")}
    assert with_para == has


def test_closed_book_has_no_document(tok, prompts):
    p = next(p for p in prompts if p.item == "lex-01" and p.cond == "closed")
    assert set(p.enc.spans) == {"question", "answer"}
    assert "Retrieved document" not in tok.decode(p.enc.ids)


def test_answer_position_is_last_token_and_letters_are_single_tokens(tok, prompts):
    p = _get(prompts, "lex-01", "none")
    assert p.enc.spans["answer"] == (len(p.enc.ids) - 1, len(p.enc.ids))
    assert set(A.answer_ids(tok)) == {"A", "B"}


def test_neutral_variant_conditions_align_with_explanation_conditions(tok, items, prompts):
    it = next(i for i in items if i["id"] == "str-03")
    n = _get(prompts, "str-03", "nB_pre")
    assert _text(tok, n, "expl") == A.PRE + it["expl_neutral"]["B"]
    assert n.relation == "wrong" and n.enc.spans["doc"] == _get(prompts, "str-03", "B_pre").enc.spans["doc"]
    assert {p.item for p in prompts if p.cond == "nA_pre"} == {i["id"] for i in items}
