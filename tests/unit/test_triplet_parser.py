"""Unit tests for REBEL triplet parser."""

from smartmemory.corpus.triplet_parser import parse_rebel_triplets


class TestParseRebelTriplets:
    def test_single_triplet(self):
        text = "<triplet> Python <subj> programming language <obj> instance of"
        result = parse_rebel_triplets(text)
        assert len(result) == 1
        assert result[0] == {
            "source": "Python",
            "relation": "instance of",
            "target": "programming language",
        }

    def test_multiple_triplets(self):
        text = (
            "<triplet> Python <subj> Guido van Rossum <obj> creator "
            "<triplet> Python <subj> programming language <obj> instance of"
        )
        result = parse_rebel_triplets(text)
        assert len(result) == 2
        assert result[0]["source"] == "Python"
        assert result[0]["target"] == "Guido van Rossum"
        assert result[0]["relation"] == "creator"
        assert result[1]["relation"] == "instance of"

    def test_multi_word_entities(self):
        text = "<triplet> United States of America <subj> Washington D.C. <obj> capital"
        result = parse_rebel_triplets(text)
        assert len(result) == 1
        assert result[0]["source"] == "United States of America"
        assert result[0]["target"] == "Washington D.C."

    def test_special_tokens_stripped(self):
        text = "<s><triplet> Foo <subj> Bar <obj> related to</s>"
        result = parse_rebel_triplets(text)
        assert len(result) == 1
        assert result[0]["source"] == "Foo"

    def test_pad_tokens_stripped(self):
        text = "<pad><triplet> A <subj> B <obj> is<pad>"
        result = parse_rebel_triplets(text)
        assert len(result) == 1

    def test_empty_string(self):
        assert parse_rebel_triplets("") == []

    def test_no_triplet_markers(self):
        assert parse_rebel_triplets("just some random text") == []

    def test_incomplete_triplet_ignored(self):
        """A triplet without <obj> shouldn't produce a result."""
        text = "<triplet> Foo <subj> Bar"
        result = parse_rebel_triplets(text)
        assert len(result) == 0

    def test_same_subject_multiple_relations(self):
        text = (
            "<triplet> React <subj> JavaScript <obj> programming language "
            "<triplet> React <subj> Meta <obj> developer"
        )
        result = parse_rebel_triplets(text)
        assert len(result) == 2
        assert all(r["source"] == "React" for r in result)
