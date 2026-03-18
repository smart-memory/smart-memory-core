"""REBEL linearized triplet parser — promoted from benchmark test code.

Parses REBEL model output format:
    <triplet> subject <subj> object <obj> relation

Into structured dicts: {"source": str, "relation": str, "target": str}
"""

from __future__ import annotations


def parse_rebel_triplets(text: str) -> list[dict[str, str]]:
    """Parse REBEL linearized triplet output into structured relation dicts.

    REBEL format: <triplet> subject <subj> object <obj> relation
    Multiple triplets can appear in a single string.

    Args:
        text: Raw REBEL output string.

    Returns:
        List of {"source": str, "relation": str, "target": str} dicts.
    """
    relations: list[dict[str, str]] = []
    subject = ""
    relation = ""
    object_ = ""
    current = "x"

    # Strip special tokens
    cleaned = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")

    for token in cleaned.split():
        if token == "<triplet>":
            # Flush previous triplet if complete
            if relation.strip():
                relations.append(
                    {
                        "source": subject.strip(),
                        "relation": relation.strip(),
                        "target": object_.strip(),
                    }
                )
            current = "t"
            subject = ""
            relation = ""
            object_ = ""
        elif token == "<subj>":
            # Flush previous triplet if we hit a new subject without <triplet>
            if relation.strip():
                relations.append(
                    {
                        "source": subject.strip(),
                        "relation": relation.strip(),
                        "target": object_.strip(),
                    }
                )
            current = "s"
            object_ = ""
        elif token == "<obj>":
            current = "o"
            relation = ""
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token

    # Flush last triplet
    if relation.strip():
        relations.append(
            {
                "source": subject.strip(),
                "relation": relation.strip(),
                "target": object_.strip(),
            }
        )

    return relations
