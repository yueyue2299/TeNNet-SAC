from pathlib import Path

import pytest

from smi_ted_light.load import MolTranBertTokenizer


VOCAB_PATH = Path(__file__).parents[1] / "smi_ted_light" / "bert_vocab_curated.txt"


@pytest.fixture
def tokenizer():
    return MolTranBertTokenizer(str(VOCAB_PATH))


@pytest.mark.parametrize(
    ("smiles", "expected_tokens", "expected_ids"),
    [
        ("CCO", ["C", "C", "O"], [0, 4, 4, 9, 1]),
        ("ClCCCl", ["Cl", "C", "C", "Cl"], [0, 20, 4, 4, 20, 1]),
        (
            "C[C@H](O)C(=O)O",
            ["C", "[C@H]", "(", "O", ")", "C", "(", "=", "O", ")", "O"],
            [0, 4, 15, 6, 9, 7, 4, 6, 12, 9, 7, 9, 1],
        ),
        ("[NH4+]", ["[NH4+]"], [0, 110, 1]),
        (
            "C%12CCCCC%12",
            ["C", "%12", "C", "C", "C", "C", "C", "%12"],
            [0, 4, 68, 4, 4, 4, 4, 4, 68, 1],
        ),
    ],
)
def test_tokenizer_preserves_legacy_smi_ted_tokens_and_ids(
    tokenizer, smiles, expected_tokens, expected_ids
):
    assert tokenizer.tokenize(smiles) == expected_tokens
    assert tokenizer(smiles, add_special_tokens=True)["input_ids"] == expected_ids


def test_tokenizer_matches_legacy_batch_padding(tokenizer):
    encoded = tokenizer(
        ["CCO", "ClCCCl"],
        padding=True,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
        max_length=8,
    )

    assert encoded["input_ids"].tolist() == [
        [0, 4, 4, 9, 1, 2],
        [0, 20, 4, 4, 20, 1],
    ]
    assert encoded["attention_mask"].tolist() == [
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
    ]


def test_tokenizer_keeps_eos_when_truncating(tokenizer):
    encoded = tokenizer(
        ["CCO", "ClCCCl"],
        padding=True,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
        max_length=5,
    )

    assert encoded["input_ids"].tolist() == [
        [0, 4, 4, 9, 1],
        [0, 20, 4, 4, 1],
    ]
