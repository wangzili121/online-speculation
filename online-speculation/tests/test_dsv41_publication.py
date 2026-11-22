import pytest

from blockspec.dsv41_publication import AtomicHeadPublisher, PublicationReceipt, validate_pair


DIGEST = "a" * 64


def receipt(phase, version, ranks, digest=DIGEST):
    return PublicationReceipt(phase, version, digest, tuple(ranks))


def test_publication_requires_one_consistent_receipt_per_rank():
    assert validate_pair(
        receipt("prepared", 1, range(8)), receipt("prepared", 1, range(8, 16)), current_version=0,
    ) == (1, DIGEST)
    with pytest.raises(ValueError, match="different"):
        validate_pair(
            receipt("prepared", 1, range(8)), receipt("prepared", 1, range(8, 16), "b" * 64),
            current_version=0,
        )
    with pytest.raises(ValueError, match="every rank"):
        validate_pair(
            receipt("prepared", 1, range(8)), receipt("prepared", 1, range(7, 15)), current_version=0,
        )


def test_atomic_publisher_stages_then_activates_and_rolls_back_failures():
    events = []

    def exchange(tag, payload):
        events.append(("exchange", tag))
        return receipt(payload["phase"], payload["version"], range(8, 16), payload["digest"]).payload()

    publisher = AtomicHeadPublisher(
        exchange=exchange,
        stage=lambda learner, version: events.append(("stage", version)) or receipt("prepared", version, range(8)),
        activate=lambda version: events.append(("activate", version)) or receipt("activated", version, range(8)),
        rollback=lambda version: events.append(("rollback", version)),
    )
    assert publisher(object(), 4) == 1
    assert publisher.current_version == 1
    assert events == [
        ("stage", 1), ("exchange", "online-head/1/prepared"),
        ("activate", 1), ("exchange", "online-head/1/activated"),
    ]

    publisher.exchange = lambda tag, payload: receipt(
        payload["phase"], payload["version"], range(8, 16), "b" * 64,
    ).payload()
    with pytest.raises(ValueError, match="different"):
        publisher(object(), 5)
    assert events[-2:] == [("stage", 2), ("rollback", 2)]
    assert publisher.current_version == 1
