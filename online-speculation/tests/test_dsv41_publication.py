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
        commit=lambda version: events.append(("commit", version)),
    )
    assert publisher(object(), 4) == 1
    assert publisher.current_version == 1
    assert events == [
        ("stage", 1), ("exchange", "online-head/1/prepared"),
        ("activate", 1), ("exchange", "online-head/1/activated"),
        ("commit", 1),
    ]

    publisher.exchange = lambda tag, payload: receipt(
        payload["phase"], payload["version"], range(8, 16), "b" * 64,
    ).payload()
    with pytest.raises(ValueError, match="different"):
        publisher(object(), 5)
    assert events[-2:] == [("stage", 2), ("rollback", 2)]
    assert publisher.current_version == 1


def test_atomic_publisher_rolls_back_after_local_activation_if_peer_activation_fails():
    events = []

    def exchange(tag, payload):
        if tag.endswith("/activated"):
            raise TimeoutError("peer activation receipt missing")
        return receipt("prepared", payload["version"], range(8, 16), payload["digest"]).payload()

    publisher = AtomicHeadPublisher(
        exchange=exchange,
        stage=lambda learner, version: receipt("prepared", version, range(8)),
        activate=lambda version: events.append(("activate", version)) or receipt("activated", version, range(8)),
        rollback=lambda version: events.append(("rollback", version)),
        commit=lambda version: events.append(("commit", version)),
    )
    with pytest.raises(TimeoutError, match="peer activation"):
        publisher(object(), 1)
    assert events == [("activate", 1), ("rollback", 1)]
    assert publisher.current_version == 0


def test_agreement_on_a_different_activated_digest_does_not_commit():
    events = []
    publisher = AtomicHeadPublisher(
        exchange=lambda tag, payload: receipt(
            payload["phase"], payload["version"], range(8, 16), payload["digest"],
        ).payload(),
        stage=lambda learner, version: receipt("prepared", version, range(8)),
        activate=lambda version: receipt("activated", version, range(8), "b" * 64),
        rollback=lambda version: events.append(("rollback", version)),
        commit=lambda version: events.append(("commit", version)),
    )
    with pytest.raises(ValueError, match="does not match the staged head"):
        publisher(object(), 4)
    assert events == [("rollback", 1)]
    assert publisher.current_version == 0
