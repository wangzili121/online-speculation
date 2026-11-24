"""Late binding of DSV4.1 online learning to the resident A3 session.

The native draft, target weights and resident workers do not exist until the
first prefill has initialized the generation.  This object is passed to the
serving session up front, exposes the immutable scheduling policy immediately,
and constructs the real learner exactly once when those resident objects are
available.
"""

from collections.abc import Callable

from .dsv41_device_update import DSV41DeviceSuffixLearner
from .dsv41_online import DSV41OnlineConfig, DSV41OnlineController
from .dsv41_resident_update import DSV41ResidentUpdateBackend
from .dsv41_serving_online import DSV41ServingOnline
from .dsv41_suffix import DSV41SuffixPlan


class DSV41OnlineBootstrap:
    """Build and proxy one resident online learner after native initialization."""

    def __init__(self, primitives_factory: Callable, config=DSV41OnlineConfig()):
        if not callable(primitives_factory):
            raise TypeError("DSV4.1 online primitives factory must be callable")
        # The implemented reverse graph is the C1A suffix (layers 20..39).
        if config.last_layers > 20:
            raise ValueError("resident DSV4.1 online update currently supports last-N <= 20")
        self.primitives_factory = primitives_factory
        self.config = config
        self.plan = DSV41SuffixPlan(config.last_layers)
        self.bridge = self.primitives = self.learner = None

    @property
    def bound(self):
        return self.bridge is not None

    def bind(self, draft, native, cluster=None):
        """Bind once after target weights, resident workers and draft exist."""
        if self.bound:
            raise RuntimeError("DSV4.1 online learner is already bound")
        if draft is None or native is None or getattr(native, "weights", None) is None:
            raise RuntimeError("resident target and draft must exist before online binding")
        draft.enable_online(self.config.last_layers)
        primitives = self.primitives_factory(
            draft=draft,
            native=native,
            cluster=cluster,
            config=self.config,
        )
        backend = DSV41ResidentUpdateBackend(primitives, last_layers=self.config.last_layers)
        learner = DSV41DeviceSuffixLearner(
            backend,
            last_layers=self.config.last_layers,
            learning_rate=self.config.learning_rate,
        )
        if cluster is None:
            # A single-node run stages and flips its local bank at a verifier
            # boundary.  The dual-node path below additionally cross-checks
            # every rank and digest before activation.
            local_head_version = 0

            def publish(owner, _update_number):
                nonlocal local_head_version
                version = local_head_version + 1
                draft.stage_online(owner, version)
                try:
                    draft.activate_online(version)
                    draft.commit_online(version)
                except BaseException:
                    draft.rollback_online(version)
                    raise
                local_head_version = version
        else:
            publish = draft.online_publisher(cluster.link.exchange)
        controller = DSV41OnlineController(learner, self.config, publisher=publish)
        self.primitives, self.learner = primitives, learner
        self.bridge = DSV41ServingOnline(controller)
        return self.bridge

    def _call(self, name, *args, **kwargs):
        if not self.bound:
            raise RuntimeError("DSV4.1 online learner has not reached resident initialization")
        return getattr(self.bridge, name)(*args, **kwargs)

    def capture_due(self):
        return self._call("capture_due")

    def select_capture(self, *args, **kwargs):
        return self._call("select_capture", *args, **kwargs)

    def register_replay(self, *args, **kwargs):
        return self._call("register_replay", *args, **kwargs)

    def verified(self, *args, **kwargs):
        return self._call("verified", *args, **kwargs)

    def continued(self, *args, **kwargs):
        return self._call("continued", *args, **kwargs)

    def synchronize(self, *args, **kwargs):
        return self._call("synchronize", *args, **kwargs)

    def finish(self, *args, **kwargs):
        return self._call("finish", *args, **kwargs)

    def summary(self):
        if not self.bound:
            return {"bound": False, "last_layers": self.config.last_layers}
        return {"bound": True} | self.bridge.summary()
