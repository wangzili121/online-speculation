"""Online feedback lifecycle. Updates occur after a round's immutable proposal is checked."""

from blockspec.feedback import Feedback


class OnlineFeedback:
    def __init__(self, *, learner=None, calibrator=None):
        self.learner, self.calibrator = learner, calibrator
        self.continuation = getattr(calibrator, "kind", None) == "continuation"

    def begin(self, prompt):
        owner = self.calibrator if self.calibrator is not None else self.learner
        self.initial = (getattr(owner, "updates", 0), getattr(owner, "update_seconds", 0.0),
                        getattr(owner, "feedback_blocks", 0), getattr(owner, "coverage_skips", 0))
        if self.continuation:
            self.calibrator.begin_request(prompt[0].tolist())
        if self.learner is not None:
            self.learner.clear_replay()

    @property
    def capture_layer(self):
        if self.learner is not None and self.learner.needs_decoder_feedback:
            return self.learner.capture_layer
        return None

    def commit(self, tokens):
        if self.continuation:
            self.calibrator.commit(tokens)

    def observe(self, proposal, teacher_logits, target, *, used, fully_covered, done):
        if self.learner is not None:
            if proposal.collect_feedback:
                feedback = Feedback(proposal.draft_inputs, proposal.draft_cache, teacher_logits[:used],
                                    used, proposal.boundary, fully_covered)
                self.learner.observe(feedback, may_update=not done)
            else:
                self.learner._skip_decoder_feedback(used)
        if self.calibrator is not None:
            kwargs = {"root": proposal.guaranteed[0]} if self.continuation else {}
            self.calibrator.observe(proposal.auxiliary_feedback, target[:used], **kwargs)

    def finish(self, result):
        if self.learner is not None:
            self.learner.clear_replay()
        owner = self.calibrator if self.calibrator is not None else self.learner
        result.updates = getattr(owner, "updates", 0) - self.initial[0]
        result.update_seconds = getattr(owner, "update_seconds", 0.0) - self.initial[1]
        result.feedback_blocks = getattr(owner, "feedback_blocks", 0) - self.initial[2]
        result.coverage_skips = getattr(owner, "coverage_skips", 0) - self.initial[3]
