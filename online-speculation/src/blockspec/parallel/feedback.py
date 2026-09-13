"""Reached-prefix feedback and post-correction publication of online updates."""

from ..feedback import Feedback


class OnlineFeedback:
    def __init__(self, *, learner):
        self.learner = learner

    def begin(self, prompt):
        owner = self.learner
        self.initial = (owner.updates, owner.update_seconds, owner.feedback_blocks, owner.coverage_skips)
        owner.clear_replay()

    @property
    def capture_layer(self):
        return self.learner.capture_layer if self.learner.needs_decoder_feedback else None

    def commit(self, tokens):
        pass

    def observe(self, proposal, teacher_logits, target, *, used, fully_covered, done):
        if proposal.collect_feedback:
            feedback = Feedback(proposal.draft_inputs, proposal.draft_cache, teacher_logits[:used],
                                used, proposal.boundary, fully_covered)
            self.learner.observe(feedback, may_update=not done)
        else:
            self.learner._skip_decoder_feedback(used)

    def finish(self, result):
        self.learner.clear_replay()
        for key, start in zip(("updates", "update_seconds", "feedback_blocks", "coverage_skips"), self.initial):
            setattr(result, key, getattr(self.learner, key) - start)
