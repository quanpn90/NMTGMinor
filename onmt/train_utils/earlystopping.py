from enum import Enum


class PatienceEnum(Enum):
    IMPROVING = 0
    DECREASING = 1
    STOPPED = 2


class Scorer(object):
    def __init__(self, best_score, name):
        self.best_score = best_score
        self.name = name

    def is_improving(self, stats):
        raise NotImplementedError()

    def is_decreasing(self, stats):
        raise NotImplementedError()

    def update(self, stats):
        self.best_score = self._caller(stats)

    def __call__(self, stats, **kwargs):
        return self._caller(stats)

    def _caller(self, stats):
        raise NotImplementedError()


class PPLScorer(Scorer):

    def __init__(self):
        super(PPLScorer, self).__init__(float("inf"), "ppl")

    def is_improving(self, stats_score):
        return stats_score < self.best_score

    def is_decreasing(self, stats_score):
        return stats_score > self.best_score

    def _caller(self, stats_score):
        return stats_score


class AccuracyScorer(Scorer):

    def __init__(self):
        super(AccuracyScorer, self).__init__(float("-inf"), "acc")

    def is_improving(self, stats):
        return stats.accuracy() > self.best_score

    def is_decreasing(self, stats):
        return stats.accuracy() < self.best_score

    def _caller(self, stats):
        return stats.accuracy()


DEFAULT_SCORERS = [PPLScorer(), AccuracyScorer()]


SCORER_BUILDER = {
    "ppl": PPLScorer,
    "accuracy": AccuracyScorer
}


def scorers_from_opts(opt):
    if opt.early_stopping_criteria is None:
        return DEFAULT_SCORERS
    else:
        scorers = []
        for criterion in set(opt.early_stopping_criteria):
            assert criterion in SCORER_BUILDER.keys(), "Criterion {} not found".format(criterion)
            scorers.append(SCORER_BUILDER[criterion]())
        return scorers


class EarlyStopping(object):

    def __init__(self, tolerance, scorers=DEFAULT_SCORERS):
        """
            Callable class to keep track of early stopping.
            Args:
                tolerance(int): number of validation steps without improving
                scorer(fn): list of scorers to validate performance on dev
        """

        self.tolerance = tolerance
        self.stalled_tolerance = self.tolerance
        self.current_tolerance = self.tolerance
        self.early_stopping_scorers = scorers  # PPLScorer
        self.status = PatienceEnum.IMPROVING  # initial status: IMPROVING
        self.current_step_best = 0

    def __call__(self, valid_score, step):
        """
            Update the internal state of early stopping mechanism, whether to
        continue training or stop the train procedure.
            Checks whether the scores from all pre-chosen scorers improved. If
        every metric improve, then the status is switched to improving and the
        tolerance is reset. If every metric deteriorate, then the status is
        switched to decreasing and the tolerance is also decreased; if the
        tolerance reaches 0, then the status is changed to stopped.
        Finally, if some improved and others not, then it's considered stalled;
        after tolerance number of stalled, the status is switched to stopped.
        :param valid_stats: Statistics of dev set

            For NMT we currently only have ppl as the metric.
        """

        if self.status == PatienceEnum.STOPPED:
            # Don't do anything
            return

        if all([scorer.is_improving(valid_score) for scorer
                in self.early_stopping_scorers]):
            self._update_increasing(valid_score, step)

        elif all([scorer.is_decreasing(valid_score) for scorer
                  in self.early_stopping_scorers]):
            self._update_decreasing()

        else:
            self._update_stalled()

    def _update_stalled(self):
        self.stalled_tolerance -= 1

        print("Stalled patience: {}/{}".format(self.stalled_tolerance, self.tolerance))

        if self.stalled_tolerance == 0:
            print("Training finished after stalled validations. Early Stop!")

            self._log_best_step()

        self._decreasing_or_stopped_status_update(self.stalled_tolerance)

    def _update_increasing(self, valid_score, step):
        self.current_step_best = step
        for scorer in self.early_stopping_scorers:
            # print("Model is improving {}: {:g} --> {:g}.".format(scorer.name, scorer.best_score, scorer(valid_score)))

            # Update best score of each criteria
            scorer.update(valid_score)

        # Reset tolerance
        self.current_tolerance = self.tolerance
        self.stalled_tolerance = self.tolerance

        # Update current status
        self.status = PatienceEnum.IMPROVING

    def _update_decreasing(self):
        # Decrease tolerance
        self.current_tolerance -= 1

        # Log
        print("Decreasing patience: {}/{}".format(self.current_tolerance, self.tolerance))

        # Log
        if self.current_tolerance == 0:
            print("Training finished after not improving. Early Stop!")
            self._log_best_step()

        self._decreasing_or_stopped_status_update(self.current_tolerance)

    def _log_best_step(self):
        print("Best model found at epoch {}".format(self.current_step_best))

    def _decreasing_or_stopped_status_update(self, tolerance):
        self.status = PatienceEnum.DECREASING \
            if tolerance > 0 \
            else PatienceEnum.STOPPED

    def is_improving(self):
        return self.status == PatienceEnum.IMPROVING

    def has_stopped(self):
        return self.status == PatienceEnum.STOPPED