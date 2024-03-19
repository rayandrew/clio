from dataclasses import dataclass

import numpy as np

from clio.utils.general import ratio_to_percentage, ratio_to_percentage_str
from clio.utils.indented_file import IndentedFile
from clio.utils.logging import log_get

log = log_get(__name__)


def get_confidence_data(probabilities: np.ndarray, threshold: float = 0.5, confidence_threshold: float = 0.1):
    high_confidence_idx = np.where(np.abs(probabilities - threshold) >= confidence_threshold)[0]
    low_confidence_idx = np.where(np.abs(probabilities - threshold) < confidence_threshold)[0]
    return high_confidence_idx, low_confidence_idx


@dataclass
class ConfidenceResult:
    high_confidence_indices: np.ndarray
    low_confidence_indices: np.ndarray
    best_case_indices: np.ndarray
    worst_case_indices: np.ndarray
    lucky_case_indices: np.ndarray
    clueless_case_indices: np.ndarray

    # high confidence
    high_conf_incorrect_reject_indices: np.ndarray
    high_conf_incorrect_accept_indices: np.ndarray
    high_conf_correct_reject_indices: np.ndarray
    high_conf_correct_accept_indices: np.ndarray

    # low confidence
    low_conf_incorrect_reject_indices: np.ndarray
    low_conf_incorrect_accept_indices: np.ndarray
    low_conf_correct_reject_indices: np.ndarray
    low_conf_correct_accept_indices: np.ndarray

    @property
    def num_data(self):
        return len(self.high_confidence_indices) + len(self.low_confidence_indices)

    @property
    def high_confidence(self):
        return len(self.high_confidence_indices)

    @property
    def low_confidence(self):
        return len(self.low_confidence_indices)

    @property
    def best_case(self):
        return len(self.best_case_indices)

    @property
    def worst_case(self):
        return len(self.worst_case_indices)

    @property
    def lucky_case(self):
        return len(self.lucky_case_indices)

    @property
    def clueless_case(self):
        return len(self.clueless_case_indices)

    @property
    def high_conf_incorrect_reject(self):
        return len(self.high_conf_incorrect_reject_indices)

    @property
    def high_conf_incorrect_accept(self):
        return len(self.high_conf_incorrect_accept_indices)

    @property
    def high_conf_correct_reject(self):
        return len(self.high_conf_correct_reject_indices)

    @property
    def high_conf_correct_accept(self):
        return len(self.high_conf_correct_accept_indices)

    @property
    def low_conf_incorrect_reject(self):
        return len(self.low_conf_incorrect_reject_indices)

    @property
    def low_conf_incorrect_accept(self):
        return len(self.low_conf_incorrect_accept_indices)

    @property
    def low_conf_correct_reject(self):
        return len(self.low_conf_correct_reject_indices)

    @property
    def low_conf_correct_accept(self):
        return len(self.low_conf_correct_accept_indices)

    @property
    def high_conf_incorrect_reject_ratio(self) -> float:
        return self.high_conf_incorrect_reject / self.high_confidence if self.high_confidence else 0.0

    @property
    def high_conf_incorrect_accept_ratio(self) -> float:
        return self.high_conf_incorrect_accept / self.high_confidence if self.high_confidence else 0.0

    @property
    def high_conf_correct_reject_ratio(self) -> float:
        return self.high_conf_correct_reject / self.high_confidence if self.high_confidence else 0.0

    @property
    def high_conf_correct_accept_ratio(self) -> float:
        return self.high_conf_correct_accept / self.high_confidence if self.high_confidence else 0.0

    @property
    def low_conf_incorrect_reject_ratio(self) -> float:
        return self.low_conf_incorrect_reject / self.low_confidence if self.low_confidence else 0.0

    @property
    def low_conf_incorrect_accept_ratio(self) -> float:
        return self.low_conf_incorrect_accept / self.low_confidence if self.low_confidence else 0.0

    @property
    def low_conf_correct_reject_ratio(self) -> float:
        return self.low_conf_correct_reject / self.low_confidence if self.low_confidence else 0.0

    @property
    def low_conf_correct_accept_ratio(self) -> float:
        return self.low_conf_correct_accept / self.low_confidence if self.low_confidence else 0.0

    @property
    def high_confidence_ratio(self) -> float:
        return len(self.high_confidence_indices) / self.num_data if self.num_data else 0.0

    @property
    def low_confidence_ratio(self) -> float:
        return len(self.low_confidence_indices) / self.num_data if self.num_data else 0.0

    @property
    def best_case_ratio(self) -> float:
        return len(self.best_case_indices) / self.num_data if self.num_data else 0.0

    @property
    def worst_case_ratio(self):
        return len(self.worst_case_indices) / self.num_data if self.num_data else 0.0

    @property
    def lucky_case_ratio(self):
        return len(self.lucky_case_indices) / self.num_data if self.num_data else 0.0

    @property
    def clueless_case_ratio(self):
        return len(self.clueless_case_indices) / self.num_data if self.num_data else 0.0

    def as_dict(self):
        return {
            "high_confidence": self.high_confidence,
            "low_confidence": self.low_confidence,
            "percent_high_confidence": ratio_to_percentage(self.high_confidence_ratio),
            "percent_low_confidence": ratio_to_percentage(self.low_confidence_ratio),
            # cases
            "best_case": self.best_case,
            "worst_case": self.worst_case,
            "lucky_case": self.lucky_case,
            "clueless_case": self.clueless_case,
            # cases percentage
            "percent_best_case": ratio_to_percentage(self.best_case_ratio),
            "percent_worst_case": ratio_to_percentage(self.worst_case_ratio),
            "percent_lucky_case": ratio_to_percentage(self.lucky_case_ratio),
            "percent_clueless_case": ratio_to_percentage(self.clueless_case_ratio),
            # high confidence
            "high_conf_incorrect_reject": self.high_conf_incorrect_reject,
            "high_conf_incorrect_accept": self.high_conf_incorrect_accept,
            "high_conf_correct_reject": self.high_conf_correct_reject,
            "high_conf_correct_accept": self.high_conf_correct_accept,
            # low confidence
            "low_conf_incorrect_reject": self.low_conf_incorrect_reject,
            "low_conf_incorrect_accept": self.low_conf_incorrect_accept,
            "low_conf_correct_reject": self.low_conf_correct_reject,
            "low_conf_correct_accept": self.low_conf_correct_accept,
            # high confidence percentage
            "percent_high_conf_incorrect_reject": ratio_to_percentage(self.high_conf_incorrect_reject_ratio),
            "percent_high_conf_incorrect_accept": ratio_to_percentage(self.high_conf_incorrect_accept_ratio),
            "percent_high_conf_correct_reject": ratio_to_percentage(self.high_conf_correct_reject_ratio),
            "percent_high_conf_correct_accept": ratio_to_percentage(self.high_conf_correct_accept_ratio),
            # low confidence percentage
            "percent_low_conf_incorrect_reject": ratio_to_percentage(self.low_conf_incorrect_reject_ratio),
            "percent_low_conf_incorrect_accept": ratio_to_percentage(self.low_conf_incorrect_accept_ratio),
            "percent_low_conf_correct_reject": ratio_to_percentage(self.low_conf_correct_reject_ratio),
            "percent_low_conf_correct_accept": ratio_to_percentage(self.low_conf_correct_accept_ratio),
        }

    def __str__(self):
        percent_high_confidence = ratio_to_percentage_str(self.high_confidence_ratio)
        percent_low_confidence = ratio_to_percentage_str(self.low_confidence_ratio)
        percent_best_case = ratio_to_percentage_str(self.best_case_ratio)
        percent_worst_case = ratio_to_percentage_str(self.worst_case_ratio)
        percent_lucky_case = ratio_to_percentage_str(self.lucky_case_ratio)
        percent_clueless_case = ratio_to_percentage_str(self.clueless_case_ratio)
        # high confidence
        percent_high_conf_incorrect_reject = ratio_to_percentage_str(self.high_conf_incorrect_reject_ratio)
        percent_high_conf_incorrect_accept = ratio_to_percentage_str(self.high_conf_incorrect_accept_ratio)
        percent_high_conf_correct_reject = ratio_to_percentage_str(self.high_conf_correct_reject_ratio)
        percent_high_conf_correct_accept = ratio_to_percentage_str(self.high_conf_correct_accept_ratio)
        # low confidence
        percent_low_conf_incorrect_reject = ratio_to_percentage_str(self.low_conf_incorrect_reject_ratio)
        percent_low_conf_incorrect_accept = ratio_to_percentage_str(self.low_conf_incorrect_accept_ratio)
        percent_low_conf_correct_reject = ratio_to_percentage_str(self.low_conf_correct_reject_ratio)
        percent_low_conf_correct_accept = ratio_to_percentage_str(self.low_conf_correct_accept_ratio)

        return (
            "High Confidence: %s (%s), Low Confidence: %s (%s), Best Case: %s (%s), Worst Case: %s (%s), Lucky Case: %s (%s), Clueless Case: %s (%s), High Confidence Incorrect Reject: %s (%s), High Confidence Incorrect Accept: %s (%s), High Confidence Correct Reject: %s (%s), High Confidence Correct Accept: %s (%s), Low Confidence Incorrect Reject: %s (%s), Low Confidence Incorrect Accept: %s (%s), Low Confidence Correct Reject: %s (%s), Low Confidence Correct Accept: %s (%s)"
            % (
                self.high_confidence,
                percent_high_confidence,
                self.low_confidence,
                percent_low_confidence,
                self.best_case,
                percent_best_case,
                self.worst_case,
                percent_worst_case,
                self.lucky_case,
                percent_lucky_case,
                self.clueless_case,
                percent_clueless_case,
                # high confidence
                self.high_conf_incorrect_reject,
                percent_high_conf_incorrect_reject,
                self.high_conf_incorrect_accept,
                percent_high_conf_incorrect_accept,
                self.high_conf_correct_reject,
                percent_high_conf_correct_reject,
                self.high_conf_correct_accept,
                percent_high_conf_correct_accept,
                # low confidence
                self.low_conf_incorrect_reject,
                percent_low_conf_incorrect_reject,
                self.low_conf_incorrect_accept,
                percent_low_conf_incorrect_accept,
                self.low_conf_correct_reject,
                percent_low_conf_correct_reject,
                self.low_conf_correct_accept,
                percent_low_conf_correct_accept,
            )
        )

    def to_indented_file(self, file: IndentedFile):
        with file.section("Best Case"):
            file.writeln("Description: High confidence and correct prediction")
            file.writeln("Total: %s", self.best_case)
            file.writeln("Percentage: %s", ratio_to_percentage_str(self.best_case_ratio))
        with file.section("Worst Case"):
            file.writeln("Description: High confidence and incorrect prediction")
            file.writeln("Total: %s", self.worst_case)
            file.writeln("Percentage: %s", ratio_to_percentage_str(self.worst_case_ratio))
        with file.section("Lucky Case"):
            file.writeln("Description: Low confidence and correct prediction")
            file.writeln("Total: %s", self.lucky_case)
            file.writeln("Percentage: %s", ratio_to_percentage_str(self.lucky_case_ratio))
        with file.section("Clueless Case"):
            file.writeln("Description: Low confidence and incorrect prediction")
            file.writeln("Total: %s", self.clueless_case)
            file.writeln("Percentage: %s", ratio_to_percentage_str(self.clueless_case_ratio))
        with file.section("High Confidence"):
            file.writeln("Total: %s", self.high_confidence)
            file.writeln("Percentage: %s", ratio_to_percentage_str(self.high_confidence_ratio))
            with file.section("High Confidence Incorrect Reject"):
                file.writeln("Total: %s", self.high_conf_incorrect_reject)
                file.writeln("Percentage: %s", ratio_to_percentage_str(self.high_conf_incorrect_reject_ratio))
            with file.section("High Confidence Incorrect Accept"):
                file.writeln("Total: %s", self.high_conf_incorrect_accept)
                file.writeln("Percentage: %s", ratio_to_percentage_str(self.high_conf_incorrect_accept_ratio))
            with file.section("High Confidence Correct Reject"):
                file.writeln("Total: %s", self.high_conf_correct_reject)
                file.writeln("Percentage: %s", ratio_to_percentage_str(self.high_conf_correct_reject_ratio))
            with file.section("High Confidence Correct Accept"):
                file.writeln("Total: %s", self.high_conf_correct_accept)
                file.writeln("Percentage: %s", ratio_to_percentage_str(self.high_conf_correct_accept_ratio))
        with file.section("Low Confidence"):
            file.writeln("Total: %s", self.low_confidence)
            file.writeln("Percentage: %s", ratio_to_percentage_str(self.low_confidence_ratio))
            with file.section("Low Confidence Incorrect Reject"):
                file.writeln("Total: %s", self.low_conf_incorrect_reject)
                file.writeln("Percentage: %s", ratio_to_percentage_str(self.low_conf_incorrect_reject_ratio))
            with file.section("Low Confidence Incorrect Accept"):
                file.writeln("Total: %s", self.low_conf_incorrect_accept)
                file.writeln("Percentage: %s", ratio_to_percentage_str(self.low_conf_incorrect_accept_ratio))
            with file.section("Low Confidence Correct Reject"):
                file.writeln("Total: %s", self.low_conf_correct_reject)
                file.writeln("Percentage: %s", ratio_to_percentage_str(self.low_conf_correct_reject_ratio))
            with file.section("Low Confidence Correct Accept"):
                file.writeln("Total: %s", self.low_conf_correct_accept)
                file.writeln("Percentage: %s", ratio_to_percentage_str(self.low_conf_correct_accept_ratio))


def get_confidence_cases(
    labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5, confidence_threshold: float = 0.1
) -> ConfidenceResult:
    assert len(labels) == len(predictions) == len(probabilities), "sanity check, all data should have the same length"
    ###########################################################################
    # to make sure we pick the right data: assess the accuracy of that specific confidence data
    #
    #
    # MODEL PERF ^
    #            |-------------------|
    #            |    I    |   II    |
    #            |-------------------|
    #            |   III   |   IV    |
    #            --------------------->
    #                              CONFIDENCE
    #
    # I.   if the prediction of that data is correct, but the confidence is low
    #      -> LUCKY/BLURRY CASE
    #      -> How to increase the confidence of that data?
    #           -> admit mentee to help that PhD student more confident to make conclusions
    # II.  if the prediction of that data is right and the confidence is high
    #      -> BEST CASE
    # III. if the prediction of that data is wrong and the confidence is low
    #      -> CLUELESS CASE
    # IV.  if the prediction of that data is wrong, but the confidence is high
    #      -> WORST CASE
    ###########################################################################

    labels = labels.copy()
    predictions = predictions.copy()

    log.debug("Assessing confidence and model performance", tab=2)
    high_conf_indices, low_conf_indices = get_confidence_data(probabilities, threshold=threshold, confidence_threshold=confidence_threshold)

    labels_high_conf = labels[high_conf_indices]
    labels_low_conf = labels[low_conf_indices]
    predictions_high_conf = predictions[high_conf_indices]
    predictions_low_conf = predictions[low_conf_indices]

    # check high confidence data
    log.debug("Calculating high confidence data", tab=2)
    high_conf_correct_indices = np.where(labels_high_conf == predictions_high_conf)[0]
    high_conf_correct_indices = high_conf_indices[high_conf_correct_indices]

    high_conf_incorrect_indices = np.where(labels_high_conf != predictions_high_conf)[0]
    high_conf_incorrect_indices = high_conf_indices[high_conf_incorrect_indices]

    # check low confidence data
    log.debug("Calculating low confidence data", tab=2)
    low_conf_correct_indices = np.where(labels_low_conf == predictions_low_conf)[0]
    low_conf_correct_indices = low_conf_indices[low_conf_correct_indices]
    low_conf_incorrect_indices = np.where(labels_low_conf != predictions_low_conf)[0]
    low_conf_incorrect_indices = low_conf_indices[low_conf_incorrect_indices]

    log.debug("Low confidence data", tab=2)
    log.debug("Total number of data: %s", len(labels), tab=3)
    log.debug("Number of low confidence data: %s", len(low_conf_indices), tab=3)
    if len(low_conf_indices) > 5:
        log.debug("Sample of low confidence data indices: %s", np.random.choice(low_conf_indices, 5), tab=3)
    else:
        log.debug("Sample of low confidence data indices: %s", low_conf_indices, tab=3)

    # Calculating model performance metrics

    ### HIGH CONFIDENCE ###
    # get number of reject (1) that is predicted incorrectly BUT with high confidence
    high_conf_incorrect_reject_indices = np.where((labels_high_conf == 1) & (predictions_high_conf == 0))[0]
    high_conf_incorrect_reject_indices = high_conf_indices[high_conf_incorrect_reject_indices]
    # get number of accept (0) that is predicted incorrectly BUT with high confidence
    high_conf_incorrect_accept_indices = np.where((labels_high_conf == 0) & (predictions_high_conf == 1))[0]
    high_conf_incorrect_accept_indices = high_conf_indices[high_conf_incorrect_accept_indices]
    # get number of reject (1) that is predicted correctly with high confidence
    high_conf_correct_reject_indices = np.where((labels_high_conf == 1) & (predictions_high_conf == 1))[0]
    high_conf_correct_reject_indices = high_conf_indices[high_conf_correct_reject_indices]

    # get number of accept (0) that is predicted correctly with high confidence
    high_conf_correct_accept_indices = np.where((labels_high_conf == 0) & (predictions_high_conf == 0))[0]
    high_conf_correct_accept_indices = high_conf_indices[high_conf_correct_accept_indices]

    ### LOW CONFIDENCE ###
    # get number of reject (1) that is predicted incorrectly BUT with low confidence
    low_conf_incorrect_reject_indices = np.where((labels_low_conf == 1) & (predictions_low_conf == 0))[0]
    low_conf_incorrect_reject_indices = low_conf_indices[low_conf_incorrect_reject_indices]
    # get number of accept (0) that is predicted incorrectly BUT with low confidence
    low_conf_incorrect_accept_indices = np.where((labels_low_conf == 0) & (predictions_low_conf == 1))[0]
    low_conf_incorrect_accept_indices = low_conf_indices[low_conf_incorrect_accept_indices]
    # get number of reject (1) that is predicted correctly with low confidence
    low_conf_correct_reject_indices = np.where((labels_low_conf == 1) & (predictions_low_conf == 1))[0]
    low_conf_correct_reject_indices = low_conf_indices[low_conf_correct_reject_indices]
    # get number of accept (0) that is predicted correctly with low confidence
    low_conf_correct_accept_indices = np.where((labels_low_conf == 0) & (predictions_low_conf == 0))[0]
    low_conf_correct_accept_indices = low_conf_indices[low_conf_correct_accept_indices]

    assert len(labels) == len(high_conf_incorrect_reject_indices) + len(high_conf_incorrect_accept_indices) + len(high_conf_correct_reject_indices) + len(
        high_conf_correct_accept_indices
    ) + len(low_conf_incorrect_reject_indices) + len(low_conf_incorrect_accept_indices) + len(low_conf_correct_reject_indices) + len(
        low_conf_correct_accept_indices
    ), "sanity check, all data should be accounted for"

    return ConfidenceResult(
        high_confidence_indices=high_conf_indices,
        low_confidence_indices=low_conf_indices,
        best_case_indices=high_conf_correct_indices,
        worst_case_indices=high_conf_incorrect_indices,
        lucky_case_indices=low_conf_correct_indices,
        clueless_case_indices=low_conf_incorrect_indices,
        high_conf_incorrect_reject_indices=high_conf_incorrect_reject_indices,
        high_conf_incorrect_accept_indices=high_conf_incorrect_accept_indices,
        high_conf_correct_reject_indices=high_conf_correct_reject_indices,
        high_conf_correct_accept_indices=high_conf_correct_accept_indices,
        low_conf_incorrect_reject_indices=low_conf_incorrect_reject_indices,
        low_conf_incorrect_accept_indices=low_conf_incorrect_accept_indices,
        low_conf_correct_reject_indices=low_conf_correct_reject_indices,
        low_conf_correct_accept_indices=low_conf_correct_accept_indices,
    )


__all__ = ["get_confidence_cases", "get_confidence_data", "ConfidenceResult"]
