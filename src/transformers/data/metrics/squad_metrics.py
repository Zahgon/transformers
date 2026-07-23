
import collections
import json
import math
import re
import string

from ...models.bert import BasicTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)


def normalize_answer(s):
    pass


def get_tokens(s):
    pass


def compute_exact(a_gold, a_pred):
    pass


def compute_f1(a_gold, a_pred):
    pass


def get_raw_scores(examples, preds):
    pass


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    pass


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    pass


def merge_eval(main_eval, new_eval, prefix):
    pass


def find_best_thresh_v2(preds, scores, na_probs, qid_to_has_ans):
    pass


def find_all_best_thresh_v2(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    pass


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    pass


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    pass


def squad_evaluate(examples, preds, no_answer_probs=None, no_answer_probability_threshold=1.0):
    pass


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    pass


def _get_best_indexes(logits, n_best_size):
    pass


def _compute_softmax(scores):
    pass


def compute_predictions_logits(
    all_examples,
    all_features,
    all_results,
    n_best_size,
    max_answer_length,
    do_lower_case,
    output_prediction_file,
    output_nbest_file,
    output_null_log_odds_file,
    verbose_logging,
    version_2_with_negative,
    null_score_diff_threshold,
    tokenizer,
):
    pass


def compute_predictions_log_probs(
    all_examples,
    all_features,
    all_results,
    n_best_size,
    max_answer_length,
    output_prediction_file,
    output_nbest_file,
    output_null_log_odds_file,
    start_n_top,
    end_n_top,
    version_2_with_negative,
    tokenizer,
    verbose_logging,
):
    pass
