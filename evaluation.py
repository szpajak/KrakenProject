import re
from collections import Counter

class ResponseEvaluator:
  def __init__(self,
               min_token_threshold = 15,
               echo_threshold = 0.7,
               ideal_length_min = 20,
               ideal_length_max = 150):

    self.min_token_threshold = min_token_threshold
    self.echo_threshold = echo_threshold
    self.ideal_length_min = ideal_length_min
    self.ideal_length_max = ideal_length_max

    self.weights = {
        'keywords': 0.5,
        'length': 0.3,
        'uncertainty': 0.2
    }

    self.uncertain_words = {
        'maybe', 'perhaps', 'possibly',
        'i think', 'could be', 'unclear', 'not sure'
    }

    self.stop_words = {
        'a', 'an', 'the', 'in', 'on', 'at', 'for', 'to',
        'of', 'is', 'am', 'are', 'what','who', 'when',
        'where', 'why', 'how', 'do', 'does', 'did'
    }



  def _tokenize(self, text: str) -> list[str]:
    if not text:
      return []
    return re.findall(r'\b\w+\b', text.lower())

  def _is_empty_or_one_word(self, answer_tokens: list[str]) -> bool:
    return len(answer_tokens) <= 1

  def _is_too_short(self, answer_tokens: list[str]) -> bool:
    return len(answer_tokens) < self.min_token_threshold

  def _is_echoing_question(self,
                           answer_tokens: list[str],
                           question_tokens: list[str]) -> bool:

    common_tokens = set(answer_tokens) & set(question_tokens)
    return len(common_tokens) / len(question_tokens) >= self.echo_threshold

  def _has_key_terms(self,
                     question_tokens: list[str],
                     answer_tokens: list[str]) -> bool:

    key_terms = {token
                 for token in question_tokens
                 if token not in self.stop_words}
    return any(term in answer_tokens for term in key_terms)

  def check_basic_criteria(self,
                           question: str,
                           answer: str) -> tuple[bool, str]:

    question_tokens = self._tokenize(question)
    answer_tokens = self._tokenize(answer)

    if self._is_empty_or_one_word(answer_tokens):
            return False, "Empty or one-word response"
    if self._is_too_short(answer_tokens):
        return False, f"Too short (less than {self.min_token_threshold} tokens)"
    if self._is_echoing_question(question_tokens, answer_tokens):
        return False, "Answer echoes the question"
    if not self._has_key_terms(question_tokens, answer_tokens):
        return False, "Answer lacks key terms from the question"

    return True, "Basic criteria passed"

  def _score_keyword_match(self,
                           question_tokens: list[str],
                           answer_tokens: list[str]) -> float:
    key_terms = {token
                 for token in question_tokens
                 if token not in self.stop_words}

    matched_keywords = key_terms & set(answer_tokens)
    return len(matched_keywords) / len(key_terms)

  def _score_length(self, answer_tokens: list[str]) -> float:
    length = len(answer_tokens)
    if length < self.ideal_length_min or length > self.ideal_length_max:
        return 0.0
    return min(1.0, length / self.ideal_length_min)

  def _score_uncertainty(self, answer_tokens: list[str]) -> float:
    uncertain_count = sum(1 for word in answer_tokens
                          if word in self.uncertain_words)
    return max(0.0, 1.0 - (uncertain_count / 10))

  def score_answer(self, question: str, answer: str) -> dict[str, float]:
    question_tokens = self._tokenize(question)
    answer_tokens = self._tokenize(answer)

    keyword_score = self._score_keyword_match(question_tokens, answer_tokens)
    length_score = self._score_length(answer_tokens)
    uncertainty_score = self._score_uncertainty(answer_tokens)

    final_score = (self.weights['keywords'] * keyword_score +
                   self.weights['length'] * length_score +
                   self.weights['uncertainty'] * uncertainty_score)

    return {"final": round(final_score, 4),
            "keyword": round(keyword_score, 4),
            "length": round(length_score, 4),
            "uncertainty": round(uncertainty_score, 4)}

  def evaluate(self, question: str, answer: str) -> dict:
    basic_criteria, reason = self.check_basic_criteria(question, answer)
    if not basic_criteria:
      return {
          'basic_criteria': False,
          'score' : 0.0,
          'message': f'Basic criteria not met: {reason}'
      }

    score = self.score_answer(question, answer)
    return {
        'basic_criteria': True,
        'final_score': score['final'],
        'keyword_score': score['keyword'],
        'length_score': score['length'],
        'uncertainty_score': score['uncertainty'],
    }