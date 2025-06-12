import re
from collections import Counter

class ResponseEvaluator:
  def __init__(self,
               echo_threshold = 0.7,
               min_length = 20,
               max_length = 150):

    self.echo_threshold = echo_threshold
    self.ideal_length_min = min_length
    self.ideal_length_max = max_length

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
    if length < self.ideal_length_min:
        return -1
    if length > self.ideal_length_max:
        return 1
    return 0

  def _score_uncertainty(self, answer_tokens: list[str]) -> float:
    uncertain_count = sum(1 for word in answer_tokens
                          if word in self.uncertain_words)
    return max(0.0, 1.0 - (uncertain_count / 10))

  def _score_context_overlap(self, answer: str, retrieved_context: str) -> float:
      answer_tokens = self._tokenize(answer)
      context_tokens = self._tokenize(retrieved_context)

      def filter_tokens(tokens: list[str]) -> list[str]:
          filtered = []
          for t in tokens:
              if t in self.stop_words:
                  continue
              if len(t) < 3:
                  continue
              filtered.append(t)
          return filtered

      filtered_ans = filter_tokens(answer_tokens)
      filtered_ctx = filter_tokens(context_tokens)

      if not filtered_ans:
          return 0.0

      ans_counter = Counter(filtered_ans)
      ctx_set = set(filtered_ctx)

      matched_count = sum(count for token, count in ans_counter.items() if token in ctx_set)
      total_count = sum(ans_counter.values())

      score = matched_count / total_count

      return max(0.0, min(1.0, score))

  def score_answer(self, question: str, answer: str, context: str) -> dict[str, float]:
    question_tokens = self._tokenize(question)
    answer_tokens = self._tokenize(answer)

    keyword_score = self._score_keyword_match(question_tokens, answer_tokens)
    length_score = self._score_length(answer_tokens)
    uncertainty_score = self._score_uncertainty(answer_tokens)
    context_overlap_score = self._score_context_overlap(answer, context)

    return {
            "keyword": round(keyword_score, 4),
            "length": round(length_score, 4),
            "uncertainty": round(uncertainty_score, 4),
            "context_overlap": round(context_overlap_score, 4)
    }

  def evaluate(self, question: str, answer: str, context: str) -> dict:
    basic_criteria, reason = self.check_basic_criteria(question, answer)
    if not basic_criteria:
      return {
          'basic_criteria': False,
          'score' : 0.0,
          'message': f'Basic criteria not met: {reason}'
      }

    score = self.score_answer(question, answer, context)
    return {
        'basic_criteria': True,
        'keyword_score': score['keyword'],
        'length_score': score['length'],
        'uncertainty_score': score['uncertainty'],
        'context_overlap_score': score['context_overlap'],
    }
