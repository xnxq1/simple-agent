from pydantic import BaseModel


class Claim(BaseModel):
    claim: str
    supported: bool
    evidence: str


class GroundnessResult(BaseModel):
    claims: list[Claim]
    faithfulness_score: float


class ContextRelevance(BaseModel):
    query: str
    document: str
    score: float


class ContextRelevanceResult(BaseModel):
    context_relevance: list[ContextRelevance]
    context_score: float


class LLMAnswerRelevance(BaseModel):
    questions: list[str]


class AnswerRelevanceResult(BaseModel):
    questions: list[str]
    score: float


evaluate_groundness_user_prompt = """
Контекст:
{context}

Ответ:
{answer}
"""
evaluate_groundness_system_prompt = """
Вы — эксперт по оценке достоверности ответов.

Разбейте ответ на фактические утверждения и проверьте каждое из них по предоставленному контексту.

ВАЖНЫЕ ПРАВИЛА:
1. Если утверждение точно подтверждено цитатой из контекста → supported: true + цитата
2. Если утверждение НЕ НАЙДЕНО или противоречит контексту → supported: false + "Нет подтверждения"
3. Если есть частичное совпадение, но смысл отличается → supported: false (требуется точное совпадение)

faithfulness_score = (кол-во утверждений с supported=true) / (общее кол-во утверждений)
- Максимальная оценка 1.0 только если ВСЕ утверждения имеют supported=true
- Если хотя бы одно утверждение имеет supported=false, оценка меньше 1.0

Ответ в JSON:
{
  "claims": [
    {"claim": "текст утверждения", "supported": true/false, "evidence": "точная цитата из контекста или 'Нет подтверждения'"}
  ],
  "faithfulness_score": 0.85
}
"""

answer_relevance_system_prompt = """
На основе приведённого ответа сгенерируйте 3 вопроса пользователя, на которые этот ответ мог бы быть корректным ответом.

Требования:
- Вопросы должны быть естественными и реалистичными.
- Они должны быть семантически связаны с ответом.
- Не повторяйте текст ответа дословно.
- Каждый вопрос должен быть понятен без дополнительного контекста.

Верните результат строго в формате JSON:

{
  "questions": ["вопрос 1", "вопрос 2", "вопрос 3"]
}
"""
answer_relevance_user_prompt = """
Ответ:
{answer}

"""
