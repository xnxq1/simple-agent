import asyncio
import json
import random
import sys
from pathlib import Path

# Add project root to path so we can import app module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.di import LLMWithoutToolsType, container
from app.infra.config import Settings
from app.infra.qdrant.repos.repos import QdrantRepo


class Question(BaseModel):
    question: str
    difficulty: str


class QuestionsResult(BaseModel):
    questions: list[Question]


async def get_random_documents(container, limit: int = 25) -> list[str]:
    qdrant_repo = container.get(QdrantRepo)
    settings = container.get(Settings)
    records, _ = await qdrant_repo.client.scroll(
        collection_name=settings.qdrant_collection,
        limit=limit * 5,
        with_payload=True,
        with_vectors=False,
    )
    sampled = random.sample(records, min(limit, len(records)))
    return [r.payload.get("text", "") for r in sampled if r.payload.get("text")]


async def generate_negative_test_questions(
    container, document: str, num_questions: int = 2
) -> QuestionsResult:
    llm_client = container.get(LLMWithoutToolsType)
    prompt = f"""
На основе следующего документа сгенерируй {num_questions} вопросов,
на которые НЕЛЬЗЯ ответить, используя только этот документ.

Документ:
{document}

Требования:
- Вопросы должны быть тематически близки к документу, но выходить за его рамки
- Вопросы не должны иметь ответа в тексте документа
- Для каждого вопроса оцени уровень сложности: "easy", "medium" или "hard"
- Все вопросы должны быть на русском языке

Формат ответа (JSON):
{{
  "questions": [
    {{
      "question": "Текст вопроса, на который нет ответа в документе?",
      "difficulty": "medium"
    }},
    {{
      "question": "Ещё один вопрос вне контекста документа?",
      "difficulty": "hard"
    }}
  ]
}}
"""
    result: QuestionsResult = await llm_client.completions_create(
        system_prompt="Ты эксперт в создании оценочных вопросов для RAG систем.",
        messages=[HumanMessage(content=prompt)],
        response_class=QuestionsResult,
    )
    return result


async def main():
    documents = await get_random_documents(container, limit=25)
    dataset = []
    for i, doc in enumerate(documents):
        try:
            result = await generate_negative_test_questions(container, doc)
            dataset.append(
                {
                    "document": doc,
                    "questions": [q.model_dump() for q in result.questions],
                }
            )
            print(
                f"[{i + 1}/{len(documents)}] generated {len(result.questions)} questions"
            )
        except Exception as e:
            print(f"[{i + 1}/{len(documents)}] failed: {e}")

    output_path = "scripts/datasets/json/negative_dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(dataset)} entries to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
