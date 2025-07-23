from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from query import answer_question
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/query")
async def query_endpoint(question_request: QuestionRequest):
    try:
        logger.info(f"Получен вопрос: {question_request.question}")
        answer = answer_question(question_request.question)
        logger.info("Ответ успешно сгенерирован")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Произошла ошибка при обработке вашего запроса.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
