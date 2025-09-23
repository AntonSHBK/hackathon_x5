import json
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from app.models.openai import OpenAITextModel
from app.settings import settings

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
    responses={500: {"description": "Internal Server Error"}},
)

chat_service = OpenAITextModel(
    api_key=settings.OPENAI_API_KEY,
    model_name="gpt-5-nano",
    max_tokens=1024,
)

# ==========================
# Requests
# ==========================

class Input(BaseModel):
    role: str = Field(..., description="Роль: 'user', 'system', 'assistant' и т.д.")
    content: str = Field(..., description="Текст сообщения")


class TextRequest(BaseModel):
    input: List[Input] = Field(..., description="Список сообщений для модели")

    model_config = {
        "json_schema_extra": {
            "example": {
                "input": [
                    {
                        "role": "user", 
                        "content": "Напиши короткий рассказ про кота"
                    }
                ]
            }
        }
    }
    
class SchemaRequest(BaseModel):
    input: List[Input] = Field(..., description="Список сообщений для модели")
    json_schema: Dict[str, Any] = Field(..., description="JSON Schema для Structured Outputs")

    model_config = {
        "json_schema_extra": {
            "example": {
                "input": [
                    {"role": "user", "content": "Какие симптомы у пациента?"},
                ],
                "json_schema": {
                    "format": {
                        "type": "json_schema",
                        "name": "diagnosis_detection",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "has_diagnosis": {"type": "boolean"},
                                "possible_diagnoses": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["has_diagnosis", "possible_diagnoses"],
                            "additionalProperties": False,
                        },
                        "strict": True
                    }
                }
            }
        }
    }
    
# ==========================
# Responses
# ==========================


class ChatResponse(BaseModel):
    response: str = Field(..., description="Сгенерированный ответ модели")

    model_config = {
        "json_schema_extra": {
            "example": {
                "response": "Жил-был кот по имени Барсик..."
            }
        }
    }
    
    
class SchemaResponse(BaseModel):
    response: dict = Field(..., description="Ответ модели в JSON, соответствующий схеме")


# ==========================
# Endpoints
# ==========================

@router.post(
    "/text",
    response_model=ChatResponse,
    summary="Обычный чат-запрос",
    description="""
    Отправляет список сообщений в модель и возвращает полный ответ целиком.
    
    **Пример**:
    ```json
    {
      "input": [
        {
            "role": "user", 
            "content": "Привет, как дела?"
        }
      ]
    }
    ```
    """,
)
def chat_endpoint(request: TextRequest) -> ChatResponse:
    try:
        resp = chat_service.generate(
            input=[msg.model_dump() for msg in request.input],
            **request.model_dump(exclude={"input"}, exclude_none=True)
        )
        text = getattr(resp, "output_text", None) or str(resp)
        return ChatResponse(response=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/text_stream",
    summary="Потоковый чат-запрос",
    description="""
    Отправляет список сообщений в модель и возвращает ответ **частями** в реальном времени.
    
    **Пример**:
    ```json
    {
      "input": [
        {
            "role": "user",
            "content": "Напиши короткий рассказ про кота"
        }
      ]
    }
    ```
    """,
)
def chat_stream_endpoint(request: TextRequest) -> StreamingResponse:
    """Обработка потокового чата."""
    try:
        stream = chat_service.generate_stream(
            input=[msg.model_dump() for msg in request.input],
            **request.model_dump(exclude={"input"}, exclude_none=True)
        )

        def stream_gen():
            for event in stream:
                if event.type == "response.output_text.delta":
                    # кусочек текста ответа
                    yield f'{ {"delta": event.delta} }\n'
                elif event.type == "response.refusal.delta":
                    # кусочек отказа
                    yield f'{ {"refusal": event.delta} }\n'
                elif event.type == "response.completed":
                    # финальный маркер
                    yield '{"event": "completed"}\n'

        return StreamingResponse(stream_gen(), media_type="application/x-ndjson")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post(
    "/text_schema",
    response_model=SchemaResponse,
    summary="Чат с использованием JSON Schema",
    description="""
    Отправляет список сообщений и JSON Schema в модель.
    Модель возвращает строго структурированный ответ по схеме.

    **Пример запроса**:
    ```json
    {
      "input": [{"role": "user", "content": "У пациента высокая температура, кашель и затруднённое дыхание. Какие симптомы у пациента?"}],
      "json_schema": {
        "format": {
          "type": "json_schema",
          "name": "diagnosis_detection",
          "schema": {
            "type": "object",
            "properties": {
              "has_diagnosis": {"type": "boolean"},
              "possible_diagnoses": {
                "type": "array",
                "items": {"type": "string"}
              }
            },
            "required": ["has_diagnosis", "possible_diagnoses"],
            "additionalProperties": false
          },
          "strict": true
        }
      }
    }
    ```
    """
)
def chat_schema_endpoint(request: SchemaRequest) -> SchemaResponse:
    try:
        resp = chat_service.generate_with_schema(
            input=[msg.model_dump() for msg in request.input],
            json_schema=request.json_schema,
            **request.model_dump(exclude={"input", "json_schema"}, exclude_none=True)
        )
        parsed = json.loads(resp.output_text)
        return SchemaResponse(response=parsed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))