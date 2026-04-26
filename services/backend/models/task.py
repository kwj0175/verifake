from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class TaskSchema(BaseModel):
    task_id: str
    status: str = "PENDING"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    # AI 추출값 통합 결과 (fake_prob, uncertainty 등 반영) [cite: 5, 6, 200]
    result: Optional[Dict[str, Any]] = None
    # 지원하지 않는 형식(limits.unsupported_reason) [cite: 36, 200]
    unsupported_reason: Optional[str] = None
    # 분석 근거 부족(limits.low_evidence_reason) [cite: 37, 200]
    low_evidence_reason: Optional[str] = None
