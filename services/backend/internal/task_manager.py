from datetime import datetime, timedelta
from typing import Dict
from ..models.task import TaskSchema

# 전역 태스크 메모리 DB
tasks_db: Dict[str, TaskSchema] = {}

class TaskManager:
    @staticmethod
    def create_task(task_id: str):
        task = TaskSchema(task_id=task_id)
        tasks_db[task_id] = task
        return task

    @staticmethod
    def initialize_tasks():
        """서버 재시작 시 PROCESSING 상태인 태스크를 FAILED로 초기화 """
        for task in tasks_db.values():
            if task.status in ["PENDING", "PROCESSING"]:
                task.status = "FAILED"
                task.low_evidence_reason = "Server restarted during analysis"

    @staticmethod
    def cleanup_expired_tasks():
        """24시간이 지난 태스크 삭제"""
        now = datetime.now()
        expired_ids = [tid for tid, t in tasks_db.items() if now - t.created_at > timedelta(hours=24)]
        for tid in expired_ids: del tasks_db[tid]
