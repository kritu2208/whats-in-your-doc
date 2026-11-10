from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import Conversation, Document, Message


class ConversationNotFoundError(LookupError):
    """Raised when a requested conversation does not exist."""


class DocumentsNotFoundError(LookupError):
    """Raised when one or more requested documents are unavailable."""


class ChatRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def validate_documents(self, document_ids: list[uuid.UUID] | None) -> None:
        if not document_ids:
            return
        requested = set(document_ids)
        statement = select(Document.id).where(
            Document.id.in_(requested),
            Document.status == "ready",
        )
        available = set(self.session.scalars(statement).all())
        if available != requested:
            raise DocumentsNotFoundError(
                "One or more selected documents do not exist or are not ready."
            )

    def get_conversation(self, conversation_id: uuid.UUID) -> Conversation:
        conversation = self.session.get(Conversation, conversation_id)
        if conversation is None:
            raise ConversationNotFoundError("Conversation not found.")
        return conversation

    def get_recent_history(self, conversation_id: uuid.UUID) -> list[dict[str, str]]:
        limit = get_settings().chat_history_messages
        if limit == 0:
            return []
        statement = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.role.in_(("user", "assistant")),
            )
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
        )
        messages = self.session.scalars(statement).all()
        return [
            {"role": message.role, "content": message.content}
            for message in reversed(messages)
        ]

    def persist_exchange(
        self,
        *,
        conversation_id: uuid.UUID | None,
        question: str,
        answer: str,
        citations: list[dict[str, Any]],
    ) -> uuid.UUID:
        if conversation_id is None:
            conversation = Conversation(title=question[:255])
            self.session.add(conversation)
            self.session.flush()
        else:
            conversation = self.get_conversation(conversation_id)

        self.session.add_all(
            [
                Message(
                    conversation_id=conversation.id,
                    role="user",
                    content=question,
                    citations=None,
                ),
                Message(
                    conversation_id=conversation.id,
                    role="assistant",
                    content=answer,
                    citations=citations,
                ),
            ]
        )
        self.session.commit()
        return conversation.id
