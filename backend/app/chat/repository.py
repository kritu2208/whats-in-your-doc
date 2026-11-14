from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

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

    @staticmethod
    def message_sort_key(message: Message) -> tuple[datetime, int, uuid.UUID]:
        return (
            message.created_at,
            0 if message.role == "user" else 1,
            message.id,
        )

    def persist_exchange(
        self,
        *,
        conversation_id: uuid.UUID | None,
        question: str,
        answer: str,
        citations: list[dict[str, Any]],
    ) -> uuid.UUID:
        now = datetime.now(timezone.utc)
        if conversation_id is None:
            conversation = Conversation(title=question[:255])
            self.session.add(conversation)
            self.session.flush()
        else:
            conversation = self.get_conversation(conversation_id)
            conversation.updated_at = now

        self.session.add_all(
            [
                Message(
                    conversation_id=conversation.id,
                    role="user",
                    content=question,
                    citations=None,
                    created_at=now,
                ),
                Message(
                    conversation_id=conversation.id,
                    role="assistant",
                    content=answer,
                    citations=citations,
                    created_at=now + timedelta(milliseconds=1),
                ),
            ]
        )
        self.session.commit()
        return conversation.id

    def list_conversations(
        self, *, offset: int = 0, limit: int = 50
    ) -> list[dict[str, Any]]:
        statement = (
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .order_by(Conversation.updated_at.desc(), Conversation.id)
            .offset(offset)
            .limit(limit)
        )
        conversations = self.session.scalars(statement).all()
        summaries: list[dict[str, Any]] = []
        for conversation in conversations:
            ordered = sorted(
                conversation.messages,
                key=self.message_sort_key,
            )
            last_message = ordered[-1] if ordered else None
            preview = (
                " ".join(last_message.content.split())[:120]
                if last_message is not None
                else ""
            )
            summaries.append(
                {
                    "id": conversation.id,
                    "title": conversation.title or "Untitled chat",
                    "message_count": len(ordered),
                    "preview": preview,
                    "created_at": conversation.created_at,
                    "updated_at": conversation.updated_at,
                }
            )
        return summaries

    def get_conversation_detail(self, conversation_id: uuid.UUID) -> Conversation:
        statement = (
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == conversation_id)
        )
        conversation = self.session.scalar(statement)
        if conversation is None:
            raise ConversationNotFoundError("Conversation not found.")
        return conversation

    def delete_conversation(self, conversation_id: uuid.UUID) -> None:
        conversation = self.get_conversation(conversation_id)
        self.session.delete(conversation)
        self.session.commit()
