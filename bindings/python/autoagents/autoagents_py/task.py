"""Task abstractions matching the Rust Task struct."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from .types import TaskImagePayload, TaskPayload


class ImageMime(str, Enum):
    JPEG = "jpeg"
    PNG = "png"
    GIF = "gif"
    WEBP = "webp"


@dataclass(slots=True)
class TaskImage:
    mime: Union[ImageMime, str]
    data: bytes

    def to_payload(self) -> TaskImagePayload:
        mime = self.mime.value if isinstance(self.mime, ImageMime) else str(self.mime)
        return {"mime": mime, "data": self.data}


@dataclass(slots=True)
class Task:
    prompt: str
    image: Optional[TaskImage] = None
    system_prompt: Optional[str] = None

    def to_payload(self) -> TaskPayload:
        payload: TaskPayload = {"prompt": self.prompt}
        if self.system_prompt is not None:
            payload["system_prompt"] = self.system_prompt
        if self.image is not None:
            payload["image"] = self.image.to_payload()
        return payload
