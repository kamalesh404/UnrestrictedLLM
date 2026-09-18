"""Prompt template manager for UnrestrictedLLM."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PromptTemplate:
    """A prompt template with variables."""
    name: str
    template: str
    description: str = ""
    variables: list = field(default_factory=list)

    def render(self, **kwargs) -> str:
        """Render the template with given variables."""
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result

    def validate(self, **kwargs) -> list[str]:
        """Return list of missing variables."""
        return [v for v in self.variables if v not in kwargs]


class TemplateManager:
    """Manages prompt templates."""

    def __init__(self):
        self._templates: dict[str, PromptTemplate] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.register(PromptTemplate(
            name="code_review",
            template="Review this code for bugs and security issues:\n\n```{language}\n{code}\n```",
            description="Code review prompt",
            variables=["language", "code"]
        ))
        self.register(PromptTemplate(
            name="explain_code",
            template="Explain what this code does step by step:\n\n```{language}\n{code}\n```",
            description="Code explanation prompt",
            variables=["language", "code"]
        ))
        self.register(PromptTemplate(
            name="refactor",
            template="Refactor this code to be more {style}:\n\n```{language}\n{code}\n```",
            description="Code refactoring prompt",
            variables=["language", "code", "style"]
        ))

    def register(self, template: PromptTemplate):
        self._templates[template.name] = template

    def get(self, name: str) -> Optional[PromptTemplate]:
        return self._templates.get(name)

    def render(self, name: str, **kwargs) -> str:
        template = self._templates.get(name)
        if not template:
            raise KeyError(f"Template '{name}' not found")
        missing = template.validate(**kwargs)
        if missing:
            raise ValueError(f"Missing variables: {', '.join(missing)}")
        return template.render(**kwargs)

    def list_templates(self) -> list[str]:
        return list(self._templates.keys())
