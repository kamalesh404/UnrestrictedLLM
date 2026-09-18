"""Tests for prompt template manager."""

import pytest
from src.utils.templates import PromptTemplate, TemplateManager


class TestPromptTemplate:
    def test_render_basic(self):
        t = PromptTemplate(name="test", template="Hello {name}!", variables=["name"])
        assert t.render(name="World") == "Hello World!"

    def test_render_multiple_vars(self):
        t = PromptTemplate(
            name="test",
            template="{greeting} {name}, today is {day}",
            variables=["greeting", "name", "day"]
        )
        result = t.render(greeting="Hi", name="Alice", day="Monday")
        assert result == "Hi Alice, today is Monday"

    def test_validate_missing(self):
        t = PromptTemplate(name="test", template="{a} {b}", variables=["a", "b"])
        missing = t.validate(a="1")
        assert missing == ["b"]

    def test_validate_all_present(self):
        t = PromptTemplate(name="test", template="{a} {b}", variables=["a", "b"])
        missing = t.validate(a="1", b="2")
        assert missing == []


class TestTemplateManager:
    def test_list_defaults(self):
        mgr = TemplateManager()
        templates = mgr.list_templates()
        assert "code_review" in templates
        assert "explain_code" in templates
        assert "refactor" in templates

    def test_get_template(self):
        mgr = TemplateManager()
        t = mgr.get("code_review")
        assert t is not None
        assert t.name == "code_review"

    def test_get_nonexistent(self):
        mgr = TemplateManager()
        assert mgr.get("nope") is None

    def test_render_template(self):
        mgr = TemplateManager()
        result = mgr.render("code_review", language="python", code="print('hi')")
        assert "python" in result
        assert "print('hi')" in result

    def test_render_missing_variable(self):
        mgr = TemplateManager()
        with pytest.raises(ValueError, match="Missing variables"):
            mgr.render("code_review", language="python")

    def test_render_nonexistent_template(self):
        mgr = TemplateManager()
        with pytest.raises(KeyError, match="nope"):
            mgr.render("nope")

    def test_register_custom(self):
        mgr = TemplateManager()
        mgr.register(PromptTemplate(
            name="custom",
            template="Do {thing}",
            variables=["thing"]
        ))
        assert "custom" in mgr.list_templates()
        assert mgr.render("custom", thing="magic") == "Do magic"
