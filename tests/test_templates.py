import os
from pathlib import Path

from rules_maker.templates import TemplateEngine
from rules_maker.models import RuleSet, Rule, RuleFormat


def test_template_listing_includes_expected():
    engine = TemplateEngine()
    names = engine.list_templates()
    assert 'cursor_rules.j2' in names
    assert 'windsurf_rules.j2' in names
    assert 'workflow.j2' in names


def test_render_cursor_rules_minimal():
    engine = TemplateEngine()
    ruleset = RuleSet(
        name='Test',
        description='Desc',
        rules=[Rule(id='r1', title='Title', description='Body')],
        format=RuleFormat.CURSOR
    )
    out = engine.render_cursor_rules(ruleset)
    assert 'Test' in out and 'Title' in out


def test_render_windsurf_rules_minimal():
    engine = TemplateEngine()
    ruleset = RuleSet(
        name='WS',
        description='WS Desc',
        rules=[Rule(id='r1', title='Do things', description='Body')],
        format=RuleFormat.WINDSURF
    )
    out = engine.render_windsurf_rules(ruleset)
    assert 'Windsurf Workflow Rules' in out
    assert 'Do things' in out

