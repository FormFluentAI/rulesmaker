from click.testing import CliRunner

from rules_maker.cli import main


def test_cli_templates_list():
    runner = CliRunner()
    result = runner.invoke(main, ["templates"])
    assert result.exit_code == 0
    assert "Available templates:" in result.output
    assert "cursor_rules.j2" in result.output


def test_cli_templates_show_specific():
    runner = CliRunner()
    # Show an existing template content
    result = runner.invoke(main, ["templates", "--template", "cursor_rules.j2"])
    assert result.exit_code == 0
    assert "Template: cursor_rules.j2" in result.output
    assert "## Rules" in result.output

