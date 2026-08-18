from blackbase.catalog import (
    Catalog,
    CatalogEntry,
    load_catalog_paths,
    load_catalog_toml,
    render_catalog_toml,
    write_catalog_shards,
)
from blackbase.project import find_catalog_scope, load_scaffold_catalog_entries


def test_shared_catalog_filters_searches_and_loads_symbols() -> None:
    entry = CatalogEntry(
        key="project.case",
        title="Case",
        kind="case",
        import_path="blackbase.project:CaseRunRequest",
        tags=("runtime", "shared"),
        context_provides=("case.result",),
    )
    catalog = Catalog((entry,))

    assert catalog.show("project.case") is entry
    assert catalog.list(kind="case") == (entry,)
    assert catalog.search("result", fields="context") == (entry,)
    assert entry.load().__name__ == "CaseRunRequest"


def test_catalog_toml_writer_round_trips_nested_contracts(tmp_path) -> None:
    entry = CatalogEntry(
        key="adapter.example",
        title="Example",
        kind="adapter",
        import_path="blackbase.catalog:Catalog",
        tags=("shared", "中文"),
        context_requires=("run.id",),
        contract={"methods": {"propose": True}, "limits": [1, 2.5]},
        metadata={"owner": "test", "optional": None},
    )

    rendered = render_catalog_toml((entry,))
    source = tmp_path / "entry.toml"
    source.write_text(rendered, encoding="utf-8")
    loaded = load_catalog_toml(source)
    assert loaded == (
        CatalogEntry(
            key=entry.key,
            title=entry.title,
            kind=entry.kind,
            import_path=entry.import_path,
            tags=entry.tags,
            context_requires=entry.context_requires,
            contract=entry.contract,
            metadata={"owner": "test"},
        ),
    )

    paths = write_catalog_shards((entry,), tmp_path / "entries")
    assert paths == ((tmp_path / "entries" / "adapter.toml").resolve(),)
    assert load_catalog_paths((tmp_path / "entries",)) == loaded


def test_project_catalog_aggregates_case_shards_with_stable_namespaces(tmp_path) -> None:
    project = tmp_path / "demo"
    (project / "cases").mkdir(parents=True)
    (project / "project_config.py").write_text("STAGES = []\n", encoding="utf-8")
    for name in ("outer", "inner"):
        case = project / "cases" / name
        case.mkdir()
        (case / ".case").write_text(f"name = {name}\nkind = solver\n", encoding="utf-8")
        (case / "build_solver.py").write_text("def build_solver(): return None\n", encoding="utf-8")
        write_catalog_shards(
            (
                CatalogEntry(
                    key="project.solver.build",
                    title="Builder",
                    kind="solver",
                    import_path="build_solver:build_solver",
                    companions=("project.plugin.audit",),
                ),
            ),
            case / "catalog" / "entries",
        )

    scope = find_catalog_scope(project / "cases" / "outer" / "build_solver.py")
    assert scope is not None and scope.kind == "case" and scope.case_name == "outer"
    entries = load_scaffold_catalog_entries(project)
    assert {entry.key for entry in entries} == {
        "project.inner.solver.build",
        "project.outer.solver.build",
    }
    assert {entry.companions[0] for entry in entries} == {
        "project.inner.plugin.audit",
        "project.outer.plugin.audit",
    }
    assert {entry.metadata["case_name"] for entry in entries} == {"inner", "outer"}
    assert {entry.import_path for entry in entries} == {
        "cases.inner.build_solver:build_solver",
        "cases.outer.build_solver:build_solver",
    }


def test_project_catalog_normalizes_repository_qualified_case_import(tmp_path) -> None:
    project = tmp_path / "demo"
    case = project / "cases" / "alpha"
    case.mkdir(parents=True)
    (project / "project_config.py").write_text("STAGES = []\n", encoding="utf-8")
    (case / ".case").write_text("name = alpha\nkind = solver\n", encoding="utf-8")
    (case / "build_solver.py").write_text("def build_solver(): return None\n", encoding="utf-8")
    write_catalog_shards(
        (
            CatalogEntry(
                key="project.example.demo",
                title="Demo",
                kind="example",
                import_path="examples.cases.alpha.cases.alpha.build_solver:build_solver",
            ),
        ),
        case / "catalog" / "entries",
    )

    entries = load_scaffold_catalog_entries(project)

    assert entries[0].import_path == "cases.alpha.build_solver:build_solver"


def test_standalone_case_catalog_preserves_local_import_and_loads_symbol(tmp_path) -> None:
    case = tmp_path / "standalone"
    module_dir = case / "pipeline"
    module_dir.mkdir(parents=True)
    (case / ".case").write_text("name = standalone\nkind = solver\n", encoding="utf-8")
    (case / "build_solver.py").write_text("def build_solver(): return None\n", encoding="utf-8")
    (module_dir / "__init__.py").write_text("", encoding="utf-8")
    (module_dir / "demo.py").write_text("VALUE = 7\n", encoding="utf-8")
    write_catalog_shards(
        (
            CatalogEntry(
                key="pipeline.demo",
                title="Demo",
                kind="representation",
                import_path="pipeline.demo:VALUE",
            ),
        ),
        case / "catalog" / "entries",
    )

    entries = load_scaffold_catalog_entries(case)

    assert entries[0].import_path == "pipeline.demo:VALUE"
    assert "project_root" not in entries[0].metadata
    assert entries[0].load() == 7


def test_catalog_path_loader_rejects_duplicate_keys(tmp_path) -> None:
    entry = CatalogEntry(
        key="duplicate.key",
        title="Duplicate",
        kind="tool",
        import_path="blackbase.catalog:Catalog",
    )
    first = tmp_path / "a.toml"
    second = tmp_path / "b.toml"
    first.write_text(render_catalog_toml((entry,)), encoding="utf-8")
    second.write_text(render_catalog_toml((entry,)), encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="duplicate catalog entry keys"):
        load_catalog_paths((tmp_path,))
