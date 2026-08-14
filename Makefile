# Developer shortcuts. The version lives only in src/ebm/__init__.py
# (hatchling reads it); these targets bump it, commit, and tag in one step.

.PHONY: test lint typecheck cov check bump-patch bump-minor bump-major

test:       ; uv run pytest
lint:       ; uv run ruff check . && uv run ruff format --check .
typecheck:  ; uv run mypy
cov:        ; uv run pytest --cov --cov-report=term-missing
check: lint typecheck test

# make bump-patch   # 0.12.0 -> 0.12.1  (also: bump-minor, bump-major)
# Update CHANGELOG.md first; after bumping, `git push --follow-tags` and cut a
# release with `gh release create vX.Y.Z --title vX.Y.Z`.
bump-patch: ; @$(MAKE) _bump PART=patch
bump-minor: ; @$(MAKE) _bump PART=minor
bump-major: ; @$(MAKE) _bump PART=major

_bump:
	uv run hatch version $(PART)
	@V=$$(uv run hatch version); \
	git commit -am "v$$V"; \
	git tag "v$$V"; \
	echo "bumped to v$$V — run: git push --follow-tags"
