PY := .venv/bin/python
.PHONY: install smoke test validate run score judge check-clean all

install:
	uv venv --python 3.11 && uv pip install -e ".[dev]"

smoke:            ## Block 0 go/no-go: models reachable, throughput measured
	$(PY) -m hreval.smoke

test:
	$(PY) -m pytest -q

validate:         ## structural gate on the eval set
	$(PY) -m hreval.scenarios.validate

run:              ## resumable matrix run; MODEL/COND/LIMIT optional
	$(PY) -m hreval.runner $(if $(MODEL),--model $(MODEL)) $(if $(COND),--condition $(COND)) $(if $(LIMIT),--limit $(LIMIT))

score:
	$(PY) -m hreval.scoring.deterministic

judge:
	$(PY) -m hreval.scoring.judge

sample-labels:    ## pick a stratified 60 for blind human labelling
	$(PY) -m hreval.scoring.agreement sample

label:            ## blind labelling CLI (native speaker only)
	$(PY) -m hreval.scoring.agreement label

kappa:            ## judge-vs-human agreement, reported on the held-out test split
	$(PY) -m hreval.scoring.agreement kappa

check-clean:      ## no client data, no secrets, anywhere in history
	@echo "== gitleaks (full history) =="
	@command -v gitleaks >/dev/null || (echo "FAIL: gitleaks not installed - install before first push"; exit 1)
	@gitleaks detect --log-opts="--all" --no-banner || (echo "FAIL: gitleaks found leaks"; exit 1)
	@echo "== forbidden strings =="
	@test -f .forbidden-strings || (echo "FAIL: .forbidden-strings missing (git-ignored; one extended regex per line)"; exit 1)
	@! grep -rniE -f .forbidden-strings --include="*.py" --include="*.json" --include="*.jsonl" --include="*.md" --include="*.toml" . 2>/dev/null || (echo "FAIL: client reference found"; exit 1)
	@echo "== .env not tracked =="
	@! git ls-files --error-unmatch .env 2>/dev/null || (echo "FAIL: .env is tracked"; exit 1)
	@echo "check-clean: OK"

all: test validate run score judge
