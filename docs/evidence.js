(function () {
  "use strict";

  const repository = "https://github.com/CmsChase/cf-difficulty-prediction/blob/main";

  const evidenceViews = {
    question: {
      kicker: "FROZEN COMPARISON",
      title: "Does statement structure add signal beyond index?",
      metric: "2 features vs 43 features",
      detail:
        "The comparator sees index_rank and index_number. The primary model adds 41 fixed statement-structure features.",
      boundary:
        "Rating is the target, not an input. Ratings, solved counts, tags, points, and participation signals are forbidden from both input matrices.",
      linkLabel: "Open the frozen configuration",
      href: `${repository}/configs/historical_statement_backtest.json`,
    },
    split: {
      kicker: "FORWARD-TIME SPLIT",
      title: "Complete contest-time buckets stay together.",
      metric: "7,375 / 1,136 / 2,468",
      detail:
        "The 1,581 unique contest start-time buckets were allocated 70/10/20; this yielded 7,375 train, 1,136 validation, and 2,468 test rows.",
      boundary:
        "No contest or shared start timestamp crosses a boundary. Chronology reduces temporal leakage, but the underlying data and pages are still historical.",
      linkLabel: "Read the split protocol",
      href: `${repository}/docs/HISTORICAL_STATEMENT_BACKTEST.md`,
    },
    selection: {
      kicker: "VALIDATION-ONLY SELECTION",
      title: "The final test did not choose the alpha.",
      metric: "α = 0.01 / 0.01",
      detail:
        "Comparator validation MAE was 493.6116; primary validation MAE was 422.2488. Each model selected alpha separately from the frozen grid.",
      boundary:
        "The selection lock records test_evaluated: false together with feature lists, selected alphas, and hashes of prepared inputs.",
      linkLabel: "Inspect the selection lock",
      href: `${repository}/outputs/historical_statement_backtest/selection/selection_lock.json`,
    },
    test: {
      kicker: "LOCKED HISTORICAL TEST",
      title: "Statement structure reduced error on the committed cohort.",
      metric: "477.2 → 401.3 MAE",
      detail:
        "Across 2,468 problems from 355 contests, primary MAE minus comparator MAE was −75.9447 rating points, about 15.9% relative to the comparator.",
      boundary:
        "This supports a narrow within-cohort comparison. It is not a prospective result and does not replace the legacy full-API metrics.",
      linkLabel: "Open the machine-readable metrics",
      href: `${repository}/outputs/historical_statement_backtest/test/test_metrics.json`,
    },
    uncertainty: {
      kicker: "PAIRED CLUSTER BOOTSTRAP",
      title: "Contest-level dependence stays together in resampling.",
      metric: "95% interval [−86.7, −65.1]",
      detail:
        "The study used 10,000 paired resamples of 355 test contests. All problems from a sampled contest travel together, and both models use the same resample.",
      boundary:
        "This percentile interval describes uncertainty in the locked historical cohort; it does not prove a future effect or a causal mechanism.",
      linkLabel: "Inspect the bootstrap output",
      href: `${repository}/outputs/historical_statement_backtest/test/paired_bootstrap.json`,
    },
    artifacts: {
      kicker: "COMMITTED EVIDENCE",
      title: "The result is preserved beyond a summary table.",
      metric: "2,468 problems × 2 models + SHA-256",
      detail:
        "Committed outputs include split membership, prepared data, validation metrics, coverage, predictions, Top-10 errors, bootstrap results, and final-result hashes.",
      boundary:
        "These files can recompute the reported metrics. The 718 MB page cache is not committed, so raw page-to-feature extraction requires a separate archival release.",
      linkLabel: "Open the result hash manifest",
      href: `${repository}/outputs/historical_statement_backtest/test/result_manifest.sha256`,
    },
  };

  function updateEvidence(selectedTab) {
    const viewName = selectedTab.dataset.evidence;
    const view = evidenceViews[viewName];

    if (!view) {
      return;
    }

    const panel = document.getElementById("evidence-panel");
    const kicker = document.getElementById("evidenceKicker");
    const title = document.getElementById("evidenceTitle");
    const metric = document.getElementById("evidenceMetric");
    const detail = document.getElementById("evidenceDetail");
    const boundary = document.getElementById("evidenceBoundary");
    const link = document.getElementById("evidenceLink");

    if (!panel || !kicker || !title || !metric || !detail || !boundary || !link) {
      return;
    }

    document.querySelectorAll(".evidence-tab").forEach(function (tab) {
      const isSelected = tab === selectedTab;
      tab.classList.toggle("active", isSelected);
      tab.setAttribute("aria-selected", String(isSelected));
      tab.tabIndex = isSelected ? 0 : -1;
    });

    panel.setAttribute("aria-labelledby", selectedTab.id);
    kicker.textContent = view.kicker;
    title.textContent = view.title;
    metric.textContent = view.metric;
    detail.textContent = view.detail;
    boundary.textContent = view.boundary;
    link.textContent = view.linkLabel;
    link.href = view.href;
  }

  function moveTabFocus(currentTab, direction) {
    const tabs = Array.from(document.querySelectorAll(".evidence-tab"));
    const currentIndex = tabs.indexOf(currentTab);

    if (currentIndex === -1 || tabs.length === 0) {
      return;
    }

    const nextIndex = (currentIndex + direction + tabs.length) % tabs.length;
    tabs[nextIndex].focus();
    updateEvidence(tabs[nextIndex]);
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".evidence-tab").forEach(function (tab) {
      tab.addEventListener("click", function () {
        updateEvidence(tab);
      });

      tab.addEventListener("keydown", function (event) {
        if (event.key === "ArrowRight" || event.key === "ArrowDown") {
          event.preventDefault();
          moveTabFocus(tab, 1);
        } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
          event.preventDefault();
          moveTabFocus(tab, -1);
        } else if (event.key === "Home") {
          event.preventDefault();
          const firstTab = document.querySelector(".evidence-tab");
          if (firstTab) {
            firstTab.focus();
            updateEvidence(firstTab);
          }
        } else if (event.key === "End") {
          event.preventDefault();
          const tabs = document.querySelectorAll(".evidence-tab");
          const lastTab = tabs[tabs.length - 1];
          if (lastTab) {
            lastTab.focus();
            updateEvidence(lastTab);
          }
        }
      });
    });
  });
})();
