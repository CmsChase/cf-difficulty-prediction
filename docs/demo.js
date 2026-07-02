(function () {
  "use strict";

  const indexBase = {
    A: 1000,
    B: 1300,
    C: 1600,
    D: 1900,
    E: 2200,
    "F+": 2500,
  };

  const tagOffset = {
    implementation: 0,
    math: 100,
    dp: 250,
    graphs: 240,
    greedy: 80,
    strings: 120,
    interactive: 360,
    geometry: 320,
  };

  const lengthOffset = {
    short: -40,
    medium: 0,
    long: 90,
  };

  const lengthUncertainty = {
    short: 120,
    medium: 160,
    long: 220,
  };

  function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, value));
  }

  function roundedToHundred(value) {
    return Math.round(value / 100) * 100;
  }

  function updateDemo() {
    const indexRank = document.getElementById("indexRank").value;
    const tagFamily = document.getElementById("tagFamily").value;
    const statementLength = document.getElementById("statementLength").value;
    const solvedAvailable = document.getElementById("solvedAvailable").value;

    const center =
      indexBase[indexRank] + tagOffset[tagFamily] + lengthOffset[statementLength];
    const solvedAdjustment = solvedAvailable === "yes" ? -80 : 0;
    const uncertainty =
      solvedAvailable === "yes"
        ? Math.max(100, lengthUncertainty[statementLength] - 60)
        : lengthUncertainty[statementLength] + 120;

    const estimate = clamp(center + solvedAdjustment, 800, 3500);
    const low = roundedToHundred(clamp(estimate - uncertainty, 800, 3500));
    const high = roundedToHundred(clamp(estimate + uncertainty, 800, 3500));

    const scenario =
      solvedAvailable === "yes"
        ? "Post-publication scenario"
        : "Cold-start scenario";
    const explanation =
      solvedAvailable === "yes"
        ? "Uses index rank, tag family, statement length, and the fact that solved-count behavior is available. The range is narrower, but this is still only a transparent heuristic."
        : "Uses index rank, tag family, and statement length. Solved-count behavior is not included, so uncertainty is wider.";

    document.getElementById("difficultyRange").textContent = `${low}–${high}`;
    document.getElementById("scenarioType").textContent = scenario;
    document.getElementById("demoExplanation").textContent = explanation;
  }

  document.addEventListener("DOMContentLoaded", function () {
    ["indexRank", "tagFamily", "statementLength", "solvedAvailable"].forEach(
      function (id) {
        document.getElementById(id).addEventListener("change", updateDemo);
      },
    );
    updateDemo();
  });
})();
