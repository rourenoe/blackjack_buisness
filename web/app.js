let currentUser = null;
let currentScenario = null;

const usernameInput = document.getElementById("username");
const startBtn = document.getElementById("startBtn");
const userStatus = document.getElementById("userStatus");
const scenarioPanel = document.getElementById("scenarioPanel");
const scenarioText = document.getElementById("scenarioText");
const scoreText = document.getElementById("scoreText");
const feedback = document.getElementById("feedback");
const progressPanel = document.getElementById("progressPanel");
const progressText = document.getElementById("progressText");

function setMessage(el, message, cssClass) {
  el.textContent = message;
  el.classList.remove("ok", "ko");
  if (cssClass) {
    el.classList.add(cssClass);
  }
}

function renderScenario(scenario) {
  currentScenario = scenario;
  scenarioPanel.hidden = false;
  scenarioText.textContent = `${scenario.category} | Player ${scenario.player_ranks[0]} + ${scenario.player_ranks[1]} vs Dealer ${scenario.dealer_rank}`;
  scoreText.textContent = `Scores: player ${scenario.player_score} | dealer ${scenario.dealer_score}`;

  document.querySelectorAll("[data-action]").forEach((btn) => {
    const action = btn.getAttribute("data-action");
    btn.disabled = !scenario.valid_actions.includes(action);
  });
}

function renderProgress(data) {
  progressPanel.hidden = false;
  progressText.textContent = `Attempts: ${data.total_attempts} | Correct: ${data.correct_attempts} | Wrong: ${data.incorrect_attempts} | Errors left: ${data.remaining_errors}`;
}

async function fetchProgress() {
  const res = await fetch(`/api/users/${encodeURIComponent(currentUser)}/progress`);
  if (!res.ok) {
    throw new Error("Failed to load progress");
  }
  const data = await res.json();
  renderProgress(data);
}

startBtn.addEventListener("click", async () => {
  const username = usernameInput.value.trim();
  if (!username) {
    setMessage(userStatus, "Enter a username first.", "ko");
    return;
  }
  setMessage(userStatus, "Loading user session...");
  feedback.textContent = "";

  const res = await fetch(`/api/users/${encodeURIComponent(username)}/start`, {
    method: "POST",
  });
  if (!res.ok) {
    const body = await res.json();
    setMessage(userStatus, body.detail || "Could not start session.", "ko");
    return;
  }

  const data = await res.json();
  currentUser = data.username;
  usernameInput.value = data.username;
  setMessage(userStatus, `User ready: ${data.username}`, "ok");
  renderScenario(data.scenario);
  await fetchProgress();
});

document.querySelectorAll("[data-action]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    if (!currentUser || !currentScenario) {
      setMessage(feedback, "Start a user session first.", "ko");
      return;
    }

    const action = btn.getAttribute("data-action");
    const res = await fetch(`/api/users/${encodeURIComponent(currentUser)}/answer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scenario_key: currentScenario.key,
        action: action,
      }),
    });

    if (!res.ok) {
      const body = await res.json();
      setMessage(feedback, body.detail || "Request failed.", "ko");
      return;
    }

    const data = await res.json();
    if (data.correct) {
      setMessage(feedback, "Correct.", "ok");
    } else {
      setMessage(feedback, `Wrong. Best action: ${data.correct_action}`, "ko");
    }

    renderScenario(data.next_scenario);
    renderProgress(data);
  });
});
