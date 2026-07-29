let currentUser = null;
let currentScenario = null;
let isActionPending = false; // Prevent double click submit API race conditions
let isReviewMode = false;    // Tracks whether player is retrying errors
let activeErrorKeys = [];
let attemptedKeys = [];
let cachedStrategyMap = null;
let currentHeatmapCategory = "hard";

const usernameInput = document.getElementById("username");
const passwordInput = document.getElementById("password");
const startBtn = document.getElementById("startBtn");
const registerBtn = document.getElementById("registerBtn");
const logoutBtn = document.getElementById("logoutBtn");
const reviewErrorsBtn = document.getElementById("reviewErrorsBtn"); // New premium error review action button
const loginFormState = document.getElementById("loginFormState");
const loggedInState = document.getElementById("loggedInState");
const loggedInUser = document.getElementById("loggedInUser");
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

// Deterministic suit helper based on card rank and index to create realistic varied suits
function getSuitForCard(rank, isDealer, index) {
  const suits = [
    { symbol: "♠", name: "spades", isRed: false },
    { symbol: "♥", name: "hearts", isRed: true },
    { symbol: "♦", name: "diamonds", isRed: true },
    { symbol: "♣", name: "clubs", isRed: false }
  ];

  if (isDealer) {
    // Keep dealers simple or deterministic
    return suits[0]; // Spades
  }

  // Choose based on card index so we don't have overlapping suits in a typical two-card hand
  if (index === 0) {
    return suits[1]; // Hearts
  } else if (index === 1) {
    return suits[2]; // Diamonds
  } else {
    return suits[3]; // Clubs
  }
}

// Generate the visual representation of a face-up card
function createCardElement(rank, isDealer, index = 0) {
  const cardDiv = document.createElement("div");
  cardDiv.className = "blackjack-card animate-deal";
  
  const suitInfo = getSuitForCard(rank, isDealer, index);
  if (suitInfo.isRed) {
    cardDiv.classList.add("red-suit");
  } else {
    cardDiv.classList.add("black-suit");
  }

  // Top Left Corner
  const cornerTop = document.createElement("div");
  cornerTop.className = "card-corner top-left";
  cornerTop.innerHTML = `<div>${rank}</div><div style="font-size: 0.9rem;">${suitInfo.symbol}</div>`;

  // Center large symbol
  const suitCenter = document.createElement("div");
  suitCenter.className = "card-suit-center";
  suitCenter.textContent = suitInfo.symbol;

  // Bottom Right Corner
  const cornerBottom = document.createElement("div");
  cornerBottom.className = "card-corner bottom-right";
  cornerBottom.innerHTML = `<div>${rank}</div><div style="font-size: 0.9rem;">${suitInfo.symbol}</div>`;

  cardDiv.appendChild(cornerTop);
  cardDiv.appendChild(suitCenter);
  cardDiv.appendChild(cornerBottom);

  return cardDiv;
}

// Generate the visual representation of a face-down dealer card
function createHiddenCardElement() {
  const cardDiv = document.createElement("div");
  cardDiv.className = "blackjack-card card-back animate-deal";
  
  const innerPattern = document.createElement("div");
  innerPattern.className = "card-back-pattern";
  
  const logo = document.createElement("div");
  logo.className = "card-back-logo";
  logo.textContent = "♠";
  
  innerPattern.appendChild(logo);
  cardDiv.appendChild(innerPattern);
  return cardDiv;
}

function renderScenario(scenario) {
  currentScenario = scenario;
  scenarioPanel.hidden = false;

  // Update legacy hidden fields
  scenarioText.textContent = `${scenario.category} | Player ${scenario.player_ranks[0]} + ${scenario.player_ranks[1]} vs Dealer ${scenario.dealer_rank}`;
  scoreText.textContent = `Scores: player ${scenario.player_score} | dealer ${scenario.dealer_score}`;

  // Update beautiful category badge
  const categoryBadge = document.getElementById("scenarioCategory");
  if (categoryBadge) {
    let cleanCategory = scenario.category.replace(/_/g, " ").toUpperCase();
    categoryBadge.textContent = cleanCategory;
    categoryBadge.className = "badge";
    
    if (scenario.category.toLowerCase().includes("hard")) {
      categoryBadge.classList.add("badge-hard");
    } else if (scenario.category.toLowerCase().includes("soft")) {
      categoryBadge.classList.add("badge-soft");
    } else {
      categoryBadge.classList.add("badge-pairs");
    }
  }

  // Render Dealer Hand
  const dealerCardsContainer = document.getElementById("dealerCards");
  const dealerScoreVal = document.getElementById("dealerScoreVal");
  if (dealerCardsContainer) {
    dealerCardsContainer.innerHTML = "";
    // Show dealer's upcard
    const cardVisible = createCardElement(scenario.dealer_rank, true, 0);
    dealerCardsContainer.appendChild(cardVisible);
    // Show one standard facedown card to simulate active game
    const cardHidden = createHiddenCardElement();
    dealerCardsContainer.appendChild(cardHidden);
  }
  if (dealerScoreVal) {
    dealerScoreVal.textContent = scenario.dealer_score;
  }

  // Render Player Hand
  const playerCardsContainer = document.getElementById("playerCards");
  const playerScoreVal = document.getElementById("playerScoreVal");
  if (playerCardsContainer) {
    playerCardsContainer.innerHTML = "";
    scenario.player_ranks.forEach((rank, index) => {
      const card = createCardElement(rank, false, index);
      playerCardsContainer.appendChild(card);
    });
  }
  if (playerScoreVal) {
    playerScoreVal.textContent = scenario.player_score;
  }

  // Manage action button states
  document.querySelectorAll("[data-action]").forEach((btn) => {
    const action = btn.getAttribute("data-action");
    btn.disabled = !scenario.valid_actions.includes(action);
  });
}

// Dynamic Level and Celebration Systems
let currentUserTierLevel = null;

const TIERS = [
  { level: 0, min: 0,   name: "🥶 Absolute Zero",  color: "#93c5fd", nextMin: 10,  theme: "ice" },
  { level: 1, min: 10,  name: "🧊 Frosty Chill",   color: "#60a5fa", nextMin: 30,  theme: "frost" },
  { level: 2, min: 30,  name: "💧 Melting Point",  color: "#38bdf8", nextMin: 60,  theme: "melt" },
  { level: 3, min: 60,  name: "💨 Rising Vapor",   color: "#a5f3fc", nextMin: 90,  theme: "vapor" },
  { level: 4, min: 90,  name: "🌱 Green Spark",    color: "#4ade80", nextMin: 120, theme: "spark" },
  { level: 5, min: 120, name: "⚡ Static Charge",  color: "#facc15", nextMin: 150, theme: "electric" },
  { level: 6, min: 150, name: "🔥 Warm Spark",     color: "#f97316", nextMin: 200, theme: "warm" },
  { level: 7, min: 200, name: "💥 Fire Charge",    color: "#ef4444", nextMin: 250, theme: "fire" },
  { level: 8, min: 250, name: "🌋 Volcanic Flow",  color: "#b91c1c", nextMin: 300, theme: "volcano" },
  { level: 9, min: 300, name: "🌌 Cosmic Supernova", color: "#ec4899", nextMin: null, theme: "supernova" }
];

function getTier(count) {
  for (let i = TIERS.length - 1; i >= 0; i--) {
    if (count >= TIERS[i].min) {
      return TIERS[i];
    }
  }
  return TIERS[0];
}

function playArcadeLevelUpSound(level) {
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  if (!AudioContext) return;
  const ctx = new AudioContext();
  
  // Retro arpeggios that rise rapidly
  const notes = [261.63, 329.63, 392.00, 523.25, 659.25, 783.99, 1046.50];
  const duration = 0.12;
  
  notes.forEach((freq, i) => {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    
    // Triangle wave for frost/electro, Sawtooth for crazy explosive volcano/supernova
    osc.type = level >= 7 ? "sawtooth" : (level >= 4 ? "triangle" : "sine");
    osc.frequency.setValueAtTime(freq, ctx.currentTime + i * 0.08);
    
    gain.gain.setValueAtTime(0.12, ctx.currentTime + i * 0.08);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + i * 0.08 + duration);
    
    osc.connect(gain);
    gain.connect(ctx.destination);
    
    osc.start(ctx.currentTime + i * 0.08);
    osc.stop(ctx.currentTime + i * 0.08 + duration);
  });
}

function playActionSound(action) {
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  if (!AudioContext) return;
  const ctx = new AudioContext();
  
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  
  osc.connect(gain);
  gain.connect(ctx.destination);
  
  const now = ctx.currentTime;
  
  if (action === "hit") {
    // Sharp click/zap
    osc.type = "triangle";
    osc.frequency.setValueAtTime(600, now);
    osc.frequency.exponentialRampToValueAtTime(150, now + 0.08);
    gain.gain.setValueAtTime(0.08, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.08);
    osc.start(now);
    osc.stop(now + 0.08);
  } else if (action === "stand") {
    // Solid clean thud
    osc.type = "sine";
    osc.frequency.setValueAtTime(250, now);
    osc.frequency.exponentialRampToValueAtTime(100, now + 0.12);
    gain.gain.setValueAtTime(0.12, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.12);
    osc.start(now);
    osc.stop(now + 0.12);
  } else if (action === "double") {
    // Double coin retro chime
    osc.type = "triangle";
    osc.frequency.setValueAtTime(523.25, now); // C5
    osc.frequency.setValueAtTime(783.99, now + 0.06); // G5
    gain.gain.setValueAtTime(0.08, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.15);
    osc.start(now);
    osc.stop(now + 0.15);
  } else if (action === "split") {
    // Quick sharp scissor slice sweep
    osc.type = "sine";
    osc.frequency.setValueAtTime(800, now);
    osc.frequency.linearRampToValueAtTime(1800, now + 0.10);
    gain.gain.setValueAtTime(0.06, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.10);
    osc.start(now);
    osc.stop(now + 0.10);
  }
}

function triggerCelebration(tier) {
  const overlay = document.getElementById("celebrationOverlay");
  const badge = document.getElementById("celebrationBadge");
  const subtitle = document.getElementById("celebrationSubtitle");
  if (!overlay) return;

  if (subtitle) subtitle.textContent = tier.name;
  if (badge) {
    const emojiMatch = tier.name.match(/[\u{1F300}-\u{1F9FF}]|[\u{2600}-\u{26FF}]/u);
    badge.textContent = emojiMatch ? emojiMatch[0] : "🌟";
  }

  overlay.className = "celebration-overlay active";
  overlay.classList.add(`theme-${tier.theme}`);

  // Create floating elements
  const container = document.createElement("div");
  container.className = "celebration-particles";
  overlay.appendChild(container);

  const emojiList = ["🎉", "🔥", "💥", "⚡", "✨", "🧊", "❄️", "🌋", "🌌"];
  const chosenEmoji = emojiList[tier.level % emojiList.length];

  for (let i = 0; i < 40; i++) {
    const particle = document.createElement("span");
    particle.className = "particle";
    particle.textContent = chosenEmoji;
    particle.style.left = `${Math.random() * 100}vw`;
    particle.style.bottom = `-50px`;
    particle.style.fontSize = `${Math.random() * 1.5 + 1}rem`;
    particle.style.animationDelay = `${Math.random() * 1.5}s`;
    particle.style.animationDuration = `${Math.random() * 1.5 + 1}s`;
    particle.style.transform = `rotate(${Math.random() * 360}deg)`;
    container.appendChild(particle);
  }

  overlay.removeAttribute("hidden");

  try {
    playArcadeLevelUpSound(tier.level);
  } catch (e) {
    console.log("Audio contexts blocked");
  }

  setTimeout(() => {
    overlay.setAttribute("hidden", "true");
    overlay.classList.remove("active");
    container.remove();
  }, 3200);
}

function updateReviewButtonUI(remainingErrorsCount = activeErrorKeys.length) {
  if (!reviewErrorsBtn) return;
  
  if (!currentUser) {
    reviewErrorsBtn.style.display = "none";
    return;
  }

  if (isReviewMode) {
    reviewErrorsBtn.style.display = "inline-block";
    reviewErrorsBtn.textContent = "🚪 Exit Review Mode";
    reviewErrorsBtn.style.background = "linear-gradient(135deg, #ef4444, #dc2626)";
    reviewErrorsBtn.style.boxShadow = "0 0 12px rgba(239, 68, 68, 0.4)";
  } else {
    if (remainingErrorsCount > 0) {
      reviewErrorsBtn.style.display = "inline-block";
      reviewErrorsBtn.textContent = `🎯 Practice Errors (${remainingErrorsCount})`;
      reviewErrorsBtn.style.background = "linear-gradient(135deg, #f59e0b, #d97706)";
      reviewErrorsBtn.style.boxShadow = "0 0 10px rgba(245, 158, 11, 0.3)";
    } else {
      reviewErrorsBtn.style.display = "none";
    }
  }
}

function renderProgress(data) {
  progressPanel.hidden = false;
  
  // Save active error keys and attempted keys for strategy map highlight
  activeErrorKeys = data.error_keys || [];
  attemptedKeys = data.attempted_keys || [];
  drawHeatmap(); // Redraw heatmap with current performance

  // Update legacy hidden text
  progressText.textContent = `Attempts: ${data.total_attempts} | Correct: ${data.correct_attempts} | Wrong: ${data.incorrect_attempts} | Errors left: ${data.remaining_errors}`;

  // Compute stats
  const total = data.total_attempts;
  const wrong = data.incorrect_attempts;
  const remaining = data.remaining_errors;
  const accuracy = total > 0 ? Math.round((data.correct_attempts / total) * 100) : 0;
  const explorationPct = Math.round((attemptedKeys.length / 310) * 100);

  // Retrieve dashboard DOM nodes
  const statAccuracy = document.getElementById("statAccuracy");
  const statAccuracyBar = document.getElementById("statAccuracyBar");
  const statExploration = document.getElementById("statExploration");
  const statExplorationBar = document.getElementById("statExplorationBar");
  const statWrong = document.getElementById("statWrong");
  const statRemaining = document.getElementById("statRemaining");

  // Determine current active tier
  const activeTier = getTier(attemptedKeys.length);

  // Update nodes with values and animations
  if (statAccuracy) statAccuracy.textContent = `${accuracy}%`;
  if (statAccuracyBar) statAccuracyBar.style.width = `${accuracy}%`;
  if (statExploration) statExploration.textContent = `${attemptedKeys.length} / 310`;
  
  if (statExplorationBar) {
    statExplorationBar.style.width = `${explorationPct}%`;
    statExplorationBar.style.background = activeTier.color; // Dynamic tier-colored progress bar!
  }
  
  if (statWrong) statWrong.textContent = wrong;
  if (statRemaining) statRemaining.textContent = remaining;

  // Update tier labels and next level requirements info
  const tierNameEl = document.getElementById("tierName");
  if (tierNameEl) {
    tierNameEl.textContent = activeTier.name;
  }

  const tierNextInfo = document.getElementById("tierNextInfo");
  if (tierNextInfo) {
    if (isReviewMode) {
      tierNextInfo.innerHTML = `<span class="text-gold" style="font-weight: 800; animation: pulse-border 1.5s infinite;">⚠️ Error Review Active: Cleared mistakes are permanently removed!</span>`;
    } else {
      if (activeTier.nextMin !== null) {
        const needed = activeTier.nextMin - attemptedKeys.length;
        tierNextInfo.textContent = `🎯 Only ${needed} more unique hand${needed === 1 ? '' : 's'} to explore to unlock the next tier!`;
      } else {
        tierNextInfo.textContent = "🏆 Master Explorer! All 310 basic hands explored! 🌟";
      }
    }
  }

  // Update Practice Errors action button HUD
  updateReviewButtonUI(remaining);

  // Apply dynamic tier glow effects and borders to exploration card HUD
  const card = document.getElementById("explorationCard");
  if (card) {
    card.className = "table-exploration-wrapper"; // Reset
    card.classList.add(`tier-${activeTier.theme}`); // Apply specific tier border and aura
    if (isReviewMode) {
      // In review mode, give a glowing red/orange warning border!
      card.style.borderColor = "var(--accent-crimson)";
      card.style.boxShadow = "0 0 15px rgba(239, 68, 68, 0.4)";
    } else {
      card.style.borderColor = "";
      card.style.boxShadow = "";
    }
  }

  // Trigger full screen celebratory effects when a level boundary is crossed
  if (currentUserTierLevel === null) {
    currentUserTierLevel = activeTier.level;
  } else if (activeTier.level > currentUserTierLevel) {
    triggerCelebration(activeTier);
    currentUserTierLevel = activeTier.level;
  }
}

async function fetchProgress() {
  const res = await fetch(`/api/users/${encodeURIComponent(currentUser)}/progress`);
  if (!res.ok) {
    throw new Error("Failed to load progress");
  }
  const data = await res.json();
  renderProgress(data);
}

function setSessionState(authenticated, username = "") {
  if (authenticated) {
    currentUser = username;
    if (loggedInUser) loggedInUser.textContent = username;
    if (loginFormState) loginFormState.hidden = true;
    if (loggedInState) loggedInState.hidden = false;
  } else {
    currentUser = null;
    currentScenario = null;
    currentUserTierLevel = null; // Clear level memory on session end
    isReviewMode = false;        // Deactivate review mode on logout
    activeErrorKeys = [];
    attemptedKeys = [];
    if (loginFormState) loginFormState.hidden = false;
    if (loggedInState) loggedInState.hidden = true;
    if (scenarioPanel) scenarioPanel.hidden = true;
    if (progressPanel) progressPanel.hidden = true;
    if (usernameInput) usernameInput.value = "";
    if (passwordInput) passwordInput.value = "";
    drawHeatmap(); // Draw clean anonymous heatmap
  }
}

async function handleAuth(mode) {
  const username = usernameInput.value.trim();
  const password = passwordInput.value.trim();

  if (!username) {
    setMessage(userStatus, "Please enter a valid nickname.", "ko");
    return;
  }
  if (!password || !/^\d{6}$/.test(password)) {
    setMessage(userStatus, "Please enter a 6-digit PIN.", "ko");
    return;
  }

  setMessage(userStatus, mode === "login" ? "Logging in..." : "Creating account...");
  feedback.textContent = "";

  try {
    const res = await fetch(`/api/users/${encodeURIComponent(username)}/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: password, mode: mode })
    });
    
    const data = await res.json();
    if (!res.ok) {
      setMessage(userStatus, data.detail || "Authentication failed.", "ko");
      return;
    }

    setSessionState(true, data.username);
    setMessage(userStatus, mode === "login" ? "Login successful!" : "Account created successfully!", "ok");
    renderScenario(data.scenario);
    await fetchProgress();
    await fetchLeaderboard();
  } catch (error) {
    setMessage(userStatus, "Connection error to training API.", "ko");
  }
}

startBtn.addEventListener("click", async () => {
  await handleAuth("login");
});

if (registerBtn) {
  registerBtn.addEventListener("click", async () => {
    await handleAuth("register");
  });
}

if (logoutBtn) {
  logoutBtn.addEventListener("click", () => {
    setSessionState(false);
    setMessage(userStatus, "You have been logged out.", "ok");
    feedback.textContent = "";
  });
}

document.querySelectorAll("[data-action]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    if (!currentUser || !currentScenario) {
      setMessage(feedback, "Start a user session first.", "ko");
      return;
    }

    // Prevent race conditions and double-clicking during active API call
    if (isActionPending) return;
    isActionPending = true;

    // Immediately disable all action buttons in the DOM for immediate user feedback
    document.querySelectorAll("[data-action]").forEach((b) => b.disabled = true);

    const action = btn.getAttribute("data-action");
    
    // Play retro action sound instantly on click!
    try {
      playActionSound(action);
    } catch (e) {
      console.log("Audio contexts blocked");
    }

    try {
      const res = await fetch(`/api/users/${encodeURIComponent(currentUser)}/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scenario_key: currentScenario.key,
          action: action,
          review_mode: isReviewMode // Sends the active review state to backend
        }),
      });

      if (!res.ok) {
        const body = await res.json();
        setMessage(feedback, body.detail || "Request failed.", "ko");
        // Re-enable valid buttons in case of failure
        renderScenario(currentScenario);
        return;
      }

      const data = await res.json();
      if (data.correct) {
        setMessage(feedback, "🎉 Correct strategy decision!", "ok");
      } else {
        setMessage(feedback, `❌ Wrong. Best action: ${data.correct_action.toUpperCase()}`, "ko");
      }

      // Check if Error Review Mode was successfully completed!
      if (data.review_mode && data.errors_cleared) {
        isReviewMode = false;
        setMessage(feedback, "🏆 Outstanding! You have successfully cleared ALL of your errors! Standard mode restored.", "ok");
        // Fire a majestic cosmic celebration for clearing all active mistakes!
        triggerCelebration({ level: 9, name: "👑 Errors Cleared!", theme: "supernova" });
      }

      renderScenario(data.next_scenario);
      renderProgress(data);
      await fetchLeaderboard(); // Update leaderboard in real-time
    } catch (error) {
      setMessage(feedback, "Communication error occurred.", "ko");
      // Re-enable valid buttons in case of connection failure
      renderScenario(currentScenario);
    } finally {
      isActionPending = false;
    }
  });
});

// --- LEADERBOARD & ADMIN LOUNGE FEATURES ---

async function fetchLeaderboard() {
  const bodyEl = document.getElementById("leaderboardBody");
  if (!bodyEl) return;

  try {
    const res = await fetch("/api/leaderboard");
    if (!res.ok) throw new Error("Could not load leaderboard");
    
    const data = await res.json();
    bodyEl.innerHTML = "";
    
    if (data.length === 0) {
      bodyEl.innerHTML = `<tr><td colspan="6" class="muted table-loading">No active players yet. Be the first!</td></tr>`;
      return;
    }

    data.forEach((row, index) => {
      const tr = document.createElement("tr");
      
      // Rank medal
      let rankText = index + 1;
      if (index === 0) rankText = "🥇";
      else if (index === 1) rankText = "🥈";
      else if (index === 2) rankText = "🥉";

      const tdRank = document.createElement("td");
      tdRank.innerHTML = `<strong>${rankText}</strong>`;
      
      const tdPlayer = document.createElement("td");
      tdPlayer.textContent = row.username;
      if (row.username === currentUser) {
        tdPlayer.innerHTML = `${row.username} <span class="badge badge-soft" style="font-size: 0.65rem; padding: 2px 6px; border: 1px solid var(--accent-blue);">YOU</span>`;
      }

      // Dynamic Player Grade/Tier matching our 10 levels of exploration
      const playerTier = getTier(row.unique_attempts || 0);
      const tdGrade = document.createElement("td");
      tdGrade.style.color = playerTier.color;
      tdGrade.style.fontWeight = "800";
      tdGrade.style.fontSize = "0.85rem";
      tdGrade.innerHTML = playerTier.name;

      const tdAccuracy = document.createElement("td");
      tdAccuracy.className = "text-ok";
      tdAccuracy.innerHTML = `<strong>${row.accuracy}%</strong>`;

      const tdAttempts = document.createElement("td");
      tdAttempts.innerHTML = `<strong>${row.unique_attempts || 0}</strong> <span class="muted" style="font-size: 0.75rem;">/ 310</span>`;

      const tdErrors = document.createElement("td");
      tdErrors.className = row.remaining_errors > 0 ? "text-ko" : "text-ok";
      tdErrors.textContent = row.remaining_errors;

      tr.appendChild(tdRank);
      tr.appendChild(tdPlayer);
      tr.appendChild(tdGrade);
      tr.appendChild(tdAccuracy);
      tr.appendChild(tdAttempts);
      tr.appendChild(tdErrors);

      bodyEl.appendChild(tr);
    });
  } catch (error) {
    bodyEl.innerHTML = `<tr><td colspan="6" class="table-loading text-ko">Error loading leaderboard.</td></tr>`;
  }
}

// Refresh leaderboard button
const refreshLeaderboardBtn = document.getElementById("refreshLeaderboardBtn");
if (refreshLeaderboardBtn) {
  refreshLeaderboardBtn.addEventListener("click", fetchLeaderboard);
}

// Practice Errors toggle button handler
if (reviewErrorsBtn) {
  reviewErrorsBtn.addEventListener("click", async () => {
    if (!currentUser) return;

    if (isReviewMode) {
      // Toggle OFF: Return to standard training mode
      isReviewMode = false;
      feedback.textContent = "";
      setMessage(feedback, "Review mode disabled. Standard practice restored.", "ok");
      
      try {
        // Fetch the player's active standard scenario transparently
        const res = await fetch(`/api/users/${encodeURIComponent(currentUser)}/current-scenario`);
        if (res.ok) {
          const scenario = await res.json();
          if (scenario) {
            renderScenario(scenario);
          }
        }
        await fetchProgress(); // Refreshes and updates indicators
      } catch (e) {
        console.error("Failed to restore standard scenario:", e);
      }
      return;
    }

    // Toggle ON: Start practicing active errors
    if (activeErrorKeys.length === 0) {
      setMessage(feedback, "No active errors to review! Keep up the great work! 🌟", "ok");
      return;
    }

    isReviewMode = true;
    setMessage(feedback, "Error Review Mode Activated! Focused on mastering your mistakes. 🎯", "ok");

    // Play a delightful double retro arcade sound on activation!
    try {
      playActionSound("double");
    } catch (e) {}

    try {
      // Load the first active mistake instantly onto the felt table
      const res = await fetch(`/api/users/${encodeURIComponent(currentUser)}/error-scenario`);
      if (res.ok) {
        const scenario = await res.json();
        if (scenario) {
          renderScenario(scenario);
        }
      }
      await fetchProgress(); // Refreshes progress bar wrapper styling
    } catch (e) {
      console.error("Failed to load error scenario:", e);
    }
  });
}

// Admin Modal and Deletion logic
const adminToggleBtn = document.getElementById("adminToggleBtn");
const adminModal = document.getElementById("adminModal");
const closeAdminBtn = document.getElementById("closeAdminBtn");

const deleteUserBtn = document.getElementById("deleteUserBtn");
const adminKeyInput = document.getElementById("adminKey");
const deleteUsernameInput = document.getElementById("deleteUsername");
const adminStatus = document.getElementById("adminStatus");

if (adminToggleBtn && adminModal) {
  adminToggleBtn.addEventListener("click", () => {
    adminModal.classList.add("open");
    if (adminStatus) adminStatus.textContent = "";
  });
}

if (closeAdminBtn && adminModal) {
  closeAdminBtn.addEventListener("click", () => {
    adminModal.classList.remove("open");
  });
}

if (adminModal) {
  adminModal.addEventListener("click", (e) => {
    if (e.target === adminModal) {
      adminModal.classList.remove("open");
    }
  });
}

if (deleteUserBtn) {
  deleteUserBtn.addEventListener("click", async () => {
    const adminKey = adminKeyInput.value.trim();
    const deleteUsername = deleteUsernameInput.value.trim();

    if (!adminKey) {
      setMessage(adminStatus, "Admin key is required.", "ko");
      return;
    }
    if (!deleteUsername) {
      setMessage(adminStatus, "Target nickname is required.", "ko");
      return;
    }

    setMessage(adminStatus, "Processing deletion request...");

    try {
      const res = await fetch("/api/admin/delete-user", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          admin_key: adminKey,
          username: deleteUsername
        })
      });

      const data = await res.json();
      if (!res.ok) {
        setMessage(adminStatus, data.detail || "Request rejected by admin protocols.", "ko");
        return;
      }

      setMessage(adminStatus, data.detail, "ok");
      deleteUsernameInput.value = "";
      
      // If deleted user is current user, reset interface
      if (currentUser && deleteUsername.toLowerCase() === currentUser.toLowerCase()) {
        setSessionState(false);
        setMessage(userStatus, "Your session was deleted by admin authority.", "ko");
      }

      await fetchLeaderboard();
      
      // Auto close modal on success after 1 second for seamless UX
      setTimeout(() => {
        if (adminModal.classList.contains("open")) {
          adminModal.classList.remove("open");
        }
      }, 1200);
    } catch (error) {
      setMessage(adminStatus, "Network failure contacting admin API.", "ko");
    }
  });
}

// --- STRATEGY HEATMAP FEATURES ---

const HEATMAP_COLUMNS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A'];

const HARD_ROWS = [
  { label: '17+', ranks: ['7', '10'] },
  { label: '16', ranks: ['6', '10'] },
  { label: '15', ranks: ['5', '10'] },
  { label: '14', ranks: ['4', '10'] },
  { label: '13', ranks: ['3', '10'] },
  { label: '12', ranks: ['2', '10'] },
  { label: '11', ranks: ['2', '9'] },
  { label: '10', ranks: ['2', '8'] },
  { label: '9', ranks: ['2', '7'] },
  { label: '8', ranks: ['2', '6'] },
  { label: '7', ranks: ['2', '5'] },
  { label: '6', ranks: ['2', '4'] },
  { label: '5', ranks: ['2', '3'] }
];

const SOFT_ROWS = [
  { label: 'A,9', ranks: ['A', '9'] },
  { label: 'A,8', ranks: ['A', '8'] },
  { label: 'A,7', ranks: ['A', '7'] },
  { label: 'A,6', ranks: ['A', '6'] },
  { label: 'A,5', ranks: ['A', '5'] },
  { label: 'A,4', ranks: ['A', '4'] },
  { label: 'A,3', ranks: ['A', '3'] },
  { label: 'A,2', ranks: ['A', '2'] }
];

const PAIR_ROWS = [
  { label: 'A,A', ranks: ['A', 'A'] },
  { label: '10,10', ranks: ['10', '10'] },
  { label: '9,9', ranks: ['9', '9'] },
  { label: '8,8', ranks: ['8', '8'] },
  { label: '7,7', ranks: ['7', '7'] },
  { label: '6,6', ranks: ['6', '6'] },
  { label: '5,5', ranks: ['5', '5'] },
  { label: '4,4', ranks: ['4', '4'] },
  { label: '3,3', ranks: ['3', '3'] },
  { label: '2,2', ranks: ['2', '2'] }
];

function getScenarioKey(row, dealerUpcard) {
  return `${row.ranks[0]}-${row.ranks[1]}|${dealerUpcard}`;
}

async function loadStrategyMap() {
  if (cachedStrategyMap) return cachedStrategyMap;
  try {
    const res = await fetch("/api/strategy-map");
    if (!res.ok) throw new Error();
    cachedStrategyMap = await res.json();
    return cachedStrategyMap;
  } catch (error) {
    console.error("Failed to load strategy map data");
    return null;
  }
}

async function drawHeatmap() {
  const container = document.getElementById("heatmapTablesContainer");
  if (!container) return;

  const strategy = await loadStrategyMap();
  if (!strategy) {
    container.innerHTML = `<p class="table-loading text-ko">Error loading strategy map data.</p>`;
    return;
  }

  let rows = [];
  if (currentHeatmapCategory === "hard") {
    rows = HARD_ROWS;
  } else if (currentHeatmapCategory === "soft") {
    rows = SOFT_ROWS;
  } else {
    rows = PAIR_ROWS;
  }

  let html = `<table class="heatmap-grid-table">`;
  html += `<thead><tr><th>Player</th>`;
  HEATMAP_COLUMNS.forEach(col => {
    html += `<th>${col}</th>`;
  });
  html += `</tr></thead><tbody>`;

  rows.forEach(row => {
    html += `<tr><td><strong>${row.label}</strong></td>`;
    HEATMAP_COLUMNS.forEach(col => {
      const key = getScenarioKey(row, col);
      const action = strategy[key];
      
      let letter = "?";
      let actionClass = "";
      if (action) {
        if (action === "stand") { letter = "S"; actionClass = "cell-stand-action"; }
        else if (action === "hit") { letter = "H"; actionClass = "cell-hit-action"; }
        else if (action === "double") { letter = "D"; actionClass = "cell-double-action"; }
        else if (action === "split") { letter = "P"; actionClass = "cell-split-action"; }
      }

      let performanceClass = "";
      if (currentUser) {
        if (!attemptedKeys.includes(key)) {
          performanceClass = "cell-unattempted";
        } else if (activeErrorKeys.includes(key)) {
          performanceClass = "cell-error";
        } else {
          performanceClass = "cell-mastered";
        }
      }

      html += `<td class="heatmap-cell ${performanceClass || actionClass}" title="Player: ${row.label} vs Dealer: ${col} -> ${action ? action.toUpperCase() : 'N/A'}">${letter}</td>`;
    });
    html += `</tr>`;
  });

  html += `</tbody></table>`;
  container.innerHTML = html;
}

// Map bindings for strategy modal
const strategyToggleBtn = document.getElementById("strategyToggleBtn");
const strategyModal = document.getElementById("strategyModal");
const closeStrategyBtn = document.getElementById("closeStrategyBtn");

if (strategyToggleBtn && strategyModal) {
  strategyToggleBtn.addEventListener("click", () => {
    strategyModal.classList.add("open");
    drawHeatmap();
  });
}

if (closeStrategyBtn && strategyModal) {
  closeStrategyBtn.addEventListener("click", () => {
    strategyModal.classList.remove("open");
  });
}

if (strategyModal) {
  strategyModal.addEventListener("click", (e) => {
    if (e.target === strategyModal) {
      strategyModal.classList.remove("open");
    }
  });
}

// Tab selections bindings
const tabHardBtn = document.getElementById("tabHardBtn");
const tabSoftBtn = document.getElementById("tabSoftBtn");
const tabPairsBtn = document.getElementById("tabPairsBtn");

function setActiveTab(activeBtn) {
  [tabHardBtn, tabSoftBtn, tabPairsBtn].forEach(btn => {
    if (btn) btn.classList.remove("active-tab");
  });
  if (activeBtn) activeBtn.classList.add("active-tab");
}

if (tabHardBtn) {
  tabHardBtn.addEventListener("click", () => {
    currentHeatmapCategory = "hard";
    setActiveTab(tabHardBtn);
    drawHeatmap();
  });
}

if (tabSoftBtn) {
  tabSoftBtn.addEventListener("click", () => {
    currentHeatmapCategory = "soft";
    setActiveTab(tabSoftBtn);
    drawHeatmap();
  });
}

if (tabPairsBtn) {
  tabPairsBtn.addEventListener("click", () => {
    currentHeatmapCategory = "pairs";
    setActiveTab(tabPairsBtn);
    drawHeatmap();
  });
}

// Fetch leaderboard on load
fetchLeaderboard();
