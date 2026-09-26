/**
 * NeuroScan AI — Single-Page Brain MRI Analysis & Research Engine
 * Manages File Uploads, REST API Inference Calls, Multi-XAI Visualizations, and Research Dashboards.
 */

const API_BASE = "/api";

const $ = id => document.getElementById(id);

let selectedFile = null;
let lastResultData = null;
let currentRequestId = 0;
let isAnalyzing = false;

const setText = (id, val) => {
  const el = $(id);
  if (el) el.textContent = val !== undefined && val !== null ? val : "—";
};

function normalizeTumorLabel(label) {
  if (!label) return "—";
  const l = label.toString().toLowerCase().trim();
  if (l === "glioma") return "Glioma";
  if (l === "meningioma") return "Meningioma";
  if (l === "notumor" || l === "no tumor") return "No Tumor";
  if (l === "pituitary" || l === "pituitary tumor") return "Pituitary Tumor";
  return label;
}

function buildAnalysisResult(data) {
  if (data.is_valid_mri === false) {
    return {
      prediction: "Not Available",
      calibratedConfidence: null,
      rawConfidence: null,
      tumorDetected: false,
      riskLevel: "Rejected",
      color: "#ef4444",
      description: data.rejection_message || data.reason || "Validation failed.",
      severity: "Rejected",
      modelUsed: "None (Pipeline Halted)",
      dateTime: new Date().toLocaleString()
    };
  }

  const rawConf = data.raw_confidence !== undefined ? data.raw_confidence : (data.confidence || 0);
  const calConf = data.calibrated_confidence !== undefined ? data.calibrated_confidence : 0;
  const predLabel = normalizeTumorLabel(data.prediction || data.display_name);
  const isNoTumor = (data.class_id === "notumor" || predLabel === "No Tumor" || data.prediction === "notumor");
  const tumorDetected = data.tumor_detected !== undefined ? data.tumor_detected : !isNoTumor;
  const riskLevel = data.risk_level || data.severity || (isNoTumor ? "None" : "Medium");

  return {
    prediction: predLabel,
    calibratedConfidence: calConf,
    rawConfidence: rawConf,
    tumorDetected: tumorDetected,
    riskLevel: riskLevel,
    color: data.color || (isNoTumor ? "#22c55e" : "#6366f1"),
    description: data.description || "",
    severity: data.severity || riskLevel,
    modelUsed: data.model_used || "EfficientNetB0_Transfer",
    dateTime: new Date().toLocaleString()
  };
}

function resetResultsState() {
  lastResultData = null;
  const resultsSec = $("resultsSection");
  if (resultsSec) resultsSec.classList.add("hidden");

  const invalidAlert = $("invalidScanAlert");
  if (invalidAlert) invalidAlert.classList.add("hidden");
  const uncertaintyBanner = $("uncertaintyBanner");
  if (uncertaintyBanner) uncertaintyBanner.classList.add("hidden");
  const xaiCard = document.querySelector(".xai-card");
  if (xaiCard) xaiCard.classList.remove("hidden");

  setText("diagnosisName", "—");
  setText("diagnosisDesc", "—");
  setText("calibratedConfVal", "0%");
  setText("rawConfVal", "0%");
  setText("severityText", "—");

  setText("resPtName", "—");
  setText("resPtId", "—");
  setText("resPtAgeGender", "—");
  setText("resRefDoctor", "—");
  setText("resFileName", "—");
  setText("resTumorStatus", "—");
  setText("resTumorType", "—");
  setText("resConfidence", "—");
  setText("resModelName", "—");
  setText("resDateTime", "—");
}

let currentUser = null;

// ─────────────────────────────────────────────
// NEUROSCAN AI AUTHENTICATION & SESSION MANAGEMENT
// ─────────────────────────────────────────────
function initAuthControls() {
  // Password Visibility Toggle
  const pwdToggleBtn = $("nsPwdToggleBtn");
  if (pwdToggleBtn) {
    pwdToggleBtn.addEventListener("click", () => {
      const pwdInput = $("nsPassword");
      if (pwdInput) {
        if (pwdInput.type === "password") {
          pwdInput.type = "text";
          pwdToggleBtn.textContent = "🙈";
        } else {
          pwdInput.type = "password";
          pwdToggleBtn.textContent = "👁️";
        }
      }
    });
  }

  // Login Form Submission with Loading & Validation
  const loginForm = $("nsLoginForm");
  if (loginForm) {
    loginForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const alertBox = $("authErrorMessage");
      const emailInput = $("nsEmail");
      const passwordInput = $("nsPassword");
      const submitBtn = $("nsSignInBtn");
      const spinner = $("nsSignInSpinner");

      const email = emailInput ? emailInput.value.trim() : "";
      const password = passwordInput ? passwordInput.value : "";

      // Reset alert
      if (alertBox) {
        alertBox.classList.add("hidden");
        alertBox.textContent = "";
      }

      // Basic Validation
      if (!email || !email.includes("@")) {
        showAuthError("Please enter a valid email address.");
        return;
      }
      if (!password || password.length < 4) {
        showAuthError("Please enter your password.");
        return;
      }

      // Simulate Authentication Loading State
      if (submitBtn) submitBtn.disabled = true;
      if (spinner) spinner.classList.remove("hidden");

      setTimeout(() => {
        if (submitBtn) submitBtn.disabled = false;
        if (spinner) spinner.classList.add("hidden");

        const namePart = email.split("@")[0].replace(".", " ").replace(/\b\w/g, c => c.toUpperCase());
        loginUser({
          name: namePart || "Clinical Researcher",
          role: "Clinical Staff",
          avatar: getInitials(namePart)
        });
      }, 600);
    });
  }

  // Google OAuth Placeholder Handler
  const googleBtn = $("googleAuthBtn");
  if (googleBtn) {
    googleBtn.addEventListener("click", () => {
      loginUser({ name: "Google User", role: "Clinical Researcher (Google Auth)", avatar: "G" });
    });
  }

  // Microsoft OAuth Placeholder Handler
  const msBtn = $("msAuthBtn");
  if (msBtn) {
    msBtn.addEventListener("click", () => {
      loginUser({ name: "Microsoft User", role: "Clinical Researcher (MS Auth)", avatar: "MS" });
    });
  }

  // Cancel Button Handler (Proceed to Workspace as Guest/Demo)
  const cancelBtn = $("nsCancelBtn");
  if (cancelBtn) {
    cancelBtn.addEventListener("click", () => {
      loginUser({ name: "Guest User", role: "Guest Evaluator", avatar: "GU" });
    });
  }

  // Forgot Password Handler
  const forgotPwdLink = $("nsForgotPwdLink");
  if (forgotPwdLink) {
    forgotPwdLink.addEventListener("click", (e) => {
      e.preventDefault();
      alert("🔒 Password Reset: Instructions have been dispatched to your registered medical institution email.");
    });
  }

  // Sign Up Handler
  const signUpLink = $("nsSignUpLink");
  if (signUpLink) {
    signUpLink.addEventListener("click", (e) => {
      e.preventDefault();
      alert("📋 Registration Portal: Contact your clinical network administrator to provision a new NeuroScan AI license.");
    });
  }

  // Sign Out Handler (From Navbar)
  const signOutBtn = $("signOutBtn");
  if (signOutBtn) {
    signOutBtn.addEventListener("click", () => {
      logoutUser();
    });
  }
}

function showAuthError(msg) {
  const alertBox = $("authErrorMessage");
  if (alertBox) {
    alertBox.textContent = `⚠️ ${msg}`;
    alertBox.classList.remove("hidden");
  }
}

function getInitials(name) {
  if (!name) return "NS";
  const parts = name.trim().split(" ");
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return name.slice(0, 2).toUpperCase();
}

function loginUser(user) {
  currentUser = user;
  
  // Hide Auth Screen, Show Main Workspace
  const authScreen = $("authScreen");
  if (authScreen) authScreen.classList.add("hidden");

  const mainWorkspace = $("mainWorkspace");
  if (mainWorkspace) mainWorkspace.classList.remove("hidden");

  const userProfileBadge = $("userProfileBadge");
  if (userProfileBadge) userProfileBadge.classList.remove("hidden");

  setText("userName", user.name);
  setText("userRole", user.role);
  setText("userAvatar", user.avatar);

  window.scrollTo({ top: 0, behavior: "smooth" });
}

function logoutUser() {
  currentUser = null;
  resetResultsState();

  const mainWorkspace = $("mainWorkspace");
  if (mainWorkspace) mainWorkspace.classList.add("hidden");

  const userProfileBadge = $("userProfileBadge");
  if (userProfileBadge) userProfileBadge.classList.add("hidden");

  const authScreen = $("authScreen");
  if (authScreen) authScreen.classList.remove("hidden");

  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ─────────────────────────────────────────────
// INIT & TAB NAVIGATION
// ─────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  if (document.activeElement) document.activeElement.blur();
  initAuthControls();
  initTabNavigation();
  initUploadControls();
  initSampleSelectors();
  initHistoryControls();
  checkServerHealth();

  // Button Listeners
  $("analyzeBtn").addEventListener("click", runAnalysisWorkflow);
  $("changeMriBtn").addEventListener("click", clearFile);
  $("downloadPdfBtn").addEventListener("click", downloadPdfReport);
  if ($("printPdfBtn")) $("printPdfBtn").addEventListener("click", printPdfReport);
  $("reloadLabBtn").addEventListener("click", loadModelComparison);
});

function initTabNavigation() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      const targetId = tab.getAttribute("data-tab");
      const pane = $(targetId);
      if (pane) pane.classList.add("active");

      // Lazy load tab data
      if (targetId === "tab-history") loadHistoryData();
      if (targetId === "tab-lab") loadModelComparison();
      if (targetId === "tab-calibration") loadCalibrationData();
      if (targetId === "tab-robustness") loadRobustnessData();
      if (targetId === "tab-errors") loadErrorAnalysisData();
      if (targetId === "tab-ablation") loadAblationData();
    });
  });
}

// ─────────────────────────────────────────────
// SERVER HEALTH CHECK
// ─────────────────────────────────────────────
async function checkServerHealth() {
  const dot = document.querySelector(".status-dot");
  const text = $("statusText");
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      if (dot) dot.classList.add("online");
      if (text) text.textContent = "Server Ready";
    } else {
      if (text) text.textContent = "AI model unavailable";
      if (dot) dot.classList.remove("online");
    }
  } catch (err) {
    if (text) text.textContent = "AI model unavailable";
    if (dot) dot.classList.remove("online");
  }
}

// ─────────────────────────────────────────────
// UPLOAD & FILE CONTROLS
// ─────────────────────────────────────────────
function initUploadControls() {
  const box = $("uploadBox");
  const fileInput = $("fileInput");

  box.addEventListener("dragover", e => { e.preventDefault(); box.style.borderColor = "#6366f1"; });
  box.addEventListener("dragleave", e => { e.preventDefault(); box.style.borderColor = ""; });
  box.addEventListener("drop", e => {
    e.preventDefault();
    box.style.borderColor = "";
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });
}

function handleFile(file) {
  selectedFile = file;
  resetResultsState();
  updateModalityCard(null);
  const reader = new FileReader();
  reader.onload = e => {
    const img = new Image();
    img.onload = () => {
      $("previewImg").src = e.target.result;
      setText("previewFname", file.name);
      setText("previewFsize", `${(file.size / 1024).toFixed(1)} KB`);
      setText("previewFdims", `${img.width} × ${img.height} px`);
      
      // Show Preview Card and Enable Analyze Button
      $("previewCard").classList.remove("hidden");
      $("analyzeBtn").disabled = false;
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

function clearFile() {
  selectedFile = null;
  $("fileInput").value = "";
  $("previewImg").src = "";
  $("previewCard").classList.add("hidden");
  $("analyzeBtn").disabled = true;
  resetResultsState();
  document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("selected"));
}

function initSampleSelectors() {
  document.querySelectorAll(".sample-card").forEach(card => {
    card.addEventListener("click", async () => {
      const cls = card.getAttribute("data-class");
      resetResultsState();
      try {
        const res = await fetch(`${API_BASE}/samples`);
        const data = await res.json();
        if (data.success && data.samples[cls]) {
          const sample = data.samples[cls];
          const blob = await (await fetch(sample.data_url)).blob();
          const file = new File([blob], sample.filename, { type: blob.type });
          handleFile(file);
        }
      } catch (err) {
        console.error("Error loading sample:", err);
      }
    });
  });
}

// ─────────────────────────────────────────────
// SINGLE-PAGE MRI INFERENCE PIPELINE
// ─────────────────────────────────────────────
async function runAnalysisWorkflow() {
  if (isAnalyzing) return;
  if (!selectedFile) {
    alert("Please select or upload an MRI scan first.");
    return;
  }

  isAnalyzing = true;
  const analyzeBtn = $("analyzeBtn");
  if (analyzeBtn) analyzeBtn.disabled = true;

  const requestId = ++currentRequestId;
  resetResultsState();

  // Reveal loading overlay and reset step active states
  $("loadingOverlay").classList.remove("hidden");
  animateLoadingSteps();

  const formData = new FormData();
  formData.append("image", selectedFile);
  formData.append("patient_name", $("ptName") ? $("ptName").value.trim() : "");
  formData.append("patient_id", $("ptId") ? $("ptId").value.trim() : "");
  formData.append("age", $("ptAge") ? $("ptAge").value.trim() : "");
  formData.append("gender", $("ptGender") ? $("ptGender").value : "");
  formData.append("referring_doctor", $("refDoctor") ? $("refDoctor").value.trim() : "");

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: formData,
    });

    if (requestId !== currentRequestId) return;

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (requestId !== currentRequestId) return;

    if (!data.success) throw new Error(data.error || "Analysis failed");

    lastResultData = data;
    lastResultData.patient_info = getPatientInfo();

    $("loadingOverlay").classList.add("hidden");
    $("resultsSection").classList.remove("hidden");
    renderResults(data);
    if (data.saved_to_history || data.is_valid_mri) {
      loadHistoryData();
    }
    $("resultsSection").scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (err) {
    if (requestId !== currentRequestId) return;
    $("loadingOverlay").classList.add("hidden");
    alert(`Analysis unavailable\n\nUnable to obtain a valid model prediction: ${err.message}`);
  } finally {
    isAnalyzing = false;
    if (selectedFile && $("previewCard") && !$("previewCard").classList.contains("hidden")) {
      if (analyzeBtn) analyzeBtn.disabled = false;
    }
  }
}

function animateLoadingSteps() {
  const steps = [1, 2, 3, 4];
  steps.forEach((s, idx) => {
    setTimeout(() => {
      steps.forEach(i => {
        const el = $(`step${i}`);
        if (el) el.classList.remove("active");
      });
      const curr = $(`step${s}`);
      if (curr) curr.classList.add("active");
    }, idx * 300);
  });
}

function buildAnalysisResult(data) {
  if (data.is_valid_mri === false) {
    return {
      prediction: "Not Available",
      calibratedConfidence: null,
      rawConfidence: null,
      tumorDetected: false,
      riskLevel: "Rejected",
      color: "#ef4444",
      description: data.rejection_message || data.reason || "Validation failed.",
      severity: "Rejected",
      modelUsed: "None (Pipeline Halted)",
      dateTime: new Date().toLocaleString()
    };
  }

  const rawConf = data.raw_confidence !== undefined ? data.raw_confidence : (data.confidence || 0);
  const calConf = data.calibrated_confidence !== undefined ? data.calibrated_confidence : 0;
  const predLabel = normalizeTumorLabel(data.prediction || data.display_name);
  const isNoTumor = (data.class_id === "notumor" || predLabel === "No Tumor" || data.prediction === "notumor");
  const tumorDetected = data.tumor_detected !== undefined ? data.tumor_detected : !isNoTumor;
  const riskLevel = data.risk_level || data.severity || (isNoTumor ? "None" : "Medium");

  return {
    prediction: predLabel,
    calibratedConfidence: calConf,
    rawConfidence: rawConf,
    tumorDetected: tumorDetected,
    riskLevel: riskLevel,
    color: data.color || (isNoTumor ? "#22c55e" : "#6366f1"),
    description: data.description || "",
    severity: data.severity || riskLevel,
    modelUsed: data.model_used || "EfficientNetB0_Transfer",
    dateTime: new Date().toLocaleString()
  };
}

function resetResultsState() {
  lastResultData = null;
  const resultsSec = $("resultsSection");
  if (resultsSec) resultsSec.classList.add("hidden");

  setText("diagnosisName", "—");
  setText("diagnosisDesc", "—");
  setText("calibratedConfVal", "0%");
  setText("rawConfVal", "0%");
  setText("severityText", "—");

  setText("resPtName", "—");
  setText("resPtId", "—");
  setText("resPtAgeGender", "—");
  setText("resRefDoctor", "—");
  setText("resFileName", "—");
  setText("resTumorStatus", "—");
  setText("resTumorType", "—");
  setText("resConfidence", "—");
  setText("resModelName", "—");
  setText("resDateTime", "—");
}

function updateModalityCard(data) {
  const card = $("modalityStatusCard");
  const icon = $("modalityIcon");
  const title = $("modalityTitle");
  const detected = $("modalityDetectedText");
  const conf = $("modalityConfText");
  const msg = $("modalityStatusMsg");
  const analyzeBtn = $("analyzeBtn");

  if (!card) return;

  if (!data) {
    // Initial upload default
    card.className = "modality-status-card mri-pending";
    if (icon) icon.textContent = "🔍";
    if (title) title.textContent = "Ready for Modality Check";
    if (detected) detected.textContent = "Pending";
    if (conf) conf.textContent = "—";
    if (msg) msg.textContent = "Click 'Analyze MRI' to run validation pipeline.";
    if (analyzeBtn) analyzeBtn.disabled = false;
    return;
  }

  if (data.is_valid_mri === false) {
    const isCt = (data.status === "rejected_ct" || data.modality === "CT");
    card.className = isCt ? "modality-status-card ct-rejected" : "modality-status-card unknown-rejected";
    if (icon) icon.textContent = "✕";
    if (title) title.textContent = data.rejection_title || (isCt ? "CT Scan Detected" : "Invalid Image Detected");
    if (detected) detected.textContent = data.modality || (isCt ? "CT" : "Invalid Image");
    if (conf) conf.textContent = data.modality_confidence ? `${data.modality_confidence}%` : "100.0%";
    if (msg) msg.textContent = isCt ? "MRI image required. Please upload a brain MRI scan." : "Invalid image detected. Please upload a valid brain MRI scan.";
    if (analyzeBtn) analyzeBtn.disabled = true;
  } else {
    card.className = "modality-status-card mri-verified";
    if (icon) icon.textContent = "✓";
    if (title) title.textContent = "MRI Scan Verified";
    if (detected) detected.textContent = "MRI";
    if (conf) conf.textContent = data.modality_confidence ? `${data.modality_confidence}%` : "—";
    if (msg) msg.textContent = "Ready for tumor analysis";
    if (analyzeBtn) analyzeBtn.disabled = false;
  }
}

function renderResults(data) {
  const analysisResult = buildAnalysisResult(data);
  updateModalityCard(data);

  // Invalid scan handling
  const invalidAlert = $("invalidScanAlert");
  if (data.is_valid_mri === false) {
    if (invalidAlert) invalidAlert.classList.remove("hidden");
    setText("invalidAlertTitle", data.rejection_title || "Validation Failed");
    setText("invalidAlertSubtitle", data.rejection_message || data.reason || "Input validation failed.");
    const rList = $("invalidReasonsList");
    if (rList) {
      rList.innerHTML = "";
      (data.characteristics || []).forEach(r => {
        const li = document.createElement("li");
        li.textContent = r;
        rList.appendChild(li);
      });
    }
  } else {
    if (invalidAlert) invalidAlert.classList.add("hidden");
  }

  // Uncertainty banner
  const uncertaintyBanner = $("uncertaintyBanner");
  if (data.is_uncertain && data.is_valid_mri) {
    if (uncertaintyBanner) uncertaintyBanner.classList.remove("hidden");
    setText("uncertaintyDesc", data.uncertainty_reason || "The model cannot confidently distinguish between top classes.");
  } else {
    if (uncertaintyBanner) uncertaintyBanner.classList.add("hidden");
  }

  // 1. Tumor Detection Summary Section
  setText("diagnosisName", analysisResult.prediction);
  const diagNameEl = $("diagnosisName");
  if (diagNameEl) diagNameEl.style.color = analysisResult.color;

  setText("diagnosisDesc", analysisResult.description);
  setText("calibratedConfVal", analysisResult.calibratedConfidence !== null ? `${analysisResult.calibratedConfidence.toFixed(1)}%` : "Not Available");
  setText("rawConfVal", analysisResult.rawConfidence !== null ? `${analysisResult.rawConfidence.toFixed(1)}%` : "Not Available");

  const badge = $("severityBadge");
  setText("severityText", analysisResult.riskLevel);
  if (badge) {
    badge.style.color = analysisResult.color;
    badge.style.borderColor = analysisResult.color;
  }

  // Multi-XAI Visualizations (Blocked/Hidden for Rejected Images)
  const xaiCard = document.querySelector(".xai-card");
  if (data.is_valid_mri === false) {
    if (xaiCard) xaiCard.classList.add("hidden");
  } else {
    if (xaiCard) xaiCard.classList.remove("hidden");
    const origSrc = data.original_b64 || "";
    const overlaySrc = data.overlay_b64 || origSrc;
    if ($("resultOriginal")) $("resultOriginal").src = origSrc;
    if ($("resultOverlay")) $("resultOverlay").src = overlaySrc;
    if ($("resultIG")) $("resultIG").src = data.ig_b64 || overlaySrc;
    if ($("resultLIME")) $("resultLIME").src = data.lime_b64 || overlaySrc;
  }

  // Faithfulness Evaluation (Blocked for Rejected Images or Fast Mode)
  const faithBox = $("faithfulnessBox");
  if (data.is_valid_mri === false || !data.faithfulness) {
    if (faithBox) faithBox.classList.add("hidden");
  } else if (data.faithfulness) {
    if (faithBox) faithBox.classList.remove("hidden");
    setText("impDropVal", `-${data.faithfulness.important_drop_pct}%`);
    setText("randDropVal", `-${data.faithfulness.random_drop_pct}%`);
    const fBadge = $("faithfulnessBadge");
    if (fBadge) {
      if (data.faithfulness.is_faithful) {
        fBadge.textContent = "FAITHFUL EXPLANATION";
        fBadge.style.color = "#22c55e";
        fBadge.style.borderColor = "#22c55e";
      } else {
        fBadge.textContent = "WEAK EVIDENCE";
        fBadge.style.color = "#f59e0b";
        fBadge.style.borderColor = "#f59e0b";
      }
    }
  }

  // Confidence Scores Bars (Blocked for Rejected Images)
  const scoresCard = $("scoresCard");
  if (data.is_valid_mri === false) {
    if (scoresCard) scoresCard.classList.add("hidden");
  } else {
    if (scoresCard) scoresCard.classList.remove("hidden");
    renderScores(data.scores);
  }

  // Characteristics & Treatment
  renderCharacteristics(data.characteristics || []);
  setText("treatmentText", data.treatment || (data.is_valid_mri === false ? "Upload a valid brain MRI scan for medical tumor assessment." : "No treatment recommended."));

  // 2. AI Analysis Result Summary Card
  const ptName = $("ptName").value || "Anonymous Patient";
  const ptId = $("ptId").value || "PT-2026-EX";
  const ptAge = $("ptAge").value || "45";
  const ptGender = $("ptGender").value || "Male";
  const refDoctor = $("refDoctor").value || "Dr. Nivas";
  const fileName = selectedFile ? selectedFile.name : "MRI_Scan.jpg";

  setText("resPtName", ptName);
  setText("resPtId", ptId);
  setText("resPtAgeGender", `${ptAge} / ${ptGender}`);
  setText("resRefDoctor", refDoctor);
  setText("resFileName", fileName);
  setText("resTumorStatus", analysisResult.tumorDetected ? "Tumor Detected" : "No Tumor Detected");
  setText("resTumorType", analysisResult.prediction);
  setText("resConfidence", `${analysisResult.calibratedConfidence.toFixed(1)}%`);
  setText("resModelName", analysisResult.modelUsed);
  setText("resDateTime", analysisResult.dateTime);
}

function renderScores(scores) {
  if (!scores) return;
  const grid = $("scoresGrid");
  grid.innerHTML = "";

  const CLASS_COLORS = {
    glioma: "#ef4444",
    meningioma: "#f59e0b",
    notumor: "#22c55e",
    pituitary: "#8b5cf6",
  };
  const CLASS_LABELS = {
    glioma: "Glioma",
    meningioma: "Meningioma",
    notumor: "No Tumor",
    pituitary: "Pituitary",
  };

  const sorted = Object.entries(scores).sort((a, b) => b[1].calibrated_confidence - a[1].calibrated_confidence);

  sorted.forEach(([key, data]) => {
    const color = CLASS_COLORS[key] || "#6366f1";
    const label = CLASS_LABELS[key] || key;
    const conf = data.calibrated_confidence || 0;

    const item = document.createElement("div");
    item.className = "score-item";
    item.innerHTML = `
      <span class="score-label">${label}</span>
      <div class="score-bar-wrap">
        <div class="score-bar" style="background: ${color}; width: 0%;" data-target="${conf}"></div>
      </div>
      <span class="score-pct">${conf.toFixed(1)}%</span>
    `;
    grid.appendChild(item);
  });

  setTimeout(() => {
    grid.querySelectorAll(".score-bar").forEach(bar => {
      bar.style.width = bar.dataset.target + "%";
    });
  }, 100);
}

function renderCharacteristics(chars) {
  const list = $("characteristicsList");
  list.innerHTML = "";
  chars.forEach(c => {
    const li = document.createElement("li");
    li.textContent = c;
    list.appendChild(li);
  });
}

// ─────────────────────────────────────────────
// DASHBOARD 1: MODEL LABORATORY
// ─────────────────────────────────────────────
async function loadModelComparison() {
  const tbody = $("modelCompTbody");
  tbody.innerHTML = '<tr><td colspan="11" class="text-center">Loading model benchmarks...</td></tr>';
  try {
    const res = await fetch(`${API_BASE}/experiments/comparison`);
    const data = await res.json();
    if (!data.success) return;

    tbody.innerHTML = "";
    let maxF1 = -1;
    let bestName = "";

    Object.entries(data.comparison).forEach(([name, m]) => {
      if (m.f1_weighted > maxF1) {
        maxF1 = m.f1_weighted;
        bestName = name;
      }

      const tr = document.createElement("tr");
      tr.id = `row-${name}`;
      tr.innerHTML = `
        <td><strong>${name}</strong></td>
        <td>${m.accuracy}%</td>
        <td>${m.precision_weighted}%</td>
        <td>${m.recall_weighted}%</td>
        <td><strong>${m.f1_weighted}%</strong></td>
        <td>${m.per_class.glioma.recall}%</td>
        <td>${m.per_class.meningioma.recall}%</td>
        <td>${m.per_class.notumor.recall}%</td>
        <td>${m.per_class.pituitary.recall}%</td>
        <td>${m.parameters.toLocaleString()}</td>
        <td>${m.inference_time_ms} ms</td>
      `;
      tbody.appendChild(tr);
    });

    if (bestName) {
      const bestRow = $(`row-${bestName}`);
      if (bestRow) bestRow.classList.add("best-row");
      $("bestModelTitle").textContent = `Selected Best Model: ${bestName}`;
      $("bestModelSubtitle").textContent = `Achieved highest balanced F1-score of ${maxF1}% across all four tumor categories.`;
    }
  } catch (err) {
    console.error("Error loading model comparison:", err);
  }
}

// ─────────────────────────────────────────────
// DASHBOARD 2: CALIBRATION DATA
// ─────────────────────────────────────────────
async function loadCalibrationData() {
  try {
    const res = await fetch(`${API_BASE}/experiments/calibration`);
    const data = await res.json();
    if (!data.success) return;

    const cal = data.calibration;
    $("eceBeforeVal").textContent = `${cal.ece_before_pct}%`;
    $("eceAfterVal").textContent = `${cal.ece_after_pct}%`;
    $("tempVal").textContent = cal.temperature_T.toFixed(3);

    const tbody = $("calibrationTbody");
    tbody.innerHTML = "";

    (cal.bins_after || []).forEach(bin => {
      const gap = Math.abs(bin.confidence - bin.accuracy) * 100.0;
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${bin.bin_lower.toFixed(1)} – ${bin.bin_upper.toFixed(1)}</td>
        <td>${bin.count}</td>
        <td>${(bin.confidence * 100.0).toFixed(1)}%</td>
        <td>${(bin.accuracy * 100.0).toFixed(1)}%</td>
        <td><span class="${gap > 5 ? 'text-danger' : 'text-success'}">${gap.toFixed(1)}%</span></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Error loading calibration:", err);
  }
}

// ─────────────────────────────────────────────
// DASHBOARD 3: ROBUSTNESS DATA
// ─────────────────────────────────────────────
async function loadRobustnessData() {
  try {
    const res = await fetch(`${API_BASE}/experiments/robustness`);
    const data = await res.json();
    if (!data.success) return;

    const tbody = $("robustnessTbody");
    tbody.innerHTML = "";

    data.robustness.forEach(r => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${r.transformation}</strong></td>
        <td>${r.accuracy}%</td>
        <td>${r.f1_score}%</td>
        <td>${r.avg_confidence}%</td>
        <td><span class="${r.confidence_change_pct < 0 ? 'text-danger' : 'text-success'}">${r.confidence_change_pct > 0 ? '+' : ''}${r.confidence_change_pct}%</span></td>
        <td><strong>${r.stability_pct}%</strong></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Error loading robustness:", err);
  }
}

// ─────────────────────────────────────────────
// DASHBOARD 4: ERROR ANALYSIS DATA
// ─────────────────────────────────────────────
async function loadErrorAnalysisData() {
  try {
    const res = await fetch(`${API_BASE}/experiments/error-analysis`);
    const data = await res.json();
    if (!data.success) return;

    const errData = data.error_analysis;
    $("totalTestVal").textContent = errData.total_test_images;
    $("correctVal").textContent = errData.correct_predictions;
    $("incorrectVal").textContent = errData.incorrect_predictions;
    $("gliomaMeningiomaErrorsVal").textContent = (errData.glioma_as_meningioma_errors + errData.meningioma_as_glioma_errors);

    const gallery = $("errorGallery");
    gallery.innerHTML = "";

    (errData.gallery || []).forEach(item => {
      const div = document.createElement("div");
      div.className = "error-item";
      div.innerHTML = `
        <div class="error-images">
          <img src="${item.original_b64}" alt="Original MRI" />
          <img src="${item.overlay_b64}" alt="Grad-CAM" />
        </div>
        <div class="error-details">
          <span><strong>True:</strong> <span class="text-success">${item.true_class.toUpperCase()}</span></span>
          <span><strong>Predicted:</strong> <span class="text-danger">${item.predicted_class.toUpperCase()}</span></span>
          <span><strong>Confidence:</strong> ${item.confidence}%</span>
        </div>
      `;
      gallery.appendChild(div);
    });
  } catch (err) {
    console.error("Error loading error analysis:", err);
  }
}

// ─────────────────────────────────────────────
// DASHBOARD 5: ABLATION STUDY
// ─────────────────────────────────────────────
async function loadAblationData() {
  try {
    const res = await fetch(`${API_BASE}/experiments/ablation`);
    const data = await res.json();
    if (!data.success) return;

    const tbody = $("ablationTbody");
    tbody.innerHTML = "";

    data.ablation.forEach(exp => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${exp.Experiment}</strong></td>
        <td>${exp.Accuracy}%</td>
        <td>${exp.Precision}%</td>
        <td>${exp.Recall}%</td>
        <td><strong>${exp.F1_Score}%</strong></td>
        <td>${exp.ECE_pct}%</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Error loading ablation study:", err);
  }
}

// ─────────────────────────────────────────────
// PDF REPORT GENERATOR
// ─────────────────────────────────────────────
function getPatientInfo() {
  const currentPt = (lastResultData && lastResultData.patient_info) ? lastResultData.patient_info : {};
  return {
    name: ($("ptName") && $("ptName").value.trim()) || currentPt.patient_name || "Anonymous Patient",
    patient_id: ($("ptId") && $("ptId").value.trim()) || currentPt.patient_id || "PT-2026-EX",
    age: ($("ptAge") && $("ptAge").value.trim()) || currentPt.age || "45",
    gender: ($("ptGender") && $("ptGender").value) || currentPt.gender || "Male",
    referring_doctor: ($("refDoctor") && $("refDoctor").value.trim()) || currentPt.referring_doctor || "Dr. Nivas",
    file_name: selectedFile ? selectedFile.name : (currentPt.image_filename || "MRI_Scan.jpg"),
    analysis_id: lastResultData ? lastResultData.analysis_id : undefined,
  };
}

async function downloadPdfReport() {
  if (!lastResultData) {
    alert("Please run an MRI analysis first before downloading a report.");
    return;
  }

  const patientInfo = getPatientInfo();

  try {
    const res = await fetch(`${API_BASE}/report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prediction_data: lastResultData,
        patient_info: patientInfo,
      }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `brain_mri_research_report_${patientInfo.patient_id}.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  } catch (err) {
    alert(`❌ PDF Generation Error: ${err.message}`);
  }
}

async function printPdfReport() {
  if (!lastResultData) {
    alert("Please run an MRI analysis first before printing a report.");
    return;
  }

  const patientInfo = getPatientInfo();

  try {
    const res = await fetch(`${API_BASE}/report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prediction_data: lastResultData,
        patient_info: patientInfo,
      }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob = await res.blob();
    const blobUrl = window.URL.createObjectURL(blob);

    // Create invisible iframe for seamless print triggering
    const printIframe = document.createElement("iframe");
    printIframe.style.position = "fixed";
    printIframe.style.right = "0";
    printIframe.style.bottom = "0";
    printIframe.style.width = "0";
    printIframe.style.height = "0";
    printIframe.style.border = "0";
    printIframe.src = blobUrl;
    document.body.appendChild(printIframe);

    printIframe.onload = () => {
      setTimeout(() => {
        try {
          printIframe.contentWindow.focus();
          printIframe.contentWindow.print();
        } catch (e) {
          window.open(blobUrl, "_blank");
        }
      }, 300);
    };
  } catch (err) {
    alert(`❌ PDF Print Error: ${err.message}`);
  }
}

// ─────────────────────────────────────────────
// ANALYSIS HISTORY MODULE & MODAL CONTROLS
// ─────────────────────────────────────────────
function initHistoryControls() {
  const searchInput = $("histSearchInput");
  const predFilter = $("histPredictionFilter");
  const dateFilter = $("histDateFilter");
  const sortOrder = $("histSortOrder");
  const clearBtn = $("clearAllHistoryBtn");
  const closeBtn = $("closeHistoryModalBtn");
  const closeFooterBtn = $("closeModalFooterBtn");
  const backdrop = $("historyModalBackdrop");

  if (searchInput) {
    let timer = null;
    searchInput.addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(loadHistoryData, 300);
    });
  }

  if (predFilter) predFilter.addEventListener("change", loadHistoryData);
  if (dateFilter) dateFilter.addEventListener("change", loadHistoryData);
  if (sortOrder) sortOrder.addEventListener("change", loadHistoryData);
  if (clearBtn) clearBtn.addEventListener("click", confirmClearAllHistory);

  if (closeBtn) closeBtn.addEventListener("click", closeHistoryModal);
  if (closeFooterBtn) closeFooterBtn.addEventListener("click", closeHistoryModal);
  if (backdrop) {
    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) closeHistoryModal();
    });
  }
}

async function loadHistoryData() {
  const search = $("histSearchInput") ? $("histSearchInput").value.trim() : "";
  const pred = $("histPredictionFilter") ? $("histPredictionFilter").value : "all";
  const dateF = $("histDateFilter") ? $("histDateFilter").value : "all";
  const sort = $("histSortOrder") ? $("histSortOrder").value : "newest";

  const url = `${API_BASE}/history?search=${encodeURIComponent(search)}&prediction=${encodeURIComponent(pred)}&date=${encodeURIComponent(dateF)}&sort=${encodeURIComponent(sort)}`;

  try {
    const res = await fetch(url);
    const data = await res.json();
    if (!data.success) throw new Error(data.error || "Failed to load history.");

    // Update Stats Cards
    if (data.stats) {
      setText("histStatTotal", data.stats.total || 0);
      setText("histStatToday", data.stats.today || 0);
      setText("histStatTumor", data.stats.tumor_detected || 0);
      setText("histStatNormal", data.stats.normal || 0);
    }

    renderHistoryTable(data.history || []);
  } catch (err) {
    console.error("Error loading history data:", err);
  }
}

function renderHistoryTable(records) {
  const tbody = $("historyTbody");
  const emptyCard = $("emptyHistoryCard");
  const table = $("historyTable");

  if (!tbody) return;

  if (!records || records.length === 0) {
    tbody.innerHTML = "";
    if (emptyCard) emptyCard.classList.remove("hidden");
    if (table) table.classList.add("hidden");
    return;
  }

  if (emptyCard) emptyCard.classList.add("hidden");
  if (table) table.classList.remove("hidden");

  tbody.innerHTML = "";

  records.forEach(r => {
    const tr = document.createElement("tr");

    const predClass = (r.prediction || "").toLowerCase().replace(/\s+/g, '');
    const badgeClass = ["glioma", "meningioma", "notumor", "pituitary"].includes(predClass)
      ? predClass
      : (r.prediction === "No Tumor" ? "notumor" : "glioma");

    const confDisplay = r.calibrated_confidence !== undefined && r.calibrated_confidence !== null
      ? `${r.calibrated_confidence.toFixed(1)}%`
      : `${(r.confidence || 0).toFixed(1)}%`;

    const statusPill = r.analysis_status === "Uncertain Prediction"
      ? '<span class="hist-status-pill uncertain">Uncertain</span>'
      : '<span class="hist-status-pill">Completed</span>';

    tr.innerHTML = `
      <td><strong>${r.date_str}</strong> <span class="text-muted" style="font-size:0.8rem;">${r.time_str}</span></td>
      <td><strong>${escapeHtml(r.patient_name)}</strong></td>
      <td><code style="color:#818cf8; font-size:0.82rem;">${escapeHtml(r.patient_id)}</code></td>
      <td><span class="text-muted">${escapeHtml(r.image_filename)}</span></td>
      <td><span class="hist-pred-badge ${badgeClass}">${escapeHtml(r.prediction)}</span></td>
      <td><strong>${confDisplay}</strong></td>
      <td>${statusPill}</td>
      <td>
        <div class="hist-action-btns">
          <button type="button" class="btn-action-view" onclick="openHistoryDetailModal('${r.analysis_id}')">👁️ View Details</button>
          <button type="button" class="btn-action-delete" title="Delete Record" onclick="confirmDeleteHistoryRecord('${r.analysis_id}')">🗑️</button>
        </div>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function escapeHtml(str) {
  if (!str) return "";
  return str.toString().replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

async function openHistoryDetailModal(analysis_id) {
  const backdrop = $("historyModalBackdrop");
  const modalBody = $("historyModalBody");
  const modalIdBadge = $("modalAnalysisId");

  if (!backdrop || !modalBody) return;

  if (modalIdBadge) modalIdBadge.textContent = analysis_id;
  modalBody.innerHTML = '<p class="text-center text-muted" style="padding: 2rem;">Loading analysis details...</p>';
  backdrop.classList.remove("hidden");

  try {
    const res = await fetch(`${API_BASE}/history/${analysis_id}`);
    const data = await res.json();
    if (!data.success || !data.detail) throw new Error(data.error || "Failed to load record detail.");

    const d = data.detail;
    setText("modalPatientTitle", `${d.patient_name} — Diagnostic Report`);

    const isNoTumor = !d.tumor_detected;
    const badgeColor = d.color || (isNoTumor ? "#22c55e" : "#ef4444");

    const charListHtml = (d.characteristics || []).map(c => `<li>${escapeHtml(c)}</li>`).join('');

    modalBody.innerHTML = `
      <!-- Top Overview Cards -->
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
        <!-- Left: Patient & Diagnostic Metadata -->
        <div class="card" style="margin: 0;">
          <div class="card-header" style="margin-bottom: 0.75rem;">
            <span class="card-icon">📋</span>
            <h3 class="card-title-sm">Patient & Diagnostic Metadata</h3>
          </div>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem 1rem; font-size: 0.88rem;">
            <div><span class="text-muted">Patient Name:</span> <br /><strong>${escapeHtml(d.patient_name)}</strong></div>
            <div><span class="text-muted">Patient ID:</span> <br /><strong style="color:#818cf8;">${escapeHtml(d.patient_id)}</strong></div>
            <div><span class="text-muted">Age / Gender:</span> <br /><strong>${escapeHtml(d.age || '—')} / ${escapeHtml(d.gender || '—')}</strong></div>
            <div><span class="text-muted">Referring Doctor:</span> <br /><strong>${escapeHtml(d.referring_doctor)}</strong></div>
            <div><span class="text-muted">MRI Filename:</span> <br /><strong style="font-size:0.82rem;">${escapeHtml(d.image_filename)}</strong></div>
            <div><span class="text-muted">Analysis Date & Time:</span> <br /><strong>${d.date_time}</strong></div>
          </div>
        </div>

        <!-- Right: AI Diagnostic Summary -->
        <div class="card" style="margin: 0;">
          <div class="card-header" style="margin-bottom: 0.75rem;">
            <span class="card-icon">🧠</span>
            <h3 class="card-title-sm">AI Prediction Summary</h3>
          </div>
          <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
            <div>
              <div style="font-size: 0.8rem; text-transform: uppercase; color: var(--text-muted);">Predicted Classification</div>
              <div style="font-size: 1.5rem; font-weight: 800; color: ${badgeColor};">${escapeHtml(d.prediction)}</div>
            </div>
            <div style="text-align: right;">
              <div style="font-size: 1.6rem; font-weight: 800; color: #fff;">${d.calibrated_confidence ? d.calibrated_confidence.toFixed(1) : (d.confidence || 0).toFixed(1)}%</div>
              <div style="font-size: 0.75rem; color: var(--text-muted);">Calibrated Confidence</div>
            </div>
          </div>
          <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.75rem;">${escapeHtml(d.description)}</p>
          <div style="display:flex; gap:0.5rem; font-size:0.8rem;">
            <span style="padding: 2px 8px; border-radius: 4px; background: rgba(255,255,255,0.08);">Risk Level: <strong>${escapeHtml(d.risk_level)}</strong></span>
            <span style="padding: 2px 8px; border-radius: 4px; background: rgba(255,255,255,0.08);">Model: <strong>${escapeHtml(d.model_used)}</strong></span>
          </div>
        </div>
      </div>

      <!-- XAI Explainability Gallery -->
      <div class="card xai-card" style="margin: 0;">
        <div class="card-header">
          <span class="card-icon">🔥</span>
          <h3 class="card-title-sm">Saved Explainable AI Visualizations (Multi-XAI)</h3>
        </div>
        <div class="xai-grid">
          <div class="xai-item">
            <img src="${d.original_b64 || ''}" alt="Original MRI" class="result-image" />
            <div class="xai-badge">Original MRI</div>
          </div>
          <div class="xai-item">
            <img src="${d.overlay_b64 || ''}" alt="Grad-CAM" class="result-image" />
            <div class="xai-badge gradcam-badge">Grad-CAM Heatmap</div>
          </div>
          <div class="xai-item">
            <img src="${d.ig_b64 || ''}" alt="Integrated Gradients" class="result-image" />
            <div class="xai-badge ig-badge">Integrated Gradients</div>
          </div>
          <div class="xai-item">
            <img src="${d.lime_b64 || ''}" alt="LIME" class="result-image" />
            <div class="xai-badge lime-badge">MRI + Overlay</div>
          </div>
        </div>
      </div>

      <!-- Characteristics & Medical Guidance -->
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
        <div class="card" style="margin: 0;">
          <div class="card-header" style="margin-bottom: 0.75rem;">
            <span class="card-icon">🧬</span>
            <h3 class="card-title-sm">Tumor Characteristics</h3>
          </div>
          <ul class="char-list" style="font-size: 0.85rem;">
            ${charListHtml || '<li>No specific characteristics recorded.</li>'}
          </ul>
        </div>
        <div class="card" style="margin: 0;">
          <div class="card-header" style="margin-bottom: 0.75rem;">
            <span class="card-icon">🩺</span>
            <h3 class="card-title-sm">Recommended Medical Management</h3>
          </div>
          <p class="treatment-text" style="font-size: 0.85rem;">${escapeHtml(d.treatment)}</p>
        </div>
      </div>
    `;

  } catch (err) {
    modalBody.innerHTML = `<div class="invalid-scan-card card"><p class="text-danger">Failed to load details: ${err.message}</p></div>`;
  }
}

function closeHistoryModal() {
  const backdrop = $("historyModalBackdrop");
  if (backdrop) backdrop.classList.add("hidden");
}

async function confirmDeleteHistoryRecord(analysis_id) {
  if (!confirm(`Are you sure you want to delete analysis history record '${analysis_id}'?`)) {
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/history/${analysis_id}`, { method: "DELETE" });
    const data = await res.json();
    if (!data.success) throw new Error(data.error || "Delete failed.");
    loadHistoryData();
  } catch (err) {
    alert(`Deletion error: ${err.message}`);
  }
}

async function confirmClearAllHistory() {
  if (!confirm("Are you sure you want to clear ALL analysis history?\n\nThis will permanently delete all saved MRI scan records and heatmaps.")) {
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/history`, { method: "DELETE" });
    const data = await res.json();
    if (!data.success) throw new Error(data.error || "Clear failed.");
    loadHistoryData();
  } catch (err) {
    alert(`Clear history error: ${err.message}`);
  }
}
