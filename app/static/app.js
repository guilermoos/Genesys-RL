const state = {
  token: null,
  user: null,
  projects: [],
  templates: [],
};

const selectors = {
  authSection: document.getElementById("auth-section"),
  mainSection: document.getElementById("main-section"),
  notification: document.getElementById("notification"),
  viewContainer: document.getElementById("view-container"),
  userStatus: document.getElementById("user-status"),
};

const apiBase = () => `${window.location.origin}/v1`;

const getAuthHeaders = () => {
  const headers = { "Content-Type": "application/json" };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  return headers;
};

const showMessage = (message, type = "info") => {
  selectors.notification.textContent = message;
  selectors.notification.className = `notification ${type === "error" ? "error" : ""}`;
  selectors.notification.classList.remove("hidden");
};

const hideMessage = () => {
  selectors.notification.classList.add("hidden");
};

const fetchJson = async (path, options = {}) => {
  const response = await fetch(`${apiBase()}${path}`, options);
  const data = await response.json().catch(() => null);

  if (!response.ok) {
    const error = data?.detail || data?.message || response.statusText;
    throw new Error(error || "Erro desconhecido");
  }

  return data;
};

const setToken = (token) => {
  state.token = token;
  localStorage.setItem("genesys_token", token);
};

const clearSession = () => {
  state.token = null;
  state.user = null;
  localStorage.removeItem("genesys_token");
};

const renderAuth = () => {
  selectors.authSection.classList.toggle("hidden", !!state.token);
  selectors.mainSection.classList.toggle("hidden", !state.token);
  selectors.userStatus.textContent = state.user ? `Logado como ${state.user.name} (${state.user.email})` : "Usuário não autenticado";
};

const handleLogin = async () => {
  hideMessage();
  const email = document.getElementById("login-email").value.trim();
  const password = document.getElementById("login-password").value;

  try {
    const data = await fetchJson("/auth/login", {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ email, password }),
    });

    setToken(data.access_token);
    state.user = data.user;
    renderAuth();
    await loadTemplates();
    await loadProjects();
    loadView("projects");
    showMessage("Login realizado com sucesso.");
  } catch (err) {
    showMessage(err.message, "error");
  }
};

const handleRegister = async () => {
  hideMessage();
  const name = document.getElementById("register-name").value.trim();
  const email = document.getElementById("register-email").value.trim();
  const password = document.getElementById("register-password").value;

  try {
    const data = await fetchJson("/auth/register", {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ name, email, password }),
    });

    showMessage(`Usuário ${data.name} criado. Faça login em seguida.`);
  } catch (err) {
    showMessage(err.message, "error");
  }
};

const handleLogout = () => {
  clearSession();
  renderAuth();
  showMessage("Sessão finalizada.");
};

const loadCurrentUser = async () => {
  if (!state.token) return;
  try {
    const user = await fetchJson("/auth/me", { headers: getAuthHeaders() });
    state.user = user;
    renderAuth();
  } catch (err) {
    clearSession();
    renderAuth();
  }
};

const loadTemplates = async () => {
  try {
    const data = await fetchJson("/templates", { headers: getAuthHeaders() });
    state.templates = data.templates || [];
  } catch (err) {
    state.templates = [];
  }
};

const loadProjects = async () => {
  try {
    const data = await fetchJson("/projects", { headers: getAuthHeaders() });
    state.projects = data.items || [];
  } catch (err) {
    state.projects = [];
  }
};

const createProject = async () => {
  hideMessage();
  const name = document.getElementById("project-name").value.trim();
  const description = document.getElementById("project-description").value.trim();
  const template = document.getElementById("project-template").value;

  if (!name || !template) {
    showMessage("Preencha nome e template.", "error");
    return;
  }

  try {
    await fetchJson("/projects", {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ name, description, template_default: template }),
    });

    await loadProjects();
    renderProjects();
    showMessage("Projeto criado com sucesso.");
    document.getElementById("project-name").value = "";
    document.getElementById("project-description").value = "";
  } catch (err) {
    showMessage(err.message, "error");
  }
};

const createTrainingJob = async (projectId, template) => {
  hideMessage();
  const episodes = Number(document.getElementById("training-episodes").value) || 100;
  const maxSteps = Number(document.getElementById("training-max-steps").value) || 100;
  const actionSpace = document.getElementById("training-action-space").value.split(",").map((item) => Number(item.trim())).filter(Boolean);

  if (!projectId || !template || actionSpace.length === 0) {
    showMessage("Projeto ou template inválido.", "error");
    return;
  }

  try {
    await fetchJson(`/jobs/projects/${projectId}/train`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({
        template,
        name: `Treino - ${new Date().toLocaleString()}`,
        config: {
          state_size: 8,
          action_space: actionSpace,
          episodes,
          max_steps: maxSteps,
          gamma: 0.99,
          learning_rate: 0.001,
          epsilon_start: 1.0,
          epsilon_end: 0.01,
          epsilon_decay: 0.995,
          batch_size: 64,
          memory_size: 10000,
          target_update_freq: 100,
        },
      }),
    });

    await renderTraining();
    showMessage("Job de treinamento agendado.");
  } catch (err) {
    showMessage(err.message, "error");
  }
};

const createInference = async () => {
  hideMessage();
  const projectId = document.getElementById("inference-project").value;
  const stateInput = document.getElementById("inference-state").value.trim();
  const modelVersionId = document.getElementById("inference-model-version").value.trim();

  if (!projectId || !stateInput) {
    showMessage("Preencha projeto e estado.", "error");
    return;
  }

  const stateValues = stateInput.split(",").map((item) => Number(item.trim())).filter((value) => !Number.isNaN(value));

  if (stateValues.length === 0) {
    showMessage("O estado deve conter valores numéricos separados por vírgula.", "error");
    return;
  }

  try {
    const result = await fetchJson(`/inference/projects/${projectId}/predict`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ state: stateValues, model_version_id: modelVersionId || null }),
    });

    document.getElementById("inference-result").textContent = JSON.stringify(result, null, 2);
    showMessage("Inferência realizada com sucesso.");
  } catch (err) {
    showMessage(err.message, "error");
  }
};

const renderProjects = () => {
  const projectRows = state.projects
    .map(
      (project) => `
      <tr>
        <td>${project.name}</td>
        <td>${project.template_default}</td>
        <td>${project.status}</td>
        <td>${project.job_count}</td>
        <td>${project.model_count}</td>
      </tr>
    `,
    )
    .join("");

  const templateOptions = state.templates
    .map((template) => `<option value="${template.name}">${template.name}</option>`)
    .join("");

  selectors.viewContainer.innerHTML = `
    <div class="view-card">
      <div class="card">
        <h2>Projetos</h2>
        <table class="table">
          <thead>
            <tr><th>Nome</th><th>Template</th><th>Status</th><th>Jobs</th><th>Modelos</th></tr>
          </thead>
          <tbody>${projectRows || `<tr><td colspan="5">Nenhum projeto encontrado.</td></tr>`}</tbody>
        </table>
      </div>
      <div class="card">
        <h3>Criar novo projeto</h3>
        <label>Nome</label>
        <input id="project-name" type="text" placeholder="Nome do projeto" />
        <label>Descrição</label>
        <textarea id="project-description" placeholder="Descrição do projeto"></textarea>
        <label>Template</label>
        <select id="project-template">${templateOptions}</select>
        <button onclick="createProject()">Criar projeto</button>
      </div>
    </div>
  `;
};

const renderTemplates = () => {
  const rows = state.templates
    .map(
      (template) => `
      <tr>
        <td>${template.name}</td>
        <td>${template.description || "-"}</td>
        <td>${template.metadata?.version || "-"}</td>
      </tr>
    `,
    )
    .join("");

  selectors.viewContainer.innerHTML = `
    <div class="view-card">
      <div class="card">
        <h2>Templates disponíveis</h2>
        <table class="table">
          <thead>
            <tr><th>Nome</th><th>Descrição</th><th>Versão</th></tr>
          </thead>
          <tbody>${rows || `<tr><td colspan="3">Nenhum template encontrado.</td></tr>`}</tbody>
        </table>
      </div>
    </div>
  `;
};

const renderTraining = async () => {
  await loadProjects();

  const projectOptions = state.projects
    .map((project) => `<option value="${project.id}">${project.name}</option>`)
    .join("");

  selectors.viewContainer.innerHTML = `
    <div class="view-card">
      <div class="card">
        <h2>Treinamento</h2>
        <label>Projeto</label>
        <select id="training-project">${projectOptions}</select>
        <label>Episódios</label>
        <input id="training-episodes" type="number" value="100" />
        <label>Max steps</label>
        <input id="training-max-steps" type="number" value="100" />
        <label>Action space</label>
        <input id="training-action-space" type="text" value="0,1,2,3" />
        <button id="training-create-button">Iniciar treino</button>
      </div>
    </div>
  `;

  document.getElementById("training-create-button").addEventListener("click", () => {
    const projectId = document.getElementById("training-project").value;
    const project = state.projects.find((item) => item.id === projectId);
    createTrainingJob(projectId, project?.template_default || "");
  });
};

const renderInference = () => {
  const projectOptions = state.projects
    .map((project) => `<option value="${project.id}">${project.name}</option>`)
    .join("");

  selectors.viewContainer.innerHTML = `
    <div class="view-card">
      <div class="card">
        <h2>Inferência</h2>
        <label>Projeto</label>
        <select id="inference-project">${projectOptions}</select>
        <label>Estado (valores separados por vírgula)</label>
        <textarea id="inference-state" placeholder="Ex: 0.1, 0.2, 0.3"></textarea>
        <label>Model version ID (opcional)</label>
        <input id="inference-model-version" type="text" placeholder="ID do modelo" />
        <button onclick="createInference()">Executar inferência</button>
      </div>
      <div class="card">
        <h3>Resultado</h3>
        <pre id="inference-result">Aguardando inferência...</pre>
      </div>
    </div>
  `;
};

const loadView = async (name) => {
  hideMessage();
  const buttons = document.querySelectorAll(".toolbar button[data-view]");
  buttons.forEach((button) => button.classList.toggle("active", button.dataset.view === name));

  switch (name) {
    case "templates":
      renderTemplates();
      break;
    case "training":
      await renderTraining();
      break;
    case "inference":
      renderInference();
      break;
    case "projects":
    default:
      renderProjects();
      break;
  }
};

const init = async () => {
  const savedToken = localStorage.getItem("genesys_token");
  if (savedToken) {
    setToken(savedToken);
    await loadCurrentUser();
    if (state.user) {
      await loadTemplates();
      await loadProjects();
      loadView("projects");
    }
  }

  document.getElementById("login-button").addEventListener("click", handleLogin);
  document.getElementById("register-button").addEventListener("click", handleRegister);
  document.getElementById("logout-button").addEventListener("click", handleLogout);

  document.querySelectorAll(".toolbar button[data-view]").forEach((button) => {
    button.addEventListener("click", () => loadView(button.dataset.view));
  });

  renderAuth();
};

window.createProject = createProject;
window.createInference = createInference;
window.init = init;

init();
