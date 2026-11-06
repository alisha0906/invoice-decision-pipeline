/* Alpine component driving the Live Run (SSE) and Dashboard views. */

function pipelineApp() {
  // Canonical stage order mirrors app/pipeline/runner.py.
  const STAGE_ORDER = [
    { name: "extract", label: "Extract" },
    { name: "structure_fields", label: "Structure Fields" },
    { name: "match_po", label: "Match PO" },
    { name: "decide", label: "Validate / Decide" },
  ];

  return {
    view: "live",

    // Live run state
    samples: [],
    selectedSample: "",
    running: false,
    error: "",
    currentFilename: "",
    stages: {}, // name -> { status, confidence, data }
    activeStage: null,
    decision: "",
    reasoningTrail: [],

    // Dashboard state
    runs: [],
    filter: "",
    detail: null,

    init() {
      this.loadSamples();
    },

    // ---- samples ----------------------------------------------------------
    async loadSamples() {
      try {
        const res = await fetch("/api/samples");
        this.samples = await res.json();
      } catch (e) {
        this.samples = [];
      }
    },

    // Split samples by source so the picker can group synthetic vs real.
    syntheticSamples() {
      return this.samples.filter((s) => s.source !== "real / HuggingFace");
    },
    realSamples() {
      return this.samples.filter((s) => s.source === "real / HuggingFace");
    },

    // ---- live run ---------------------------------------------------------
    resetRun(filename) {
      this.error = "";
      this.decision = "";
      this.reasoningTrail = [];
      this.stages = {};
      this.activeStage = null;
      this.currentFilename = filename || "";
    },

    onFileChange(event) {
      const file = event.target.files && event.target.files[0];
      if (file) this.uploadAndRun(file);
    },

    async uploadAndRun(file) {
      this.resetRun(file.name);
      this.running = true;
      try {
        const fd = new FormData();
        fd.append("file", file);
        const res = await fetch("/api/upload", { method: "POST", body: fd });
        if (!res.ok) throw new Error("upload failed");
        const { id } = await res.json();
        this.openStream(id);
      } catch (e) {
        this.error = "Upload failed: " + e.message;
        this.running = false;
      }
    },

    async runSample() {
      if (!this.selectedSample) return;
      this.resetRun(this.selectedSample);
      this.running = true;
      try {
        const res = await fetch("/api/run-sample/" + encodeURIComponent(this.selectedSample), {
          method: "POST",
        });
        if (!res.ok) throw new Error("sample not found");
        const { id } = await res.json();
        this.openStream(id);
      } catch (e) {
        this.error = "Could not run sample: " + e.message;
        this.running = false;
      }
    },

    openStream(id) {
      const source = new EventSource("/api/stream/" + id);
      source.onmessage = (evt) => {
        let payload;
        try {
          payload = JSON.parse(evt.data);
        } catch (e) {
          return;
        }
        if (payload.type === "stage") {
          this.stages[payload.name] = {
            status: payload.status,
            confidence: payload.confidence,
            message: payload.message,
            data: payload.data,
          };
          this.activeStage = payload.name;
        } else if (payload.type === "decision") {
          this.decision = payload.decision;
          this.reasoningTrail = payload.reasoning_trail || [];
          this.activeStage = null;
          this.running = false;
          source.close();
          if (this.view === "dashboard") this.loadRuns();
        }
      };
      source.onerror = () => {
        // Stream ends (or fails) -> stop spinner; decision event already closed it on success.
        this.running = false;
        source.close();
      };
    },

    orderedStages() {
      let seenActive = false;
      return STAGE_ORDER.map((s, i) => {
        const done = this.stages[s.name];
        let state = "pending";
        if (done) {
          state = "done";
        } else if (this.running && !seenActive) {
          state = "active";
          seenActive = true;
        }
        return {
          name: s.name,
          label: s.label,
          index: i + 1,
          state,
          status: done ? done.status : null,
          confidence: done ? done.confidence : null,
          message: done ? done.message : null,
          data: done ? done.data : null,
        };
      });
    },

    // ---- dashboard --------------------------------------------------------
    async loadRuns() {
      try {
        const url = this.filter ? "/api/runs?decision=" + this.filter : "/api/runs";
        const res = await fetch(url);
        this.runs = await res.json();
      } catch (e) {
        this.runs = [];
      }
    },

    async openDetail(id) {
      try {
        const res = await fetch("/api/runs/" + id);
        this.detail = await res.json();
      } catch (e) {
        this.detail = null;
      }
    },

    // ---- styling helpers --------------------------------------------------
    badgeClass(decision) {
      return {
        APPROVE: "badge-approve",
        REVIEW: "badge-review",
        REJECT: "badge-reject",
      }[decision] || "badge-neutral";
    },

    statusBadgeClass(status) {
      return {
        OK: "badge-approve",
        DEGRADED: "badge-review",
        SKIPPED: "badge-neutral",
        FAILED: "badge-reject",
      }[status] || "badge-neutral";
    },

    stageDotClass(st) {
      if (st.state === "done") return "bg-emerald-600 text-white";
      if (st.state === "active") return "bg-indigo-600 text-white";
      return "bg-slate-700 text-slate-300";
    },

    bannerClass(decision) {
      return {
        APPROVE: "bg-emerald-600/15 ring-emerald-500/40 text-emerald-200",
        REVIEW: "bg-amber-500/15 ring-amber-400/40 text-amber-200",
        REJECT: "bg-rose-600/15 ring-rose-500/40 text-rose-200",
      }[decision] || "bg-slate-800 ring-slate-700 text-slate-200";
    },
  };
}
