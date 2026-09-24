// Cockpit / HUD for the 3D booster simulator: DOM glass panels + 2D canvases.
// Reads the state snapshot only; never commands anything.  All captions are
// Chinese; identifiers coming from the server are translated via zh.js.
import { FDAI, LandingND } from "./instruments.js";
import { TunnelView } from "./tunnel.js";
import * as ZH from "./zh.js";

const $ = (id) => document.getElementById(id);
const PHASES = ["COAST", "BOOSTBACK", "LANDING BURN", "LANDED"];
const PHASE_COL = { STANDBY: "#7f93a6", MANUAL: "#46d6ff", COAST: "#46d6ff", BOOSTBACK: "#ff5ef0", "LANDING BURN": "#ffb43c", TERMINAL: "#5dff9a", LANDED: "#5dff9a", FALLBACK: "#ff5a4e" };

function fmt(x, d = 1) { return x === null || x === undefined || !isFinite(x) ? "--" : x.toFixed(d); }
function kN(n) { return Math.abs(n) >= 1e5 ? (n / 1000).toFixed(0) + " kN" : (n / 1000).toFixed(1) + " kN"; }
function setLamp(el, cls) { el.className = "lamp" + (cls ? " " + cls : ""); }

export class HUD {
  constructor() {
    this.fdai = new FDAI($("navball"));
    this.nd = new LandingND($("scopeCv"));
    this.thrG = $("thrG").getContext("2d");
    this.vsG = $("vsG").getContext("2d");
    this.tunnel = new TunnelView({ canvas: $("tunnel"), wrap: $("tnWrap"), modeBar: $("tnMode"), planeBar: $("tnPlane"),
      partBtn: $("tnPart"), forceBtn: $("tnForce"), foot: $("tnFoot"), tag: $("tnTag") });
    this.bannerUntil = 0;
    this.lastState = null;
    const lad = $("ladder");
    for (const p of PHASES) {
      const d = document.createElement("div");
      d.className = "step"; d.dataset.p = p;
      d.innerHTML = `<span class="dot"></span><span>${ZH.phase(p)}</span>`;
      lad.appendChild(d);
    }
    this._initDrag();
  }

  // Panels can be dragged by their title bar (the attitude cluster by its grip).
  _initDrag() {
    const KEY = "booster3d.layout.v1";
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { saved = {}; }
    const place = (el, p) => {
      el.style.left = p.x + "px"; el.style.top = p.y + "px";
      el.style.right = "auto"; el.style.bottom = "auto"; el.style.transform = "none";
    };
    const clampPos = (el, x, y) => [Math.max(0, Math.min(window.innerWidth - 40, x)), Math.max(0, Math.min(window.innerHeight - 30, y))];
    this.resetLayout = () => {
      try { localStorage.removeItem(KEY); } catch (e) { /* ignore */ }
      for (const el of document.querySelectorAll(".panel")) {
        el.style.left = el.style.top = el.style.right = el.style.bottom = el.style.transform = "";
        if (el.classList.contains("resizable")) el.style.width = el.style.height = "";
      }
      saved = {};
    };
    // Resizable panels remember their size.
    const persist = () => { try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) { /* private mode */ } };
    for (const el of document.querySelectorAll(".panel.resizable")) {
      const sz = saved[el.id + ":size"];
      if (sz) { el.style.width = sz.w + "px"; el.style.height = sz.h + "px"; }
      let t = null;
      new ResizeObserver(() => {
        if (!el.style.width && !el.style.height) return;          // default size, nothing to save
        clearTimeout(t);
        t = setTimeout(() => { saved[el.id + ":size"] = { w: el.offsetWidth, h: el.offsetHeight }; persist(); }, 300);
      }).observe(el);
    }
    for (const el of document.querySelectorAll(".panel")) {
      if (!el.id || el.id === "help") continue;
      if (saved[el.id]) place(el, saved[el.id]);
      const handle = el.querySelector(":scope > .grip") || el.querySelector(":scope > h3") || (el.id === "top" ? el : null);
      if (!handle) continue;
      handle.addEventListener("mousedown", (ev) => {
        if (ev.button !== 0) return;
        ev.preventDefault(); ev.stopPropagation();
        const r = el.getBoundingClientRect();
        const off = [ev.clientX - r.left, ev.clientY - r.top];
        place(el, { x: r.left, y: r.top });
        el.classList.add("dragging");
        const move = (e) => { const [x, y] = clampPos(el, e.clientX - off[0], e.clientY - off[1]); place(el, { x, y }); };
        const up = () => {
          window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up);
          el.classList.remove("dragging");
          const q = el.getBoundingClientRect();
          saved[el.id] = { x: q.left, y: q.top };
          persist();
        };
        window.addEventListener("mousemove", move); window.addEventListener("mouseup", up);
      });
    }
  }

  toggleHelp() { const h = $("help"); h.style.display = h.style.display === "block" ? "none" : "block"; }

  banner(text, sub, color, secs = 4) {
    const b = $("banner");
    b.innerHTML = text + (sub ? `<small>${sub}</small>` : "");
    b.style.color = color; b.style.display = "block";
    this.bannerUntil = performance.now() + secs * 1000;
  }

  // ------------------------------------------------------------------ update
  update(st, view) {
    if (!st) return;
    const phase = st.phase;
    // Top strip.
    const m = Math.floor(st.t / 60), s = st.t - 60 * m;
    $("met").textContent = `T+${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
    const alt = st.alt;
    if (alt >= 10000) { $("alt").textContent = (alt / 1000).toFixed(2); $("altU").textContent = "km"; }
    else { $("alt").textContent = alt.toFixed(alt < 100 ? 1 : 0); $("altU").textContent = "m"; }
    const vs = st.vel[2];
    $("vs").textContent = (vs >= 0 ? "+" : "") + vs.toFixed(1) + " m/s";
    $("vs").style.color = vs < -60 ? "var(--amber)" : "";
    $("tgo").textContent = st.tgo === null || st.tgo < 0 ? "--.-" : st.tgo.toFixed(1) + " s";
    const pb = $("phaseBadge");
    const shown = st.state === "CRASHED" ? "CRASHED" : st.state === "LANDED" ? "LANDED" : st.state === "ON PAD" ? "ON PAD" : phase;
    pb.textContent = ZH.phase(shown);
    const pc = st.state === "CRASHED" ? "#ff5a4e" : PHASE_COL[shown] || "#46d6ff";
    pb.style.color = pc; pb.style.borderColor = pc; pb.style.background = pc + "22";
    $("scen").textContent = ZH.scenario(st.scenario);

    // Autopilot panel.
    const visited = new Set(st.visited || []);
    for (const el of $("ladder").children) {
      const p = el.dataset.p;
      const on = st.autopilot && (p === phase || (p === "LANDED" && st.state === "LANDED"));
      el.className = "step" + (on ? " on" : visited.has(p) ? " done" : "");
    }
    $("apMode").textContent = st.autopilot ? "G-FOLD 3D" : "手动";
    $("solver").textContent = ZH.backend(st.solver.backend);
    $("solStat").textContent = ZH.solverStatus(st.solver.status) + (st.solver.locked ? " · 触地时间已锁定" : "");
    $("solves").textContent = st.solver.solves;
    $("margin").textContent = (st.solver.margin * 100).toFixed(0) + " %";
    $("cmdThr").textContent = st.cmd_throttle === null ? "--" : (st.cmd_throttle * 100).toFixed(0) + " %";
    $("gimbal").textContent = `${st.gimbal[0].toFixed(1)} / ${st.gimbal[1].toFixed(1)}°`;
    $("att").textContent = `${st.heading.toFixed(0)}° / ${st.pitch.toFixed(1)}° / ${st.roll.toFixed(0)}°`;
    setLamp($("lAP"), st.autopilot ? "on" : "");
    setLamp($("lSAS"), st.sas !== "OFF" ? "cy" : "");
    $("lSAS").textContent = st.sas === "RETROGRADE" ? "增稳 逆速" : st.sas === "STABILITY" ? "增稳 保持" : "增稳";
    setLamp($("lLEGS"), st.legs > 0.98 ? "on" : st.legs > 0.02 ? "amb" : "");
    setLamp($("lENG"), st.throttle > 0.01 ? "amb" : "");
    const rcsOn = Math.abs(st.rcs[0]) + Math.abs(st.rcs[1]) + Math.abs(st.rcs[2]) > 0.05;
    setLamp($("lRCS"), rcsOn ? "cy" : "");
    setLamp($("lCFD"), st.aero.cfd_forces ? "cy" : "");
    setLamp($("lWIND"), st.wind_on ? "on" : "");
    const fuelFrac = (st.fuel.lox + st.fuel.rp1) / (st.fuel.lox_cap + st.fuel.rp1_cap);
    setLamp($("lFUEL"), fuelFrac < 0.03 ? "red" : fuelFrac < 0.1 ? "amb" : "");
    setLamp($("lCONT"), st.feet > 0 ? "on" : "");

    // Resources.
    const f = st.fuel;
    $("loxF").style.width = (100 * f.lox / f.lox_cap).toFixed(1) + "%";
    $("rp1F").style.width = (100 * f.rp1 / f.rp1_cap).toFixed(1) + "%";
    $("loxT").textContent = `${f.lox.toFixed(0)} / ${f.lox_cap.toFixed(0)} kg`;
    $("rp1T").textContent = `${f.rp1.toFixed(0)} / ${f.rp1_cap.toFixed(0)} kg`;
    const wet = f.dry + f.lox_cap + f.rp1_cap;
    $("mbDry").style.width = (100 * f.dry / wet) + "%";
    $("mbLox").style.width = (100 * f.lox / wet) + "%";
    $("mbRp1").style.width = (100 * f.rp1 / wet) + "%";
    $("mass").textContent = `${(f.mass / 1000).toFixed(2)} t（干重 ${(f.dry / 1000).toFixed(1)} t）`;
    $("dv").textContent = f.dv.toFixed(0) + " m/s";
    $("dv").className = "v " + (f.dv < 150 ? "r" : f.dv < 400 ? "a" : "g");
    $("burn").textContent = f.burn_left === null ? "--（发动机关机）" : f.burn_left.toFixed(1) + " s";
    $("mdot").textContent = f.mdot.toFixed(1) + " kg/s";
    $("isp").textContent = f.isp.toFixed(0) + " s";
    $("twr").textContent = `${kN(f.thrust)} / ${f.twr.toFixed(2)}`;
    $("com").textContent = f.com_z.toFixed(2) + " m";

    // Aero.
    const a = st.aero;
    $("aeroSrc").textContent = a.cfd_forces ? (a.source === "CFD" ? "3D CFD 驱动" : "3D CFD（等待数据）") : "工程模型";
    $("mach").textContent = `${a.mach.toFixed(2)} / ${(a.q / 1000).toFixed(2)} kPa`;
    $("alpha").textContent = `${a.alpha.toFixed(1)}° / ${a.tail_first ? "尾部" : "头部"}`;
    $("forces").textContent = `${kN(a.drag)} / ${kN(a.normal)}`;
    const sm = a.cp_z - a.com_z;
    const stable = a.tail_first ? sm > 0 : sm < 0;
    $("margin2").textContent = `${sm >= 0 ? "+" : ""}${sm.toFixed(2)} m${a.q > 50 ? (stable ? " 静稳定" : " 静不稳定") : ""}`;
    $("margin2").className = "v " + (a.q > 50 ? (stable ? "g" : "a") : "");
    const fc = view.cfdCoef;
    $("cfdC").textContent = fc ? `${fc.cn.toFixed(2)} / ${fc.ca.toFixed(2)}` : "-";
    $("cfdCp").textContent = fc && fc.zcp !== null && fc.zcp !== undefined && a.q > 5 ? fc.zcp.toFixed(1) + " m" : "-";
    $("flowView").textContent = view.flowMode ? ["染料", "涡量", "速度"][view.volField] + "（F 关闭）" : "关（F 打开）";

    // Events.
    $("evList").innerHTML = (st.events || []).slice().reverse().map(([t, e]) => `<div class="e">${t.toFixed(1).padStart(6, " ")} s  <b>${ZH.event(e)}</b></div>`).join("");
    $("warp").textContent = st.paused ? "暂停" : st.warp + "×";

    // Speed box.
    const vrel = [st.vel[0], st.vel[1], st.vel[2]];
    $("spd").textContent = Math.hypot(...vrel).toFixed(1) + " m/s";

    // Banners on state change.
    if (this.lastState && this.lastState !== st.state) {
      const td = st.touchdown;
      if (st.state === "CRASHED") this.banner("箭体损毁", ZH.crash(st.crash), "#ff5a4e", 6);
    }
    this.lastState = st.state;
    if (performance.now() > this.bannerUntil) $("banner").style.display = "none";

    this.fdai.draw(st);
    this._throttle(st);
    this._vspeed(st);
    this.nd.draw(st);
    this.tunnel.draw(st, view, view.flowTime || 0);
  }

  // ----------------------------------------------------------------- navball
  _mkPrograde(ctx, x, y) {
    ctx.strokeStyle = "#d4ff3c"; ctx.lineWidth = 3.5;
    ctx.beginPath(); ctx.arc(x, y, 11, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x - 11, y); ctx.lineTo(x - 21, y); ctx.moveTo(x + 11, y); ctx.lineTo(x + 21, y); ctx.moveTo(x, y - 11); ctx.lineTo(x, y - 20); ctx.stroke();
    ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fillStyle = "#d4ff3c"; ctx.fill();
  }

  _mkRetro(ctx, x, y) {
    ctx.strokeStyle = "#d4ff3c"; ctx.lineWidth = 3.5;
    ctx.beginPath(); ctx.arc(x, y, 11, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x - 8, y - 8); ctx.lineTo(x + 8, y + 8); ctx.moveTo(x + 8, y - 8); ctx.lineTo(x - 8, y + 8); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x - 11, y); ctx.lineTo(x - 20, y + 8); ctx.moveTo(x + 11, y); ctx.lineTo(x + 20, y + 8); ctx.moveTo(x, y - 11); ctx.lineTo(x, y - 21); ctx.stroke();
  }

  // --------------------------------------------------------------- throttle
  _throttle(st) {
    const ctx = this.thrG, W = 92, H = 400;
    ctx.clearRect(0, 0, W, H);
    const top = 34, bot = H - 30, x0 = 30, w = 32;
    ctx.fillStyle = "#7f93a6"; ctx.font = "15px \"Microsoft YaHei\", \"PingFang SC\", sans-serif"; ctx.textAlign = "center";
    ctx.fillText("油门", W / 2, 20);
    ctx.fillStyle = "rgba(0,0,0,0.5)"; ctx.fillRect(x0, top, w, bot - top);
    const h = (bot - top) * st.throttle;
    const g = ctx.createLinearGradient(0, bot, 0, top);
    g.addColorStop(0, "#1e8f4d"); g.addColorStop(0.7, "#5dff9a"); g.addColorStop(1, "#ffb43c");
    ctx.fillStyle = g; ctx.fillRect(x0, bot - h, w, h);
    ctx.strokeStyle = "rgba(140,210,255,0.45)"; ctx.lineWidth = 1.5; ctx.strokeRect(x0, top, w, bot - top);
    ctx.fillStyle = "#9fb2c4"; ctx.font = "12px monospace"; ctx.textAlign = "right";
    for (let i = 0; i <= 10; i++) {
      const y = bot - (bot - top) * i / 10;
      ctx.fillRect(x0 - (i % 5 ? 5 : 10), y - 1, i % 5 ? 5 : 10, 2);
      if (i % 5 === 0) ctx.fillText(String(i * 10), x0 - 12, y + 4);
    }
    if (st.cmd_throttle !== null && st.autopilot) {
      const y = bot - (bot - top) * Math.min(1, st.cmd_throttle);
      ctx.fillStyle = "#ff5ef0";
      ctx.beginPath(); ctx.moveTo(x0 + w + 2, y); ctx.lineTo(x0 + w + 14, y - 8); ctx.lineTo(x0 + w + 14, y + 8); ctx.closePath(); ctx.fill();
    }
    ctx.fillStyle = "#fff"; ctx.font = "bold 16px monospace"; ctx.textAlign = "center";
    ctx.fillText((st.throttle * 100).toFixed(0) + "%", W / 2 + 4, H - 8);
  }

  // ------------------------------------------------------------ vert. speed
  _vspeed(st) {
    const ctx = this.vsG, W = 92, H = 400;
    ctx.clearRect(0, 0, W, H);
    const top = 34, bot = H - 30, x0 = 30, w = 32, mid = (top + bot) / 2;
    ctx.fillStyle = "#7f93a6"; ctx.font = "15px \"Microsoft YaHei\", \"PingFang SC\", sans-serif"; ctx.textAlign = "center";
    ctx.fillText("升降", W / 2, 20);
    ctx.fillStyle = "rgba(0,0,0,0.5)"; ctx.fillRect(x0, top, w, bot - top);
    // Log-ish scale: +-300 m/s.
    const map = (v) => mid - Math.sign(v) * Math.log10(1 + Math.abs(v)) / Math.log10(301) * (mid - top);
    for (const v of [-300, -100, -30, -10, -3, 0, 3, 10, 30, 100, 300]) {
      const y = map(v);
      ctx.fillStyle = "#9fb2c4"; ctx.fillRect(x0 + w, y - 1, 7, 2);
      ctx.font = "11px monospace"; ctx.textAlign = "left";
      if (Math.abs(v) !== 3) ctx.fillText(String(Math.abs(v)), x0 + w + 9, y + 4);
    }
    const y = map(st.vel[2]);
    ctx.fillStyle = st.vel[2] < 0 ? "#ffb43c" : "#46d6ff";
    ctx.fillRect(x0 + 3, Math.min(y, mid), w - 6, Math.abs(y - mid));
    ctx.fillStyle = "#fff"; ctx.fillRect(x0, mid - 1, w, 2);
    ctx.strokeStyle = "rgba(140,210,255,0.45)"; ctx.lineWidth = 1.5; ctx.strokeRect(x0, top, w, bot - top);
    ctx.fillStyle = "#fff"; ctx.font = "bold 15px monospace"; ctx.textAlign = "center";
    ctx.fillText(st.vel[2].toFixed(1), W / 2 + 4, H - 8);
  }

}
