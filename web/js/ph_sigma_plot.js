// ph_sigma_plot.js v984 -- ONE place that draws a sigma curve.
//
// v976 drew the Sigma List's curve inside ph_sigma_list.js. v984 gives the
// Sigma Curve and the Dual Sigma Curve the same picture; a second copy of the
// drawing would drift from the first (a different axis, a different dot size,
// a refusal shown differently). So the drawing lives here and every sigma
// node calls it. This file never CALCULATES a curve -- it draws what the
// backend's preview route answered, which is the run's own arithmetic.
//
// spec = {
//   state:   null (no answer yet / unreachable) | {error: "..."} | undefined
//   header:  one line of text above the plot
//   warn:    optional second line in amber (e.g. "sigma_min >= sigma_max -- swapped")
//   series:  [{ sigmas:[...], color:"#8fff8f",
//               from:0, to:N   -- the knots this series' sampler really uses;
//                                 drawn bold with dots. Outside: faint line.
//               label:"HIGH" }]
//   marker:  optional { index, text } -- a vertical line at a knot (the split)
// }

export const PLOT_MIN_H = 120;          // a 2-point grid still deserves a picture
export const PLOT_MAX_H = 260;
export const PLOT_PER_POINT = 6;        // 20 points is a different curve than 8
export const PLOT_PAD = 8;

// The plot's height follows the number of knots: the curve adapts to its
// points instead of squeezing 60 dots into the space 8 get.
export function plotHeightFor(nPoints) {
    const n = Number.isFinite(nPoints) && nPoints > 0 ? nPoints : 9;
    return Math.max(PLOT_MIN_H, Math.min(PLOT_MAX_H, PLOT_MIN_H + (n - 9) * PLOT_PER_POINT));
}

export function dotRadius(nPoints) {
    // With many points the dots shrink rather than merge into a caterpillar,
    // so 20 steps still reads as 20 steps.
    return nPoints > 24 ? 1.5 : nPoints > 14 ? 2 : 2.5;
}

function wrapText(ctx, text, x, y, maxW, maxY, lineH) {
    const words = String(text).split(" ");
    let line = "", ty = y;
    for (const word of words) {
        const test = line ? line + " " + word : word;
        if (ctx.measureText(test).width > maxW && line) {
            ctx.fillText(line, x, ty); line = word; ty += lineH;
            if (ty > maxY) return;
        } else line = test;
    }
    if (line && ty <= maxY) ctx.fillText(line, x, ty);
}

function fmt(v) {
    const a = Math.abs(v);
    return a >= 100 ? v.toFixed(1) : a >= 10 ? v.toFixed(2) : v.toFixed(3);
}

export function drawSigmaPlot(ctx, widgetWidth, y, height, spec) {
    const x0 = PLOT_PAD;
    const y0 = y;
    const w = widgetWidth - PLOT_PAD * 2;
    const h = height - 4;
    if (w < 40 || h < 24) return;

    ctx.save();
    ctx.fillStyle = "#181818";
    ctx.fillRect(x0, y0, w, h);

    const st = spec ? spec.state : null;
    if (!spec || st === null) {
        ctx.fillStyle = "#777";
        ctx.font = "11px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("no preview", x0 + w / 2, y0 + h / 2);
        ctx.restore();
        return;
    }
    if (st && st.error !== undefined) {           // the refusal the run raises
        ctx.fillStyle = "#ff6b6b";
        ctx.font = "11px sans-serif";
        ctx.textAlign = "left";
        wrapText(ctx, st.error || "invalid", x0 + 6, y0 + 16, w - 12, y0 + h - 4, 13);
        ctx.restore();
        return;
    }

    const series = (spec.series || []).filter((s) => s && s.sigmas && s.sigmas.length >= 2);
    if (!series.length) { ctx.restore(); return; }

    ctx.font = "10px sans-serif";
    ctx.textAlign = "left";
    let top = y0 + 13;
    if (spec.header) {
        ctx.fillStyle = "#9aa";
        ctx.fillText(spec.header, x0 + 6, top);
        top += 12;
    }
    if (spec.warn) {
        ctx.fillStyle = "#ffb347";
        ctx.fillText(spec.warn, x0 + 6, top);
        top += 12;
    }

    const n = Math.max(...series.map((s) => s.sigmas.length));
    const max = Math.max(...series.map((s) => Math.max(...s.sigmas)));
    const min = 0;
    const lblW = 34;                               // room for the value axis
    const gx = x0 + lblW, gw = w - lblW - 6;
    const gy = top - 3, gh = y0 + h - 6 - gy;
    if (gh < 12) { ctx.restore(); return; }
    const px = (i) => gx + gw * (i / (n - 1));
    const py = (v) => gy + gh - gh * ((v - min) / (max - min || 1));

    ctx.strokeStyle = "#2a2a2a";                   // one rule per step boundary
    ctx.lineWidth = 1;
    for (let i = 0; i < n; i++) {
        ctx.beginPath();
        ctx.moveTo(px(i), gy);
        ctx.lineTo(px(i), gy + gh);
        ctx.stroke();
    }
    ctx.strokeStyle = "#3a3a3a";
    ctx.beginPath();
    ctx.moveTo(gx, gy + gh);
    ctx.lineTo(gx + gw, gy + gh);
    ctx.stroke();

    ctx.fillStyle = "#778";                        // value axis: top and 0
    ctx.textAlign = "right";
    ctx.fillText(fmt(max), gx - 4, gy + 8);
    ctx.fillText("0", gx - 4, gy + gh);

    const r = dotRadius(n);
    for (const s of series) {
        const sig = s.sigmas;
        const last = sig.length - 1;
        const from = Math.max(0, Math.min(last, s.from ?? 0));
        const to = Math.max(from, Math.min(last, s.to ?? last));
        const path = (a, b) => {
            ctx.beginPath();
            for (let i = a; i <= b; i++) {
                if (i === a) ctx.moveTo(px(i), py(sig[i]));
                else ctx.lineTo(px(i), py(sig[i]));
            }
            ctx.stroke();
        };
        // the whole curve, faint -- what the list holds
        if (from > 0 || to < last) {
            ctx.globalAlpha = 0.3;
            ctx.strokeStyle = s.color;
            ctx.lineWidth = 1;
            path(0, last);
            ctx.globalAlpha = 1;
        }
        // the part the sampler runs, bold with a dot per knot
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 1.8;
        path(from, to);
        ctx.fillStyle = s.color;
        for (let i = from; i <= to; i++) {
            ctx.beginPath();
            ctx.arc(px(i), py(sig[i]), r, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    const m = spec.marker;
    if (m && Number.isFinite(m.index) && m.index > 0 && m.index < n - 1) {
        const mx = px(m.index);
        ctx.strokeStyle = "#ddd";
        ctx.lineWidth = 1;
        if (ctx.setLineDash) ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(mx, gy);
        ctx.lineTo(mx, gy + gh);
        ctx.stroke();
        if (ctx.setLineDash) ctx.setLineDash([]);
        if (m.text) {
            ctx.fillStyle = "#ddd";
            ctx.textAlign = mx > gx + gw * 0.6 ? "right" : "left";
            // at the TOP of the rule: the tail of every schedule runs along the
            // bottom, so a label there sits on the curve it explains
            ctx.fillText(m.text, mx + (ctx.textAlign === "right" ? -4 : 4), gy + 9);
        }
    }

    // a small legend for multi-series plots
    const named = series.filter((s) => s.label);
    if (named.length > 1) {
        ctx.textAlign = "right";
        let ly = gy + 22;
        for (const s of named) {
            ctx.fillStyle = s.color;
            ctx.fillText(s.label, gx + gw - 2, ly);
            ly += 11;
        }
    }
    ctx.restore();
}
